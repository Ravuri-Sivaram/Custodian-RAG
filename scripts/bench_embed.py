#!/usr/bin/env python3
"""G4 throughput gate: concurrent load test on the embed endpoint, comparing FastAPI (serial
gpu_lock) vs vLLM (continuous batching). Decides the number that determines "performs better,
promote to Plan A" (docs/VLLM_PLAN.md §4 G4).

**Fair-comparison rules**: each request = 1 query -> 1 vector (simulating query load). Both
backends have the **chat template pre-applied** (so the client never becomes the bottleneck) --
only GPU forward-pass throughput is compared. Runs across a set of concurrency levels, reporting
QPS (successes / wall clock) + p50/p95 latency + error count (FastAPI backpressure 503s count too).

Run (FastAPI runs in compose; vLLM needs `vllm serve --runner pooling --max-model-len 8192` started
on :8000):
  # FastAPI (already published on :8900)
  python scripts/bench_embed.py --mode fastapi --url http://localhost:8900 --conc 1 8 16 32 --n 200
  # vLLM (:8000)
  python scripts/bench_embed.py --mode vllm    --url http://localhost:8000 --conc 1 8 16 32 --n 200 \
      --model ~/models/Qwen3-VL-Embedding-8B
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import unicodedata

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(REPO, "eval", "gold.jsonl")
QUERY_INSTRUCTION = "Retrieve relevant documents for the query."


def _queries(n_cycle: int) -> list[str]:
    qs = []
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line)["query"])
    # Cycle through to fill n_cycle entries (gold only has 88; if the load test needs more requests, reuse them in a loop)
    return [qs[i % len(qs)] for i in range(n_cycle)]


def _templated(model: str, texts: list[str]) -> list[str]:
    """vLLM mode: the client pre-applies the chat template (system=instruction, user=text,
    add_generation_prompt). Uses the same recipe as the official wrapper, ensuring the token
    sequence sent to vLLM matches what FastAPI's server does internally (fair comparison + semantically equivalent)."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    instr = QUERY_INSTRUCTION.strip()
    instr = instr if unicodedata.category(instr[-1]).startswith("P") else instr + "."
    out = []
    for t in texts:
        conv = [{"role": "system", "content": [{"type": "text", "text": instr}]},
                {"role": "user", "content": [{"type": "text", "text": t}]}]
        out.append(proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
    return out


def _pctl(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


async def _bench_one(url: str, mode: str, model: str, payloads: list[dict], conc: int) -> dict:
    """Fires len(payloads) requests with a concurrency cap of conc. Returns QPS/p50/p95/errors."""
    sem = asyncio.Semaphore(conc)
    lat: list[float] = []
    errors = 0
    path = "/embed" if mode == "fastapi" else "/v1/embeddings"

    async with httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        async def one(pl: dict):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(path, json=pl)
                    if r.status_code == 200:
                        lat.append(time.perf_counter() - t0)
                    else:
                        errors += 1
                except Exception:
                    errors += 1
        t_start = time.perf_counter()
        await asyncio.gather(*(one(pl) for pl in payloads))
        wall = time.perf_counter() - t_start

    ok = len(lat)
    return {"conc": conc, "ok": ok, "errors": errors, "wall_s": round(wall, 2),
            "qps": round(ok / wall, 1) if wall > 0 else 0,
            "p50_ms": round(_pctl(lat, 50) * 1000, 1), "p95_ms": round(_pctl(lat, 95) * 1000, 1)}


async def main_async(args) -> None:
    raw = _queries(args.n)
    if args.mode == "fastapi":
        payloads = [{"texts": [q], "instruction": QUERY_INSTRUCTION} for q in raw]
    else:
        print(f"[bench] Pre-applying chat template to {len(raw)} queries (vLLM fair-comparison rules)…", flush=True)
        templated = _templated(args.model, raw)
        payloads = [{"input": s, "model": args.model, "encoding_format": "float"} for s in templated]

    print(f"\n=== G4 throughput: {args.mode}  {args.url}  (n={args.n}/level) ===")
    print(f"{'conc':>5} {'qps':>8} {'p50_ms':>9} {'p95_ms':>9} {'ok':>5} {'err':>5} {'wall_s':>7}")
    results = []
    for c in args.conc:
        # warm up once (rules out lazy-init/JIT effects on the first request)
        await _bench_one(args.url, args.mode, args.model, payloads[: min(4, len(payloads))], min(4, c))
        r = await _bench_one(args.url, args.mode, args.model, payloads, c)
        results.append(r)
        print(f"{r['conc']:>5} {r['qps']:>8} {r['p50_ms']:>9} {r['p95_ms']:>9} {r['ok']:>5} {r['errors']:>5} {r['wall_s']:>7}", flush=True)
    peak = max(results, key=lambda x: x["qps"])
    print(f"\nPeak QPS={peak['qps']} @ conc={peak['conc']}  (p95={peak['p95_ms']}ms)")


def main() -> None:
    ap = argparse.ArgumentParser(description="G4 embed throughput load test (FastAPI vs vLLM)")
    ap.add_argument("--mode", required=True, choices=["fastapi", "vllm"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--n", type=int, default=200, help="requests to fire per concurrency level")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-VL-Embedding-8B"),
                    help="vLLM mode: model path (used both for templating and the payload's model field)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
