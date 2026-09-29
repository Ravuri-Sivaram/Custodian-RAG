"""Concurrent load test: fires a mixed workload at the live daemon, reports p50/p95/max/throughput/error count.

Usage (WSL custodian):
  python scripts/bench.py --key <API_KEY> --clients 5 --n 10 [--ask-ratio 0.1]

Design: serialization via a global lock is a known part of the architecture (retrieval stage
serialized, LLM stage parallel). The point of this load test isn't to "prove it's fast" — it's to
get an **honest queuing curve** (how long each request waits under team-level concurrency), to
feed into OPERATIONS for capacity planning.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time

import httpx

QUERIES = [
    "Netflix 2015 total revenues", "Chevron net income 2021", "Amazon 2017 cash and equivalents",
    "what does DDoS attack protection cover", "semiconductor equipment sector performance", "TSMC capital expenditure", "IBM 2020 operating income",
    "gender tagging NMT experiments", "ProgramFC HOVER results", "bank account opening process in the international student guide",
    "ThinkPad T480 battery replacement", "IRS form 1040 filing requirements", "policy vendor onboarding steps",
    "UnitedHealth 2020 revenues", "memory chip price increases", "GPT-4V input modes", "LVMH 2021 revenue",
]


def worker(base, key, n, ask_ratio, out, wid):
    c = httpx.Client(base_url=base, timeout=300, headers={"X-API-Key": key} if key else {})
    rng = random.Random(wid)
    for i in range(n):
        q = rng.choice(QUERIES)
        do_ask = rng.random() < ask_ratio
        t0 = time.time()
        try:
            if do_ask:
                r = c.post("/v1/ask", json={"query": q, "top_k": 4})
            else:
                r = c.post("/v1/retrieve", json={"query": q, "top_k": 6})
            ok = r.status_code == 200 and r.json().get("status") in ("ok", "empty")
        except Exception:
            ok = False
        out.append({"ep": "ask" if do_ask else "retrieve", "ms": (time.time() - t0) * 1000, "ok": ok})


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return round(vals[min(len(vals) - 1, round(p / 100 * (len(vals) - 1)))], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8787")
    ap.add_argument("--key", default="")
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--n", type=int, default=10, help="requests per client")
    ap.add_argument("--ask-ratio", type=float, default=0.0, dest="ask_ratio")
    args = ap.parse_args()

    out: list = []
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(args.url, args.key, args.n, args.ask_ratio, out, w))
               for w in range(args.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    total = len(out)
    errs = sum(1 for r in out if not r["ok"])
    print(f"\nclients={args.clients} n/client={args.n} ask_ratio={args.ask_ratio} "
          f"total_requests={total} errors={errs} wall_clock={wall:.1f}s throughput={total / wall:.2f} req/s")
    for ep in ("retrieve", "ask"):
        ms = [r["ms"] for r in out if r["ep"] == ep]
        if ms:
            print(f"  {ep:9s} n={len(ms):3d}  p50={pct(ms, 50)}ms  p95={pct(ms, 95)}ms  max={pct(ms, 100)}ms")
    print(json.dumps({"clients": args.clients, "total": total, "errors": errs,
                      "wall_s": round(wall, 1), "rps": round(total / wall, 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
