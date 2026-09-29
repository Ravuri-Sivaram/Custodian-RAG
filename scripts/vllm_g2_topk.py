#!/usr/bin/env python3
"""G2 cross-build/cross-query test: vLLM-encoded queries vs. officially-encoded queries, both run
against **the same real store** (built with the official encoder), comparing whether top-k flips
(docs/VLLM_PLAN.md §4 G2). This is the **production gate** that decides whether vLLM can be Plan A
-- G1's element-wise micro-drift (cosine 0.9997) does not by itself imply top-k stays unchanged
(HNSW approximation + RRF rank fusion can amplify tiny differences between near-duplicate chunks).

**Isolation design**: vLLM only produces the **query vector** (that's its only job here); the
retrieval pipeline (dense+BM25+RRF+ACL) **runs exactly once, in custodian**. The only difference
between the two sides is the source of the dense vector (official encode_query vs. vLLM's budget
encoding). Same sparse vector, same store, same RRF -> this cleanly isolates the effect of the
encoder alone.

**Time-sliced run** (to avoid double-loading the 8B model and OOMing, same as the earlier probe):
  docker compose --env-file .env.compose stop inference          # free the GPU
  # 1) vLLM produces 88 query vectors (vllm env; truncate 4096->1024+renorm to match the store's dim)
  conda activate vllm && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python scripts/vllm_g2_topk.py --step vllm --out /tmp/vllm_g2
  # 2) custodian produces the official vectors + runs retrieval on both sides + compares (custodian;
  #    opens the embedded store at ~/rag_real)
  conda activate custodian && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
      python scripts/vllm_g2_topk.py --step retrieve --out /tmp/vllm_g2
  docker compose --env-file .env.compose start inference          # restore the stack
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_MODEL = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
EMBED_DB = os.path.expanduser("~/rag_real")            # embedded store (built with the official encoder, 7652 points; the server was migrated from it)
COLLECTION = "real"
TENANT = "demo"                                         # store ACL: tenant=demo/visibility=public
DENSE_DIM = 1024                                        # production MRL truncation dim; the store is 1024-dim, so query vectors must match
QUERY_INSTRUCTION = "Retrieve relevant documents for the query."
TOPK = 10
GOLD = os.path.join(REPO, "eval", "gold.jsonl")


def _load_queries() -> list[str]:
    qs = []
    with open(GOLD, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line)["query"])
    return qs


def _mrl_np(full: np.ndarray, d: int = DENSE_DIM) -> np.ndarray:
    """Full 4096-dim -> first d dims + L2 renorm (mirrors embedder.remote.RemoteDense._mrl_np, to match the store's dim)."""
    v = full[:d]
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def step_vllm(out_dir: str) -> None:
    """vllm env: 88 queries -> vLLM pooling vectors -> truncate to 1024 + renorm -> save as npz. Only depends on vllm+numpy."""
    import unicodedata

    from vllm import LLM
    queries = _load_queries()

    def sys_instr(instr: str) -> str:
        instr = instr.strip()
        return instr if unicodedata.category(instr[-1]).startswith("P") else instr + "."

    print(f"[vllm] Loading {EMB_MODEL} (pooling, max_model_len=8192)…", flush=True)
    llm = LLM(model=EMB_MODEL, runner="pooling", dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, max_model_len=8192, gpu_memory_utilization=0.90)
    tok = llm.get_tokenizer()
    prompts = []
    for q in queries:
        conv = [{"role": "system", "content": [{"type": "text", "text": sys_instr(QUERY_INSTRUCTION)}]},
                {"role": "user", "content": [{"type": "text", "text": q}]}]
        prompts.append({"prompt": tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)})
    outs = llm.embed(prompts)
    vecs = np.stack([_mrl_np(np.asarray(o.outputs.embedding, dtype=np.float32)) for o in outs])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "vllm_qvecs.npz"), vecs=vecs)   # only stores numeric arrays, no pickle on the read side
    print(f"[vllm] Saved {vecs.shape} query vectors (1024-dim, already renormed) -> {out_dir}/vllm_qvecs.npz", flush=True)


def step_retrieve(out_dir: str) -> None:
    """custodian: runs both the official vectors and the vLLM vectors against the same embedded store, compares top-k. Retrieval pipeline runs only once here, purely isolating the effect of the encoder."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from embedder.config import EmbedConfig
    from embedder.dense import Dense
    from embedder.sparse import query_sparse
    from embedder.store import Store
    from embedder.types import User

    p = os.path.join(out_dir, "vllm_qvecs.npz")
    if not os.path.exists(p):
        raise SystemExit(f"vLLM query vectors not found at {p}; run --step vllm in the vllm env first.")
    vllm_vecs = np.load(p)["vecs"]

    queries = _load_queries()
    if len(queries) != len(vllm_vecs):
        raise SystemExit(f"Query count {len(queries)} != vLLM vector count {len(vllm_vecs)}; re-run --step vllm.")

    cfg = EmbedConfig(qdrant_path=os.path.join(EMBED_DB, "qdrant"),
                      sidecar_dir=os.path.join(EMBED_DB, "sidecar"),
                      collection=COLLECTION, dense_dim=DENSE_DIM)
    print(f"[retrieve] Opening embedded store {cfg.qdrant_path} + the official encoder…", flush=True)
    store = Store(cfg)
    dense = Dense(cfg)                                  # local official Qwen3VLEmbedder (GPU)
    user = User(tenant=TENANT, principals=[])

    def topk(dvec: np.ndarray, q: str) -> list[str]:
        sparse = query_sparse(q, cfg.stopwords)         # same sparse vector on both sides (CPU, regex-tokenized), isolating the dense-vector difference
        hits = store.hybrid_search(dvec.tolist(), sparse, user, top_k=TOPK)
        return [h.chunk_id for h in hits]

    exact_seq = exact_set = top1 = 0
    jaccard_sum = 0.0
    diverged = []
    for i, q in enumerate(queries):
        off = topk(dense.encode_query(q), q)            # official (the store was built with it, so it's the authoritative baseline)
        vll = topk(vllm_vecs[i], q)                     # vLLM's budget vector
        if not off:                                     # empty recall (shouldn't happen) -- skip from the stats
            continue
        if off == vll:
            exact_seq += 1
        if set(off) == set(vll):
            exact_set += 1
        if off[0] == (vll[0] if vll else None):
            top1 += 1
        inter = len(set(off) & set(vll))
        jaccard_sum += inter / len(set(off) | set(vll))
        if off != vll:
            # record the first divergent rank (first position in top-k where the two differ)
            fd = next((r for r in range(min(len(off), len(vll))) if off[r] != vll[r]), min(len(off), len(vll)))
            diverged.append((i, fd, q[:44]))

    n = len(queries)
    print("\n=== G2 cross-build/cross-query top-%d (store built with the official encoder, official vs vLLM query encoding) ===" % TOPK)
    print(f"  sample count           : {n}")
    print(f"  top-1 match            : {top1}/{n}  ({100*top1/n:.1f}%)")
    print(f"  top-{TOPK} set match     : {exact_set}/{n}  ({100*exact_set/n:.1f}%)   (order may differ)")
    print(f"  top-{TOPK} order match   : {exact_seq}/{n}  ({100*exact_seq/n:.1f}%)")
    print(f"  mean Jaccard@{TOPK}      : {jaccard_sum/n:.4f}")
    print(f"\n  divergent samples       : {len(diverged)}; distribution of first-divergence rank (rank: count):")
    from collections import Counter
    for rank, cnt in sorted(Counter(fd for _, fd, _ in diverged).items()):
        print(f"    rank {rank}: {cnt}")
    print("\n  divergent samples (first 8, with first-divergence rank):")
    for i, fd, qt in diverged[:8]:
        print(f"    q#{i} first divergence@rank{fd}  «{qt}»")

    # Verdict thresholds (honest, not overconfident): stable top-1 + high Jaccard = strong evidence
    # that vLLM can be Plan A; fully matching order is the strictest bar.
    j = jaccard_sum / n
    print("\n--- Verdict ---")
    if top1 == n and j >= 0.95:
        print("PASS (strong): G2 strongly passes: top-1 fully matches + Jaccard>=0.95 -- micro-drift does not flip "
              "effective ranking, vLLM can be Plan A (small order jitter is mostly among near-duplicate tail chunks, harmless).")
    elif top1 >= 0.95 * n and j >= 0.85:
        print("PASS (marginal): G2 basically passes: top-1 highly consistent but with some tail jitter -- acceptable "
              "for production (top-k serves as an evidence pool, tail-order jitter doesn't change the answer); "
              "for a stricter check, align transformers versions or rebuild the whole store with vLLM.")
    else:
        print("FAIL: G2 fails: top-1 or Jaccard dropped too much -- the micro-drift genuinely flips recall. The "
              "existing store should not be queried directly with vLLM; align transformers versions and re-test, "
              "or rebuild the whole store with vLLM (see VLLM_PLAN §8).")


def main() -> None:
    ap = argparse.ArgumentParser(description="G2 cross-build/cross-query top-k stability (vLLM vs official query encoding)")
    ap.add_argument("--step", required=True, choices=["vllm", "retrieve"])
    ap.add_argument("--out", default="/tmp/vllm_g2")
    args = ap.parse_args()
    (step_vllm if args.step == "vllm" else step_retrieve)(args.out)


if __name__ == "__main__":
    main()
