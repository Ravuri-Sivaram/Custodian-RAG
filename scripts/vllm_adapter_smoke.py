#!/usr/bin/env python3
"""Phase 1 end-to-end smoke test: custodian (remote backend) -> vLLM adapter -> vLLM serve,
querying the real store and getting real hits. Proves that "not a single line of the application
layer changes -- pointing CUSTODIAN_INFERENCE_URL at the adapter is enough to switch from FastAPI
to vLLM" (docs/VLLM_PLAN.md Phase 1).

Prerequisites (already started within the same WSL invocation): vllm serve on :8000 + the adapter
on :8900. This script (custodian, 0 GPU -- remote mode) builds a Retriever (inference_url=adapter)
and queries the real embedded store. Run:
  CUSTODIAN_INFERENCE_URL=http://localhost:8900 python scripts/vllm_adapter_smoke.py
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from embedder.config import EmbedConfig      # noqa: E402
from embedder.retriever import Retriever      # noqa: E402
from embedder.types import User              # noqa: E402

ADAPTER = os.environ.get("CUSTODIAN_INFERENCE_URL", "http://localhost:8900")
EMBED_DB = os.path.expanduser("~/rag_real")
QUERIES = ["What was IBM's total revenue growth?", "Netflix 2015 revenue and net profit",
           "unsupervised cross-lingual transfer"]


def main() -> None:
    cfg = EmbedConfig(qdrant_path=os.path.join(EMBED_DB, "qdrant"),
                      sidecar_dir=os.path.join(EMBED_DB, "sidecar"),
                      collection="real", dense_dim=1024,
                      inference_url=ADAPTER)          # non-empty -> RemoteDense (doesn't load a local GPU model)
    r = Retriever(cfg)
    assert type(r.dense).__name__ == "RemoteDense", f"expected RemoteDense, got {type(r.dense).__name__}"
    print(f"[smoke] Retriever.dense = {type(r.dense).__name__}  (inference_url={ADAPTER})", flush=True)
    user = User(tenant="demo", principals=[])
    ok = 0
    for q in QUERIES:
        hits = r.store.hybrid_search(r.dense.encode_query(q).tolist(),
                                     __import__("embedder.sparse", fromlist=["query_sparse"]).query_sparse(q, cfg.stopwords),
                                     user, top_k=3)
        cids = [h.chunk_id for h in hits]
        real = bool(cids and any(c for c in cids))
        ok += real
        print(f"[smoke] «{q[:40]}» -> {len(cids)} hits: {cids[:2]}{' …' if len(cids)>2 else ''}", flush=True)
    print(f"\n{'PASS' if ok==len(QUERIES) else 'FAIL'} Phase 1 end-to-end: {ok}/{len(QUERIES)} queries got real hits via the vLLM adapter "
          f"({'application layer unchanged, vLLM backend works' if ok==len(QUERIES) else 'some query got empty recall, check the adapter/template'})", flush=True)


if __name__ == "__main__":
    main()
