"""GPU equivalence end-to-end test (phase B go/no-go, docs/SCALE_OUT.md §5-B). Proves that
local index-building vectors == remote query vectors.

**Run time-sliced** (to avoid OOM: inference's 2x8B already takes ~33GB on the 4090, and local
Dense needs another 16GB -- loading both at once blows out VRAM):
  1) start the inference service:  conda activate custodian && python -m embedder.inference_server   # 127.0.0.1:8900
  2) save remote output:           python scripts/equiv_gpu.py --step remote   # remote encode/query/rerank -> /tmp/equiv_remote.npz
  3) stop the inference service (frees the GPU)
  4) compare locally:              python scripts/equiv_gpu.py --step local    # local vs saved output -> E1/E3 GO/NO-GO

Verdict thresholds: normalized encode/query cosine > 0.9999 (tolerating minor bf16-local vs
fp32-remote rounding, same bar used for the vLLM migration gate); rerank maxdiff < 1e-3.
Measured (2026-07): encode cosine=1.0000000 / maxdiff 2.98e-08 / query cosine=1.0 / rerank
maxdiff 0.00 -> E1/E3 GO.
Note: E2 (cross-build/cross-query top-k) is a **separate, not-yet-done item** -- E1's element-wise
equivalence does not imply that top-k order is unchanged under hybrid RRF+HNSW (near-duplicate
chunks can differ by less than maxdiff and still swap ranks); that needs a separate real-query
build/query comparison, see §5-B."""
import argparse

import numpy as np

from embedder.config import EmbedConfig

URL = "http://127.0.0.1:8900"
ARCHIVE = "/tmp/equiv_remote.npz"
D = 1024
TEXTS = [
    "What was Netflix's revenue in 2015",
    "The company reported strong quarterly revenue growth in Q3.",
    "Table comparing third-quarter net profit with the same period last year",
    "mixed script test 123 !@#$% special chars",
    "a",
]
QUERY = "Netflix revenue 2015"
DOCS = [
    "Netflix 2015 full-year revenue was $6.78 billion, up from prior year.",
    "cats are cute animals that sleep a lot",
    "The quarterly report shows steady subscriber growth.",
]


def remote_step():
    from embedder.remote import RemoteDense, RemoteReranker
    rd = RemoteDense(EmbedConfig(inference_url=URL, dense_dim=D))
    v = rd.encode_text(TEXTS)
    q = rd.encode_query(QUERY)
    rr = RemoteReranker(EmbedConfig(inference_url=URL))
    s = np.array(rr.score(QUERY, DOCS), dtype=np.float64)
    np.savez(ARCHIVE, v=v, q=q, s=s)
    print(f"remote output saved -> {ARCHIVE}: encode{v.shape} query{q.shape} rerank={s.tolist()}")


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def local_step():
    from embedder.dense import Dense
    from embedder.rerank import Reranker
    d = Dense(EmbedConfig(dense_dim=D))
    v_local, q_local = d.encode_text(TEXTS), d.encode_query(QUERY)
    s_local = np.array(Reranker(EmbedConfig()).score(QUERY, DOCS), dtype=np.float64)
    a = np.load(ARCHIVE)
    v_r, q_r, s_r = a["v"], a["q"], a["s"]

    e1 = [_cos(v_local[i], v_r[i]) for i in range(len(TEXTS))]
    print("[norm] local :", [f"{float(np.linalg.norm(v_local[i])):.5f}" for i in range(len(TEXTS))])
    print("[norm] remote:", [f"{float(np.linalg.norm(v_r[i])):.5f}" for i in range(len(TEXTS))])
    print("[E1 encode] cosine:", [f"{c:.7f}" for c in e1], " maxdiff=", f"{float(np.abs(v_local - v_r).max()):.2e}")
    print(f"[E1 query] cosine={_cos(q_local, q_r):.7f}")
    e3_max = float(np.abs(s_local - s_r).max())
    print(f"[E3 rerank] local={s_local.tolist()} remote={s_r.tolist()} maxdiff={e3_max:.2e}")
    E1 = all(c > 0.9999 for c in e1) and _cos(q_local, q_r) > 0.9999
    print(f"\nE1 {'PASS GO' if E1 else 'FAIL NO-GO'} (encode equivalence)   "
          f"E3 {'PASS GO' if e3_max < 1e-3 else 'FAIL NO-GO'} (rerank equivalence)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", choices=["remote", "local"], required=True)
    args = ap.parse_args()
    (remote_step if args.step == "remote" else local_step)()
