"""embedder configuration. This component runs in the WSL `custodian` conda environment (GPU=4090,
pinned by name -- see dense.py)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# The sidecar JSON schema version. Stamped on write, checked on read: a mismatch (including a
# missing version at all) raises and requires re-running index_document.
# **Bump this whenever the sidecar's structure changes** (Element/Section fields, the acl_index
# encoding, etc.), or an old sidecar would be silently deserialized into wrong data -- at best,
# small-to-big assembly comes back empty; at worst, it misaligns by position. This turns a silent
# failure into a loud one.
SIDECAR_VERSION = 1


@dataclass
class EmbedConfig:
    # --- dense: Qwen3-VL-Embedding-8B (local, downloaded via modelscope to ~/models) ---
    dense_model_path: str = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
    dense_dim: int = 1024                  # The MRL truncation dimension; 1024 is a 4x storage saving starting point, tune further against measured recall (up to 4096)
    query_instruction: str = "Retrieve relevant documents for the query."
    gpu_name_must_contain: str = "4090"    # FASTEST_FIRST numbering can shift -> assert by name to pin the 4090 (never a 5070)

    # --- sparse: BM25 (regex-based Telugu/alphanumeric tokenization) ---
    stopwords: frozenset[str] = field(default_factory=frozenset)   # Chinese stopwords can be injected here

    # --- Qdrant (embedded to start; switch to server mode once the scale calls for it) ---
    qdrant_path: str = os.path.expanduser("~/qdrant_data")
    qdrant_url: str = ""           # Non-empty = server mode (the real Phase D multi-replica switch; takes priority over qdrant_path -- see store.py's three-way branch)
    collection: str = "rag_chunks"

    # --- sidecar: stores elements+sections+acl_index per doc_id (required for assemble_big's
    #     small-to-big assembly at query time; Qdrant only stores chunk vectors, the raw
    #     elements/sections never go into the index) ---
    sidecar_dir: str = os.path.expanduser("~/rag_sidecar")

    # --- rerank: Qwen3-VL-Reranker-8B (optional cross-encoder reranking; from the same model family as embedding) ---
    rerank_model_path: str = os.path.expanduser("~/models/Qwen3-VL-Reranker-8B")
    rerank_instruction: str = "Given a search query, retrieve relevant candidates that answer the query."
    rerank_top_n: int = 50                 # Cross-encoder reranks the top N candidates from the hybrid recall

    # --- indexing throughput ---
    embed_batch_size: int = 16             # index_document groups chunks into batches of this size per
                                            # dense.encode_text/encode_image call, instead of one chunk per
                                            # forward pass. Batching amortizes fixed per-call overhead across
                                            # many chunks, which matters a lot on a GPU (thousands of
                                            # chunks/doc = thousands of forward passes otherwise). Tune down
                                            # if you hit GPU OOM on very large images; tune up if you have
                                            # headroom and want faster indexing.

    # --- retrieval ---
    prefetch_limit: int = 50               # How many candidates each path (dense/sparse) recalls before RRF fusion
    top_k: int = 8

    # --- model inference backend (for the production split) ---
    # Empty = local (loads Qwen3-VL onto the GPU in-process, the default, backward compatible);
    # non-empty = remote: dense/rerank calls go over HTTP to a separate inference service
    # (inference_server.py), and this process never loads a model or touches the GPU -> the
    # application tier becomes stateless and can run multiple replicas. See remote.py.
    inference_url: str = ""
    inference_timeout: float = 120.0       # The inference service's HTTP **read** timeout (the first call includes a lazy model load, so this is generous)
    # Failure modes (a split timeout plus bounded retry, absorbing the inference service's normal
    # warm-up / rolling-restart transients; see docs/SCALE_OUT.md §5-A):
    inference_connect_timeout: float = 3.0 # Connect timeout: short, so a hung server fails fast into the retry path instead of waiting the full 120s read timeout
    inference_retries: int = 2             # Bounded retry count for 503 / connection errors / read timeouts (a 4xx client error raises immediately, no retry)
    inference_backoff: float = 0.5         # Exponential backoff base: sleeps backoff*2^n seconds before the nth retry (0.5/1.0)
