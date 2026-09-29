"""Wrapper around Qwen3-VL-Reranker-8B: cross-encoder fine reranking (an optional second stage).

Reuses the model's own `Qwen3VLReranker` from `scripts/qwen3_vl_reranker.py` rather than
reimplementing it: it feeds the whole (query, document) pair into the model and derives a
0-1 relevance score from the yes/no token logits plus a sigmoid. This is more precise than a
bi-encoder (dense) but slower, so it only reranks the top-N results from hybrid retrieval.
Uses the GPU, pinned to the 4090 by name (same as dense).

Note: reranking the **image** itself for image_only (pure-image) chunks is still a TODO --
at retrieval time we only have a relative image_path and lack the absolute path, so we
currently rerank on the text placeholder, which underrates pure-image chunks. The evaluation
corpus has zero image_only chunks, so this doesn't affect validation; production usage with
pure-image chunks would need a way to resolve the image path at retrieval time.
"""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import replace

from .config import EmbedConfig
from .types import Hit


class Reranker:
    def __init__(self, cfg: EmbedConfig | None = None):
        self.cfg = cfg or EmbedConfig()
        self._model = None
        self._gpu_error: Exception | None = None   # Caches a permanent GPU configuration error (wrong card / missing CUDA), mirroring Dense, so we don't re-run the assertion on every retry
        # Locks pushed down to the local implementation (RemoteReranker overrides this with nullcontext -> HTTP concurrency):
        self._fwd_lock = threading.Lock()      # Serializes GPU forward passes (local only)
        self._load_lock = threading.Lock()     # Single-flight lazy load (kept separate to avoid nesting with _fwd_lock)

    def _assert_gpu(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available: Qwen3-VL-Reranker requires a GPU")
        name = torch.cuda.get_device_name(0)   # Qwen3VLReranker always uses torch device 0
        if self.cfg.gpu_name_must_contain not in name:
            raise RuntimeError(
                f"torch device 0 = {name!r}, expected it to contain {self.cfg.gpu_name_must_contain!r}; "
                f"FASTEST_FIRST device numbering can drift -- refusing to run on the wrong card.")

    def _load(self) -> None:
        if self._model is not None:            # Fast path: already loaded, no lock needed
            return
        with self._load_lock:                  # Single-flight (separate lock, not nested with _fwd_lock)
            if self._model is not None:        # double-check
                return
            if self._gpu_error is not None:    # A permanent GPU config error was already cached -> fail fast, don't retry the assertion/load
                raise self._gpu_error
            try:
                self._assert_gpu()
            except RuntimeError as e:          # Missing CUDA / wrong card is permanent for this session, so cache it (a model-load error is not cached, since it could be transient)
                self._gpu_error = e
                raise
            import torch
            scripts = os.path.join(self.cfg.rerank_model_path, "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)    # Reuse the official Qwen3VLReranker (shipped alongside the model)
            from qwen3_vl_reranker import Qwen3VLReranker
            self._model = Qwen3VLReranker(
                model_name_or_path=self.cfg.rerank_model_path, torch_dtype=torch.bfloat16)

    def score(self, query: str, docs_text: list[str], instruction: str | None = None) -> list[float]:
        """Scores (query, each doc text) pair with a cross-encoder relevance score (0-1), in the
        same order as docs_text.
        instruction: when None, uses the server-side cfg default; when not None, uses the value
        passed by the caller (the inference server consumes the client payload's instruction on
        that basis, so the client is the single source of truth -- this removes the earlier
        footgun where the server could silently ignore the client's value, leaving two
        potentially conflicting sources)."""
        self._load()
        inputs = {"instruction": instruction or self.cfg.rerank_instruction, "query": {"text": query},
                  "documents": [{"text": t or ""} for t in docs_text], "fps": 1.0}
        with self._fwd_lock:                # Serializes local GPU forward passes; RemoteReranker's override of score goes over HTTP and doesn't hit this
            scores = self._model.process(inputs)
        if len(scores) != len(docs_text):   # API contract: scores must correspond 1:1 with documents. Fail loudly on violation rather than silently misordering or going out of bounds
            raise RuntimeError(f"reranker returned {len(scores)} scores != {len(docs_text)} documents, API contract violated")
        return scores

    def rerank(self, query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
        """Reranks hits from hybrid retrieval with the cross-encoder and returns the top top_k (all of them by default). Empty hits are returned unchanged."""
        if not hits:
            return hits
        scores = self.score(query, [h.text or "" for h in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        # Write the cross-encoder score back into `score` and mark score_kind='rerank' -- otherwise
        # the ranking would follow the rerank scores while Hit.score still held the RRF score, and
        # that mismatch would mislead the agent's confidence judgment. Use replace() to build a new
        # Hit rather than mutating in place, avoiding side effects on shared objects.
        out = [replace(hits[i], score=float(scores[i]), score_kind="rerank") for i in order]
        return out[:top_k] if top_k is not None else out   # "is not None": top_k=0 should return empty, so we can't treat it as falsy-meaning-all
