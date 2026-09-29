"""Wrapper around Qwen3-VL-Embedding-8B: the only part of embedder that touches the GPU.

Reuses the model's own official `Qwen3VLEmbedder` from `scripts/qwen3_vl_embedding.py`
(last-token pooling + a shared text/image space) rather than reimplementing pooling/forward.
Text and image are both encoded into the same 4096-dim space (the official example computes
similarity directly as query@doc.T); MRL: truncate to the first `cfg.dense_dim` dims and
re-apply L2 normalize (the standard approach for the Qwen3-Embedding family; the README says
it supports custom dimensions from 64 to 4096). The 4090 is pinned by name, because under
FASTEST_FIRST torch's device numbering can drift, so we trust the card's name rather than
assuming device 0 is always the right one.
"""
from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict

import numpy as np

from .config import EmbedConfig

_QUERY_CACHE_CAP = 256


class Dense:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self._model = None
        self._gpu_error: Exception | None = None   # Cache a permanent GPU configuration error (assertion failure) so we don't re-run it on every retry
        self._query_cache: OrderedDict = OrderedDict()   # LRU cache of query text -> vector (user-independent; ACL is applied after retrieval, so this is safe)
        # Locks pushed down to the local implementation (RemoteDense overrides the forward/load locks with nullcontext -> HTTP is naturally concurrent, and backoff doesn't hold a lock):
        self._fwd_lock = threading.Lock()      # Serializes GPU forward passes (local only; a single card's model.process is not thread-safe)
        self._load_lock = threading.Lock()     # Single-flight lazy load (a concurrent first query loads the 8B model only once; kept separate from _fwd_lock to avoid nested-lock deadlock)
        self._cache_lock = threading.Lock()    # Query LRU lock: the real benefit is the two-section structure that keeps encode() outside the lock (so backoff doesn't block other queries);
        #                                        under CPython's GIL a single dict operation is already atomic (even without a lock it wouldn't corrupt), so this lock's extra job is keeping the two sections consistent and being ready for free-threading (3.13+)

    def _assert_gpu(self) -> None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available: Qwen3-VL dense requires a GPU")
        name = torch.cuda.get_device_name(0)   # Qwen3VLEmbedder always uses torch device 0
        if self.cfg.gpu_name_must_contain not in name:
            raise RuntimeError(
                f"torch device 0 = {name!r}, expected it to contain {self.cfg.gpu_name_must_contain!r}. "
                f"FASTEST_FIRST device numbering can drift -- refusing to run on the wrong card; "
                f"lock it down with CUDA_VISIBLE_DEVICES and retry.")

    def _load(self) -> None:
        if self._model is not None:            # Fast path: already loaded, no lock needed
            return
        with self._load_lock:                  # Single-flight: only one thread actually loads on a concurrent first query (separate lock, not nested with _fwd_lock)
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
            scripts = os.path.join(self.cfg.dense_model_path, "scripts")
            if scripts not in sys.path:
                sys.path.insert(0, scripts)     # Reuse the official Qwen3VLEmbedder (shipped alongside the model)
            from qwen3_vl_embedding import Qwen3VLEmbedder
            self._model = Qwen3VLEmbedder(
                model_name_or_path=self.cfg.dense_model_path, torch_dtype=torch.bfloat16)

    def _mrl(self, emb) -> np.ndarray:
        """Matryoshka: truncate to the first dense_dim dims and re-apply L2 normalize (no truncation if dense_dim is >= the full dimension).
        **Truncate and normalize in fp32** (call .float() first): this matches the remote backend's _mrl_np
        (numpy fp32), eliminating the norm bias that bf16 normalize introduces (equivalence testing showed
        that converting to fp32 after normalizing in bf16 gives norm ~= 1.002, while doing it in fp32 gives
        exactly 1.0 -- otherwise the vectors stored locally and the vectors queried remotely would differ
        numerically; the COSINE direction is still equivalent, but it's a latent risk). Converting the
        truncated prefix from bf16 to fp32 is lossless and doesn't change direction, so old stores stay compatible."""
        import torch.nn.functional as F
        emb = emb.float()                                    # bf16 -> fp32, matching the remote fp32 path
        if emb.shape[-1] > self.cfg.dense_dim:
            emb = F.normalize(emb[:, :self.cfg.dense_dim], p=2, dim=-1)
        return emb.cpu().numpy()

    def encode_text(self, texts: list[str], instruction: str | None = None) -> np.ndarray:
        self._load()
        with self._fwd_lock:       # Serializes local GPU forward passes (including the GPU-side normalize inside _mrl); RemoteDense's override of encode_text goes over HTTP and doesn't hit this
            emb = self._model.process([{"text": t, "instruction": instruction} for t in texts], normalize=True)
            return self._mrl(emb)

    def encode_image(self, image_paths: list[str], instruction: str | None = None) -> np.ndarray:
        """image_paths must be **absolute paths** (the official format converts them to file://). Joining relative paths happens in embed.py."""
        self._load()
        with self._fwd_lock:       # Same as encode_text: serializes local GPU forward passes
            emb = self._model.process([{"image": p, "instruction": instruction} for p in image_paths], normalize=True)
            return self._mrl(emb)

    def encode_query(self, query: str) -> np.ndarray:
        """Query side: adds the retrieval instruction (per the README, using a task-specific English
        instruction for queries gives a 1-5% improvement).
        LRU cache, since an agent doing multiple hops often resends the same or a near-identical query,
        so this avoids redundant 8B forward passes; the returned vector is safe to reuse read-only."""
        c = self._query_cache
        with self._cache_lock:                 # Read section: only wraps dict operations
            if query in c:
                c.move_to_end(query)
                return c[query]
        # encode_text (HTTP/GPU plus any remote backoff) runs outside the lock -- this is the key point:
        # backoff must not hold any retrieval lock.
        # Benign race: if two threads miss on the same query at once, each computes it and the later write
        # wins; the result is identical either way (MRL is deterministic), so this is acceptable -- we
        # never wrap the whole section in a lock just to eliminate this race.
        v = self.encode_text([query], instruction=self.cfg.query_instruction)[0]
        with self._cache_lock:                 # Write section: only wraps dict operations
            c[query] = v
            c.move_to_end(query)
            if len(c) > _QUERY_CACHE_CAP:
                c.popitem(last=False)
        return v
