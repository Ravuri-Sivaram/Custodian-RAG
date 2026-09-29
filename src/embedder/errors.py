"""Semantic exceptions for the embedder layer (pure definitions, no third-party dependencies).

**Why this is its own file**: `custodian.toolcore` is a stdlib-only tool-semantics layer (see the
convention at the top of toolcore.py: even a thin adapter running in a GPU-less environment
must be able to import it), so it cannot import embedder (doing so would transitively trigger
embedder/__init__ to pull in dense/qdrant deps). So the exception in this layer exposes a
**marker class attribute** that toolcore can detect via duck-typing (getattr) instead of
importing by type -- this lets "inference unavailable" be routed differently from the generic
backend_unavailable case, without giving toolcore a dependency on embedder.
"""
from __future__ import annotations


class InferenceUnavailable(RuntimeError):
    """Raised when the remote inference service is unavailable -- i.e. after the client has
    exhausted its retries (503 while warming up / connection failure / read timeout).

    Callers use this to return a **retryable** structured error (`inference_unavailable`),
    distinct from:
      - 4xx client errors (remote.py raises httpx.HTTPStatusError immediately, without
        retrying or converting to this exception);
      - other backend failures (Qdrant/sidecar etc.), which still fall under the generic
        backend_unavailable case.

    `inference_unavailable = True` is a duck-typing marker for the stdlib-only toolcore:
    `getattr(e, "inference_unavailable", False)` being true is enough to route it, with no
    need to import this module.
    """
    inference_unavailable = True
