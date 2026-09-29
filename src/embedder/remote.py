"""Remote inference backend (for the production split): dense/rerank GPU forward passes go over
HTTP to a separate inference service, and this process never loads a model.

**Why**: loading the model in-process ties the application layer to a GPU and blocks horizontal
scaling. Splitting the GPU forward pass out into its own service (inference_server.py) lets the
application layer (custodian serve) become stateless and GPU-free, so it can run multiple replicas
behind an orchestrator -- a prerequisite for running elastically on K8s (see docs/SCALE_OUT.md).

**Equivalence guarantee (the important part)**: the inference service is configured to return
**full-width**, un-truncated, normalized vectors; the client (RemoteDense) then applies its own
real dense_dim via MRL truncation. Mathematically, local and remote produce **the exact same final
vector** -- so the same index can be built locally and queried remotely (or vice versa) without
misalignment. RemoteDense/RemoteReranker **inherit** from the local base classes and only override
the "forward pass" methods (which go over HTTP); MRL truncation, the query LRU cache, and writing
scores back onto a Hit are all **inherited business logic**, so there's no duplication and behavior
stays identical between local and remote.

**Failure modes** (see docs/SCALE_OUT.md §5-A): the inference service warming up (1-2 minutes) or a
rolling replica restart are normal operational states, not failures. `_post_retry` absorbs these
transients with a bounded retry plus exponential backoff on 503/connection errors/read timeouts;
once retries are exhausted, it raises the semantic exception `InferenceUnavailable` (errors.py),
which the caller (via toolcore's duck-typing) turns into a retriable `inference_unavailable`
rather than the non-retriable, generic `backend_unavailable`. A 4xx (client error) raises
httpx.HTTPStatusError immediately, with no retry. dense and reranker share one connection pool
(cached by url), closed together at process exit via atexit.

The backend choice is decided by EmbedConfig.inference_url (empty = local), via the make_dense/
make_reranker factories below.
"""
from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time

import numpy as np

from .config import EmbedConfig
from .dense import Dense
from .errors import InferenceUnavailable
from .rerank import Reranker

# A module-level client cache: dense and reranker for the same inference_url **share one
# connection pool** (avoiding wasting a second one). One instance per process, closed together at
# atexit. A naive check-then-set here would race under concurrent construction of Remote* objects
# (the loser's client would be orphaned and never closed by atexit -- a connection leak), so
# get-or-create and close share the same lock.
_CLIENTS: dict[str, "object"] = {}
_CLIENTS_LOCK = threading.Lock()


def _get_client(cfg: EmbedConfig):
    """Reuses an httpx.Client keyed by inference_url (shared between dense and reranker); timeouts
    are split -- connect is short, read is long. A short connect timeout means a hung/down server
    fails fast into retry rather than waiting out the full 120s read timeout; a long read timeout
    tolerates long-text forward passes. get-or-create happens inside the lock (constructing an
    httpx.Client is cheap, so the lock cost is negligible): this eliminates both duplicate pool
    creation and orphaned-client leaks under concurrency."""
    import httpx
    url = cfg.inference_url.rstrip("/")
    with _CLIENTS_LOCK:
        c = _CLIENTS.get(url)
        if c is None:
            c = httpx.Client(base_url=url, timeout=httpx.Timeout(
                connect=cfg.inference_connect_timeout, read=cfg.inference_timeout, write=10.0, pool=5.0))
            _CLIENTS[url] = c
        return c


@atexit.register
def _close_clients() -> None:
    """Closes every connection pool together at process exit (avoiding a leak); idempotent and
    swallows exceptions (the exit path shouldn't fail just because closing a connection errored)."""
    with _CLIENTS_LOCK:
        for c in list(_CLIENTS.values()):
            try:
                c.close()
            except Exception:
                pass
        _CLIENTS.clear()


def _post_retry(client, cfg: EmbedConfig, path: str, payload: dict) -> dict:
    """POST with a bounded retry. Retriable (with exponential backoff):
      - **any 5xx** (503 not-ready / 502 or 504 from a gateway with no healthy backend / a 500 from
        a one-off inference crash that self-heals) -- all of these are transients typical of a
        rolling restart;
      - **any httpx transport-level exception** (`TransportError`: covers Connect/Read/Write/
        Close/RemoteProtocol errors plus every kind of Timeout).
    Not retried: a 4xx client error -- `raise_for_status()` raises httpx.HTTPStatusError
    immediately (which is **not** a TransportError, so it keeps propagating; retrying it wouldn't
    help).
    Once retries are exhausted, raises InferenceUnavailable (a semantic exception the caller uses
    to return a retriable `inference_unavailable`).

    A few specific choices worth calling out: (1) `>= 500` rather than hardcoding `== 503` -- when
    a backend gets killed, K8s/nginx often returns 502/504, and hardcoding 503 would treat those as
    non-retriable client errors and raise them bare; (2) `except httpx.TransportError` rather than
    only catching `ConnectError`/`TimeoutException` -- the most common disconnect shape from
    `docker kill`/restarting the inference service is actually a `RemoteProtocolError` ("Server
    disconnected without sending a response", from reusing a dead keep-alive connection) or a
    `ReadError` (the peer sent RST), and neither of those is a ConnectError or a TimeoutException --
    missing them would silently break the "a killed backend is handled gracefully" chain, letting
    it fall through to the generic, non-retriable backend_unavailable instead. TransportError is
    the shared parent class of all of these plus timeouts, and it deliberately does not include
    `HTTPStatusError` (so 4xx still propagates immediately) -- one except clause catches everything
    it should, without accidentally swallowing a client error; (3) `max(0, retries)` clamps a
    negative config value so it doesn't produce an empty range and then `raise None`, hiding the
    real cause."""
    import httpx
    retries = max(0, cfg.inference_retries)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = client.post(path, json=payload)
            if r.status_code >= 500:                       # Every 5xx is treated as transient -> retriable
                last = InferenceUnavailable(f"Inference service returned {r.status_code} @ {path}")
            else:
                r.raise_for_status()                       # A 4xx client error -> raise immediately, no retry
                return r.json()
        except httpx.TransportError as e:                  # Connection refused/reset (including a dead reused keep-alive connection) + every kind of timeout
            last = InferenceUnavailable(f"Inference service {type(e).__name__} @ {path}")
        if attempt < retries:                              # Retries remain -> exponential backoff
            time.sleep(cfg.inference_backoff * (2 ** attempt))
    raise last                                             # Retries exhausted (last is guaranteed to be an InferenceUnavailable)


class RemoteDense(Dense):
    """Dense forward passes go over HTTP. Inherits from Dense to reuse _mrl (the client-side
    dense_dim truncation) and the encode_query cache.
    Only overrides: _load (never loads a model), encode_text/encode_image (calls the inference
    service, then applies _mrl locally to the returned full-width vector)."""

    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        self._client = _get_client(cfg)         # A shared connection pool (keyed by url); tests can replace self._client with a mock
        # Remote goes over HTTP with no local GPU forward pass or model load, so the forward and
        # load locks are no-ops -- neither backoff nor the HTTP call ever holds a lock
        # (_cache_lock is still inherited and still active).
        self._fwd_lock = contextlib.nullcontext()
        self._load_lock = contextlib.nullcontext()
        self._handshake_done = False            # A one-time model/dimension handshake (_verify_server); once verified, no more GET calls are sent
        self._handshake_lock = threading.Lock()

    def _load(self) -> None:                    # The remote backend never loads a model or touches the GPU in this process
        return

    def _verify_server(self) -> None:
        """A one-time handshake before the first query: GET /healthz and compare the server's
        model identity and full width. Without this, swapping the server's model without
        rebuilding the index would only be caught by _mrl_np's lower-bound assertion
        (full_dim<dense_dim) -- a same-dimension but different embedding space would silently
        misalign, and recall would collapse with no error, discoverable only after the fact via
        eval.
        Edge cases: an unreachable probe, or a missing/None field (full_dim=None while the server
        is still warming up, per inference_server.py) **does not block** -- that's a transient,
        left to _post_retry's existing retry chain; only once both fields are present and actually
        mismatched does this raise loudly (a configuration mismatch is not a transient and
        shouldn't be silently absorbed by retries). _handshake_done is only set once both fields
        have been checked, so during warm-up, every query retries the handshake until it succeeds."""
        if self._handshake_done:
            return
        with self._handshake_lock:              # Single-flight: a concurrent first query only sends one GET (repeating it would be harmless too; the lock just saves bandwidth)
            if self._handshake_done:
                return
            try:
                r = self._client.get("/healthz")
                if r.status_code != 200:
                    return
                data = r.json()
            except Exception:                   # Unreachable or non-JSON: don't block, leave it to the business request's own retry chain
                return
            model, full_dim = data.get("model_dense"), data.get("full_dim")
            if model is not None and model != os.path.basename(self.cfg.dense_model_path):
                raise RuntimeError(
                    f"Inference service model identity mismatch: server model_dense={model!r} != client "
                    f"{os.path.basename(self.cfg.dense_model_path)!r} (CUSTODIAN_DENSE_MODEL_PATH). "
                    f"The vector space would silently misalign -- align the models on both sides, or rebuild the index.")
            if full_dim is not None and int(full_dim) < self.cfg.dense_dim:
                raise RuntimeError(
                    f"Inference service full_dim={full_dim} < client dense_dim={self.cfg.dense_dim}, a configuration mismatch -- "
                    f"check whether CUSTODIAN_DENSE_DIM matches the inference service's model.")
            if model is not None and full_dim is not None:
                self._handshake_done = True

    def _post_vectors(self, path: str, payload: dict) -> np.ndarray:
        self._verify_server()                                       # Lazy handshake: a model/dimension mismatch fails loud (only actually sends a GET the first time)
        data = _post_retry(self._client, self.cfg, path, payload)   # With retries; raises InferenceUnavailable once exhausted
        return np.asarray(data["vectors"], dtype=np.float32)        # The inference service returns **full-width**, normalized vectors

    def encode_text(self, texts: list[str], instruction: str | None = None) -> np.ndarray:
        full = self._post_vectors("/embed", {"texts": texts, "instruction": instruction})
        return self._mrl_np(full)               # The client truncates to its real dense_dim + renormalizes (mathematically equivalent to local)

    def encode_image(self, image_paths: list[str], instruction: str | None = None) -> np.ndarray:
        # image_paths must be paths the inference service can actually read (same machine or a
        # shared volume). A cross-machine deployment should switch to base64 -- see the TODO in
        # inference_server.py.
        full = self._post_vectors("/embed_image", {"image_paths": image_paths, "instruction": instruction})
        return self._mrl_np(full)

    def _mrl_np(self, full: np.ndarray) -> np.ndarray:
        """MRL-truncates a full-width numpy vector plus L2 renormalization (pure numpy,
        mathematically equivalent to Dense._mrl's torch version). numpy rather than torch is
        deliberate: the remote backend shouldn't need a torch dependency just to truncate a vector
        (the whole point is that the application layer can run with no GPU and no torch)."""
        d = self.cfg.dense_dim
        if full.shape[-1] < d:              # A client dense_dim larger than the server's full width is a configuration mismatch -- fail loud rather than silently indexing an under-length vector
            raise RuntimeError(
                f"dense_dim={d} > the inference service's returned full width={full.shape[-1]}, a configuration mismatch -- "
                f"check whether CUSTODIAN_DENSE_DIM matches the inference service's model.")
        if full.shape[-1] > d:
            v = full[:, :d]
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            v = v / np.clip(norm, 1e-12, None)
            return v.astype(np.float32)
        return full.astype(np.float32)      # == d: the full width is already the target width and already normalized, so it's returned as-is


class RemoteReranker(Reranker):
    """Reranking goes over HTTP. Inherits from Reranker to reuse rerank() (sorting, plus writing
    scores back onto Hit.score/score_kind); only overrides _load (never loads a model) and score
    (calls the inference service). A score() failure (once retries are exhausted and
    InferenceUnavailable is raised) gets caught by retrieve.py's existing rerank try/except and
    degrades to plain hybrid search -- rerank is an enhancement signal, so degrading it is safe;
    this asymmetry with dense's loud failure is intentional."""

    def __init__(self, cfg: EmbedConfig):
        super().__init__(cfg)
        self._client = _get_client(cfg)
        self._fwd_lock = contextlib.nullcontext()      # Remote reranking goes over HTTP, so the forward and load locks are no-ops
        self._load_lock = contextlib.nullcontext()

    def _load(self) -> None:
        return

    def score(self, query: str, docs_text: list[str], instruction: str | None = None) -> list[float]:
        # The instruction parameter matches the base Reranker.score signature (this used to be
        # missing, which meant calling it per the base class's contract raised a bare TypeError,
        # and remote could only ever use the cfg default -- narrowing the accepted input
        # compared to the base class it's supposed to substitute for). None falls back to cfg,
        # matching the old behavior exactly.
        data = _post_retry(self._client, self.cfg, "/rerank",
                           {"query": query, "documents": list(docs_text),
                            "instruction": instruction or self.cfg.rerank_instruction})
        scores = data["scores"]
        if len(scores) != len(docs_text):       # The same contract check as the local backend: scores must correspond 1:1 to documents, or fail loud
            raise RuntimeError(f"The remote reranker returned {len(scores)} scores != {len(docs_text)} documents; the API contract was violated")
        return [float(s) for s in scores]


# ---------- Backend factories: the single place that decides local vs. remote ----------
def make_dense(cfg: EmbedConfig | None = None) -> Dense:
    """Picks the dense backend from cfg.inference_url: empty -> local Dense (in-process GPU, the default, backward compatible); non-empty -> RemoteDense."""
    cfg = cfg or EmbedConfig()
    return RemoteDense(cfg) if cfg.inference_url else Dense(cfg)


def make_reranker(cfg: EmbedConfig | None = None) -> Reranker:
    cfg = cfg or EmbedConfig()
    return RemoteReranker(cfg) if cfg.inference_url else Reranker(cfg)
