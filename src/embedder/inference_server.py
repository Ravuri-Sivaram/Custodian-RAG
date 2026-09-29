"""The GPU model inference service (for the production split): splits the only GPU-bound part of
the embedder (Qwen3-VL embedding + reranking) out into its own HTTP service. Once the application
layer (custodian serve) is pointed at this service via EmbedConfig.inference_url, it never loads a
model and never touches the GPU -> it becomes stateless and can run multiple replicas. See
remote.py for the client side.

**Pure GPU forward pass, no business logic**: only exposes model.process's raw output (full-width
normalized vectors / raw rerank scores); MRL truncation, the query cache, and writing scores back
onto a Hit all stay on the client side (remote.py). This keeps the service simple enough to swap
out for TEI/vLLM/Triton in the future.

**Returns full width**: constructs Dense with dense_dim set to an enormous value, so its own _mrl
never truncates -> it returns the model's raw, full-width vector, and the client truncates to its
own real dense_dim. This guarantees local and remote produce equivalent final vectors (see
remote.py for the full explanation).

**Genuine readiness semantics**: a background thread warms up both 8B models at startup (1-2
minutes each); /readyz returns 503 until they're warm, so an orchestrator (compose/K8s) doesn't
route traffic before the models are actually ready. This is a good illustration of "liveness (the
process is alive) vs. readiness (it can actually serve)".

**Serialized GPU forward passes**: concurrent model.process calls on a single card are not
thread-safe, so a lock serializes them (the same approach as custodian's request-locking).

To run: conda activate custodian && python -m embedder.inference_server   # defaults to 0.0.0.0:8900
Requires pip install fastapi uvicorn.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import replace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import EmbedConfig
from .dense import Dense
from .rerank import Reranker

_FULL_DIM = 10 ** 9        # An enormous dense_dim -> _mrl never truncates, so the model's raw full width is returned (the client truncates to its real dimension)
# Warm-up self-healing: retry count and backoff base for a transient (non-permanent-config)
# failure; once retries are exhausted and it's still failing, os._exit(1) lets the orchestrator's
# restart policy bring it back up.
_WARMUP_RETRIES = int(os.environ.get("INFERENCE_WARMUP_RETRIES", "3"))
_WARMUP_BACKOFF = float(os.environ.get("INFERENCE_WARMUP_BACKOFF", "5.0"))
# Backpressure: the cap on requests in flight at once (executing, or waiting on the GPU lock). GPU
# forward passes are serialized, so a deep queue is pure waste plus a congestion feedback loop; once
# full, requests get a fast-failing 503 overloaded instead.
_MAX_INFLIGHT = int(os.environ.get("INFERENCE_MAX_INFLIGHT", "16"))


def _readiness(ready: bool, err: str | None) -> tuple[dict, int]:
    """The readiness decision (a pure function, easy to unit test). **A load failure takes priority
    over "not ready yet"** -- this is the key point:
    - err (a permanent load failure: wrong GPU, missing model) -> error, 503. This must never be
      reported as "loading", or an operator would see "warming up forever" instead of "load
      failed", misdirecting troubleshooting;
    - not ready (still warming up) -> loading, 503;
    - ready -> ready, 200.
    /readyz and every endpoint's _guard share this single function, so there's no drift from
    duplicating the same judgment in three separate places (an earlier version of _guard had
    exactly this drift -- it missed the err branch)."""
    if err:
        return {"status": "error", "detail": err}, 503
    if not ready:
        return {"status": "loading", "hint": "The model is warming up; please retry shortly"}, 503
    return {"status": "ready"}, 200


class EmbedReq(BaseModel):
    texts: list[str] = []
    instruction: str | None = None


class EmbedImageReq(BaseModel):
    image_paths: list[str] = []          # Must be paths the inference service can read (same machine or a shared volume); a cross-machine deployment should switch to base64 (TODO)
    instruction: str | None = None


class RerankReq(BaseModel):
    query: str = ""
    documents: list[str] = []
    instruction: str | None = None       # The client's payload is the single source of truth for this: a non-None value is used as-is, None falls back to the server's cfg default


def create_app(cfg: EmbedConfig | None = None) -> FastAPI:
    cfg = cfg or EmbedConfig()
    full = replace(cfg, dense_dim=_FULL_DIM, inference_url="")   # Full width, and forced to local (so it never recurses into remote)
    dense = Dense(full)
    reranker = Reranker(full)

    app = FastAPI(title="custodian-inference")
    state = app.state
    state.ready = False
    state.err: str | None = None
    state.gpu_lock = threading.Lock()    # Serializes GPU forward passes on a single card
    state.full_dim: int | None = None    # The model's full width (obtained by the warm-up probe); exposed via /healthz for troubleshooting plus the client's first-query handshake check (remote._verify_server)
    # Bounded admission (backpressure): GPU forward passes are serialized, so under overload an
    # unbounded queue means a client that gave up on a read-timeout leaves its request still queued
    # -- burning GPU time on work nobody's waiting for, plus the client's own retry re-entering the
    # queue -- 3x wasted work and a congestion feedback loop. A semaphore pins in-flight requests
    # (executing + waiting on the lock) at _MAX_INFLIGHT; once full, it fails fast with a 503
    # overloaded, and the client already treats any 5xx as InferenceUnavailable and backs off before
    # retrying, so this composes naturally and keeps the queue bounded.
    state.inflight = threading.BoundedSemaphore(_MAX_INFLIGHT)

    @app.on_event("startup")
    def _warmup():
        def load():
            for attempt in range(_WARMUP_RETRIES + 1):
                try:
                    dense._load()            # 1-2 minutes each (8B models), warmed up in the background; /readyz returns 503 until this finishes
                    reranker._load()
                    probe = dense._model.process([{"text": "dim-probe", "instruction": None}], normalize=True)
                    state.full_dim = int(probe.shape[-1])   # The full width (e.g. 4096); dense_dim=_FULL_DIM here means _mrl never truncates it
                    state.ready = True
                    state.err = None
                    return
                except Exception as e:       # A load failure: distinguish a permanent configuration error from a transient one
                    state.err = f"{type(e).__name__}: {e}"
                    # A permanent configuration error (wrong GPU, missing CUDA -- Dense has already
                    # cached this as _gpu_error) won't be fixed by retrying -> stop immediately
                    # rather than spinning and flooding the logs.
                    if getattr(dense, "_gpu_error", None) is not None:
                        break
                    if attempt < _WARMUP_RETRIES:   # A transient issue (e.g. a shared-volume hiccup, an occasional transitive-dependency glitch) -> retry with exponential backoff
                        time.sleep(_WARMUP_BACKOFF * (2 ** attempt))
            # Self-healing: once retries for a transient failure are exhausted and it's still
            # failing, the process kills itself, letting an "unless-stopped" restart policy bring it
            # back up (the orchestrator's start_period absorbs the restart's warm-up window). A
            # permanent configuration error does not trigger a self-kill (a crash loop wouldn't help
            # anyway; the error is left in state for /healthz to surface for troubleshooting).
            if not state.ready and getattr(dense, "_gpu_error", None) is None:
                import sys
                print(f"FATAL: inference warm-up retries exhausted, exiting so the orchestrator restarts it: {state.err}", file=sys.stderr, flush=True)
                os._exit(1)
        threading.Thread(target=load, daemon=True).start()

    @app.get("/healthz")                 # Liveness: is the process alive (not necessarily that the model is warm). **Still reports ok even after a permanent load failure** --
    async def healthz():                 # async: a pure in-memory read, never touches the 40-thread pool -> the probe never starves under high load
        return {"status": "ok", "service": "inference", "ready": state.ready, "error": state.err,
                "full_dim": state.full_dim, "model_dense": os.path.basename(cfg.dense_model_path)}
        # err is for human troubleshooting; full_dim/model_dense are consumed by remote._verify_server's
        # first-query handshake (a model identity/dimension mismatch fails loud; during warm-up, when
        # full_dim is None, the client doesn't block and just retries on the next query). This doesn't
        # change the liveness verdict -- traffic routing decisions are made from /readyz instead.

    @app.get("/readyz")                  # Readiness: is the model warm and able to actually serve -- the orchestrator routes traffic based on this
    async def readyz():                  # async: a pure in-memory read of state.ready/err, never touches the thread pool (sharing a pool with GPU business traffic would starve the probe under load, causing a false unhealthy)
        body, code = _readiness(state.ready, state.err)
        return body if code == 200 else JSONResponse(body, status_code=code)

    def _guard():                        # An endpoint guard: not ready (including a load failure) always returns 503, distinguishing error from loading
        body, code = _readiness(state.ready, state.err)
        return None if code == 200 else JSONResponse(body, status_code=code)

    def _admit():                        # A backpressure guard: an already-full in-flight count returns 503 overloaded immediately (a non-blocking attempt); returns a token (for release) or None+a 503 response
        if not state.inflight.acquire(blocking=False):
            return None, JSONResponse({"status": "overloaded",
                                       "hint": "The inference service is at capacity; please back off and retry."}, status_code=503)
        return state.inflight, None

    @app.post("/embed")
    def embed(q: EmbedReq):
        g = _guard()
        if g is not None:
            return g
        tok, over = _admit()
        if over is not None:
            return over
        try:
            with state.gpu_lock:
                vecs = dense.encode_text(q.texts, instruction=q.instruction)   # Full width (dense_dim is set enormous, so nothing is truncated)
            return {"vectors": vecs.tolist()}
        finally:
            tok.release()

    @app.post("/embed_image")
    def embed_image(q: EmbedImageReq):
        g = _guard()
        if g is not None:
            return g
        tok, over = _admit()
        if over is not None:
            return over
        try:
            with state.gpu_lock:
                vecs = dense.encode_image(q.image_paths, instruction=q.instruction)
            return {"vectors": vecs.tolist()}
        finally:
            tok.release()

    @app.post("/rerank")
    def rerank(q: RerankReq):
        g = _guard()
        if g is not None:
            return g
        tok, over = _admit()
        if over is not None:
            return over
        try:
            with state.gpu_lock:
                # Uses the instruction from the client's own payload (the client is the single
                # source of truth here); None falls back to the server's cfg default inside score().
                scores = reranker.score(q.query, q.documents, instruction=q.instruction)
            return {"scores": scores}
        finally:
            tok.release()

    return app


def main() -> None:
    import uvicorn
    host = os.environ.get("INFERENCE_HOST", "0.0.0.0")
    port = int(os.environ.get("INFERENCE_PORT", "8900"))
    print(f"custodian-inference  http://{host}:{port}  (the model is warming up in the background; /readyz reports ready once it's done)", flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
