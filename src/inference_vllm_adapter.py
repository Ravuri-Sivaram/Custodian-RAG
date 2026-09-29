"""vLLM inference backend adapter (Plan B/A, docs/VLLM_PLAN.md Phase 1).

**Purpose**: present exactly the same contract as `inference_server.py`
(`/embed`/`/embed_image`/`/rerank`/`/readyz`/`/healthz`), with embeddings actually produced behind the
scenes by `vllm serve --runner pooling` (continuous batching). The application layer (`remote.py`)
needs zero changes — pointing `CUSTODIAN_INFERENCE_URL` at this adapter switches the FastAPI backend over
to the vLLM backend. **Throughput is ~12x** (measured on G4: 297 vs 24 QPS).

**Why an adapter instead of having custodian talk to vLLM's `/v1/embeddings` directly**:
  1. Contract boundary: custodian's contract is `/embed {texts,instruction} -> {vectors}` (full-dimension,
     normalized; the client truncates dimensions). vLLM's is the OpenAI-style
     `/v1/embeddings {input} -> {data:[{embedding}]}`. The adapter translates between them so the
     application layer never has to touch this difference.
  2. **The chat template is the crux of equivalence**: the official Qwen3VLEmbedder wraps a query as
     `[{system:instruction},{user:text}]` and calls
     `apply_chat_template(add_generation_prompt=True)` (confirmed by reading the official wrapper's
     source). vLLM's `/v1/embeddings` takes raw strings and does **not** apply any template on its
     own -> the adapter must **replicate this exact recipe**, pre-applying the template before sending
     to vLLM, or last-token pooling gets misaligned and the resulting vectors drift. The tokenizer
     therefore runs in the adapter (on CPU).

**Phase 1 scope (stated plainly)**:
  - `/embed`: routed through vLLM (this is the QPS bottleneck on the query path, so it's migrated
    first to capture most of the win).
  - `/embed_image`: returns 501 (image encoding is index-build-only, runs on local torch, and never
    goes through vLLM; the query path never calls it).
  - `/rerank`: proxied to the torch reranker container when `RERANK_PROXY_URL` is configured;
    otherwise returns 503 — this is a safe degradation for custodian (retrieve.py's rerank-except path
    falls back to hybrid retrieval), so having no reranker in Phase 1 doesn't crash anything.
    Migrating the reranker to vLLM is planned for Phase 3 (not yet validated on G3).

Environment variables:
  INFERENCE_VLLM_URL    vLLM serve address (default http://localhost:8000)
  INFERENCE_MODEL_PATH  Model path (used to load the tokenizer for the chat template; default
                         ~/models/Qwen3-VL-Embedding-8B)
  RERANK_PROXY_URL      Optional: the torch reranker's /rerank address (if unset, /rerank returns 503
                         as a graceful degradation)
  ADAPTER_HOST/ADAPTER_PORT  Bind address (default 0.0.0.0:8900, same port as inference_server -> a
                         drop-in replacement)

**Why this lives at the src/ root rather than inside the embedder/ package**: the adapter has to run
in the **vllm environment** (which doesn't have custodian/qdrant_client installed), so it must avoid
triggering `embedder/__init__` (which imports qdrant_client). Also, running a file directly with
`python <file>` from inside a package would push `src/embedder/` onto `sys.path[0]`, which shadows the
stdlib `import types` with `embedder/types.py` (both of these pitfalls were confirmed empirically).
This adapter has zero dependency on the embedder package, so keeping it as a standalone top-level
module is cleanest (the container CMD likewise runs `python src/inference_vllm_adapter.py`).

To run (bare-metal verification, part of the same phase-E pivot):
  conda activate vllm && vllm serve ~/models/Qwen3-VL-Embedding-8B --runner pooling --max-model-len 8192 --port 8000 &
  python src/inference_vllm_adapter.py    # serves on :8900; point custodian at it via CUSTODIAN_INFERENCE_URL=http://localhost:8900
"""
from __future__ import annotations

import os
import unicodedata

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class EmbedReq(BaseModel):
    texts: list[str] = []
    instruction: str | None = None


class EmbedImageReq(BaseModel):
    image_paths: list[str] = []
    instruction: str | None = None


class RerankReq(BaseModel):
    query: str = ""
    documents: list[str] = []
    instruction: str | None = None


def _sys_instruction(instr: str | None, default: str) -> str:
    """Replicates the official Qwen3VLEmbedder.format_model_input: if the instruction doesn't end in
    punctuation (Unicode category P*), append '.'."""
    instr = (instr or default).strip()
    if instr and not unicodedata.category(instr[-1]).startswith("P"):
        instr = instr + "."
    return instr


def create_app(cfg=None) -> FastAPI:
    vllm_url = os.environ.get("INFERENCE_VLLM_URL", "http://localhost:8000").rstrip("/")
    model_path = os.environ.get("INFERENCE_MODEL_PATH", os.path.expanduser("~/models/Qwen3-VL-Embedding-8B"))
    rerank_proxy = os.environ.get("RERANK_PROXY_URL", "").rstrip("/")
    # Default instruction aligned with the official wrapper's default_instruction (on the query path,
    # custodian explicitly passes query_instruction anyway)
    default_instr = os.environ.get("INFERENCE_DEFAULT_INSTRUCTION", "Represent the user's input.")

    app = FastAPI(title="custodian-inference-vllm-adapter")
    state = app.state
    state.tok = None           # lazy-load the tokenizer (keeps startup fast; only needed on the first /embed call)
    state.err = None

    def _tokenizer():
        if state.tok is None:
            from transformers import AutoProcessor
            state.tok = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return state.tok

    # vLLM serve exposes OpenAI's /v1/embeddings; continuous batching happens on the vLLM side, so the
    # adapter is a pure async forwarder here (no gpu_lock needed)
    state.client = httpx.Client(base_url=vllm_url, timeout=httpx.Timeout(120.0, connect=3.0))

    def _templated(texts: list[str], instruction: str | None) -> list[str]:
        """Render {texts,instruction} into chat-template strings following the official recipe (this is
        the crux of matching the official embedder's output)."""
        tok = _tokenizer()
        sys_txt = _sys_instruction(instruction, default_instr)
        out = []
        for t in texts:
            conv = [{"role": "system", "content": [{"type": "text", "text": sys_txt}]},
                    {"role": "user", "content": [{"type": "text", "text": t}]}]
            out.append(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
        return out

    @app.get("/healthz")
    async def healthz():                       # liveness: is the adapter process alive (async, in-memory only)
        return {"status": "ok", "service": "inference-vllm-adapter", "vllm_url": vllm_url,
                "model_dense": os.path.basename(model_path), "error": state.err}

    @app.get("/readyz")
    async def readyz():                        # readiness: probe the downstream vLLM /health endpoint (async, does not block the event loop)
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get(vllm_url + "/health", timeout=httpx.Timeout(1.5, connect=1.0))
            if r.status_code == 200:
                return {"status": "ready"}
            return JSONResponse({"status": "vllm_not_ready"}, status_code=503)
        except Exception:
            return JSONResponse({"status": "vllm_unavailable"}, status_code=503)   # don't echo back the internal host:port (same security discipline as elsewhere)

    @app.post("/embed")
    def embed(q: EmbedReq):
        prompts = _templated(q.texts, q.instruction)
        r = state.client.post("/v1/embeddings",
                              json={"input": prompts, "model": model_path, "encoding_format": "float"})
        r.raise_for_status()
        data = r.json()["data"]
        vecs = [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]   # restore original order by index
        return {"vectors": vecs}               # vLLM returns full-dimension, normalized vectors; the client
                                                # (remote._mrl_np) truncates dimensions (same contract as inference_server)

    @app.post("/embed_image")
    def embed_image(q: EmbedImageReq):         # Phase 1: image encoding is index-build-only and runs on local torch, not through vLLM
        return JSONResponse({"status": "not_implemented",
                             "detail": "Image encoding is index-build-only (local torch); the query path never calls /embed_image."}, status_code=501)

    @app.post("/rerank")
    def rerank(q: RerankReq):
        if not rerank_proxy:                   # Phase 1 has no reranker: 503 -> custodian degrades gracefully to hybrid
            return JSONResponse({"status": "rerank_unavailable",
                                 "detail": "No reranker wired up in Phase 1 (safe degradation; custodian falls back to hybrid)."}, status_code=503)
        r = state.client.post(rerank_proxy + "/rerank", json=q.model_dump())   # proxy to the torch reranker container
        return JSONResponse(r.json(), status_code=r.status_code)

    return app


def main() -> None:
    import uvicorn
    host = os.environ.get("ADAPTER_HOST", "0.0.0.0")
    port = int(os.environ.get("ADAPTER_PORT", "8900"))
    print(f"custodian-inference-vllm-adapter  http://{host}:{port}  -> vLLM "
          f"{os.environ.get('INFERENCE_VLLM_URL', 'http://localhost:8000')}", flush=True)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
