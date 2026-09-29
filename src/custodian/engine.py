"""Engine wiring: connects the RAG engine components (embedder/generator) into the Custodian service
layer.

The engine's three packages (chunker/embedder/generator) have been folded into this repo's `src/`
and are installable packages alongside custodian -- so they're imported directly, with no more
cross-repo sys.path injection or file-path loading of the kind toolcore used to do (that historical
D12 boundary has been removed; see docs/PROVENANCE.md).

Concurrency (lock sinking, tracked as M1; docs/SCALE_OUT.md phase B): we **no longer use a single
big LockedRetriever lock to serialize all retrieval** -- that big lock let a remote backend's
dense-HTTP retry/backoff (during warmup or a rolling restart) hold the lock and block every query
on the whole replica (a full-replica avalanche). Instead, **locks now live on the resource classes
that actually need them**:
  - the single-client Qdrant section -> `Store._lock` (needed for both local and remote; the
    embedded single client is not thread-safe);
  - GPU model forward passes -> `Dense`/`Reranker._fwd_lock` (local only; the remote override
    becomes a nullcontext, since HTTP is naturally concurrent);
  - query LRU / lazy loading -> `Dense._cache_lock` / `_load_lock`.
As a result, a remote encode's HTTP call plus its backoff sit outside every lock and don't block
other queries. LLM network calls are not held inside any lock either (see service.py). Once we
switch to Qdrant server mode, `Store._lock` can be removed entirely (phase F/Q3), at which point
the remote `Retriever` becomes truly lock-free and horizontally scalable.
"""
from __future__ import annotations

import os

from embedder import EmbedConfig, Retriever, User, acl_admits
from generator import Generator, OpenAICompatibleLLM


def build_retriever(cfg) -> Retriever:
    """Build the real retriever (opens the embedded Qdrant; the dense model is lazy-loaded on
    first query). Locks now live on the individual resource classes, so this no longer wraps a
    LockedRetriever."""
    # Only the **local backend** needs local models: dense/rerank's official scripts/ directory is
    # a hard runtime requirement (dense.py/rerank.py inject qwen3_vl_* into sys.path), so if it's
    # missing we report a clear error up front rather than a ModuleNotFoundError buried deep in
    # the first query. **The remote backend (inference_url set) loads no model in this process and
    # needs no model files locally** -- so the check is skipped; this is exactly what "GPU/model
    # independence at the application layer" means in practice.
    if not cfg.inference_url:
        scripts = os.path.join(cfg.dense_model_path, "scripts")
        if not os.path.isdir(scripts):
            raise SystemExit(
                f"dense model scripts directory is missing: {scripts}\n"
                f"The model (including its official scripts/) must live at CUSTODIAN_DENSE_MODEL_PATH (currently {cfg.dense_model_path}); "
                f"downloading Qwen3-VL-Embedding-8B via modelscope includes scripts/. Alternatively, set CUSTODIAN_INFERENCE_URL to use a remote inference service.")
    ecfg = EmbedConfig(qdrant_path=cfg.qdrant_path, qdrant_url=cfg.qdrant_url, sidecar_dir=cfg.sidecar_dir,
                       collection=cfg.collection, dense_dim=cfg.dense_dim,
                       dense_model_path=cfg.dense_model_path, rerank_model_path=cfg.rerank_model_path,
                       gpu_name_must_contain=cfg.gpu_name, inference_url=cfg.inference_url,
                       # Passing through the 4 failure-mode fields (per the phase-F review): a
                       # production replica is entirely env-driven, and without passing these
                       # through, retry/timeout behavior would be pinned to their defaults and any
                       # tuning would require a code change plus rebuilding the image.
                       # CUSTODIAN_INFERENCE_* -> EmbedConfig; phase-F load testing tunes
                       # backpressure/backoff based on these.
                       inference_timeout=cfg.inference_timeout,
                       inference_connect_timeout=cfg.inference_connect_timeout,
                       inference_retries=cfg.inference_retries,
                       inference_backoff=cfg.inference_backoff)
    return Retriever(ecfg)


def build_user(cfg):
    """The ACL identity bound at startup (clients/agents cannot change it via request
    parameters). An empty tenant means everything fail-closes to empty results."""
    return User(tenant=cfg.tenant, principals=cfg.principals)


def build_generator(retriever, cfg):
    """Closed-pipeline generator: generator.Generator + DeepSeek (OpenAI-compatible).
    Injects acl_admits for defense-in-depth at the exit point (a second fail-closed ACL check on
    every hit chunk that goes into the prompt). If the API key is missing, OpenAICompatibleLLM
    raises ValueError -- the caller turns that into a structured llm_unconfigured response."""
    llm = OpenAICompatibleLLM(model=cfg.llm_model, base_url=cfg.llm_base_url,
                              api_key_env=cfg.llm_api_key_env, thinking=False,
                              max_tokens=cfg.llm_max_tokens)
    # CUSTODIAN_ASK_MAX_CONTEXT_TOKENS (0 = unlimited) -> a soft budget on total context size for the
    # closed pipeline (guards against a 400 from exceeding the context window when switching to a
    # smaller-context LLM backend).
    return Generator(retriever, llm, acl_check=acl_admits,
                     max_context_tokens=cfg.ask_max_context_tokens or None)
