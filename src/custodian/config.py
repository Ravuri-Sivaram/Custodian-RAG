"""Custodian configuration: everything comes from environment variables (CUSTODIAN_*), read once at
startup into an immutable dataclass.

Design: no toml/yaml -- a single-instance deployment is well served by one .env file; .env parsing
is a minimal implementation (no python-dotenv dependency needed), and a variable already set in
the real environment always wins. Every setting lives under one unified CUSTODIAN_* namespace (the
old, separate RAG_* namespace disappeared once the engine was folded into this repo).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# The repository root (three levels up from custodian/src/custodian/config.py -- used to locate .env)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_INLINE_COMMENT = re.compile(r"\s+#.*$")


def _parse_env_value(v: str) -> str:
    """Tolerates common .env authoring quirks: a matching pair of quotes is stripped (the quoted
    content, including any '#', is kept as-is); an unquoted value has its trailing inline comment
    stripped (`8787  # service port` -> `8787`, otherwise the later int() call crashes startup)."""
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return _INLINE_COMMENT.sub("", v).strip()


def _load_env_file(path: str) -> None:
    """A minimal .env loader; setdefault semantics (a real environment variable always beats the .env file)."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = _parse_env_value(v)
            if v:
                os.environ.setdefault(k.strip(), v)


def _int_env(name: str, default: int) -> int:
    """A bad integer value raises a named error instead of a bare ValueError crashing startup."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Environment variable {name} must be an integer (got {raw!r}); check your .env.")


def _float_env(name: str, default: float) -> float:
    """Same as _int_env but for floats (used for the inference timeout/backoff settings)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"Environment variable {name} must be a number (got {raw!r}); check your .env.")


def _ns(suffix: str, default: str = "") -> str:
    """CUSTODIAN_<suffix> first, falling back to RAG_<suffix> (a deprecated alias from the engine's
    original naming, kept so an older deployment's .env doesn't silently stop working)."""
    return os.environ.get("CUSTODIAN_" + suffix) or os.environ.get("RAG_" + suffix) or default


def _int_ns(suffix: str, default: int) -> int:
    """Integer version of _ns: CUSTODIAN_<suffix> first, RAG_<suffix> as a fallback, a named error on a bad value."""
    raw = (os.environ.get("CUSTODIAN_" + suffix) or os.environ.get("RAG_" + suffix) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Environment variable CUSTODIAN_{suffix}/RAG_{suffix} must be an integer (got {raw!r}); check your .env.")


def load_env() -> None:
    """Loads custodian/.env (the engine has been folded into this repo, so there is no second .env to fall back to)."""
    _load_env_file(os.path.join(REPO_ROOT, ".env"))


@dataclass(frozen=True)
class CustodianConfig:
    # Identity (ACL): fail-closed -- an empty tenant means every retrieval/listing call returns empty
    tenant: str = ""
    principals: list[str] = field(default_factory=list)
    # Index
    index_dir: str = ""
    qdrant_path: str = ""
    qdrant_url: str = ""          # Server mode (CUSTODIAN_QDRANT_URL); empty = embedded (default, single process). See docs/SCALE_OUT.md Phase D for multi-replica setups.
    sidecar_dir: str = ""
    collection: str = "real"
    dense_dim: int = 1024
    corpus_dir: str = ""         # Source corpus directory for indexing (MinerU parser output; lives outside the repo, via CUSTODIAN_CORPUS_DIR)
    # GPU / local models (dense embedder + reranker; default ~/models, must include the vendor's scripts/ directory; GPU is pinned by name match)
    dense_model_path: str = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
    rerank_model_path: str = os.path.expanduser("~/models/Qwen3-VL-Reranker-8B")
    gpu_name: str = "4090"       # torch device 0's name must contain this string (guards against landing on the wrong GPU under CUDA's fastest-first ordering); leave empty to skip the check
    # Model inference backend: empty = local (loads the GPU model in-process); non-empty = a remote inference service URL, letting the application tier run without a GPU and scale horizontally
    inference_url: str = ""
    # Remote failure-mode knobs (same names/meaning as EmbedConfig; all environment-driven so load testing can tune retries/backoff/backpressure without a code change + image rebuild)
    inference_timeout: float = 120.0          # HTTP read timeout (covers the first, lazy-load call)
    inference_connect_timeout: float = 3.0    # Connect timeout -- short, so a hung server fails fast into retry
    inference_retries: int = 2                # Bounded retry count for 5xx / transport-level errors
    inference_backoff: float = 0.5            # Exponential backoff base
    # Service
    host: str = "127.0.0.1"
    port: int = 8787
    api_key: str = ""            # Legacy single-key mode (a basic access gate for one user)
    keys_file: str = ""          # Keys mode (team deployment): path to a JSON file; setting this enables multi-identity auth
    log_dir: str = ""            # Request log directory; empty = logging off
    log_queries: bool = True     # Whether logs include (truncated) query text; on by default for internal deployments
    max_context_tokens: int = 12000
    # Soft token budget for the total context handed to the LLM by the closed-pipeline /v1/ask
    # (estimated tokens; the whole request is truncated once over budget). 0 = unlimited (default).
    # Distinct from max_context_tokens (the tool-facing retrieval delivery budget) -- only needs
    # changing when swapping in a smaller-context LLM backend.
    ask_max_context_tokens: int = 0
    # smart-ask: the "smarter" product-layer logic on top of /v1/ask (numeric-question table
    # re-retrieval + actionable refusal hints); off = pure/minimal mode.
    smart_ask: bool = True
    # LLM (closed-pipeline /v1/ask)
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_api_key_env: str = "DEEPSEEK_API_KEY"
    llm_max_tokens: int = 2000


def from_env() -> CustodianConfig:
    load_env()
    index_dir = os.path.expanduser(os.environ.get("CUSTODIAN_INDEX_DIR", "~/rag_real"))
    principals = [p.strip() for p in _ns("PRINCIPALS").split(",") if p.strip()]
    # Explicit path overrides also get expanduser() applied (otherwise a literal "~/x" would open
    # an empty store under a literal "./~" directory).
    qdrant = _ns("QDRANT_PATH").strip()
    sidecar = _ns("SIDECAR_DIR").strip()
    return CustodianConfig(
        tenant=_ns("TENANT").strip(),
        principals=principals,
        index_dir=index_dir,
        qdrant_path=os.path.expanduser(qdrant) if qdrant else os.path.join(index_dir, "qdrant"),
        qdrant_url=_ns("QDRANT_URL").strip(),            # CUSTODIAN_QDRANT_URL; non-empty selects server mode (see store.py's three-way branch)
        sidecar_dir=os.path.expanduser(sidecar) if sidecar else os.path.join(index_dir, "sidecar"),
        collection=_ns("COLLECTION", "real"),
        dense_dim=_int_ns("DENSE_DIM", 1024),
        corpus_dir=os.path.expanduser(os.environ.get("CUSTODIAN_CORPUS_DIR", "").strip()),
        dense_model_path=os.path.expanduser(os.environ.get("CUSTODIAN_DENSE_MODEL_PATH", "~/models/Qwen3-VL-Embedding-8B")),
        rerank_model_path=os.path.expanduser(os.environ.get("CUSTODIAN_RERANK_MODEL_PATH", "~/models/Qwen3-VL-Reranker-8B")),
        gpu_name=os.environ.get("CUSTODIAN_GPU_NAME", "4090"),
        inference_url=os.environ.get("CUSTODIAN_INFERENCE_URL", "").strip(),
        inference_timeout=_float_env("CUSTODIAN_INFERENCE_TIMEOUT", 120.0),
        inference_connect_timeout=_float_env("CUSTODIAN_INFERENCE_CONNECT_TIMEOUT", 3.0),
        inference_retries=_int_env("CUSTODIAN_INFERENCE_RETRIES", 2),
        inference_backoff=_float_env("CUSTODIAN_INFERENCE_BACKOFF", 0.5),
        host=os.environ.get("CUSTODIAN_HOST", "127.0.0.1"),
        port=_int_env("CUSTODIAN_PORT", 8787),
        api_key=os.environ.get("CUSTODIAN_API_KEY", "").strip(),
        keys_file=os.path.expanduser(os.environ.get("CUSTODIAN_KEYS_FILE", "").strip()),
        log_dir=os.path.expanduser(os.environ.get("CUSTODIAN_LOG_DIR", "~/custodian_logs").strip()),
        log_queries=os.environ.get("CUSTODIAN_LOG_QUERIES", "on").strip().lower() not in ("off", "0", "false"),
        smart_ask=os.environ.get("CUSTODIAN_SMART_ASK", "on").strip().lower() not in ("off", "0", "false"),
        max_context_tokens=_int_ns("MAX_CONTEXT_TOKENS", 12000),
        ask_max_context_tokens=_int_env("CUSTODIAN_ASK_MAX_CONTEXT_TOKENS", 0),
        llm_base_url=os.environ.get("CUSTODIAN_LLM_BASE_URL", "https://api.deepseek.com"),
        llm_model=os.environ.get("CUSTODIAN_LLM_MODEL", "deepseek-v4-flash"),
        llm_api_key_env=os.environ.get("CUSTODIAN_LLM_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        llm_max_tokens=_int_env("CUSTODIAN_LLM_MAX_TOKENS", 2000),
    )


def adapter_base_url() -> str:
    """The daemon address the thin MCP adapter process needs to connect to (plus an optional API key)."""
    load_env()
    return os.environ.get("CUSTODIAN_URL", f"http://127.0.0.1:{os.environ.get('CUSTODIAN_PORT', '8787')}")
