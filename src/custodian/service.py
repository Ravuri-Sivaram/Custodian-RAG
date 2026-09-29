"""The Custodian HTTP daemon (FastAPI): the **only** process in the system that touches the embedded
Qdrant instance and the GPU model.

Why a daemon (rather than each client opening its own index) -- two hard constraints (see
docs/DESIGN.md §2):
  1. The embedded Qdrant client holds an exclusive lock: a second process opening the same path
     fails outright;
  2. The dense model (Qwen3-VL 8B) takes 1-2 minutes to load: with one process per stdio MCP
     session, every new session would pay that cost again.
The daemon holds the index exclusively and keeps the model warm, exposing an HTTP exit shared by
every consumer: curl/scripts (the closed-pipeline /v1/ask), and the thin MCP adapter (agentic,
connecting in a fraction of a second per session).

Security model: matches the engine's stdio server -- **ACL identity is bound at startup**
(CUSTODIAN_TENANT/CUSTODIAN_PRINCIPALS), and an HTTP client cannot change identity via request
parameters; with no tenant set, everything fails closed and returns empty. An optional
CUSTODIAN_API_KEY adds an access gate (when set, /v1/* requires X-API-Key; /healthz is exempt).
**Deploying this service means granting whoever can reach the port everything that identity can
see** -- it binds only to 127.0.0.1 by default.

Tool semantics (argument validation, structured results, dedup, budgeting, error mapping) all come
from the engine's toolcore layer -- this file only does the HTTP binding: routing, identity
injection, per-session dedup (X-Custodian-Session, see sessions.py), and the closed-pipeline
/v1/ask.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import __version__, config, engine, identity as identity_mod, smart_ask, toolcore
from .observability import RequestLog, Stats
from .sessions import SessionRegistry
from embedder import User
from generator import DEFAULT_TABLE_LEG, looks_numeric

log = logging.getLogger("custodian")

# A dedicated thread pool for health probes: /readyz's blocking Qdrant call runs here, kept
# separate from FastAPI's default 40-thread business pool -- so under full load, the business pool
# being saturated can't starve the health probe (which would otherwise make nginx pull a perfectly
# healthy replica out of rotation -- a global outage).
_PROBE_LIMITER = anyio.CapacityLimiter(8)


# ---------- Request models (enum validation for mode/strategy/etc. is left to toolcore, which
# returns a structured bad_arg instead of a bare 422) ----------
class RetrieveReq(BaseModel):
    query: str = ""
    top_k: int | None = None
    rerank: bool = False
    doc_ids: list[str] | None = None
    doc_type: str | None = None
    kind: str | None = None
    mode: str = "full"
    strategy: str = "hybrid"
    rerank_top_n: int | None = None


class AskReq(BaseModel):
    query: str = ""
    top_k: int | None = None
    rerank: bool = False
    include_contexts: bool = False   # when true, citations include the cited passage's raw text (large; by default only provenance metadata comes back)
    # Retrieval filter/routing (same semantics as /v1/retrieve). Motivating case: a question whose
    # numeric answer is buried in a table -- phrased generically, the table chunk gets pushed out
    # of the top-k by prose, and kind='table' fixes that in one shot.
    doc_ids: list[str] | None = None
    doc_type: str | None = None
    kind: str | None = None
    strategy: str | None = None      # hybrid|dense|sparse; None = the engine default (hybrid)


class ExpandReq(BaseModel):
    chunk_id: str = ""
    target_tokens: int = 1500


class GroupedReq(BaseModel):
    query: str = ""
    doc_ids: list[str] = []
    top_k: int = 3
    rerank: bool = False


def create_app(cfg: config.CustodianConfig | None = None, retriever=None, user=None,
               generator_factory=None, keys=None) -> FastAPI:
    """The app factory. Production: everything default (config built from env, the real index
    opened at startup). Tests: inject a fake retriever/user/generator/keys."""
    cfg = cfg or config.from_env()
    # toolcore's delivery budget is read from an environment variable; this line is where the
    # Custodian-level config value gets pushed into that shared namespace (cfg stays the single
    # source of truth).
    os.environ["CUSTODIAN_MAX_CONTEXT_TOKENS"] = str(cfg.max_context_tokens)
    tc = toolcore

    # ---------- Identity mode: keys (team, default) / legacy (single key) / open (loopback only) ----------
    if keys is None and cfg.keys_file:
        keys = identity_mod.load_keys(cfg.keys_file)     # A malformed file -> SystemExit, startup refused
    mode = "keys" if keys else ("legacy" if cfg.api_key else "open")
    if not identity_mod.is_loopback(cfg.host) and mode != "keys":
        # Fail-closed startup guard: deploying is granting access -- binding to a non-loopback
        # address requires multi-identity auth; the whole library is never allowed to run bare on
        # the LAN.
        raise SystemExit(f"CUSTODIAN_HOST={cfg.host} is a non-loopback address; CUSTODIAN_KEYS_FILE (keys mode) must be configured to start.")
    if mode == "keys" and cfg.api_key:
        log.warning("CUSTODIAN_API_KEY is ignored in keys mode (identity comes from the keys file).")

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        if state.user is None:
            state.user = engine.build_user(cfg)
        if state.retriever is None:
            log.info("Opening index %s (collection=%s)...", cfg.qdrant_path, cfg.collection)
            state.retriever = engine.build_retriever(cfg)   # This is where the exclusive embedded-Qdrant lock is acquired
        if not cfg.tenant:
            log.warning("CUSTODIAN_TENANT is not set -- all retrieval will fail closed and return empty (no_identity).")
        yield
        # Graceful shutdown: during drain, flush the background write thread's queued request logs
        # to disk so the last batch isn't lost (observability completeness). This has a timeout: if
        # the disk hangs (e.g. a stuck bind-mount), give up rather than freezing the event loop --
        # an unbounded join could eat the whole stop_grace_period (30s) and get SIGKILLed (uvicorn's
        # 25s drain + this 5s flush stays within that 30s budget).
        try:
            state.reqlog.flush(timeout=5.0)
        except Exception:
            log.warning("shutdown: reqlog flush failed", exc_info=True)

    app = FastAPI(title="Custodian", version=__version__, lifespan=_lifespan)
    state = app.state
    state.cfg, state.tc = cfg, tc
    state.retriever = retriever            # None -> actually built during lifespan (acquires the Qdrant lock)
    state.user = user
    state.gen_local = threading.local()    # Generator/LLM is kept per-thread deliberately: a shared
    state.generator_factory = generator_factory or engine.build_generator   # singleton's last_finish_reason would bleed across concurrent requests
    state.sessions = SessionRegistry()
    state.stats = Stats()                  # In-process metrics plus a JSONL request log
    state.reqlog = RequestLog(cfg.log_dir, log_queries=cfg.log_queries)

    def _current_user(request: Request):
        """The engine User for this request. In keys mode, built fresh from the resolved identity
        (the core of multi-identity support); legacy/open reuse the identity bound at startup."""
        iden = getattr(request.state, "identity", None)
        if mode == "keys" and iden is not None:
            return User(tenant=iden.tenant, principals=list(iden.principals))
        return state.user

    def _iden_name(request: Request) -> str:
        iden = getattr(request.state, "identity", None)
        return iden.name if iden is not None else ("local" if mode == "open" else "default")

    # toolcore's no_identity hint is already worded in terms of CUSTODIAN_* names (the unified
    # naming), so this binding layer doesn't need to translate it further.
    no_id_hint = tc._NO_IDENTITY_HINT

    def _adapt(d: dict) -> dict:
        if isinstance(d, dict) and d.get("status") == "no_identity":
            d["hint"] = no_id_hint
        return d

    # ---------- Auth + identity resolution (/healthz is exempt) ----------
    @app.middleware("http")
    async def _auth(request: Request, call_next):
        if request.url.path not in ("/healthz", "/readyz"):   # readyz gets the same exemption as healthz: orchestrator/nginx probes carry no API key
            k = request.headers.get("x-api-key", "")
            if mode == "keys":
                iden = keys.get(k)
                if iden is None:                        # Unknown or missing -> always 401, never leaking whether the key ever existed
                    return JSONResponse({"status": "unauthorized",
                                         "hint": "Missing or invalid X-API-Key (this service runs in multi-identity keys mode)."},
                                        status_code=401)
                request.state.identity = iden
            elif mode == "legacy":
                if k != cfg.api_key:
                    return JSONResponse({"status": "unauthorized",
                                         "hint": "Missing or incorrect X-API-Key (the server has CUSTODIAN_API_KEY set)."},
                                        status_code=401)
        return await call_next(request)

    # ---------- Observability: timing + request logging (never logs the key itself; truncation happens in the obs layer) ----------
    @app.middleware("http")
    async def _observe(request: Request, call_next):
        t0 = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            # try/finally: still record even if the handler crashed with an uncaught exception --
            # otherwise a serious error would be invisible in observability.
            ms = (time.time() - t0) * 1000
            ep = request.url.path
            if ep.startswith("/v1/") or ep == "/healthz":
                code = response.status_code if response is not None else 500
                extra = dict(getattr(request.state, "log_extra", None) or {})
                biz = extra.get("status")
                # An "error" counts either an HTTP 4xx/5xx **or** a structured business failure
                # (no_access/bad_arg/ask_failed, ...); ok/empty count as success. This used to only
                # look at http>=400, which missed every structured failure that still returned 200.
                err = code >= 400 or (biz is not None and biz not in ("ok", "empty"))
                # The stats key is the **route template** (/v1/documents/{doc_id}), not the raw
                # path: keying on the raw path would let doc_id values give the key space unbounded
                # cardinality (Stats' defaultdict allocates a new deque per key, which in a
                # long-running daemon is a slow leak / a low-effort DoS vector). Requests with no
                # matched route (a 404, or one short-circuited by _auth's 401 -- _observe wraps
                # everything, so both land here) fall into one fixed bucket, so the key space stays
                # bounded at (registered routes + 1). The JSONL record's "ep" field still keeps the
                # raw path (useful for debugging, and it only lives on disk, not in memory).
                route = request.scope.get("route")
                stat_ep = getattr(route, "path", None) or ("/v1/_unmatched" if ep.startswith("/v1/") else ep)
                state.stats.record(stat_ep, ms, err)
                rec = {"ts": round(time.time(), 3), "ep": ep, "user": _iden_name(request),
                       "http": code, "ms": round(ms, 1)}
                if response is None:
                    rec["crashed"] = True
                rec.update(extra)
                state.reqlog.write(rec, state.stats)

    def _log(request: Request, out: dict, **extra):
        """Records log_extra in one place (carrying the business status, which _observe uses for
        both the error count and the on-disk record). Returns `out` unchanged, so this can wrap a
        return statement directly."""
        request.state.log_extra = {"status": out.get("status") if isinstance(out, dict) else None, **extra}
        return out

    def _session_keys(request: Request):
        """Dedup is opt-in: only active when the X-Custodian-Session header is present. The
        registration key is prefixed with the identity name, so under multiple identities, even a
        forged/duplicate session id from a different identity stays invisible to others (the
        assumption that "session-id collisions are harmless" only holds under a single identity)."""
        sid = request.headers.get("x-custodian-session")
        return state.sessions.get(f"{_iden_name(request)}|{sid}") if sid else None

    # ---------- Health / metrics ----------
    @app.get("/healthz")
    async def healthz():
        """Liveness: is the process alive. **async** (a pure in-memory read) -- if this probe
        shared the default 40-thread anyio pool with /v1/ask (an LLM call taking tens of seconds)
        or /v1/retrieve (up to ~361s worst case with retries when inference hangs), under high load
        or a downstream failure the probe could starve in that pool's queue, get reported unhealthy
        by a Docker healthcheck or K8s liveness probe, and trigger a crash loop. An async endpoint
        runs directly on the event loop and never touches the thread pool, sidestepping this
        entirely.
        Information boundary matches /readyz (deliberately): this endpoint is unauthenticated, so
        it never returns reconnaissance-useful details like the collection name, llm_model, or
        identity_mode (the same review conclusion that keeps readyz from returning the collection
        name applies here -- returning it from healthz would undermine that). Those fields live
        behind the admin-gated /v1/stats instead."""
        return {"status": "ok", "service": "custodian", "version": __version__,
                "tenant_bound": bool(cfg.tenant) or mode == "keys",
                "uptime_s": round(time.time() - state.stats.started, 1)}

    @app.get("/readyz")
    async def readyz():
        """Readiness: is the downstream reachable **and is the collection actually ready**. Used
        by a load balancer to route traffic; kept separate from /healthz (liveness) -- a downstream
        blip should only pull this one replica's traffic temporarily, not trigger a crash loop
        (healthz still returns 200). Exempt from auth (same as healthz), so an orchestrator probe
        can call it without a key.
        **async plus a dedicated limiter**: the endpoint itself is async (doesn't consume the
        business 40-thread pool); the blocking Qdrant call is offloaded to _PROBE_LIMITER (8
        threads, isolated from business traffic), and the inference liveness check uses an
        AsyncClient -- neither contends with /v1/* for the default pool, so the probe stays prompt
        even under full load. Without this, a starved probe would make nginx pull a replica that is
        actually working fine out of rotation entirely (a global outage).
        Security: any exception is only logged server-side; the response body **never includes
        str(e)** -- otherwise an unauthenticated probe could read back the internal Qdrant/inference
        host:port (the same "details stay server-side" discipline as /v1/ask). This endpoint is
        excluded from the _observe metrics allowlist -- a high-frequency probe shouldn't pollute
        business metrics."""
        if state.retriever is None:                          # lifespan hasn't finished yet (very early startup): not ready
            return JSONResponse({"status": "starting"}, status_code=503)
        try:
            exists = await anyio.to_thread.run_sync(         # Qdrant's blocking HTTP call goes to the dedicated limiter thread, not the business pool
                lambda: state.retriever.store.client.collection_exists(cfg.collection),
                limiter=_PROBE_LIMITER)
        except Exception:
            log.warning("readyz: Qdrant liveness check failed", exc_info=True)   # details stay server-side only, never leaking internal network topology
            return JSONResponse({"status": "qdrant_unavailable"}, status_code=503)
        if not exists:                                       # A missing collection (a freshly started server that hasn't been indexed yet) means not ready -- don't report healthy while querying an empty store
            return JSONResponse({"status": "collection_missing"}, status_code=503)  # security: never returns the collection name to an unauthenticated probe
        if cfg.inference_url:                                 # Only probe the inference service's /readyz in remote mode
            try:
                import httpx
                # An explicit short timeout: httpx.Timeout(3) budgets 3s per phase, worst case ~9s
                # > a typical 5s healthcheck window -- long enough to get this replica wrongly
                # flagged unhealthy and restarted, so it's kept tight instead.
                async with httpx.AsyncClient() as client:
                    r = await client.get(cfg.inference_url.rstrip("/") + "/readyz",
                                          timeout=httpx.Timeout(1.5, connect=1.0))
                if r.status_code != 200:
                    return JSONResponse({"status": "inference_not_ready"}, status_code=503)
            except Exception:
                log.warning("readyz: inference liveness check failed", exc_info=True)
                return JSONResponse({"status": "inference_unavailable"}, status_code=503)
        return {"status": "ready"}

    @app.get("/v1/stats")
    def stats(request: Request):
        """An in-process metrics snapshot. In keys mode, only an admin key can read it (even the
        pattern of what's being queried is itself information worth gating)."""
        iden = getattr(request.state, "identity", None)
        if mode == "keys" and not (iden is not None and iden.admin):
            return JSONResponse({"status": "forbidden", "hint": "Stats requires an admin key."}, status_code=403)
        snap = state.stats.snapshot()
        snap.update({"status": "ok", "identity_mode": mode, "sessions": len(state.sessions),
                     "collection": cfg.collection, "llm_model": cfg.llm_model,   # moved here from /healthz (an unauthenticated probe shouldn't see this)
                     "log_path": state.reqlog.path if state.reqlog.enabled else ""})
        return snap

    @app.get("/v1/instructions")
    def instructions():
        """The agent usage contract (identical text to the engine's stdio server's MCPServer instructions)."""
        return {"status": "ok", "instructions": tc._INSTRUCTIONS}

    # ---------- Retrieval tool surface (six endpoints, one-to-one with the MCP tools, same semantics as toolcore) ----------
    @app.post("/v1/retrieve")
    def retrieve(q: RetrieveReq, request: Request):
        out = _adapt(tc._retrieve_impl(state.retriever, _current_user(request), q.query, q.top_k, q.rerank,
                                       q.doc_ids, q.doc_type, q.kind, q.mode, q.strategy,
                                       q.rerank_top_n, returned_keys=_session_keys(request)))
        return _log(request, out, query=q.query, n=out.get("meta", {}).get("returned_n"))

    @app.get("/v1/documents")
    def list_documents(request: Request):
        return _log(request, _adapt(tc._list_impl(state.retriever, _current_user(request))))

    @app.get("/v1/documents/{doc_id}")
    def get_document(doc_id: str, request: Request, max_tokens: int = 6000):
        return _log(request, _adapt(tc._get_document_impl(state.retriever, _current_user(request), doc_id, max_tokens)))

    @app.get("/v1/documents/{doc_id}/outline")
    def get_outline(doc_id: str, request: Request):
        return _log(request, _adapt(tc._outline_impl(state.retriever, _current_user(request), doc_id)))

    @app.post("/v1/expand")
    def expand(q: ExpandReq, request: Request):
        return _log(request, _adapt(tc._expand_impl(state.retriever, _current_user(request), q.chunk_id, q.target_tokens)))

    @app.post("/v1/retrieve_grouped")
    def retrieve_grouped(q: GroupedReq, request: Request):
        return _log(request, _adapt(tc._grouped_impl(state.retriever, _current_user(request), q.query, q.doc_ids, q.top_k, q.rerank)))

    # ---------- Closed-pipeline Q&A (generator: retrieval + a grounding prompt + DeepSeek + citation parsing) ----------
    def _get_generator():
        """Lazily built per thread (the thread pool is bounded, so the instance count is bounded
        too): within a single thread, answer() followed by reading finish_reason has no concurrent
        window to race against."""
        gen = getattr(state.gen_local, "gen", None)
        if gen is None:
            gen = state.generator_factory(state.retriever, cfg)
            state.gen_local.gen = gen
        return gen

    @app.post("/v1/ask")
    def ask(q: AskReq, request: Request):
        req_user = _current_user(request)
        if not req_user or not req_user.tenant:
            return _log(request, {"status": "no_identity", "retriable": False, "hint": no_id_hint})
        if not (q.query or "").strip():
            return _log(request, {"status": "empty_query", "retriable": False, "hint": "query is empty; please provide a specific question."})
        if q.strategy is not None and q.strategy not in ("hybrid", "dense", "sparse"):
            return _log(request, {"status": "bad_arg", "retriable": False,
                                  "hint": f"strategy must be hybrid|dense|sparse (got {q.strategy})."})
        try:
            gen = _get_generator()
        except ValueError:
            return _log(request, {"status": "llm_unconfigured", "retriable": False,
                                  "hint": f"Missing LLM API key (set the {cfg.llm_api_key_env} environment variable in .env)."})
        except Exception:                  # Missing openai package, a broken engine path, etc. no longer surface as a bare 500
            log.exception("Generator construction failed")
            return _log(request, {"status": "ask_failed", "retriable": False,
                                  "hint": "Generator initialization failed (a dependency or engine configuration issue); see the server logs."})
        # smart-ask's second layer: **failure-driven** table re-retrieval -- the first pass stays
        # clean; only when a numeric question gets refused (fully or partially) does a second pass
        # add a kind=table leg (hard-capped at one retry; the retry is recorded in `auto`; an
        # explicit kind from the caller is always respected as-is).
        # The earlier "always add the table leg up front" approach was ruled out by a 88-question
        # test (it mishandled 5 prose questions); see docs/TESTING.md §3 for the record -- don't
        # revert to it.
        auto: list[str] = []
        numeric = False
        if cfg.smart_ask:
            numeric = looks_numeric(q.query)
        try:
            # The Qdrant/GPU part of retrieval runs inside the resource locks (Store._lock /
            # Dense._fwd_lock); the LLM network call itself runs outside any lock, so it never
            # blocks other retrieval requests.
            ans = gen.answer(q.query, req_user, top_k=q.top_k, rerank=q.rerank,
                             doc_ids=q.doc_ids, doc_type=q.doc_type, kind=q.kind, strategy=q.strategy)
            if cfg.smart_ask and numeric and q.kind is None and smart_ask.is_refusal(ans.text):
                ans2 = gen.answer(q.query, req_user, top_k=q.top_k, rerank=q.rerank,
                                  doc_ids=q.doc_ids, doc_type=q.doc_type, strategy=q.strategy,
                                  extra_legs=[dict(DEFAULT_TABLE_LEG)])
                # Only a **fully answered** retry is adopted (no lingering refusal or missing-data
                # caveat). The 88-question test showed that a partial answer sometimes carries a
                # wrong missing-data claim ("X was not provided", when X actually is in context),
                # dropping faithfulness from 1.0 to 0.93 -- better to keep the first pass's honest
                # refusal plus hints than to say something wrong. Faithfulness is this system's top
                # priority, ranked above "answer a bit more."
                if not smart_ask.is_refusal(ans2.text):
                    ans = ans2
                    auto.append("table_leg_retry")
                else:
                    auto.append("table_leg_retry_discarded")   # recorded for the trail: a retry was attempted but discarded per policy
        except Exception:
            log.exception("ask failed")   # details stay server-side only, never leak to the client
            return _log(request, {"status": "ask_failed", "retriable": True,
                                  "hint": "Generation failed (a retrieval backend or upstream LLM error); please retry shortly."}, query=q.query)
        citations = []
        for c in ans.citations:
            d = {"marker": c.marker, "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                 "title": c.title, "section": c.section, "page": c.page}
            if q.include_contexts:
                d["text"] = c.text
            citations.append(d)
        # smart-ask's first layer: actionable hints on a refusal or partial refusal (a normal
        # answer is left undisturbed).
        hints = (smart_ask.build_hints(q.query, auto=auto, req_kind=q.kind, req_rerank=q.rerank,
                                   numeric=numeric)
                 if cfg.smart_ask and smart_ask.is_refusal(ans.text) else [])
        # finish_reason is read from the Answer snapshot rather than an instance attribute on
        # gen.llm: if a retry was discarded, an instance attribute would still carry the second
        # round's value (mismatched against the first round's answer that was actually returned),
        # and with zero recall (no LLM call at all) it would carry a stale value from a previous
        # request on the same thread -- the snapshot travels with the answer, so it's always
        # correctly aligned.
        return _log(request, {"status": "ok", "answer": ans.text, "citations": citations,
                              "n_contexts": ans.n_contexts, "model": cfg.llm_model,
                              "finish_reason": ans.finish_reason,
                              "auto": auto, "hints": hints},
                    query=q.query, auto=auto or None, n_citations=len(citations), refusal=bool(hints))

    return app
