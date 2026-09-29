"""Regression tests for the adversarial review P1 (2026-07-02) fixes: one test pins down each
finding that was confirmed / self-verified as real."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from _fakes import FakeRetriever, make_app, make_cfg, make_user
from custodian import config as pconfig
from custodian import mcp_adapter as A
from custodian.indexer import run_index


# ---------- C2: the no_identity hint must name CUSTODIAN_TENANT (not the engine's RAG_TENANT) ----------
def test_no_identity_hint_names_custodian_vars():
    app = make_app(user=make_user(tenant=""), cfg=make_cfg(tenant=""))
    with TestClient(app) as c:
        r1 = c.post("/v1/retrieve", json={"query": "q"}).json()
        r2 = c.post("/v1/ask", json={"query": "q"}).json()
    for r in (r1, r2):
        assert r["status"] == "no_identity"
        assert "CUSTODIAN_TENANT" in r["hint"] and "RAG_TENANT" not in r["hint"]


# ---------- C1: indexer rejects restricted + empty allow (a silent invisibility trap) ----------
def test_indexer_rejects_restricted_without_allow():
    with pytest.raises(SystemExit, match="allow"):
        run_index(make_cfg(), visibility="restricted", allow="")


def test_indexer_restricted_with_allow_passes_guard():
    # With principals set, the guard lets it through (it then exits because the corpus directory
    # doesn't exist -- proving it's the ACL guard that's blocking, not something else)
    with pytest.raises(SystemExit, match="Corpus directory does not exist"):
        run_index(make_cfg(), corpus="Z:/definitely/not/exist", visibility="restricted", allow="g_hr")


# ---------- Model path surfaced to config: build_retriever fails clearly at startup when scripts/ is missing ----------
def test_build_retriever_asserts_model_scripts(tmp_path):
    from custodian.engine import build_retriever
    with pytest.raises(SystemExit, match="scripts directory is missing"):
        build_retriever(make_cfg(dense_model_path=str(tmp_path / "no-such-model")))


# ---------- Stage D review M1: CUSTODIAN_QDRANT_URL must actually reach Store (QdrantClient(url=)), or the server chain doesn't connect (a case where I claimed it was forwarded but missed updating engine) ----------
def test_qdrant_url_reaches_store_via_engine():
    """Guards engine.build_retriever forwarding qdrant_url. Mocks QdrantClient to avoid a real
    connection, pure CPU; deleting engine's qdrant_url= forwarding fails this test."""
    from unittest import mock

    from custodian.engine import build_retriever
    cfg = make_cfg(qdrant_url="http://fake:6333", inference_url="http://fake:8900")   # inference_url is non-empty -> skips the scripts check
    with mock.patch("embedder.store.QdrantClient") as MockQC:
        build_retriever(cfg)
    MockQC.assert_called_once()
    assert MockQC.call_args.kwargs.get("url") == "http://fake:6333", \
        f"engine did not forward qdrant_url to Store (QdrantClient call={MockQC.call_args})"


# ---------- custodian parse: fails clearly when the manifest is missing (not silent, not a bare traceback) ----------
def test_custodian_parse_missing_manifest(tmp_path):
    from custodian.parser import run_parse
    with pytest.raises(SystemExit, match="manifest does not exist"):
        run_parse(str(tmp_path / "nope.csv"), str(tmp_path / "out"), str(tmp_path))


# ---------- ask: a non-ValueError exception from the factory no longer raises a bare 500 ----------
def test_ask_factory_runtime_error_degrades():
    def boom(retriever, cfg):
        raise ModuleNotFoundError("openai")
    with TestClient(make_app(generator_factory=boom)) as c:
        r = c.post("/v1/ask", json={"query": "q"})
    assert r.status_code == 200 and r.json()["status"] == "ask_failed"


# ---------- Adapter: doc_id URL-encoding / empty doc_id rejected locally / 3xx is structured ----------
class _StubClient:
    def __init__(self, responder):
        self.responder = responder
        self.base_url = "http://stub:8787"
        self.calls: list = []

    def request(self, method, path, json=None, params=None):
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        return self.responder(method, path, json, params)


def test_adapter_quotes_doc_id(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(200, json={"status": "ok"}))
    monkeypatch.setattr(A, "_client", sc)
    A.get_document("Q1#report/v2")
    assert sc.calls[0]["path"] == "/v1/documents/Q1%23report%2Fv2"   # # and / no longer truncate the id or change the route
    A.get_outline("research-report 2026")
    assert "%" in sc.calls[1]["path"] and sc.calls[1]["path"].endswith("/outline")


def test_adapter_empty_doc_id_local_bad_arg(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(200, json={"status": "ok"}))
    monkeypatch.setattr(A, "_client", sc)
    assert A.get_document("")["status"] == "bad_arg"
    assert A.get_outline("")["status"] == "bad_arg"
    assert sc.calls == []                    # no HTTP was sent (otherwise an empty doc_id would hit the list endpoint / a 307)


def test_adapter_3xx_structured(monkeypatch):
    sc = _StubClient(lambda m, p, j, q: httpx.Response(307, headers={"location": "/v1/documents"}))
    monkeypatch.setattr(A, "_client", sc)
    out = A.get_document("d1")
    assert out["status"] == "backend_unavailable" and out["retriable"] is True


# ---------- config: .env inline comments/quotes, named int-parse errors, override paths expanduser ----------
def test_parse_env_value():
    assert pconfig._parse_env_value("8787  # service port") == "8787"
    assert pconfig._parse_env_value('"sk-abc # not comment"') == "sk-abc # not comment"
    assert pconfig._parse_env_value("'  spaced  '") == "  spaced  "
    assert pconfig._parse_env_value("plain") == "plain"


def test_int_env_named_error(monkeypatch):
    monkeypatch.setenv("CUSTODIAN_PORT", "8787  x")
    with pytest.raises(SystemExit, match="CUSTODIAN_PORT"):
        pconfig._int_env("CUSTODIAN_PORT", 8787)


def test_override_paths_expanduser(monkeypatch):
    monkeypatch.setenv("CUSTODIAN_QDRANT_PATH", "~/qtest")
    monkeypatch.setenv("CUSTODIAN_SIDECAR_DIR", "~/stest")
    cfg = pconfig.from_env()
    assert cfg.qdrant_path == os.path.expanduser("~/qtest")     # no longer lands in a literal "./~" directory
    assert cfg.sidecar_dir == os.path.expanduser("~/stest")


# ---------- Contract has no drift: the six tools' docstrings are identical, word for word, across
#            the HTTP adapter and stdio-direct transports, and _INSTRUCTIONS has a single source
#            of truth in the in-repo toolcore (no longer exec'ing an external engine repo) ----------
def test_transports_contract_no_drift():
    from custodian import mcp_stdio as S
    from custodian import toolcore
    for name in ["retrieve", "list_documents", "get_document", "get_outline", "expand", "retrieve_grouped"]:
        adp_doc = (getattr(A, name).__doc__ or "").strip()
        std_doc = (getattr(S, name).__doc__ or "").strip()
        assert adp_doc == std_doc, f"{name} docstring differs between the two transports"
    assert A._tc is toolcore                          # the HTTP adapter's tool semantics source of truth = in-repo toolcore
    assert S._INSTRUCTIONS is toolcore._INSTRUCTIONS  # stdio's instructions come from the same source (enforced by the single-source-of-truth structure)


# ---------- N3: /v1/ask forwards retrieval filters ----------
from generator import Generator, MockLLM
from _fakes import make_res, make_hit


def test_ask_forwards_filters_to_retriever():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="table data 6,779,511")])
    app = make_app(retriever=ret, generator_factory=lambda r, c: Generator(r, MockLLM()))
    with TestClient(app) as c:
        r = c.post("/v1/ask", json={"query": "revenue?", "kind": "table",
                                    "doc_ids": ["d1"], "strategy": "sparse"}).json()
    assert r["status"] == "ok"
    call = ret.calls[0]
    assert call["kind"] == "table" and call["doc_ids"] == ["d1"] and call["strategy"] == "sparse"


def test_ask_bad_strategy_structured():
    app = make_app(generator_factory=lambda r, c: Generator(r, MockLLM()))
    with TestClient(app) as c:
        r = c.post("/v1/ask", json={"query": "q", "strategy": "weird"}).json()
    assert r["status"] == "bad_arg" and "strategy" in r["hint"]


# ---------- Stage E: /readyz probes downstream + is exempt from auth (a prerequisite for F's nginx health routing; the only code change) ----------
def _ret_with_qdrant(collection_exists):
    """Adds a store.client.collection_exists to FakeRetriever (the fake has no .client by default)."""
    ret = FakeRetriever()
    ret.store = SimpleNamespace(client=SimpleNamespace(collection_exists=collection_exists))
    return ret


def test_readyz_ready_when_downstream_ok():
    """local mode (inference_url empty): store.client.collection_exists is callable -> ready 200."""
    with TestClient(make_app(retriever=_ret_with_qdrant(lambda name: True))) as c:
        r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_readyz_503_when_qdrant_down():
    """collection_exists raises -> a structured 503 qdrant_unavailable (not a bare crash; nginx
    pulls traffic based on this, healthz still 200)."""
    def boom(name):
        raise RuntimeError("connection refused to http://qdrant:6333")
    with TestClient(make_app(retriever=_ret_with_qdrant(boom))) as c:
        r = c.get("/readyz")
        assert c.get("/healthz").status_code == 200      # liveness is unaffected by downstream (no crashloop)
    assert r.status_code == 503 and r.json()["status"] == "qdrant_unavailable"
    # Review sec-2: the 503 body must [not] echo str(e) -- otherwise an unauthenticated probe
    # could read internal host:port. Removing exc_info from log.warning doesn't matter, but
    # loosening the source's deterrence here should fail this test
    assert "qdrant:6333" not in r.text and "connection refused" not in r.text


def test_readyz_collection_missing_503():
    """Review compose-B: the collection doesn't exist (a freshly started server hasn't migrated
    yet) -> 503 collection_missing, doesn't green-light querying an empty store.
    Removing the `if not exists` branch in service.py should fail this test (reverting to the
    false-green "ready as soon as qdrant is reachable")."""
    with TestClient(make_app(retriever=_ret_with_qdrant(lambda name: False))) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "collection_missing"


def test_readyz_exempt_from_auth():
    """legacy auth mode, no API key: /readyz doesn't 401 (a probe with no key can reach it); /v1/
    still 401s with no key (proving auth is genuinely enforced).
    Removing /readyz from the auth allowlist should fail this test -- the healthcheck would be
    permanently unhealthy in keys/legacy mode."""
    app = make_app(retriever=_ret_with_qdrant(lambda name: True), cfg=make_cfg(api_key="secret"))
    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 200                              # 200 even without a key (exempt)
        assert c.post("/v1/retrieve", json={"query": "q"}).status_code == 401   # no key -> 401 (auth is genuinely enforced)


# ---------- Stage F review: /readyz's inference liveness-probe branch (remote mode; nginx traffic
#            routing genuinely depends on it, previously had zero test coverage) ----------
class _FakeAsyncClient:
    """Fake httpx.AsyncClient: the async-with context + get() (returns a status or raises).
    Mimics how service.readyz uses AsyncClient."""
    def __init__(self, status=None, exc=None):
        self._status, self._exc = status, exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, timeout=None):
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(status_code=self._status)


def test_readyz_inference_ready(monkeypatch):
    """remote mode: qdrant has the collection + inference /readyz 200 -> overall ready 200."""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(status=200))
    app = make_app(retriever=_ret_with_qdrant(lambda name: True),
                   cfg=make_cfg(inference_url="http://inf:8900"))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_readyz_inference_not_ready(monkeypatch):
    """inference is still warming up (/readyz 503) -> custodian /readyz 503 inference_not_ready
    (nginx uses this to hold traffic off this replica)."""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeAsyncClient(status=503))
    app = make_app(retriever=_ret_with_qdrant(lambda name: True),
                   cfg=make_cfg(inference_url="http://inf:8900"))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "inference_not_ready"


def test_readyz_inference_unavailable_no_leak(monkeypatch):
    """inference is unreachable (the liveness probe raises) -> 503 inference_unavailable, and the
    response body doesn't leak internal host:port (same discipline as sec-2)."""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda *a, **k: _FakeAsyncClient(exc=httpx.ConnectError("refused to http://inf:8900")))
    app = make_app(retriever=_ret_with_qdrant(lambda name: True),
                   cfg=make_cfg(inference_url="http://inf:8900"))
    with TestClient(app) as c:
        r = c.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "inference_unavailable"
    assert "inf:8900" not in r.text and "refused" not in r.text


# ---------- Stage F review: the mcp_stdio exit point forwards inference_url (the same "missed
#            updating an exit point" bug from Stage D recurred on the remote switch) ----------
def test_mcp_stdio_config_passes_inference_url(monkeypatch):
    """Guards mcp_stdio._config forwarding inference_url + the model path/gpu_name. Deleting
    either forward should fail this test -- the agentic exit point silently falling back to local."""
    from custodian import mcp_stdio as S
    monkeypatch.setenv("CUSTODIAN_INFERENCE_URL", "http://inf:8900")
    monkeypatch.setenv("CUSTODIAN_TENANT", "demo")
    monkeypatch.setenv("CUSTODIAN_DENSE_MODEL_PATH", "/custom/emb")
    monkeypatch.setenv("CUSTODIAN_GPU_NAME", "5090")
    ec = S._config()
    assert ec.inference_url == "http://inf:8900"        # core: the remote switch is wired up
    assert ec.dense_model_path == "/custom/emb"          # model path comes from the same source
    assert ec.gpu_name_must_contain == "5090"            # gpu name comes from the same source


# ---------- Stage F review: the 4 failure-mode fields CUSTODIAN_* -> CustodianConfig -> forwarded
#            through engine to EmbedConfig ----------
def test_inference_failure_fields_env_to_embedconfig(monkeypatch):
    """CUSTODIAN_INFERENCE_RETRIES/TIMEOUT/... go through from_env into CustodianConfig, then
    through engine.build_retriever to EmbedConfig (otherwise retries/timeout are pinned to
    defaults and tuning in Stage F would require code changes and rebuilding the image)."""
    from unittest import mock

    from custodian.engine import build_retriever
    monkeypatch.setenv("CUSTODIAN_INFERENCE_RETRIES", "5")
    monkeypatch.setenv("CUSTODIAN_INFERENCE_BACKOFF", "1.5")
    monkeypatch.setenv("CUSTODIAN_INFERENCE_TIMEOUT", "90")
    cfg = pconfig.from_env()
    assert cfg.inference_retries == 5 and cfg.inference_backoff == 1.5 and cfg.inference_timeout == 90.0
    # Forwarded to EmbedConfig: mock Retriever intercepts ecfg (avoids actually building Store/Dense)
    cfg2 = make_cfg(inference_url="http://inf:8900", inference_retries=5, inference_backoff=1.5)
    with mock.patch("custodian.engine.Retriever") as MockRet:
        build_retriever(cfg2)
    ecfg = MockRet.call_args.args[0]
    assert ecfg.inference_retries == 5 and ecfg.inference_backoff == 1.5


# ---------- Fix: CUSTODIAN_ASK_MAX_CONTEXT_TOKENS goes through config -> engine to Generator
#            (the closed-pipeline context budget) ----------
def test_ask_max_context_tokens_env_to_generator(monkeypatch):
    """Guards the forwarding chain: env -> CustodianConfig.ask_max_context_tokens ->
    build_generator(max_context_tokens=). Deleting engine's max_context_tokens= forwarding should
    fail this test -- setting the env var would silently have no effect (the same "missed
    updating an exit point" bug from Stage D)."""
    from unittest import mock

    from custodian.engine import build_generator
    monkeypatch.setenv("CUSTODIAN_ASK_MAX_CONTEXT_TOKENS", "6000")
    cfg = pconfig.from_env()
    assert cfg.ask_max_context_tokens == 6000
    with mock.patch("custodian.engine.OpenAICompatibleLLM"), \
         mock.patch("custodian.engine.Generator") as MockGen:
        build_generator(None, make_cfg(ask_max_context_tokens=6000))
        build_generator(None, make_cfg())                      # default 0 -> None (behavior fully unchanged)
    assert MockGen.call_args_list[0].kwargs.get("max_context_tokens") == 6000
    assert MockGen.call_args_list[1].kwargs.get("max_context_tokens") is None


# ---------- Stage F review: P2-3 the server-side /rerank endpoint consumes the client's
#            instruction (converges on "the client is the single source of truth") ----------
def test_reranker_score_consumes_client_instruction():
    """Reranker.score(instruction=) uses the caller's value when it's not None, falling back to
    the server cfg when None. inference_server /rerank relies on this to feed the payload's
    q.instruction into the model (fixes the P2-3 dual-source footgun). Deleting rerank.py's
    instruction parameter should fail this test."""
    import contextlib

    from embedder.config import EmbedConfig
    from embedder.rerank import Reranker
    rr = Reranker.__new__(Reranker)
    rr.cfg = EmbedConfig(rerank_instruction="SERVER_DEFAULT")
    rr._load = lambda: None
    rr._fwd_lock = contextlib.nullcontext()
    captured = {}
    rr._model = SimpleNamespace(process=lambda inp: (captured.update(inp), [0.5, 0.5])[1])
    rr.score("q", ["a", "b"], instruction="CLIENT_WINS")
    assert captured["instruction"] == "CLIENT_WINS"       # client value takes priority (converges on a single source of truth)
    rr.score("q", ["a", "b"], instruction=None)
    assert captured["instruction"] == "SERVER_DEFAULT"     # None falls back to the server default (backward compatible)


# ========== Embedder cluster adversarial-verification fixes (2026-07-07) ==========
# Fix 1: index_document reorders when the old entry is deleted (encoding/sidecar tmp are pure
# preparation -> delete->upsert->replace is a millisecond-scale close)
from dataclasses import dataclass, field


@dataclass
class _Chunk:
    """A minimal chunk double: covers every field that embed._payload reads."""
    chunk_id: str
    doc_id: str
    text: str
    kind: str = "text"
    content_raw: str = ""
    breadcrumb: list = field(default_factory=list)
    section_path: str = ""
    section_id: str = "s1"
    section_anchor: str | None = None
    page_start: int = 0
    page_end: int = 0
    source_indices: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    lang: str = "en"
    doc_type: str = "unknown"
    image_path: str | None = None
    doc_meta: dict = field(default_factory=dict)
    acl: dict = field(default_factory=lambda: {"tenant": "t1", "allow": [], "visibility": "public", "unset": False})


@dataclass
class _El:
    idx: int
    text: str


class _ChunkResult:
    def __init__(self, chunks):
        self.chunks = chunks
        self.sections = []
        self.banners = frozenset()

    def acl_index(self):
        return {0: {"tenant": "t1", "allow": [], "visibility": "public", "unset": False}}


class _FakeDense:
    """A fake dense model: fail_at=N means encode_text starts raising exc on the Nth call
    (simulates remote retries being exhausted / a GPU assertion fast-failing)."""
    def __init__(self, dim=8, fail_at=None, exc=None):
        self.dim, self.fail_at, self.exc = dim, fail_at, exc
        self.n = 0

    def encode_text(self, texts, instruction=None):
        import numpy as np
        self.n += 1
        if self.fail_at is not None and self.n >= self.fail_at:
            raise self.exc or RuntimeError("boom")
        return np.array([[0.1] * self.dim] * len(texts), dtype="float32")


def _mk_embedder(tmp_path, dense):
    from embedder import EmbedConfig, Embedder
    cfg = EmbedConfig(qdrant_path=":memory:", dense_dim=8, collection="t", sidecar_dir=str(tmp_path))
    return Embedder(cfg, dense=dense)


def test_index_document_encode_failure_keeps_old_index(tmp_path):
    """Fix 1's core regression: reindexing fails during encoding (the old implementation had
    already called delete_by_doc) -> the old points/old sidecar must remain usable as-is.
    Before the fix: old vectors were wiped and the doc permanently disappeared from the store
    until manually re-run; after the fix: encoding is pure preparation, a failure has zero side
    effects."""
    from embedder.errors import InferenceUnavailable
    emb = _mk_embedder(tmp_path, _FakeDense())
    chunks = [_Chunk("doc#0", "doc", "paragraph one"), _Chunk("doc#1", "doc", "paragraph two")]
    emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")
    assert emb.store.client.count("t").count == 2
    sidecar_v1 = open(os.path.join(str(tmp_path), "doc.json"), encoding="utf-8").read()
    emb.dense = _FakeDense(fail_at=2, exc=InferenceUnavailable("retries exhausted"))   # blows up encoding the 2nd chunk
    with pytest.raises(InferenceUnavailable):
        emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")
    assert emb.store.client.count("t").count == 2, "an encoding failure is pure preparation; the old vectors must still be searchable (was 0 before the fix)"
    assert open(os.path.join(str(tmp_path), "doc.json"), encoding="utf-8").read() == sidecar_v1, "the old sidecar must not be touched"
    assert not os.path.exists(os.path.join(str(tmp_path), "doc.json.tmp")), "no leftover tmp file"


def test_index_document_post_delete_failure_loud(tmp_path, capsys):
    """Fix 1: a failure after delete (upsert blows up) -> the doc has genuinely fallen out of the
    store, so it must loudly warn "fallen out of the index, needs a re-run" before raising, and
    the sidecar must not be half-updated (replace never runs, tmp is cleaned up)."""
    emb = _mk_embedder(tmp_path, _FakeDense())
    chunks = [_Chunk("doc#0", "doc", "paragraph one")]
    emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")

    def boom(points):
        raise RuntimeError("qdrant write failed")
    emb.store.upsert = boom
    with pytest.raises(RuntimeError, match="qdrant write failed"):
        emb.index_document("doc", [_El(0, "x")], _ChunkResult(chunks), image_root=".")
    err = capsys.readouterr().err
    assert "fallen out of the index" in err and "doc" in err, "a post-delete failure must warn loudly, not be silently swallowed by an upstream 'skip'"
    assert os.path.exists(os.path.join(str(tmp_path), "doc.json")), "the old sidecar is kept (consistent with the new vectors never having landed)"
    assert not os.path.exists(os.path.join(str(tmp_path), "doc.json.tmp")), "tmp is cleaned up"


def test_run_index_collects_failures_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """Companion to fix 1: the indexer no longer silently "skips" a failed doc -- it collects a
    list, summarizes it in the DONE line, and exits nonzero via SystemExit."""
    import custodian.indexer as IX
    corpus = tmp_path / "corpus"
    (corpus / "typeA__doc_bad").mkdir(parents=True)
    (corpus / "typeA__doc_ok").mkdir()
    monkeypatch.setattr(IX, "from_mineru_dir", lambda d: [SimpleNamespace(text="body text")])
    monkeypatch.setattr(IX, "Chunker", lambda: SimpleNamespace(chunk=lambda els, **kw: SimpleNamespace(chunks=[1, 2])))

    class _FakeEmb:
        def __init__(self, cfg):
            pass

        def index_document(self, d, els, res, image_root):
            if "bad" in d:
                raise RuntimeError("inference service unavailable")
            return {}
    monkeypatch.setattr(IX, "Embedder", _FakeEmb)
    with pytest.raises(SystemExit) as ei:
        IX.run_index(make_cfg(), corpus=str(corpus), dest=str(tmp_path / "idx"))
    out = capsys.readouterr().out
    assert "failed typeA__doc_bad" in out                       # per-failure line
    assert "1 failed" in out                                    # DONE summary
    assert "typeA__doc_bad" in str(ei.value), "the exit message must carry the failure list so it can be re-run"


# Fix 6 (narrowed version): search_with_context exposes the same-section fold count (SearchResults.section_folded_n)
def test_search_with_context_counts_section_folds():
    from embedder.retriever import Retriever, SearchResults
    from embedder.types import Hit, User

    pub = {"tenant": "t", "visibility": "public", "allow": [], "unset": False}

    def hit(cid, sid):
        return Hit(chunk_id=cid, doc_id="d", kind="text", text="t", score=0.9, payload={"section_id": sid})

    def mk(hits):
        r = Retriever.__new__(Retriever)                     # bypasses __init__ (doesn't build Store/Dense)
        r.search = lambda q, u, top_k=None, **kw: hits
        r._assemble = lambda h, u, cache=None: SimpleNamespace(acl=pub, text="ctx", anchor=None)
        return r
    out = mk([hit("c1", "s1"), hit("c2", "s1"), hit("c3", "s1"), hit("c4", "s2")]).search_with_context(
        "q", User("t", []))
    assert isinstance(out, SearchResults) and [o["hit"].chunk_id for o in out] == ["c1", "c4"]
    assert out.section_folded_n == 2, "c2/c3 fold into the same s1 and don't appear in the return value; the count must be exposed (to distinguish 'genuinely exhausted' from 'folded')"
    assert mk([hit("c1", "s1")]).search_with_context("q", User("t", [])).section_folded_n == 0
