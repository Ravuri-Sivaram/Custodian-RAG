"""Remote inference backend unit tests (pure CPU, mocks HTTP, doesn't touch a real GPU/service). Covers:
 factory routing / _load no-op / _mrl_np <-> _mrl equivalence / encode_text + query caching / the
 reranker contract;
 P0-1 retries: a transient 503 retries successfully / 4xx raises immediately with zero retries /
 exhausting retries raises InferenceUnavailable / ConnectError is retried;
 P0-1 toolcore dispatch: InferenceUnavailable -> inference_unavailable, ordinary exceptions -> backend_unavailable;
 P0-2 _readiness priority: err takes priority over loading;
 P2-1 dense/reranker share a connection pool; asymmetry: InferenceUnavailable is an Exception
 subclass (so it gets swallowed by retrieve's rerank degradation).

Note: the test itself may use CPU torch for the _mrl(torch) <-> _mrl_np(numpy) comparison (the
application code path under test, remote.py, doesn't import torch)."""
import httpx
import numpy as np
import pytest

from embedder import remote as R
from embedder.config import EmbedConfig
from embedder.dense import Dense
from embedder.errors import InferenceUnavailable
from embedder.remote import RemoteDense, RemoteReranker, make_dense, make_reranker


class _Resp:
    """A fake httpx.Response: carries status_code; raise_for_status raises httpx.HTTPStatusError at >=400 (simulates a 4xx)."""
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._d = data if data is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}",
                                        request=httpx.Request("POST", "http://x"),
                                        response=httpx.Response(self.status_code))

    def json(self):
        return self._d


class _ScriptClient:
    """Responds step-by-step from a script: each item is either a _Resp (returned) or an Exception
    instance (raised). Once exhausted, repeats the last item. Records calls.
    get (health handshake fix3): health defaults to returning 200 with an empty dict (missing
    fields = does not block); health_exc simulates the health check being unreachable."""
    def __init__(self, *steps, health=None, health_exc=None):
        self.steps = list(steps) or [_Resp(200, {})]
        self.calls = []
        self.health, self.health_exc = health, health_exc
        self.get_calls = 0

    def post(self, path, json=None):
        self.calls.append({"path": path, "json": json})
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step

    def get(self, path):
        self.get_calls += 1
        if self.health_exc is not None:
            raise self.health_exc
        return _Resp(200, self.health or {})


def _fast_cfg(**kw):
    """Remote cfg + backoff=0 (retries don't actually sleep, so the test stays fast)."""
    return EmbedConfig(inference_url="http://x", inference_backoff=0.0, **kw)


# ---------- Skeleton contract (original 4 tests) ----------
def test_factory_selects_backend():
    # The single place that decides local vs remote: inference_url empty = Local, non-empty = Remote
    assert type(make_dense(EmbedConfig())).__name__ == "Dense"
    assert type(make_dense(EmbedConfig(inference_url="http://x:8900"))).__name__ == "RemoteDense"
    assert type(make_reranker(EmbedConfig())).__name__ == "Reranker"
    assert type(make_reranker(EmbedConfig(inference_url="http://x:8900"))).__name__ == "RemoteReranker"


def test_remote_dense_load_noop_and_mrl_equivalent_to_local():
    # RemoteDense._load does not load a model; the client's numpy truncation == Dense._mrl's torch
    # truncation (mathematically equivalent => building locally / querying remotely never misaligns)
    torch = pytest.importorskip("torch", reason="the control group needs CPU torch (it's in the [gpu] extra, not [dev]); installed separately in CI")
    rd = RemoteDense(EmbedConfig(inference_url="http://x", dense_dim=1024))
    rd._load()
    assert rd._model is None
    rng = np.random.default_rng(0)
    full = rng.standard_normal((3, 4096)).astype(np.float32)
    full = full / np.linalg.norm(full, axis=-1, keepdims=True)
    remote_out = rd._mrl_np(full)
    local_out = Dense(EmbedConfig(dense_dim=1024))._mrl(torch.from_numpy(full))
    assert remote_out.shape == (3, 1024)
    assert np.allclose(remote_out, local_out, atol=1e-5)


def test_encode_text_and_query_cache():
    rd = RemoteDense(_fast_cfg(dense_dim=1024, query_instruction="Q-INSTR"))
    full = np.random.default_rng(1).standard_normal((1, 4096)).astype(np.float32)
    rd._client = _ScriptClient(_Resp(200, {"vectors": full.tolist()}))
    out = rd.encode_text(["hello"], instruction="I")
    c0 = rd._client.calls[0]
    assert c0["path"] == "/embed" and c0["json"]["texts"] == ["hello"] and c0["json"]["instruction"] == "I"
    assert out.shape == (1, 1024)                                       # server returns full dim -> client truncates to dense_dim
    rd.encode_query("qq")
    assert rd._client.calls[1]["json"]["instruction"] == "Q-INSTR"      # encode_query inherits: query_instruction is added automatically
    rd.encode_query("qq")
    assert len(rd._client.calls) == 2                                   # LRU cache hit, no new HTTP call


def test_reranker_score_contract():
    rr = RemoteReranker(_fast_cfg())
    rr._client = _ScriptClient(_Resp(200, {"scores": [0.1, 0.9]}))
    assert rr.score("q", ["a", "b"]) == [0.1, 0.9]
    c = rr._client.calls[0]
    assert c["path"] == "/rerank" and c["json"]["query"] == "q" and c["json"]["documents"] == ["a", "b"]
    rr2 = RemoteReranker(_fast_cfg())                                   # contract: score count != document count -> fail-loud
    rr2._client = _ScriptClient(_Resp(200, {"scores": [0.5]}))
    with pytest.raises(RuntimeError, match="contract was violated"):
        rr2.score("q", ["a", "b"])


# ---------- P0-1 client retries ----------
def test_retry_succeeds_after_transient_503():
    rd = RemoteDense(_fast_cfg(dense_dim=8, inference_retries=2))
    rd._client = _ScriptClient(_Resp(503), _Resp(503), _Resp(200, {"vectors": [[0.0] * 8]}))
    out = rd.encode_text(["x"])
    assert out.shape == (1, 8)
    assert len(rd._client.calls) == 3                                  # succeeds on the 3rd try, after exactly 2 retries


def test_retry_4xx_no_retry():
    rd = RemoteDense(_fast_cfg(inference_retries=3))
    rd._client = _ScriptClient(_Resp(400))
    with pytest.raises(httpx.HTTPStatusError):                         # a 4xx client error raises immediately, not converted to InferenceUnavailable
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 1                                  # zero retries


def test_retry_exhausts_raises_inference_unavailable():
    rd = RemoteDense(_fast_cfg(inference_retries=2))
    rd._client = _ScriptClient(_Resp(503))                            # persistent 503
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 3                                  # all retries+1 attempts were tried


def test_retry_connect_error_then_exhaust():
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(httpx.ConnectError("refused"))         # can't connect (service down)
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2


def test_inference_unavailable_is_exception_with_marker():
    e = InferenceUnavailable("x")
    assert getattr(e, "inference_unavailable", False) is True          # basis for toolcore's duck-typing
    assert isinstance(e, Exception)                                    # ensures retrieve.py's rerank `except Exception` swallows it (degrades gracefully)


# ---------- P0-1 toolcore dispatch (dense fails loud) ----------
def test_toolcore_dispatches_inference_unavailable():
    from types import SimpleNamespace

    from embedder.types import User
    from custodian import toolcore as tc

    def boom_inf(*a, **k):
        raise InferenceUnavailable("503 warming up")

    def boom_other(*a, **k):
        raise RuntimeError("qdrant down")

    u = User("t", [])
    r_inf = SimpleNamespace(search_with_context=boom_inf, search_grouped=boom_inf)
    assert tc._retrieve_impl(r_inf, u, "q", None, False)["status"] == "inference_unavailable"
    assert tc._grouped_impl(r_inf, u, "q", ["d1"], None, False)["status"] == "inference_unavailable"
    # Negative case: an ordinary backend exception still falls under generic backend_unavailable (not misclassified as inference)
    r_other = SimpleNamespace(search_with_context=boom_other, search_grouped=boom_other)
    assert tc._retrieve_impl(r_other, u, "q", None, False)["status"] == "backend_unavailable"
    assert tc._grouped_impl(r_other, u, "q", ["d1"], None, False)["status"] == "backend_unavailable"


# ---------- P0-2 readiness priority (err takes priority over loading) ----------
def test_readiness_error_precedence():
    from embedder.inference_server import _readiness
    body_err, code_err = _readiness(False, "CUDA error")
    assert code_err == 503 and body_err["status"] == "error"          # load failure -> error, not reported as loading
    assert _readiness(False, None)[0]["status"] == "loading"          # warming up -> loading
    assert _readiness(True, None) == ({"status": "ready"}, 200)       # ready -> ready


# ---------- P2-1 shared connection pool ----------
def test_get_client_shared_by_url():
    R._CLIENTS.clear()
    try:
        cfg = EmbedConfig(inference_url="http://shared:8900")
        d = RemoteDense(cfg)
        rr = RemoteReranker(cfg)
        assert d._client is rr._client                                # dense+reranker for the same url share one connection pool
    finally:
        R._close_clients()                                            # tear down the real client this test created


# ---------- Review fixes: M2/M3/S1/S6b/M4 ----------
def test_retry_5xx_gateway_retried():                                 # M3: 502/504 transient gateway errors (backend was killed) should also be retried
    rd = RemoteDense(_fast_cfg(dense_dim=8, inference_retries=2))
    rd._client = _ScriptClient(_Resp(502), _Resp(504), _Resp(200, {"vectors": [[0.0] * 8]}))
    out = rd.encode_text(["x"])
    assert out.shape == (1, 8) and len(rd._client.calls) == 3


def test_retry_500_exhausts_to_inference_unavailable():              # M3: 500 is also transient (should not be raised bare as a client error)
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(_Resp(500))
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2


def test_retry_connect_timeout_retried():                            # M2: ConnectTimeout is a TimeoutException, not a ConnectError
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(httpx.ConnectTimeout("timeout"))      # before the M2 fix this would raise bare and bypass retries
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2                                # caught -> retried until exhausted


def test_negative_retries_no_crash():                                # S1: a negative config value is clamped to 0, doesn't raise None (TypeError)
    rd = RemoteDense(_fast_cfg(inference_retries=-1))
    rd._client = _ScriptClient(_Resp(503))
    with pytest.raises(InferenceUnavailable):                        # a semantic exception, not a TypeError
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 1                                # clamped to 0 -> tries once


def test_backoff_exponential_sequence(monkeypatch):                  # S6b: backoff really is exponential (2^attempt), not linear
    slept = []
    monkeypatch.setattr(R.time, "sleep", lambda s: slept.append(s))
    rd = RemoteDense(EmbedConfig(inference_url="http://x", inference_retries=2, inference_backoff=0.5))
    rd._client = _ScriptClient(_Resp(503))                          # persistent 503 -> exhausted after backing off twice
    with pytest.raises(InferenceUnavailable):
        rd.encode_text(["x"])
    assert slept == [0.5, 1.0]                                       # backoff*2^0, backoff*2^1


def test_rerank_degrades_on_inference_unavailable():                 # M4: the other half of the asymmetry -- rerank raising InferenceUnavailable is swallowed and degrades gracefully
    from types import SimpleNamespace

    from embedder.retriever import Retriever
    from embedder.types import Hit, User

    hits = [Hit(f"c{i}", "d", "text", "t", 0.1, {}, "rrf") for i in range(3)]
    r = Retriever.__new__(Retriever)                                 # bypass __init__ (don't build a Store/Dense)
    r.cfg = EmbedConfig()
    r.dense = SimpleNamespace(encode_query=lambda q: SimpleNamespace(tolist=lambda: [0.0] * 8))
    r.store = SimpleNamespace(hybrid_search=lambda *a, **k: list(hits))

    def boom(*a, **k):
        raise InferenceUnavailable("inference service 503")

    r._get_reranker = lambda: SimpleNamespace(rerank=boom)
    out = r.search("x", User("t", []), top_k=2, rerank=True)
    assert [h.chunk_id for h in out] == ["c0", "c1"]                 # rerank failure degrades gracefully and returns hybrid top-2, no loud crash
    assert all(h.score_kind == "rrf" for h in out)                   # still hybrid units (the asymmetry of dense failing loud vs rerank degrading)


def test_mrl_fp32_normalize_on_bf16_input():
    """Stage B review M1: the real build-index entry point feeds **bf16 tensors** into Dense._mrl
    (model.process returns bf16). The fp32 fix (.float() first, then truncate+normalize) -> norm=1.0;
    a reverse guard proves that normalizing on bf16 breaks (norm ~= 1.002). Removing dense.py's
    _mrl .float() call makes this test fail immediately -- it guards the P1-2 fix. Pure CPU
    (torch CPU), runs in CI, no GPU needed."""
    torch = pytest.importorskip("torch", reason="needs CPU torch (it's in the [gpu] extra, not [dev]); installed separately in CI")
    import torch.nn.functional as F

    from embedder.dense import Dense
    torch.manual_seed(0)                                             # fix the random seed (bf16 rounding bias depends on the exact values, otherwise flaky)
    emb_bf16 = torch.randn(32, 4096).bfloat16()                     # simulate model.process's bf16 output (the production build-index entry point)
    v = Dense(EmbedConfig(dense_dim=1024))._mrl(emb_bf16)            # production path: bf16 -> _mrl (.float() then truncate+normalize)
    assert v.dtype == np.float32
    v_dev = float(np.abs(np.linalg.norm(v, axis=-1) - 1.0).max())   # deviation from 1 (either direction)
    assert v_dev < 1e-5, f"the fp32 fix should make norm=1.0 (deviation<1e-5), measured deviation {v_dev:.2e}"
    bad = F.normalize(emb_bf16[:, :1024], p=2, dim=-1).float().cpu().numpy()   # old path: normalize on bf16
    bad_dev = float(np.abs(np.linalg.norm(bad, axis=-1) - 1.0).max())
    assert bad_dev > 5e-4, \
        f"normalizing on bf16 should produce a clear norm deviation (proving the fp32 fix is necessary and this test catches regressions), measured deviation {bad_dev:.2e}"


# ---------- P1-1 lower-bound assertion (pure CPU; moved from test_equivalence_gpu.py so a GPU skipif can't swallow it into a false green) ----------
def test_mrl_np_lower_bound_assertion():
    """P1-1: client dense_dim > server's full dim -> fail-loud (does not silently return a too-short
    vector and feed it into the index). Pure CPU, part of the standing regression net."""
    rd = RemoteDense.__new__(RemoteDense)                     # skip __init__ (don't build an httpx client)
    rd.cfg = EmbedConfig(dense_dim=8192)                      # > 4096 full dim
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        rd._mrl_np(np.zeros((2, 4096), dtype=np.float32))


# ---------- Stage F review: broadening the TransportError spectrum (disconnect shapes from docker kill/restart of inference) ----------
def test_retry_read_error_retried():
    """ReadError (peer RST) is a TransportError but **not** a ConnectError/TimeoutException -- the old
    narrow catch would bypass retries and bypass InferenceUnavailable semantics, getting swallowed
    into a non-retried backend_unavailable. After changing to `except httpx.TransportError` it should
    be absorbed."""
    rd = RemoteDense(_fast_cfg(dense_dim=8, inference_retries=2))
    rd._client = _ScriptClient(httpx.ReadError("connection reset"),
                               _Resp(200, {"vectors": [[0.0] * 8]}))
    out = rd.encode_text(["x"])
    assert out.shape == (1, 8) and len(rd._client.calls) == 2      # the first ReadError is retried, the second attempt succeeds


def test_retry_remote_protocol_error_then_exhaust():
    """RemoteProtocolError("Server disconnected without sending a response") is the most typical
    exception when a keep-alive dead connection gets reused after a docker kill/restart. The old
    narrow catch missed it -> the "kill goes unnoticed" chain breaks. After switching to
    TransportError it should retry to exhaustion and raise the semantic exception."""
    rd = RemoteDense(_fast_cfg(inference_retries=1))
    rd._client = _ScriptClient(httpx.RemoteProtocolError("Server disconnected"))
    with pytest.raises(InferenceUnavailable):                     # not a bare RemoteProtocolError bubbling up
        rd.encode_text(["x"])
    assert len(rd._client.calls) == 2                             # all retries+1 attempts were tried


# ---------- Fix 3: first-query handshake (healthz's model_dense/full_dim validation, to prevent the vector space silently misaligning after a model swap) ----------
_OK_VEC = _Resp(200, {"vectors": [[0.0] * 8]})


def test_handshake_model_mismatch_fail_loud():
    """The server swapped models (model_dense doesn't match the client's dense_model_path basename)
    -> the first query raises RuntimeError, and no business POST is sent (otherwise the wrong-space
    vectors would already have been written). Previously only the _mrl_np lower-bound assertion
    existed, and a same-dimension-different-model swap was completely silent."""
    rd = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health={"model_dense": "OtherModel", "full_dim": 4096})
    with pytest.raises(RuntimeError, match="model identity mismatch"):
        rd.encode_text(["x"])
    assert rd._client.calls == []                                 # the check happens before the business request


def test_handshake_full_dim_below_dense_dim_fail_loud():
    rd = RemoteDense(_fast_cfg(dense_dim=1024, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health={"model_dense": "MyEmb", "full_dim": 512})
    with pytest.raises(RuntimeError, match="full_dim"):
        rd.encode_text(["x"])


def test_handshake_ok_verifies_once():
    """Once both fields pass validation -> a flag is set, so subsequent queries have zero GET overhead."""
    rd = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health={"model_dense": "MyEmb", "full_dim": 4096})
    rd.encode_text(["a"])
    rd.encode_text(["b"])
    assert rd._client.get_calls == 1                              # handshake happens only once
    assert len(rd._client.calls) == 2                             # business calls proceed as usual


def test_handshake_unreachable_or_warming_not_blocking():
    """Warm-up edge case: healthz being unreachable / full_dim still being None (warming up) should
    not block business calls (transient failures are left to the _post_retry retry chain), and it
    doesn't count as a completed handshake -- the next query keeps trying until validation succeeds."""
    rd = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd._client = _ScriptClient(_OK_VEC, health_exc=httpx.ConnectError("refused"))
    assert rd.encode_text(["a"]).shape == (1, 8)                  # health check failure doesn't block business calls
    rd2 = RemoteDense(_fast_cfg(dense_dim=8, dense_model_path="/models/MyEmb"))
    rd2._client = _ScriptClient(_OK_VEC, health={"model_dense": "MyEmb", "full_dim": None})   # warming up
    rd2.encode_text(["a"])
    rd2.encode_text(["b"])
    assert rd2._client.get_calls == 2                             # not yet validated -> retries the handshake on every query (stops once validated, see previous test)


# ---------- Fix 5: RemoteReranker.score's signature aligned with the base class (instruction parameter + payload passthrough) ----------
def test_remote_reranker_score_instruction_passthrough():
    """Explicit instruction goes into the payload (base class contract: per-call override); when not
    passed, falls back to cfg.rerank_instruction (old behavior unchanged).
    Removing remote.py score's instruction parameter makes this test fail immediately -- guards
    against an LSP contravariance regression that narrows the input domain."""
    rr = RemoteReranker(_fast_cfg(rerank_instruction="CFG_DEFAULT"))
    rr._client = _ScriptClient(_Resp(200, {"scores": [0.5]}))
    rr.score("q", ["a"], instruction="PER_CALL")
    assert rr._client.calls[0]["json"]["instruction"] == "PER_CALL"
    rr.score("q", ["a"])
    assert rr._client.calls[1]["json"]["instruction"] == "CFG_DEFAULT"
