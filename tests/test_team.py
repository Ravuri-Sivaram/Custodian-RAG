"""Team service surface unit tests (D10 multi-identity + D11 observability): keys parsing is
fail-closed / 401 / identity flows through to the engine / cross-user session isolation / stats
admin gating / non-loopback startup guard / request logging (never logs the key, query is
truncated/can be turned off)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from types import SimpleNamespace

from _fakes import FakeRetriever, make_app, make_cfg, make_hit, make_res, make_user
from custodian import identity as I

ALICE = I.Identity(name="alice", tenant="demo", principals=["g_eng"], admin=True)
BOB = I.Identity(name="bob", tenant="other", principals=[])
KEYS = {"pk_alice_0123456789abcdef": ALICE, "pk_bob_0123456789abcdef": BOB}


# ---------- keys file parsing: fail-closed ----------
def test_load_keys_validation(tmp_path):
    p = tmp_path / "keys.json"
    p.write_text(json.dumps({"keys": [{"key": "short", "name": "a", "tenant": "t"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="too short"):
        I.load_keys(str(p))
    p.write_text(json.dumps({"keys": [{"key": "x" * 20, "name": "", "tenant": "t"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="name/tenant"):
        I.load_keys(str(p))
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="JSON"):
        I.load_keys(str(p))
    with pytest.raises(SystemExit, match="does not exist"):
        I.load_keys(str(tmp_path / "nope.json"))


def test_append_key_roundtrip(tmp_path):
    p = str(tmp_path / "keys.json")
    key = I.append_key(p, name="carol", tenant="demo", principals=["g_hr"], admin=False)
    loaded = I.load_keys(p)
    assert key in loaded and loaded[key].name == "carol" and loaded[key].principals == ["g_hr"]
    assert key.startswith("pk_carol_") and len(key) >= 32


# ---------- keys mode: authentication + identity flows through to the engine ----------
def test_keys_mode_401_and_identity_reaches_engine():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="ctx")])
    with TestClient(make_app(retriever=ret, keys=dict(KEYS))) as c:
        assert c.post("/v1/retrieve", json={"query": "q"}).status_code == 401          # missing key
        assert c.post("/v1/retrieve", json={"query": "q"},
                      headers={"X-API-Key": "wrong"}).status_code == 401               # wrong key
        assert c.get("/healthz").status_code == 200                                    # healthz is exempt
        r = c.post("/v1/retrieve", json={"query": "q"},
                   headers={"X-API-Key": "pk_alice_0123456789abcdef"})
        assert r.status_code == 200 and r.json()["status"] == "ok"
        c.post("/v1/retrieve", json={"query": "q"}, headers={"X-API-Key": "pk_bob_0123456789abcdef"})
    # Identity flows to the engine per request: alice uses tenant demo, bob uses tenant other
    # (what each can see is enforced by the engine's ACL)
    assert ret.calls[0]["user_tenant"] == "demo" and ret.calls[0]["user_principals"] == ["g_eng"]
    assert ret.calls[1]["user_tenant"] == "other"


def test_keys_mode_session_isolated_across_users():
    # Two users spoofing the **same** X-Custodian-Session: the registration key carries an
    # identity-name prefix, so they stay invisible to each other
    def fresh():
        return [make_res(make_hit(cid="c1"), ctx_text="big", anchor=[1, 5])]
    ret = FakeRetriever(results_factory=fresh)
    with TestClient(make_app(retriever=ret, keys=dict(KEYS))) as c:
        a = c.post("/v1/retrieve", json={"query": "q"},
                   headers={"X-API-Key": "pk_alice_0123456789abcdef", "X-Custodian-Session": "S"}).json()
        b = c.post("/v1/retrieve", json={"query": "q"},
                   headers={"X-API-Key": "pk_bob_0123456789abcdef", "X-Custodian-Session": "S"}).json()
        a2 = c.post("/v1/retrieve", json={"query": "q"},
                    headers={"X-API-Key": "pk_alice_0123456789abcdef", "X-Custodian-Session": "S"}).json()
    assert a["hits"][0]["context_status"] == "full_section"
    assert b["hits"][0]["context_status"] == "full_section"       # bob is unaffected by alice
    assert a2["hits"][0]["context_status"] == "already_returned"  # alice's own session dedups normally


# ---------- stats: admin gating ----------
def test_stats_admin_gate():
    with TestClient(make_app(keys=dict(KEYS))) as c:
        assert c.get("/v1/stats", headers={"X-API-Key": "pk_bob_0123456789abcdef"}).status_code == 403
        r = c.get("/v1/stats", headers={"X-API-Key": "pk_alice_0123456789abcdef"})
        assert r.status_code == 200 and r.json()["identity_mode"] == "keys"


def test_stats_open_mode_accessible_and_counts():
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="x")])
    with TestClient(make_app(retriever=ret)) as c:
        c.post("/v1/retrieve", json={"query": "q"})
        snap = c.get("/v1/stats").json()
    assert snap["status"] == "ok" and snap["endpoints"]["/v1/retrieve"]["n"] == 1
    assert snap["endpoints"]["/v1/retrieve"]["p50_ms"] is not None
    # Fix (healthz narrowing): collection/llm_model moved off the unauthenticated /healthz onto
    # this endpoint (admin-gated in keys mode)
    assert snap["collection"] == "real" and snap["llm_model"] == "deepseek-v4-flash"


def test_stats_unauthorized_requests_bounded():
    # Fix (stats key cardinality): in keys mode, 401 requests with no key (short-circuited by
    # _auth, no route template) are merged into a fixed bucket, so a network client can't blow up
    # the stats key set with arbitrary /v1/* paths
    with TestClient(make_app(keys=dict(KEYS))) as c:
        for i in range(5):
            c.post(f"/v1/bogus{i}", json={})
        snap = c.get("/v1/stats", headers={"X-API-Key": "pk_alice_0123456789abcdef"}).json()
    assert snap["endpoints"]["/v1/_unmatched"]["n"] == 5
    assert not any("bogus" in k for k in snap["endpoints"])


# ---------- Non-loopback startup guard ----------
def test_non_loopback_requires_keys():
    with pytest.raises(SystemExit, match="non-loopback"):
        make_app(cfg=make_cfg(host="0.0.0.0"))
    make_app(cfg=make_cfg(host="0.0.0.0"), keys=dict(KEYS))       # keys mode lets it through, no raise


# ---------- Request logging: identity name rather than key; query truncated; can be turned off ----------
def test_request_log_written_no_key_material(tmp_path):
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="x")])
    cfg = make_cfg(log_dir=str(tmp_path))
    app = make_app(retriever=ret, cfg=cfg, keys=dict(KEYS))
    with TestClient(app) as c:
        c.post("/v1/retrieve", json={"query": "secret question" * 40},
               headers={"X-API-Key": "pk_alice_0123456789abcdef"})
    app.state.reqlog.flush()                                       # writing to disk goes through a background writer thread (stage F review), wait for it to drain before reading
    lines = open(os.path.join(str(tmp_path), "requests.jsonl"), encoding="utf-8").read().strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["user"] == "alice" and rec["ep"] == "/v1/retrieve" and rec["http"] == 200
    assert "pk_alice" not in json.dumps(rec)                       # the key itself is never written to disk
    assert len(rec["query"]) <= 120                                # truncated


def test_request_log_queries_off(tmp_path):
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="x")])
    cfg = make_cfg(log_dir=str(tmp_path), log_queries=False)
    app = make_app(retriever=ret, cfg=cfg)
    with TestClient(app) as c:
        c.post("/v1/retrieve", json={"query": "question text that should not appear"})
    app.state.reqlog.flush()                                       # same as above: wait for the background writer thread to drain before reading
    rec = json.loads(open(os.path.join(str(tmp_path), "requests.jsonl"), encoding="utf-8").readline())
    assert "query" not in rec                                      # the privacy boundary runs before enqueueing (log_queries=off strips query), unaffected by the async write


# ---------- Review fixes: name must be unique / no '|'; keys new never raises a bare traceback; observability crash-safety + structured errors counted ----------
def test_load_keys_rejects_dup_name_and_pipe(tmp_path):
    p = tmp_path / "keys.json"
    p.write_text(json.dumps({"keys": [
        {"key": "x" * 20, "name": "dup", "tenant": "t1"},
        {"key": "y" * 20, "name": "dup", "tenant": "t2"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicated"):
        I.load_keys(str(p))
    p.write_text(json.dumps({"keys": [{"key": "x" * 20, "name": "a|b", "tenant": "t"}]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="'\\|'"):
        I.load_keys(str(p))


def test_append_key_rejects_dup_name_and_corrupt(tmp_path):
    p = str(tmp_path / "keys.json")
    I.append_key(p, name="alice", tenant="demo", principals=[])
    with pytest.raises(SystemExit, match="already exists"):
        I.append_key(p, name="alice", tenant="demo", principals=[])
    corrupt = str(tmp_path / "bad.json")
    open(corrupt, "w").write("{not json")
    with pytest.raises(SystemExit, match="JSON"):        # reuses load_keys's validation, doesn't raise a bare traceback
        I.append_key(corrupt, name="bob", tenant="demo", principals=[])


# ---------- Fix: RequestLog.flush has a real timeout (doesn't freeze graceful shutdown when disk hangs) ----------
def test_reqlog_flush_timeout_returns_and_warns(tmp_path, monkeypatch, caplog):
    import logging
    import threading
    import time

    from custodian import observability as obs_mod
    never = threading.Event()

    def stuck_open(*a, **kw):
        never.wait(8)                          # simulates a hung bind-mount IO (far longer than the flush timeout; the daemon thread doesn't block process exit)
        raise OSError("simulated stuck disk")

    monkeypatch.setattr(obs_mod, "open", stuck_open, raising=False)   # only affects open() inside the obs module (the writer thread)
    rl = obs_mod.RequestLog(str(tmp_path))
    rl.write({"ts": 1, "ep": "/v1/x"})
    t0 = time.time()
    with caplog.at_level(logging.WARNING, logger="custodian"):
        rl.flush(timeout=0.5)                  # previously the timeout was silently ignored -> an unbounded join would hang here
    assert time.time() - t0 < 3                # returns on a real timeout, doesn't eat the whole stop_grace_period
    assert any("flush timed out" in rec.getMessage() for rec in caplog.records)
    never.set()                                # release the writer thread so the hang doesn't leak into later tests


def test_reqlog_flush_timeout_normal_drain(tmp_path):
    from custodian.observability import RequestLog
    rl = RequestLog(str(tmp_path))
    rl.write({"ts": 1, "ep": "/v1/x"})
    rl.flush(timeout=5.0)                      # normal disk: returns as soon as drained (doesn't wait out the whole timeout)
    assert open(os.path.join(str(tmp_path), "requests.jsonl"), encoding="utf-8").read().strip()


def test_structured_failure_counted_in_errors():
    # no_identity (HTTP 200 + status=no_identity) must count toward stats.errors (previously only
    # checking http>=400 missed it)
    with TestClient(make_app(user=make_user(tenant=""), cfg=make_cfg(tenant=""))) as c:
        c.post("/v1/retrieve", json={"query": "q"})
        snap = c.get("/v1/stats").json()
    assert snap["endpoints"]["/v1/retrieve"]["n"] == 1 and snap["endpoints"]["/v1/retrieve"]["errors"] == 1


def test_observe_records_on_handler_crash(tmp_path):
    # A handler crash (an uncaught exception propagating to middleware) is still logged and
    # counted (try/finally), it doesn't go invisible in observability.
    # Uses the list_documents path: toolcore._list_impl has no try/except, so a store exception
    # genuinely propagates (retrieve would swallow it).
    def boom(user):
        raise RuntimeError("store boom")
    ret = FakeRetriever()
    ret.store = SimpleNamespace(list_documents=boom)
    cfg = make_cfg(log_dir=str(tmp_path))
    client = TestClient(make_app(retriever=ret, cfg=cfg), raise_server_exceptions=False)
    with client as c:
        assert c.get("/v1/documents").status_code == 500
        snap = c.get("/v1/stats").json()
    assert snap["endpoints"]["/v1/documents"]["errors"] == 1
    recs = [json.loads(l) for l in open(os.path.join(str(tmp_path), "requests.jsonl"), encoding="utf-8")]
    assert any(r.get("crashed") and r["http"] == 500 and r["ep"] == "/v1/documents" for r in recs)
