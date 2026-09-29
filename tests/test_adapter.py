"""MCP thin-adapter unit tests: HTTP forwarding mapping + structured degradation for every
failure kind (never a raw throw to the agent). Replaces mcp_adapter._client with a stub client
(does not start a real service; adapter x real daemon is covered by the GPU smoke test)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
import pytest

from custodian import mcp_adapter as A


class StubClient:
    """Records calls and produces a response via responder (or raises)."""

    def __init__(self, responder):
        self.responder = responder
        self.base_url = "http://stub:8787"
        self.calls: list = []

    def request(self, method, path, json=None, params=None):
        self.calls.append({"method": method, "path": path, "json": json, "params": params})
        return self.responder(method, path, json, params)


@pytest.fixture
def stub(monkeypatch):
    holder = {}

    def install(responder):
        sc = StubClient(responder)
        monkeypatch.setattr(A, "_client", sc)
        holder["c"] = sc
        return sc
    return install


def _ok(payload):
    return lambda m, p, j, q: httpx.Response(200, json=payload)


def test_retrieve_forwards_args(stub):
    sc = stub(_ok({"status": "ok", "hits": []}))
    out = A.retrieve("capex", top_k=5, rerank=True, doc_type="law", strategy="sparse")
    assert out["status"] == "ok"
    call = sc.calls[0]
    assert call["method"] == "POST" and call["path"] == "/v1/retrieve"
    assert call["json"]["query"] == "capex" and call["json"]["top_k"] == 5
    assert call["json"]["rerank"] is True and call["json"]["strategy"] == "sparse"


def test_get_document_uses_params(stub):
    sc = stub(_ok({"status": "ok", "text": "x"}))
    A.get_document("d1", max_tokens=1234)
    call = sc.calls[0]
    assert call["method"] == "GET" and call["path"] == "/v1/documents/d1"
    assert call["params"] == {"max_tokens": 1234}


def test_backend_down_structured(stub):
    def refuse(m, p, j, q):
        raise httpx.ConnectError("connection refused")
    stub(refuse)
    out = A.retrieve("q")
    assert out["status"] == "backend_unavailable" and out["retriable"] is True
    assert "custodian serve" in out["hint"]                     # hint points at the recovery action


def test_401_maps_unauthorized(stub):
    stub(lambda m, p, j, q: httpx.Response(401, json={"status": "unauthorized"}))
    assert A.list_documents()["status"] == "unauthorized"


def test_5xx_maps_backend_unavailable(stub):
    stub(lambda m, p, j, q: httpx.Response(503, text="boom"))
    out = A.expand("c1")
    assert out["status"] == "backend_unavailable" and out["retriable"] is True


def test_422_maps_contract_mismatch_not_retriable(stub):
    # Fix: a 4xx other than 401 (a 422 from version drift breaking the field contract) is a
    # permanent error -- must not be marked retriable=True, or the toolcore contract would teach
    # the agent to loop ineffectively retrying the same request; the hint points at version/URL
    # troubleshooting, not "retry shortly"
    stub(lambda m, p, j, q: httpx.Response(422, json={"detail": "field type error"}))
    out = A.retrieve("q")
    assert out["status"] == "contract_mismatch" and out["retriable"] is False
    assert "CUSTODIAN_URL" in out["hint"] and "versions" in out["hint"]
    assert "retry shortly" not in out["hint"]


def test_404_maps_contract_mismatch_not_retriable(stub):
    # A 404 from CUSTODIAN_URL mistakenly pointing at another service is likewise not retriable
    # (the 3xx branch was pointed at CUSTODIAN_URL after review; 404 was previously missing the same treatment)
    stub(lambda m, p, j, q: httpx.Response(404, text="not found"))
    out = A.get_document("d1")
    assert out["status"] == "contract_mismatch" and out["retriable"] is False


def test_non_json_maps_backend_unavailable(stub):
    stub(lambda m, p, j, q: httpx.Response(200, text="<html>not json</html>"))
    assert A.get_outline("d1")["status"] == "backend_unavailable"


def test_session_header_present_on_real_client():
    # Process-level uuid session header on the adapter: the daemon uses it for per-session dedup isolation
    keys = {k.lower() for k in A._client.headers}
    assert "x-custodian-session" in keys


def test_instructions_same_source_as_engine():
    # Same-source contract: the instructions the adapter ships match the engine's toolcore text exactly (no drift)
    assert "grounding" in A._tc._INSTRUCTIONS and "chunk_id" in A._tc._INSTRUCTIONS
