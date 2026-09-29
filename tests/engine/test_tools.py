"""RAG MCP server tool-logic unit tests (pure CPU, mock retriever, no dependency on GPU/Qdrant/a
real MCP transport). Since Batch1.A, tools return a **structured dict** (not plain text). Covers:
structured fields / big-block preference + fallback / context_status / deduped counting /
score_kind / trust markers (1.D), input validation (no_identity/empty_query/bad_arg fail-closed),
and list. Real ACL filtering (zero cross-tenant results) is covered by
embedder/tests/test_store.py::test_list_documents_acl_scoped."""
import os
from types import SimpleNamespace


from custodian import mcp_stdio as server
from embedder import User


def _hit(cid="c1", doc="d1", title="Title A", section="Chapter 1 > 1.1", score=0.83, text="raw hit chunk text", kind="text"):
    return SimpleNamespace(chunk_id=cid, doc_id=doc, text=text, score=score, kind=kind,
                           payload={"doc_meta": {"title": title}, "section_path": section,
                                    "page_start": 4, "page_end": 4})


def _res(hit, ctx_text=None, status="full_section", anchor=None, n_tokens=42):
    ctx = None
    if ctx_text is not None:
        ctx = SimpleNamespace(text=ctx_text, anchor=anchor, resolved_section="sec#x", n_tokens=n_tokens, climbed=0)
    return {"hit": hit, "context": ctx, "context_status": status}


class _MockRet:
    def __init__(self, results=None, docs=None, document=None, outline=None, expand_big=None, grouped=None):
        self._results, self._docs = results or [], docs or []
        self._document, self._outline, self._expand, self._grouped = document, outline, expand_big, grouped
        self.last_assemble = None
        self.store = SimpleNamespace(list_documents=lambda user: self._docs)

    def search_with_context(self, query, user, top_k=None, rerank=False, doc_ids=None,
                            doc_type=None, kind=None, assemble=True, strategy="hybrid", rerank_top_n=None):
        self.last_assemble = assemble
        return self._results

    def get_document(self, doc_id, user, max_tokens=6000):
        if self._document is None:
            raise PermissionError
        return dict(self._document)

    def get_outline(self, doc_id, user):
        if self._outline is None:
            raise PermissionError
        return self._outline

    def expand(self, chunk_id, user, target_tokens=1500):
        return self._expand

    def search_grouped(self, query, user, doc_ids, top_k=3, rerank=False):
        return self._grouped or {}


def test_structured_prefers_bigblock_then_falls_back():
    ret = _MockRet(results=[_res(_hit(), ctx_text="big-block expanded context", status="full_section"),
                            _res(_hit(cid="c2"), status="single_chunk_no_sidecar")])   # second hit has no ctx -> falls back to the hit chunk
    out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
    assert out["status"] == "ok" and out["meta"]["returned_n"] == 2
    h0, h1 = out["hits"]
    assert "big-block expanded context" in h0["text"] and h0["chunk_id"] == "c1"
    assert h0["context_status"] == "full_section" and h0["title"] == "Title A"
    assert h0["section_path"] == "Chapter 1 > 1.1" and h0["page_start"] == 4
    assert h0["score_kind"] == "rrf" and h0["trust"] == "untrusted"     # 1.D: body text is marked untrusted
    assert "raw hit chunk text" in h1["text"] and h1["context_status"] == "single_chunk_no_sidecar"  # fallback uses hit.text
    assert "warning" in out                                              # top-level untrusted warning


def test_structured_empty():
    out = server._build_retrieve_result(_MockRet(results=[]), User("t", []), "q", 5, False)
    assert out["status"] == "empty" and out["retriable"] and out["hits"] == []


def test_deduped_counted():
    ret = _MockRet(results=[_res(_hit(), ctx_text="a", status="full_section"),
                            _res(_hit(cid="c2"), status="deduped")])
    out = server._build_retrieve_result(ret, User("t", []), "q", 5, False)
    assert out["meta"]["deduped_n"] == 1


def test_bound_user_reads_env(monkeypatch):
    monkeypatch.setenv("CUSTODIAN_TENANT", "t1")
    monkeypatch.setenv("CUSTODIAN_PRINCIPALS", "g_hr, g_fin ,")   # comma-separated, tolerating whitespace/empty entries
    u = server._bound_user()
    assert u.tenant == "t1" and u.principals == ["g_hr", "g_fin"]


def test_retrieve_impl_validation_fail_closed():
    boom = _MockRet()                                     # validation runs before calling the retriever; empty identity/empty query/bad top_k never trigger retrieval
    assert server._retrieve_impl(boom, User("", []), "q", 5, False)["status"] == "no_identity"
    assert server._retrieve_impl(boom, User("t", []), "   ", 5, False)["status"] == "empty_query"
    assert server._retrieve_impl(boom, User("t", []), "q", 0, False)["status"] == "bad_arg"


def test_retrieve_impl_ok():
    ret = _MockRet(results=[_res(_hit(), ctx_text="TSMC2025 capex $38-42 billion")])
    out = server._retrieve_impl(ret, User("t", ["g"]), "capex", 5, False)
    assert out["status"] == "ok" and "380-420" in out["hits"][0]["text"]


def test_list_impl_and_fail_closed():
    ret = _MockRet(docs=[{"doc_id": "dPub", "title": "public report"}, {"doc_id": "dHr", "title": "HR doc"}])
    out = server._list_impl(ret, User("t", ["g"]))
    assert out["status"] == "ok" and {d["doc_id"] for d in out["documents"]} == {"dPub", "dHr"}
    assert server._list_impl(ret, User("", []))["status"] == "no_identity"


def test_retrieve_concise_and_filters():
    # 3.E concise -> assemble=False is forwarded + meta.mode=concise; 3.A/3.F filters go into meta
    ret = _MockRet(results=[_res(_hit(), ctx_text="x")])
    out = server._retrieve_impl(ret, User("t", ["g"]), "q", 5, False, doc_ids=["d1"], doc_type="dt", kind="table", mode="concise")
    assert ret.last_assemble is False and out["meta"]["mode"] == "concise"
    assert out["meta"]["filters"] == {"doc_ids": ["d1"], "doc_type": "dt", "kind": "table"}
    assert server._retrieve_impl(ret, User("t", []), "q", 5, False, mode="weird")["status"] == "bad_arg"


def test_hit_dict_multimodal():
    # 3.H: table/chart carries content_raw; image/chart carries image_path
    th = SimpleNamespace(chunk_id="t1", doc_id="d1", text="table placeholder", score=0.5, kind="table",
                         payload={"content_raw": "<table>...</table>", "doc_meta": {}})
    d = server._hit_dict(1, {"hit": th, "context": None, "context_status": "concise"})
    assert d["kind"] == "table" and d["content_raw"] == "<table>...</table>"
    ih = SimpleNamespace(chunk_id="i1", doc_id="d1", text="image placeholder", score=0.5, kind="image",
                         payload={"image_path": "images/fig1.png", "doc_meta": {}})
    assert server._hit_dict(1, {"hit": ih, "context": None})["image_path"] == "images/fig1.png"


def test_get_document_impl():
    ret = _MockRet(document={"doc_id": "d1", "text": "full text...", "n_tokens": 3,
                             "n_elements_visible": 4, "truncated": False})   # R4.F: does not include n_elements_total (the implementation doesn't return it)
    out = server._get_document_impl(ret, User("t", ["g"]), "d1", 6000)
    assert out["status"] == "ok" and out["text"] == "full text..." and out["trust"] == "untrusted"
    # unauthorized/nonexistent -> get_document raises PermissionError -> no_access (same response, doesn't leak existence)
    assert server._get_document_impl(_MockRet(document=None), User("t", ["g"]), "d1", 6000)["status"] == "no_access"
    assert server._get_document_impl(ret, User("t", ["g"]), "", 6000)["status"] == "bad_arg"
    assert server._get_document_impl(ret, User("", []), "d1", 6000)["status"] == "no_identity"


def test_outline_impl():
    ret = _MockRet(outline=[{"sec_id": "s1", "title": "Chapter 1", "level": 1}])
    out = server._outline_impl(ret, User("t", ["g"]), "d1")
    assert out["status"] == "ok" and out["sections"][0]["title"] == "Chapter 1"
    assert server._outline_impl(_MockRet(outline=None), User("t", ["g"]), "d1")["status"] == "no_access"


def test_expand_impl():
    big = SimpleNamespace(text="larger context", anchor=[2, 8], resolved_section="sec#1", n_tokens=420, climbed=1)
    out = server._expand_impl(_MockRet(expand_big=big), User("t", ["g"]), "c1", 1500)
    assert out["status"] == "ok" and out["text"] == "larger context" and out["anchor"] == [2, 8] and out["trust"] == "untrusted"
    assert server._expand_impl(_MockRet(expand_big=None), User("t", ["g"]), "c1", 1500)["status"] == "no_access"
    assert server._expand_impl(_MockRet(), User("", []), "c1", 1500)["status"] == "no_identity"


def test_grouped_impl():
    ret = _MockRet(grouped={"d1": [_hit(cid="a")], "d2": []})
    out = server._grouped_impl(ret, User("t", ["g"]), "compare", ["d1", "d2"], 3, False)
    assert out["status"] == "ok" and out["groups"]["d1"][0]["chunk_id"] == "a" and out["groups"]["d2"] == []
    assert server._grouped_impl(ret, User("t", ["g"]), "q", [], 3, False)["status"] == "bad_arg"      # no doc_ids
    assert server._grouped_impl(ret, User("", []), "q", ["d1"], 3, False)["status"] == "no_identity"


def test_strategy_validation_and_rerank_degraded():
    # 4.A strategy validation + 4.E top_k None + 4.B rerank_degraded (mock result score_kind=rrf, rerank=True -> degraded)
    ret = _MockRet(results=[_res(_hit(), ctx_text="x")])
    assert server._retrieve_impl(ret, User("t", []), "q", 5, False, strategy="weird")["status"] == "bad_arg"
    out = server._retrieve_impl(ret, User("t", ["g"]), "q", None, True, strategy="dense")
    assert out["status"] == "ok" and out["meta"]["rerank_degraded"] is True
    assert out["meta"]["strategy"] == "dense" and out["meta"]["requested_k"] is None


def test_cross_call_dedup():
    # 5.A: the same returned_keys set across two calls -- a repeated (doc_id, anchor) degrades to an already_returned pointer the second time
    keys = set()
    ret = _MockRet(results=[_res(_hit(cid="c1"), ctx_text="big", anchor=[1, 5])])
    o1 = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o1["hits"][0]["context_status"] == "full_section" and o1["hits"][0]["text"]
    o2 = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o2["hits"][0]["context_status"] == "already_returned" and o2["hits"][0]["text"] == ""
    assert o2["meta"]["already_returned_n"] == 1


def test_budget_truncation():
    # 5.B: hits past the soft token cap have their body cleared but their address kept, marked
    # omitted_budget (400 tokens each, budget=500: first hit kept, second exceeds)
    os.environ["CUSTODIAN_MAX_CONTEXT_TOKENS"] = "500"   # note: _max_ctx_tokens has a max(500,..) floor
    try:
        ret = _MockRet(results=[_res(_hit(cid="a"), ctx_text="x", anchor=[1, 2], n_tokens=400),
                                _res(_hit(cid="b"), ctx_text="y", anchor=[3, 4], n_tokens=400)])
        out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
        assert out["hits"][0]["context_status"] == "full_section" and out["hits"][0]["text"]
        assert out["hits"][1]["context_status"] == "omitted_budget" and out["hits"][1]["text"] == ""
        assert out["meta"]["budget_truncated"] is True and out["hits"][1]["chunk_id"] == "b"   # address is kept
    finally:
        del os.environ["CUSTODIAN_MAX_CONTEXT_TOKENS"]


def test_list_coverage():
    # 6.B: list_documents returns coverage (a document count per doc_type, so the agent can judge scope)
    ret = _MockRet(docs=[{"doc_id": "d1", "title": "A", "doc_type": "academic_paper"},
                         {"doc_id": "d2", "title": "B", "doc_type": "academic_paper"},
                         {"doc_id": "d3", "title": "C", "doc_type": "financial_research_te"}])
    out = server._list_impl(ret, User("t", ["g"]))
    assert out["coverage"] == {"academic_paper": 2, "financial_research_te": 1}


def test_list_truncated_passthrough_and_hint():
    # Review fix (fix 2): store.list_documents's (docs, truncated) shape -> truncated is forwarded
    # + a truncation hint is added; a bare list (test doubles / the old implementation) is treated
    # compatibly as not truncated, no regression forced onto them.
    docs = [{"doc_id": "d1", "title": "A", "doc_type": "x"}]
    out = server._list_impl(_MockRet(docs=(docs, True)), User("t", ["g"]))
    assert out["status"] == "ok" and out["truncated"] is True
    assert "incomplete" in out["hint"], "when truncated, the agent must be told the listing/coverage is incomplete"
    out2 = server._list_impl(_MockRet(docs=(docs, False)), User("t", ["g"]))
    assert out2["truncated"] is False and out2["hint"] == ""
    out3 = server._list_impl(_MockRet(docs=docs), User("t", ["g"]))       # bare-list compatibility
    assert out3["truncated"] is False


def test_section_folded_n_in_meta():
    # Review fix (fix 6): when search_with_context returns a SearchResults (a list subclass), its
    # section_folded_n goes into meta (alongside deduped_n, letting the agent distinguish "the
    # store is genuinely exhausted" from "hits were folded into the same section"); a bare list
    # (mock) defaults to 0.
    from embedder.retriever import SearchResults
    results = SearchResults([_res(_hit(), ctx_text="a")])
    results.section_folded_n = 3
    out = server._build_retrieve_result(_MockRet(results=results), User("t", ["g"]), "q", 5, False)
    assert out["meta"]["section_folded_n"] == 3
    out2 = server._build_retrieve_result(_MockRet(results=[_res(_hit(cid="c9"), ctx_text="b")]),
                                         User("t", ["g"]), "q", 5, False)
    assert out2["meta"]["section_folded_n"] == 0                          # a bare list has no such attribute -> 0


def test_recovery_hint():
    # 6.C: when over budget, the hint points at expand (a closed error-recovery loop)
    os.environ["CUSTODIAN_MAX_CONTEXT_TOKENS"] = "500"
    try:
        ret = _MockRet(results=[_res(_hit(cid="a"), ctx_text="x", anchor=[1, 2], n_tokens=400),
                                _res(_hit(cid="b"), ctx_text="y", anchor=[3, 4], n_tokens=400)])
        out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
        assert "omitted_budget" in out["hint"] and "expand" in out["hint"]
    finally:
        del os.environ["CUSTODIAN_MAX_CONTEXT_TOKENS"]


def test_asset_content_raw_demoted_and_budgeted():
    # R4.A: an asset hit's (large) content_raw counts against the budget; going over marks
    # omitted_budget and **clears content_raw too** (no longer emitted as-is)
    os.environ["CUSTODIAN_MAX_CONTEXT_TOKENS"] = "500"
    try:
        th = SimpleNamespace(chunk_id="t1", doc_id="d1", text="", score=0.8, kind="table",
                             payload={"content_raw": "X" * 4000, "doc_meta": {}})   # a table worth ~1000 tokens
        asset_res = {"hit": th, "context": SimpleNamespace(text="", anchor=[3, 4], resolved_section="sec#a",
                                                           n_tokens=0, climbed=0), "context_status": "asset_no_prose"}
        ret = _MockRet(results=[_res(_hit(cid="p1"), ctx_text="p", anchor=[1, 2], n_tokens=400), asset_res])
        out = server._build_retrieve_result(ret, User("t", ["g"]), "q", 5, False)
        asset = out["hits"][1]
        assert asset["context_status"] == "omitted_budget"                 # content_raw counts against the budget -> demoted when over
        assert asset.get("content_raw") is None and asset["text"] == ""    # the large field is cleared too, not emitted as-is around the budget
    finally:
        os.environ.pop("CUSTODIAN_MAX_CONTEXT_TOKENS", None)


def test_omitted_not_registered_returned_keys():
    # R4.C: a hit demoted to omitted_budget (body never delivered) is not registered in
    # returned_keys -> next time it's not mistakenly treated as already_returned
    os.environ["CUSTODIAN_MAX_CONTEXT_TOKENS"] = "500"
    try:
        keys = set()
        res = lambda: [_res(_hit(cid="a"), ctx_text="x", anchor=[1, 2], n_tokens=400),
                       _res(_hit(cid="b"), ctx_text="y", anchor=[3, 4], n_tokens=400)]
        o1 = server._build_retrieve_result(_MockRet(results=res()), User("t", ["g"]), "q", 5, False, returned_keys=keys)
        assert o1["hits"][1]["context_status"] == "omitted_budget"         # b went over budget, never delivered
        o2 = server._build_retrieve_result(_MockRet(results=res()), User("t", ["g"]), "q", 5, False, returned_keys=keys)
        assert o2["hits"][1]["context_status"] != "already_returned"       # b wasn't delivered last time, so it shouldn't say "already returned" this time
    finally:
        os.environ.pop("CUSTODIAN_MAX_CONTEXT_TOKENS", None)


def test_section_window_cross_call_dedup_by_resolved_section():
    # R4.B: section_window dedups across calls by resolved_section (anchor drifts with the seed);
    # the same section window is already_returned the second time
    keys = set()
    r1 = _res(_hit(cid="w1"), ctx_text="window A", status="section_window", anchor=[1, 5])   # resolved_section=sec#x
    o1 = server._build_retrieve_result(_MockRet(results=[r1]), User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o1["hits"][0]["context_status"] == "section_window"
    r2 = _res(_hit(cid="w2"), ctx_text="window B", status="section_window", anchor=[2, 7])   # same section, different anchor
    o2 = server._build_retrieve_result(_MockRet(results=[r2]), User("t", ["g"]), "q", 5, False, returned_keys=keys)
    assert o2["hits"][0]["context_status"] == "already_returned"          # dedup hits by resolved_section (anchor alone would miss it)


def test_instructions_cover_contract():
    # 6.A/6.B/6.D: server instructions cover the key contract points (grounding/untrusted data/routing/citation anchors/status)
    ins = server._INSTRUCTIONS
    for kw in ["grounding", "not instructions", "chunk_id", "no relevant information", "context_status"]:
        assert kw in ins, f"instructions missing key point: {kw}"


if __name__ == "__main__":
    for fn in [test_structured_prefers_bigblock_then_falls_back, test_structured_empty, test_deduped_counted,
               test_bound_user_reads_env, test_retrieve_impl_validation_fail_closed, test_retrieve_impl_ok,
               test_list_impl_and_fail_closed, test_retrieve_concise_and_filters, test_hit_dict_multimodal,
               test_get_document_impl, test_outline_impl, test_expand_impl, test_grouped_impl,
               test_strategy_validation_and_rerank_degraded, test_cross_call_dedup, test_budget_truncation,
               test_list_coverage, test_list_truncated_passthrough_and_hint, test_section_folded_n_in_meta,
               test_recovery_hint,
               test_asset_content_raw_demoted_and_budgeted, test_omitted_not_registered_returned_keys,
               test_section_window_cross_call_dedup_by_resolved_section, test_instructions_cover_contract]:
        fn()
    print("MCP tool (toolcore/mcp_stdio) tests OK")
