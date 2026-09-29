"""Retriever.search_with_context contract/safety unit tests (pure CPU, mocks search+_assemble, no
dependency on GPU/Qdrant). Covers: dedup doesn't fold None (seal#6), exit-point ACL double-check
(seal#2/hard rule 5), sidecar-missing degradation (seal#10), same-anchor big-block dedup
(lazy-tree review), and sidecar version binding (stamped on write, loudly fails on mismatch when read)."""
import json
import os
import tempfile


from embedder.config import SIDECAR_VERSION, EmbedConfig
from embedder.retriever import Retriever
from embedder.types import Hit, User


class _Ctx:
    def __init__(self, acl=None, anchor=None):
        self.acl = acl
        self.text = "ctx"
        self.anchor = anchor


def _hit(cid, doc, sid):
    return Hit(chunk_id=cid, doc_id=doc, kind="text", text="t", score=0.9, payload={"section_id": sid})


def _ret(hits, assemble):
    r = Retriever.__new__(Retriever)               # bypasses __init__ (doesn't build Store/Dense)
    r.search = lambda q, u, top_k=None, rerank=False, doc_ids=None, doc_type=None, kind=None, strategy="hybrid", rerank_top_n=None: hits
    r._assemble = assemble
    return r


def test_none_section_not_folded():
    # seal#6: section_id=None means "no section", not "the same section" -- it must not be folded
    # (otherwise multiple section-less hits in the same doc would collapse to just the first one)
    hits = [_hit("c1", "d", None), _hit("c2", "d", None), _hit("c3", "d", "s1"), _hit("c4", "d", "s1")]
    out = _ret(hits, lambda h, u, cache=None: _Ctx()).search_with_context("q", User(tenant="t", principals=[]))
    ids = [o["hit"].chunk_id for o in out]
    assert ids == ["c1", "c2", "c3"], ids          # c1, c2 both kept (None isn't folded); c3 kept, c4 dedups against it (same s1)


def _hit_kind(cid, doc, sid, kind):
    return Hit(chunk_id=cid, doc_id=doc, kind=kind, text="t", score=0.9, payload={"section_id": sid, "kind": kind})


def test_asset_not_section_deduped():
    # 3: asset blocks (charts/tables) don't participate in section dedup -- otherwise they'd get
    # folded away by a prose sibling in the same section, while the big-block doesn't include
    # their data (_gather doesn't pull asset_content), so it would be lost entirely ("retrieved
    # but the table/chart's numbers can't be answered from"). Assets key independently by chunk_id.
    pub = {"tenant": "t", "visibility": "public", "allow": []}
    hits = [_hit_kind("c1", "d", "s1", "text"), _hit_kind("c2", "d", "s1", "chart"), _hit_kind("c3", "d", "s1", "text")]
    out = _ret(hits, lambda h, u, cache=None: _Ctx(pub)).search_with_context("q", User(tenant="t", principals=[]))
    ids = [o["hit"].chunk_id for o in out]
    assert ids == ["c1", "c2"], ids        # c1 (prose) kept; c2 (chart) kept too, not deduped; c3 (prose, same s1) is deduped away


class _Win:
    """A windowed-block mock (windowed=True), used by R2#1/#5."""
    def __init__(self, rs, anchor):
        self.acl = {"tenant": "t", "visibility": "public", "allow": []}
        self.text = "windowed text"
        self.anchor = anchor
        self.climbed = 0
        self.windowed = True
        self.resolved_section = rs


def test_window_block_status_section_window():
    # R2#1: a windowed block must be marked section_window, and must not pass itself off as
    # full_section (or the agent won't bother calling expand)
    out = _ret([_hit("c1", "d", "s1")], lambda h, u, cache=None: _Win("s0#window", [0, 3])).search_with_context(
        "q", User(tenant="t", principals=[]))
    assert out[0]["context_status"] == "section_window", out[0]["context_status"]


def test_window_dedup_by_resolved_section():
    # R2#5: two windows within the same bound (different anchors, same resolved_section) should
    # fold, with the second one demoted
    ctxs = {"c1": _Win("s0#window", [0, 3]), "c2": _Win("s0#window", [1, 4])}
    out = _ret([_hit("c1", "d", "s1"), _hit("c2", "d", "s2")],
               lambda h, u, cache=None: ctxs[h.chunk_id]).search_with_context("q", User(tenant="t", principals=[]))
    assert out[0]["context"] is not None and out[1]["context"] is None      # c2 folds away (same resolved_section)
    assert out[1]["context_status"] == "deduped"


def test_empty_text_bigblock_status_asset_no_prose():
    # R2#4: a big-block with completely empty text (an asset page with no prose) -> asset_no_prose,
    # doesn't pass itself off as full_section
    empty = _Ctx({"tenant": "t", "visibility": "public", "allow": []}, anchor=[2, 4])
    empty.text = ""
    out = _ret([_hit("c1", "d", "s1")], lambda h, u, cache=None: empty).search_with_context(
        "q", User(tenant="t", principals=[]))
    assert out[0]["context_status"] == "asset_no_prose", out[0]["context_status"]


def test_exit_acl_check_blocks():
    # seal#2/hard rule 5: big.acl is not visible to the user -> context is set to None (not
    # delivered), even though the hit already passed the hard filter
    bad = _Ctx({"tenant": "t2", "visibility": "restricted", "allow": []})   # user is unauthorized
    out = _ret([_hit("c1", "d", "s")], lambda h, u, cache=None: bad).search_with_context(
        "q", User(tenant="t1", principals=["g"]))
    assert out[0]["context"] is None


def test_sidecar_missing_degrades():
    # seal#10: a hit whose sidecar is missing -> that hit degrades (context=None), without
    # dragging down the whole query
    def assemble(h, u, cache=None):
        if h.doc_id == "bad":
            raise FileNotFoundError("sidecar gone")
        return _Ctx({"tenant": "t", "visibility": "public", "allow": []})   # a normal ctx carries a visible acl (acl=None is now rejected at the exit point, review #4)
    out = _ret([_hit("c1", "bad", "s"), _hit("c2", "ok", "s")], assemble).search_with_context(
        "q", User(tenant="t", principals=[]))
    assert len(out) == 2 and out[0]["context"] is None and out[1]["context"] is not None


def test_dedup_same_anchor_bigblock():
    # A same-doc, same-anchor big-block (climbed to the same parent) keeps only the first hit,
    # the rest degrade to a bare hit (lazy-tree review: overlapping hits)
    acl = {"tenant": "t", "visibility": "public", "allow": []}
    amap = {"c1": [0, 5], "c2": [0, 5], "c3": [9, 12]}     # c1, c2 same anchor; c3 different
    out = _ret([_hit("c1", "d", "s1"), _hit("c2", "d", "s2"), _hit("c3", "d", "s3")],
               lambda h, u, cache=None: _Ctx(acl, anchor=amap[h.chunk_id])).search_with_context(
        "q", User(tenant="t", principals=[]))
    ctxs = [o["context"] for o in out]
    assert ctxs[0] is not None and ctxs[1] is None and ctxs[2] is not None   # c2's duplicate anchor -> degraded


def test_writer_stamps_version():
    # Write side: _write_sidecar must stamp the current SIDECAR_VERSION (from the same source the
    # read-side check uses)
    from dataclasses import dataclass

    from embedder.embed import Embedder       # import inside the function: the torch/qdrant chain
                                               # only loads when this test actually runs (import doesn't touch the GPU)

    @dataclass
    class _El:
        idx: int
        text: str

    @dataclass
    class _Sec:
        sec_id: str

    class _Res:                                # mimics ChunkResult: _write_sidecar only reads sections/banners/acl_index()
        sections = [_Sec("s1")]
        banners = frozenset(["banner"])

        def acl_index(self):
            return {0: {"tenant": "t", "visibility": "public", "allow": []}}

    d = tempfile.mkdtemp()
    e = Embedder.__new__(Embedder)             # bypasses __init__ (doesn't build Store/Dense)
    e.cfg = EmbedConfig(sidecar_dir=d)
    e._write_sidecar("doc", [_El(0, "x")], _Res())
    with open(os.path.join(d, "doc.json"), encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == SIDECAR_VERSION, data.get("version")


def _load_with_sidecar(payload: dict):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "doc.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)
    r = Retriever.__new__(Retriever)           # only needs cfg; _load_sidecar's version check runs before the chunker import, doesn't touch the GPU
    r.cfg = EmbedConfig(sidecar_dir=d)
    return r._load_sidecar("doc")


def test_sidecar_version_mismatch_loud():
    # Read side: an explicit old version number -> ValueError (not FileNotFoundError/JSONDecodeError,
    # search_with_context won't catch it -> fails loudly)
    try:
        _load_with_sidecar({"version": 999, "elements": [], "sections": [], "banners": [], "acl_index": {}})
        assert False, "a version mismatch should raise ValueError"
    except ValueError as ex:
        assert "version mismatch" in str(ex), str(ex)


def test_sidecar_missing_version_loud():
    # Read side: an old sidecar with no version field (grandfather decision: missing counts as a
    # mismatch, forcing a rebuild of the baseline) -> also fails loudly
    try:
        _load_with_sidecar({"elements": [], "sections": [], "banners": [], "acl_index": {}})
        assert False, "a missing version field should raise ValueError"
    except ValueError as ex:
        assert "version mismatch" in str(ex), str(ex)


def test_load_sidecar_doc_acl_precheck():
    # Batch2.B (revised after the B2 adversarial review): the doc-level precheck is only used by
    # the **doc_id direct-read tools** (which pass user); an empty acl_index also fail-closes;
    # user=None (hit-driven) skips the precheck -- guards against chunker's _stricter rewriting a
    # hit's acl out of acl_index and wrongly blocking a public hit (HIGH).
    d = tempfile.mkdtemp()

    def _write(name, acl_index):
        p = {"version": SIDECAR_VERSION, "elements": [], "sections": [], "banners": [], "acl_index": acl_index}
        with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(p, f)

    _write("doc", {"0": {"tenant": "t1", "visibility": "restricted", "allow": ["g_fin"], "unset": False}})
    _write("empty", {})                                   # empty acl_index
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig(sidecar_dir=d)
    for label, kw in [("unauthorized user", dict(user=User("t1", ["g_hr"]))),
                      ("direct read with empty acl_index", dict(user=User("t1", ["g_fin"])))]:
        name = "doc" if "unauthorized" in label else "empty"
        try:
            r._load_sidecar(name, **kw)
            assert False, f"{label} should fail-closed and raise PermissionError"
        except PermissionError:
            pass
    assert r._load_sidecar("doc", user=User("t1", ["g_fin"]))[3], "an authorized user's direct read -> succeeds"
    assert r._load_sidecar("doc")[3] is not None, "hit-driven (user=None) skips the precheck, loads normally"
    assert r._load_sidecar("empty"), "an empty acl_index with no user (hit-driven) also loads"


_HR = {"tenant": "t1", "visibility": "restricted", "allow": ["g_hr"], "unset": False}
_PUB = {"tenant": "t1", "visibility": "public", "allow": [], "unset": False}


def test_visible_own_sections_no_parent_leak():
    # B3 review HIGH: a restricted parent section (own-body entirely unauthorized) must not be
    # listed just because a child section is visible -- judged by own-body
    from chunker.types import Section
    secs = {  # parent s0[0,4) own body idx0,1=HR; child s1[2,4) idx2,3=public
        "s0": Section("s0", "d", 1, "confidential HR", ["confidential HR"], 0, 4, None),
        "s1": Section("s1", "d", 2, "public notice", ["confidential HR", "public notice"], 2, 4, "s0"),
    }
    acl_index = {0: _HR, 1: _HR, 2: _PUB, 3: _PUB}
    r = Retriever.__new__(Retriever)
    vis_pub = {s.sec_id for s in r._visible_own_sections(secs, acl_index, User("t1", ["g_pub"]))}
    assert vis_pub == {"s1"}, f"a public-only user should only see the child section s1; the restricted parent s0's title must not leak; got {vis_pub}"
    vis_hr = {s.sec_id for s in r._visible_own_sections(secs, acl_index, User("t1", ["g_hr"]))}
    assert vis_hr == {"s0", "s1"}, "an HR user sees both sections"


def _write_full_sidecar(d, name, elements, sections, acl_index):
    p = {"version": SIDECAR_VERSION, "elements": elements, "sections": sections,
         "banners": [], "acl_index": {str(k): v for k, v in acl_index.items()}}
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(p, f)


def test_get_document_mixed_acl_and_titles():
    # B3 review: get_document gates element-by-element on ACL + includes visible section titles
    # (titles are not in acl_index) + does not return n_elements_total
    from types import SimpleNamespace
    d = tempfile.mkdtemp()
    els = [{"idx": 0, "kind": "text", "text": "confidential HR"}, {"idx": 1, "kind": "text", "text": "HR-BODY"},
           {"idx": 2, "kind": "text", "text": "public notice"}, {"idx": 3, "kind": "text", "text": "PUB-BODY"}]
    secs = [{"sec_id": "s0", "doc_id": "d", "level": 1, "title": "confidential HR", "breadcrumb": ["confidential HR"],
             "start_idx": 0, "end_idx": 4, "parent_sec_id": None},
            {"sec_id": "s1", "doc_id": "d", "level": 2, "title": "public notice", "breadcrumb": ["confidential HR", "public notice"],
             "start_idx": 2, "end_idx": 4, "parent_sec_id": "s0"}]
    _write_full_sidecar(d, "doc", els, secs, {1: _HR, 3: _PUB})   # title idx0/2 are not in acl_index (mimics a real chunker)
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig(sidecar_dir=d)
    r.store = SimpleNamespace(get_doc_lang=lambda doc_id: "en")
    out = r.get_document("doc", User("t1", ["g_pub"]))            # public-only
    assert "PUB-BODY" in out["text"] and "public notice" in out["text"], "should include the visible body + its section title"
    assert "HR-BODY" not in out["text"] and "confidential HR" not in out["text"], "must not include the unauthorized HR body/title"
    assert "n_elements_total" not in out, "must not return a total that would leak an unauthorized count (info leak)"
    assert out["n_elements_visible"] == 2                          # the public notice title + PUB-BODY
    out_hr = r.get_document("doc", User("t1", ["g_hr"]))
    assert "HR-BODY" in out_hr["text"] and "PUB-BODY" in out_hr["text"], "an HR user sees the full text"
    # get_outline: public-only lists only the child section s1, not the restricted parent s0; no breadcrumb returned
    ol = r.get_outline("doc", User("t1", ["g_pub"]))
    assert {s["sec_id"] for s in ol} == {"s1"} and "breadcrumb" not in ol[0]


def test_load_sidecar_sparse_idx_raises():
    # B3 review LEAK fix: elements not densely ordered (idx!=position) -> _load_sidecar's assertion raises (prevents ACL misalignment by position)
    d = tempfile.mkdtemp()
    _write_full_sidecar(d, "sparse", [{"idx": 5, "kind": "text", "text": "a"}, {"idx": 0, "kind": "text", "text": "b"}],
                        [], {})
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig(sidecar_dir=d)
    try:
        r._load_sidecar("sparse")                                 # asserts even with no user (safety precondition for content retrieval)
        assert False, "sparse idx should raise ValueError"
    except ValueError as ex:
        assert "densely ordered" in str(ex)


def test_rerank_graceful_degrade():
    # 4.B: when rerank fails (reranker.rerank raises) -> degrades to returning the hybrid retrieval's hits[:k], without crashing; the returned hits are still in hybrid's scoring dimension
    from types import SimpleNamespace
    hits = [Hit(f"c{i}", "d", "text", "t", 0.1, {}, "rrf") for i in range(3)]
    r = Retriever.__new__(Retriever)
    r.cfg = EmbedConfig()
    r.dense = SimpleNamespace(encode_query=lambda q: SimpleNamespace(tolist=lambda: [0.0] * 8))
    r.store = SimpleNamespace(hybrid_search=lambda *a, **k: list(hits))

    def boom(*a, **k):
        raise RuntimeError("reranker OOM")
    r._get_reranker = lambda: SimpleNamespace(rerank=boom)
    out = r.search("x", User("t", []), top_k=2, rerank=True)
    assert len(out) == 2 and [h.chunk_id for h in out] == ["c0", "c1"], "a rerank failure should degrade to hybrid's top-2"
    assert all(h.score_kind == "rrf" for h in out), "the degraded hits are still in hybrid's dimension (_build uses this to mark rerank_degraded)"


def test_dense_construct_no_gpu():
    # 4.D: constructing Dense doesn't load or touch the GPU (so pure-metadata paths like list_documents still work when the GPU is down)
    from embedder.dense import Dense
    d = Dense(EmbedConfig())
    assert d._model is None and d._gpu_error is None, "must not load the model / touch the GPU at construction time"


def test_dense_query_cache():
    # 5.C: the same query hits the LRU cache the second time, no re-encoding (avoids a redundant 8B forward pass)
    import numpy as np
    from embedder.dense import Dense
    d = Dense(EmbedConfig())
    calls = []
    d.encode_text = lambda texts, instruction=None: (calls.append(texts), np.array([[0.1] * 4]))[1]
    d.encode_query("hello"); d.encode_query("hello")
    assert len(calls) == 1, "the same query should hit the cache the second time"
    d.encode_query("world")
    assert len(calls) == 2, "a different query re-encodes"


if __name__ == "__main__":
    test_none_section_not_folded()
    test_asset_not_section_deduped()
    test_window_block_status_section_window()
    test_window_dedup_by_resolved_section()
    test_empty_text_bigblock_status_asset_no_prose()
    test_exit_acl_check_blocks()
    test_sidecar_missing_degrades()
    test_dedup_same_anchor_bigblock()
    test_writer_stamps_version()
    test_sidecar_version_mismatch_loud()
    test_sidecar_missing_version_loud()
    test_load_sidecar_doc_acl_precheck()
    test_visible_own_sections_no_parent_leak()
    test_get_document_mixed_acl_and_titles()
    test_load_sidecar_sparse_idx_raises()
    test_rerank_graceful_degrade()
    test_dense_construct_no_gpu()
    test_dense_query_cache()
    print("retrieve tests OK")
