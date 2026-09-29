"""Qdrant store + ACL hard-filter unit tests (:memory: embedded, fake dense + real BM25 sparse,
pure CPU). Focus: ACL fail-closed -- cross-tenant/unauthorized/unset documents must never be
retrievable."""
import random


from qdrant_client import models

from embedder.config import EmbedConfig
from embedder.sparse import doc_sparse, query_sparse
from embedder.store import Store
from embedder.types import User

DIM = 8


def _vec():
    return [random.random() for _ in range(DIM)]


def _pt(i, text, acl):
    """acl = nested {tenant, allow, visibility, unset}; the payload carries both the split fields
    (for filtering) and the nested acl (for the Batch2.A exit point's acl_admits re-check), matching
    a real embed payload."""
    split = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
             "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(
        id=i, vector={"dense": _vec(), "sparse": doc_sparse(text)},
        payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": "text", "text": text, "acl": acl, **split})


_A = lambda tenant, allow, vis, unset=False: {"tenant": tenant, "allow": allow, "visibility": vis, "unset": unset}


def test_acl_hard_filter():
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()
    s.upsert([
        _pt(1, "public revenue report", _A("t1", [], "public")),                # OK: public
        _pt(2, "t1 hr revenue report", _A("t1", ["g_hr"], "restricted")),     # OK: t1 + g_hr
        _pt(3, "t1 finance revenue", _A("t1", ["g_fin"], "restricted")),    # denied: not authorized (g_fin)
        _pt(4, "t2 hr revenue", _A("t2", ["g_hr"], "restricted")),         # denied: cross-tenant
        _pt(5, "unauthorized revenue", _A("t1", [], "restricted", unset=True)),   # denied: fail-closed (unset)
    ])
    user = User(tenant="t1", principals=["g_hr"])
    hits = s.hybrid_search(_vec(), query_sparse("revenue report"), user, top_k=10)
    ids = {h.chunk_id for h in hits}
    assert "c1" in ids, "the public document should be retrieved"
    assert "c2" in ids, "same tenant + authorized (g_hr) should be retrieved"
    assert "c3" not in ids, "unauthorized (g_fin) leaked!"
    assert "c4" not in ids, "cross-tenant (t2) leaked!"
    assert "c5" not in ids, "fail-closed is broken: an unset document leaked!"


def test_hybrid_search_doc_ids_acl_and():
    # Batch2.B: the doc_ids filter must AND with ACL -- naming an unauthorized doc must still
    # fail-closed to zero results
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()
    s.upsert([
        _pt(1, "public revenue report", _A("t1", [], "public")),                # d1 public, authorized
        _pt(2, "t1 finance revenue", _A("t1", ["g_fin"], "restricted")),    # d2, the g_hr user is not authorized
    ])
    user = User(tenant="t1", principals=["g_hr"])
    only_d1 = s.hybrid_search(_vec(), query_sparse("revenue report"), user, top_k=10, doc_ids=["d1"])
    assert {h.doc_id for h in only_d1} == {"d1"}, "doc_ids=[d1] should only return the authorized d1"
    none = s.hybrid_search(_vec(), query_sparse("revenue"), user, top_k=10, doc_ids=["d2"])
    assert none == [], "naming an unauthorized doc_id (d2) must return zero results (ACL ANDs with doc_id, doc_id cannot bypass ACL)"


def _pt_doc(i, doc, title, acl):
    """A point carrying both nested acl (for the exit point / list_documents re-check) and the
    split fields (for filtering), matching a real embed payload."""
    split = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
             "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(id=i, vector={"dense": _vec()},
        payload={"chunk_id": f"c{i}", "doc_id": doc, "doc_meta": {"title": title}, "acl": acl, **split})


def test_list_documents_acl_scoped():
    # list_documents is scoped by ACL: unauthorized/cross-tenant/unset documents don't appear in
    # the listing (fail-closed, used by the agentic RAG tools)
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()
    A = lambda tenant, allow, vis, unset=False: {"tenant": tenant, "allow": allow,
                                                 "visibility": vis, "unset": unset}
    s.upsert([
        _pt_doc(1, "dPub", "public report", A("t1", [], "public")),
        _pt_doc(2, "dHr", "HR doc", A("t1", ["g_hr"], "restricted")),
        _pt_doc(3, "dFin", "finance doc", A("t1", ["g_fin"], "restricted")),     # t1 has no g_hr access
        _pt_doc(4, "dT2", "other-company doc", A("t2", ["g_hr"], "restricted")),       # cross-tenant
        _pt_doc(5, "dUnset", "unauthorized", A("t1", [], "restricted", unset=True)),  # fail-closed
    ])
    ids = lambda u: {d["doc_id"] for d in s.list_documents(u)[0]}    # returns (docs, truncated)
    assert ids(User("t1", ["g_hr"])) == {"dPub", "dHr"}, "t1/g_hr should only see public + authorized"
    assert ids(User("t1", ["g_fin"])) == {"dPub", "dFin"}, "t1/g_fin sees public + finance"
    assert ids(User("t2", ["g_hr"])) == {"dT2"}, "t2 only sees its own tenant"
    docs, truncated = s.list_documents(User("t1", ["g_hr"]))
    assert {d["doc_id"]: d["title"] for d in docs}["dHr"] == "HR doc", "title should be backfilled"
    assert truncated is False, "5 points is far below the default scan cap, must not report truncation"


def test_list_documents_truncated_signal():
    # Review fix (fix 2): limit is a **cap on visible-chunk scanning**, not a document count --
    # stopping early after hitting the cap must surface truncated=True, or a large store (>10k
    # chunks) would silently produce an incomplete listing, misleading the agent's coverage
    # judgment with no way to detect it.
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()
    pub = _A("t1", [], "public")
    s.upsert([_pt_doc(i, f"doc{i:02d}", f"title{i}", pub) for i in range(1, 11)])   # 10 visible points / 10 docs
    u = User("t1", ["g_hr"])
    docs, truncated = s.list_documents(u, limit=5)
    assert truncated is True, "there's another page (next_page_offset is not None) and the limit was hit -> must report truncation"
    assert len(docs) <= 5
    docs_all, truncated_all = s.list_documents(u)                     # the default limit is enough to scan everything
    assert truncated_all is False and len(docs_all) == 10


def test_ensure_collection_existing_branch_backfills_indexes():
    # Review fix (fix 4): when the collection already exists, idempotently backfill all payload
    # indexes too -- so a collection half-initialized by a mid-build crash (in server mode) can
    # self-heal on the next startup, and any newly added index automatically gets backfilled onto
    # an existing collection.
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()                                             # create branch
    created = []
    s.client.create_payload_index = (
        lambda coll, field_name, field_schema: created.append(field_name))   # instance-level spy (local indexing is a no-op anyway)
    s.ensure_collection()                                             # existing branch: previously returned immediately, indexes were never backfilled
    assert set(created) == {"acl_tenant", "acl_visibility", "acl_allow", "acl_unset",
                            "doc_id", "doc_type", "kind"}, f"the existing branch did not idempotently backfill indexes: {created}"


def test_doc_type_kind_filter():
    # 3.F: doc_type/kind filtering narrows with an AND against ACL
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()

    def pt(i, dt, kind):
        return models.PointStruct(id=i, vector={"dense": _vec(), "sparse": doc_sparse("revenue report")},
            payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": kind, "doc_type": dt, "text": "revenue report",
                     "acl": _A("t1", [], "public"),
                     "acl_tenant": "t1", "acl_allow": [], "acl_visibility": "public", "acl_unset": False})
    s.upsert([pt(1, "financial_research_te", "text"), pt(2, "academic_paper", "text"), pt(3, "financial_research_te", "table")])
    u = User("t1", [])
    q = query_sparse("revenue report")
    assert {h.doc_id for h in s.hybrid_search(_vec(), q, u, top_k=10, doc_type="financial_research_te")} == {"d1", "d3"}
    assert {h.doc_id for h in s.hybrid_search(_vec(), q, u, top_k=10, kind="table")} == {"d3"}
    assert {h.doc_id for h in s.hybrid_search(_vec(), q, u, top_k=10, doc_type="financial_research_te", kind="table")} == {"d3"}


def test_get_by_chunk_id_acl():
    # 3.C prerequisite: get_by_chunk_id fetches the payload directly by chunk_id (uuid5), **since
    # retrieve-by-id bypasses the filter, acl_admits must re-check it**
    import uuid as _uuid
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()

    def pt(cid, acl):
        return models.PointStruct(id=str(_uuid.uuid5(_uuid.NAMESPACE_URL, cid)),
            vector={"dense": _vec()},
            payload={"chunk_id": cid, "doc_id": "d1", "kind": "text", "text": "x", "acl": acl,
                     "acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
                     "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)})
    s.upsert([pt("pub#1", _A("t1", [], "public")), pt("fin#1", _A("t1", ["g_fin"], "restricted"))])
    u = User("t1", ["g_hr"])
    assert s.get_by_chunk_id("pub#1", u)["chunk_id"] == "pub#1", "public + authorized -> gets the payload"
    assert s.get_by_chunk_id("fin#1", u) is None, "unauthorized chunk -> None (retrieve-by-id does not bypass ACL)"
    assert s.get_by_chunk_id("nonexist#9", u) is None, "doesn't exist -> None (same as unauthorized)"


def test_strategy_modes_and_score_kind():
    # 4.A: strategy routing + native score_kind (rrf/cosine/bm25)
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t", prefetch_limit=20))
    s.ensure_collection()
    s.upsert([_pt(1, "public revenue report", _A("t1", [], "public"))])
    u = User("t1", [])
    q = query_sparse("revenue report")
    assert s.hybrid_search(_vec(), q, u, top_k=5)[0].score_kind == "rrf"
    assert s.hybrid_search(_vec(), q, u, top_k=5, strategy="dense")[0].score_kind == "cosine"
    assert s.hybrid_search(_vec(), q, u, top_k=5, strategy="sparse")[0].score_kind == "bm25"
    assert s.hybrid_search(_vec(), None, u, top_k=5, strategy="sparse") == [], "sparse strategy but no sparse query -> empty"


def test_delete_by_doc():
    # deletes a doc's old points before reindexing, deleting only that doc without touching others (lazy-tree review #4)
    s = Store(EmbedConfig(qdrant_path=":memory:", dense_dim=DIM, collection="t"))
    s.ensure_collection()

    def pt(i, doc):
        return models.PointStruct(id=i, vector={"dense": _vec()}, payload={"chunk_id": f"c{i}", "doc_id": doc})

    s.upsert([pt(1, "dA"), pt(2, "dA"), pt(3, "dB")])
    s.delete_by_doc("dA")
    assert s.client.count("t").count == 1                       # dA's 2 points deleted, dB's 1 point kept
    remain = s.client.scroll("t", limit=10)[0]
    assert all(p.payload["doc_id"] == "dB" for p in remain)


if __name__ == "__main__":
    test_acl_hard_filter()
    test_delete_by_doc()
    print("store tests OK")
