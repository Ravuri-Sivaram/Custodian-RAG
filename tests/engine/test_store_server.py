"""Qdrant **server-mode** regression tests (Stage D go/no-go). Requires a Qdrant server @
localhost:6333; automatically skips in CI / without a server.
Manual run: start the server (~/qdrant/qdrant), then pytest tests/engine/test_store_server.py -v
Force no silent skip (used by the CI compose job): CUSTODIAN_REQUIRE_QDRANT_SERVER=1 pytest ...

Covers (revised after the Stage D adversarial review):
  Q2 ACL: the embedded QdrantLocal's RRF fusion drops the top-level should (a hard rule ->
    unauthorized-access leak), which store.py works around by pushing ACL down into every prefetch.
    - Pipeline level: all three of hybrid/dense/sparse fail-closed on final delivery (including
      the exit point's acl_admits re-check);
    - **Fusion level (M4): bypasses the exit-point acl_admits and directly asserts the server RRF's
      raw output contains no unauthorized access** -- this is the real claim behind "the hard rule
      doesn't recur on the server" (otherwise the exit-point re-check would mask a fusion-level leak);
    - list_documents(scroll)'s should semantics + a raw scroll probe (S1).
  Multi-replica locking: the embedded backend is exclusive to a single process -> the server lets
    two clients open the same collection (note: same process -- this verifies the embedded file-lock
    ceiling was removed at the client layer, not a true multi-process replica scenario, which is
    left to Stage E's compose end-to-end test)."""
import os
import random
import shutil
import tempfile

import pytest
from qdrant_client import QdrantClient, models

from embedder.config import EmbedConfig
from embedder.sparse import doc_sparse, query_sparse
from embedder.store import Store
from embedder.types import User

URL = "http://localhost:6333"
DIM = 8
random.seed(0)   # S4: deterministic (fake-vector ordering doesn't flake)


def _server_up() -> bool:
    try:
        QdrantClient(url=URL, timeout=2).get_collections()   # a real connection (closer to what the tests actually use than raw HTTP)
        return True
    except Exception:
        return False


_UP = _server_up()
if os.getenv("CUSTODIAN_REQUIRE_QDRANT_SERVER") == "1" and not _UP:   # M5: CI forbids a silent skip
    raise RuntimeError("CUSTODIAN_REQUIRE_QDRANT_SERVER=1 but the Qdrant server is unreachable -- the D go/no-go must not silently skip")
pytestmark = pytest.mark.skipif(not _UP, reason=f"requires a Qdrant server @ {URL} (set CUSTODIAN_REQUIRE_QDRANT_SERVER=1 to forbid skipping)")


def _vec():
    return [random.random() for _ in range(DIM)]


def _A(t, a, v, unset=False):
    return {"tenant": t, "allow": a, "visibility": v, "unset": unset}


def _pt(i, text, acl):
    split = {"acl_tenant": acl["tenant"], "acl_allow": acl["allow"],
             "acl_visibility": acl["visibility"], "acl_unset": acl.get("unset", False)}
    return models.PointStruct(id=i, vector={"dense": _vec(), "sparse": doc_sparse(text)},
        payload={"chunk_id": f"c{i}", "doc_id": f"d{i}", "kind": "text", "text": text, "acl": acl, **split})


def _seed_acl_data(s):
    s.upsert([
        _pt(1, "public revenue report", _A("t1", [], "public")),               # OK: public
        _pt(2, "t1 hr revenue report", _A("t1", ["g_hr"], "restricted")),    # OK: same tenant + g_hr
        _pt(3, "t1 finance revenue", _A("t1", ["g_fin"], "restricted")),   # denied: not authorized
        _pt(4, "t2 hr revenue", _A("t2", ["g_hr"], "restricted")),        # denied: cross-tenant
        _pt(5, "unauthorized revenue", _A("t1", [], "restricted", unset=True)),  # denied: fail-closed (unset)
    ])


def _fresh(coll):
    s = Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection=coll, prefetch_limit=20))
    try:
        s.client.delete_collection(coll)
    except Exception:
        pass
    s.ensure_collection()
    return s


def test_server_mode_acl_pipeline_fail_closed_all_strategies():
    """Pipeline level: in server mode, all three of hybrid (RRF)/dense/sparse fail-closed on final
    delivery (including the exit-point acl_admits check)."""
    s = _fresh("test_acl_server")
    try:
        _seed_acl_data(s)
        user = User(tenant="t1", principals=["g_hr"])
        q = query_sparse("revenue report")
        for strat in ("hybrid", "dense", "sparse"):
            ids = {h.chunk_id for h in s.hybrid_search(_vec(), q, user, top_k=10, strategy=strat)}
            assert "c1" in ids and "c2" in ids, f"{strat}: public + same-tenant-authorized should be retrieved"
            assert not (ids & {"c3", "c4", "c5"}), f"{strat}: unauthorized-access leak! retrieved={sorted(ids)}"
    finally:
        s.client.delete_collection("test_acl_server")


def test_server_fusion_no_should_leak_raw():
    """M4 (the real claim): **bypasses the exit-point acl_admits** and directly asserts the server
    RRF fusion's raw query_points output contains no unauthorized points.
    The exit-point re-check is a mode-independent hard backstop that would mask a fusion-level
    leak; this observes directly whether the server filter is genuinely fail-closed."""
    s = _fresh("test_acl_raw")
    try:
        _seed_acl_data(s)
        user = User(tenant="t1", principals=["g_hr"])
        acl = s.acl_filter(user)
        prefetch = [models.Prefetch(query=_vec(), using="dense", filter=acl, limit=20),
                    models.Prefetch(query=query_sparse("revenue report"), using="sparse", filter=acl, limit=20)]
        raw = s.client.query_points("test_acl_raw", prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=acl, limit=10, with_payload=True).points
        raw_ids = {p.payload["chunk_id"] for p in raw}                # not filtered through acl_admits
        assert not (raw_ids & {"c3", "c4", "c5"}), \
            f"the server RRF fusion layer leaked should (prefetch filter did not take effect)! raw retrieval={sorted(raw_ids)}"
        assert "c1" in raw_ids and "c2" in raw_ids, "authorized points should be present in fusion's raw output"
    finally:
        s.client.delete_collection("test_acl_raw")


def test_server_list_documents_acl_scoped():
    """S1: list_documents(scroll) is ACL-scoped in server mode, plus a raw scroll probe that
    bypasses acl_admits (should semantics + no missed rows across pagination)."""
    s = _fresh("test_list_server")
    try:
        _seed_acl_data(s)
        user = User(tenant="t1", principals=["g_hr"])
        docs, truncated = s.list_documents(user)
        ids = {d["doc_id"] for d in docs}
        assert ids == {"d1", "d2"}, f"t1/g_hr should only see public (d1) + authorized (d2), got {sorted(ids)}"
        assert truncated is False, "a single-digit point count should not report truncation"
        # Raw scroll probe: bypasses the exit-point acl_admits and asserts the server scroll
        # filter layer doesn't let unauthorized access through
        raw, _ = s.client.scroll("test_list_server", scroll_filter=s.acl_filter(user), limit=100, with_payload=True)
        raw_ids = {p.payload["chunk_id"] for p in raw}
        assert not (raw_ids & {"c3", "c4", "c5"}), f"the server scroll layer leaked should! raw={sorted(raw_ids)}"
    finally:
        s.client.delete_collection("test_list_server")


def test_server_ensure_collection_heals_missing_payload_indexes():
    """Review fix (fix 4): manually calling create_collection (without building indexes) simulates
    "a half-initialized collection from a mid-build crash" -- ensure_collection's existing branch
    must idempotently backfill all 7 payload indexes (otherwise acl_*/doc_* filtering degrades to a
    full scan on the server and never self-heals)."""
    coll = "test_index_heal"
    raw = QdrantClient(url=URL)
    try:
        raw.delete_collection(coll)
    except Exception:
        pass
    raw.create_collection(coll,
        vectors_config={"dense": models.VectorParams(size=DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    try:
        assert not raw.get_collection(coll).payload_schema, "precondition: a half-initialized collection should have no indexes at all"
        Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection=coll)).ensure_collection()
        schema = set(raw.get_collection(coll).payload_schema)
        expected = {"acl_tenant", "acl_visibility", "acl_allow", "acl_unset", "doc_id", "doc_type", "kind"}
        assert expected <= schema, f"the existing branch did not backfill indexes, missing {expected - schema}"
    finally:
        raw.delete_collection(coll)


def test_server_unlocks_multi_client():
    """Multi-replica locking: the embedded backend is exclusive to a single process (a second
    client errors) -> the server lets two clients open the same collection at once.
    Caveat (S2 limitation): two Stores in the same process verify that "the embedded file-lock
    ceiling was removed at the client layer", not a true multi-replica scenario with two independent
    `custodian serve` processes (which also involves sidecar sharing etc., left to Stage E's compose
    end-to-end test). The counter-example's "already accessed" is QdrantLocal's in-process registry check."""
    tmp = tempfile.mkdtemp()
    s1 = Store(EmbedConfig(qdrant_path=tmp, dense_dim=DIM, collection="c"))
    s1.ensure_collection()
    with pytest.raises(RuntimeError, match="already accessed"):
        Store(EmbedConfig(qdrant_path=tmp, dense_dim=DIM, collection="c"))
    del s1
    shutil.rmtree(tmp, ignore_errors=True)

    sa = _fresh("test_multi_replica")
    try:
        sb = Store(EmbedConfig(qdrant_url=URL, dense_dim=DIM, collection="test_multi_replica"))
        assert sa.client.get_collection("test_multi_replica").points_count == 0
        assert sb.client.get_collection("test_multi_replica").points_count == 0   # the second client reads the same collection
    finally:
        sa.client.delete_collection("test_multi_replica")
