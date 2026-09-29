"""Qdrant storage + retrieval: dense+sparse hybrid search (RRF) plus hard ACL filtering.
ACL enforces chunker's INTEGRATION contract (§6): fail-closed, tenant isolation, and (allow ANY OR
public), with nested filters to keep it safe.

Concurrency: the embedded QdrantLocal single client is **not thread-safe** (concurrent reads are
safe; a read racing a write is not -- upsert rebinds arrays via np.append, and _update_point writes
in place). Every method that touches self.client serializes through `self._lock`. `acl_filter` is a
pure constructor (never touches the client) and takes no lock. This lock can be removed entirely
once the deployment moves to Qdrant server mode.
Note: `self._lock` only guarantees atomicity **within a single method**; `embed.py`'s
index_document spans delete_by_doc and upsert across two separate lock acquisitions, so there is a
brief window during re-indexing where that document is temporarily unrecallable (this is an
existing property, not something introduced by this locking scheme -- the previous, coarser lock
didn't wrap the indexing path either; delete happens only after all encoding has completed, so the
window is millisecond-scale; a failure inside that window means the doc has fallen out of the
index, and embed.py raises a loud warning requiring a re-run). Concurrent indexing and querying in
the same process is outside this module's current contract."""
from __future__ import annotations

import threading
import uuid

from qdrant_client import QdrantClient, models

from .acl import acl_admits
from .config import EmbedConfig
from .types import Hit, User


class Store:
    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        # Three branches, in priority order (url > :memory: > path): server mode (for multi-replica
        # deployments) > in-memory (tests; must be checked before path, since many tests depend on
        # :memory:) > embedded (the default, single-process mode).
        if cfg.qdrant_url:
            self.client = QdrantClient(url=cfg.qdrant_url)
        elif cfg.qdrant_path == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=cfg.qdrant_path)
        self._lock = threading.Lock()   # Serializes the embedded single client; in server mode Qdrant handles concurrency itself, so this lock could be relaxed there

    def ensure_collection(self) -> None:
        with self._lock:
            c = self.cfg
            if self.client.collection_exists(c.collection):
                # Never silently return early here: dense_dim can change after the collection's
                # dimension was fixed at creation time -- a dimension drift would either crash
                # upsert (if it grew) or silently corrupt retrieval semantics (if it shrank, since
                # COSINE similarity in a different subspace is meaningless). The config comment
                # already anticipates dense_dim being tuned, so this check exists to catch that.
                size = self.client.get_collection(c.collection).config.params.vectors["dense"].size
                if size != c.dense_dim:
                    raise ValueError(
                        f"Collection {c.collection!r} already exists with dense size={size} != cfg.dense_dim={c.dense_dim}; "
                        f"changing the dimension requires rebuilding the collection (delete {c.qdrant_path}) or realigning dense_dim back to {size}.")
            else:
                self.client.create_collection(
                    c.collection,
                    vectors_config={"dense": models.VectorParams(size=c.dense_dim, distance=models.Distance.COSINE)},
                    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
                )
            # Payload indexes are idempotently (re-)created in **both** branches (recreating the
            # same schema is just an overwrite, no need to check first): if this only ran in the
            # creation branch, a server-mode collection that crashed after create_collection
            # succeeded but before its indexes finished would be permanently left half-initialized
            # (missing acl_* indexes among others, degrading every filtered query to a full scan)
            # with no way to self-heal, and any future new index would never get backfilled onto an
            # existing collection either. ensure_collection runs on every Embedder/Store
            # initialization, so the next startup self-heals it automatically, at the cost of 7
            # idempotent RPCs.
            for fld, schema in [("acl_tenant", "keyword"), ("acl_visibility", "keyword"),
                                ("acl_allow", "keyword"), ("acl_unset", "bool"), ("doc_id", "keyword"),
                                ("doc_type", "keyword"), ("kind", "keyword")]:   # doc_type/kind filter indexes (no-op locally, but needed in server mode)
                self.client.create_payload_index(c.collection, field_name=fld, field_schema=schema)

    def acl_filter(self, user: User, doc_ids: list[str] | None = None,
                   doc_type: str | None = None, kind: str | None = None) -> models.Filter:
        """Fail-closed: acl_unset==False AND acl_tenant==user.tenant AND (acl_allow intersects
        principals OR the doc is public). (allow OR public) is expressed via a **nested Filter**
        rather than relying on top-level must+should semantics -- this closes off a class of
        permission drift confirmed during adversarial review.
        Optional filters: doc_ids / doc_type / kind are always added as FieldConditions in
        **must** (ANDed with ACL, **never a top-level should**, avoiding a known issue where the
        embedded backend's fusion mode drops top-level should clauses); any filter only ever
        narrows results, never loosens them -- unauthorized content stays fail-closed regardless.
        A pure construction of models.Filter that never touches the client -> **no lock needed**."""
        must = [
            models.FieldCondition(key="acl_unset", match=models.MatchValue(value=False)),
            models.FieldCondition(key="acl_tenant", match=models.MatchValue(value=user.tenant)),
            models.Filter(should=[
                models.FieldCondition(key="acl_allow", match=models.MatchAny(any=user.principals or [])),
                models.FieldCondition(key="acl_visibility", match=models.MatchValue(value="public")),
            ]),
        ]
        if doc_ids:
            must.append(models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(doc_ids))))
        if doc_type:
            must.append(models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type)))
        if kind:
            must.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
        return models.Filter(must=must)

    def upsert(self, points: list[models.PointStruct]) -> None:
        with self._lock:
            self.client.upsert(self.cfg.collection, points=points)

    def delete_by_doc(self, doc_id: str) -> None:
        """Deletes every point for this doc. This clears the old points before re-indexing: without
        it, if the new version has fewer chunks, the old higher-numbered points would remain as
        orphan vectors -- still recallable, but with anchor/source_indices pointing at indices in
        the new sidecar that no longer mean the same thing, pulling the wrong passage and
        misaligning ACL. doc_id already has a payload index, so filtering on it for deletion is
        efficient."""
        with self._lock:
            self.client.delete(self.cfg.collection, points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))])))

    def hybrid_search(self, dense_vec, sparse_vec, user: User, top_k: int | None = None,
                      doc_ids: list[str] | None = None, doc_type: str | None = None,
                      kind: str | None = None, strategy: str = "hybrid") -> list[Hit]:
        c = self.cfg
        acl = self.acl_filter(user, doc_ids, doc_type, kind)
        limit = c.top_k if top_k is None else max(1, top_k)   # Guards against the 0-falsy bug (0 silently becoming the default) and negative values
        # strategy lets the caller choose a route: a single-path query returns the **native** score
        # (cosine/bm25, interpretable), while hybrid uses RRF fusion.
        # The ACL filter must be pushed down onto every prefetch in hybrid mode: the embedded
        # QdrantLocal backend is known to drop a top-level query_filter's should clause under
        # fusion (confirmed via a diagnostic ACL test), and only a prefetch-level filter reliably
        # applies the should clause; the top-level query_filter is kept too as a second layer of
        # defense. A single-path direct query has no fusion and doesn't hit this issue, and the
        # exit-side acl_admits re-check is a further backstop regardless.
        with self._lock:                                 # Serializes the Qdrant client's forward calls (dense_vec is an argument here; encoding already happened outside the lock)
            if strategy == "dense":
                res = self.client.query_points(c.collection, query=dense_vec, using="dense",
                                               query_filter=acl, limit=limit, with_payload=True).points
                score_kind = "cosine"
            elif strategy == "sparse":
                if sparse_vec is None:                   # The query has no valid exact-match token -> the sparse path has no results
                    return []
                res = self.client.query_points(c.collection, query=sparse_vec, using="sparse",
                                               query_filter=acl, limit=limit, with_payload=True).points
                score_kind = "bm25"
            else:                                        # hybrid (the default)
                prefetch = [models.Prefetch(query=dense_vec, using="dense", filter=acl, limit=c.prefetch_limit)]
                if sparse_vec is not None:               # An image_only chunk has no sparse vector; the query itself may also have none
                    prefetch.append(models.Prefetch(query=sparse_vec, using="sparse", filter=acl, limit=c.prefetch_limit))
                res = self.client.query_points(
                    c.collection, prefetch=prefetch, query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=acl, limit=limit, with_payload=True).points
                score_kind = "rrf"
        # An exit-side per-hit re-check, symmetric with list_documents: this is a second ACL gate on
        # the bare-hit delivery path (a nested acl that's missing or not admitted gets dropped here).
        return [Hit(chunk_id=p.payload.get("chunk_id", ""), doc_id=p.payload.get("doc_id", ""),
                    kind=p.payload.get("kind", ""), text=p.payload.get("text", ""),
                    score=p.score, payload=dict(p.payload), score_kind=score_kind)
                for p in res if acl_admits(p.payload.get("acl") or {}, user)]

    def list_documents(self, user: User, limit: int = 10000) -> tuple[list[dict], bool]:
        """Lists the documents visible to `user` (doc_id + title), ACL-scoped -- backs the agentic
        RAG list_documents tool. The scroll uses acl_filter pushed down, plus **a per-point
        acl_admits re-check** (a second layer of defense: the embedded QdrantLocal backend's
        handling of nested should under fusion is known to drop it in hybrid_search, and its
        behavior under scroll hasn't been separately verified -- fail-closed doesn't take that
        gamble, even at the cost of one extra check per point).
        `limit` is a **scan cap on visible chunks**, not a document count (a production corpus
        easily exceeds the default 10,000-chunk cap), so this returns (docs, truncated):
        truncated=True means the scan stopped early at the cap and the listing may be incomplete --
        the caller (toolcore) must pass that signal through to the agent."""
        docs: dict[str, tuple] = {}
        flt = self.acl_filter(user)
        with self._lock:                                 # The whole scroll loop shares one client call sequence, serialized
            offset, seen = None, 0
            while seen < limit:
                points, offset = self.client.scroll(
                    self.cfg.collection, scroll_filter=flt, limit=min(256, limit - seen),
                    offset=offset, with_payload=True)
                if not points:
                    break
                for p in points:
                    if not acl_admits(p.payload.get("acl") or {}, user):   # Second layer of defense: re-check ACL per point
                        continue
                    did = p.payload.get("doc_id", "")
                    if did and did not in docs:                            # Also carries doc_type, so the routing layer can compute a coverage summary
                        docs[did] = ((p.payload.get("doc_meta") or {}).get("title") or did, p.payload.get("doc_type") or "")
                seen += len(points)
                if offset is None:
                    break
        # scroll's next_page_offset semantics: it's non-None only when there's another page --
        # this naturally distinguishes "scanned the whole library" from "stopped early at the cap".
        truncated = offset is not None and seen >= limit
        return [{"doc_id": k, "title": t, "doc_type": dt} for k, (t, dt) in sorted(docs.items())], truncated

    def get_by_chunk_id(self, chunk_id: str, user: User) -> dict | None:
        """Fetches a point's payload by chunk_id (used by expand). point_id=uuid5(chunk_id) is
        deterministic to compute -> an O(1) direct lookup, no chunk_id index needed.
        **A retrieve-by-id call bypasses acl_filter, so it must re-check acl_admits itself** (no
        point, or no access, both return None -- fail-closed, and deliberately not distinguishing
        "doesn't exist" from "no access")."""
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
        with self._lock:
            pts = self.client.retrieve(self.cfg.collection, ids=[pid], with_payload=True)
        if not pts:
            return None
        payload = pts[0].payload or {}
        if not acl_admits(payload.get("acl") or {}, user):
            return None
        return payload

    def get_doc_lang(self, doc_id: str) -> str:
        """Fetches a document's authoritative lang (from any chunk's payload; consistent across the
        whole document). Used by get_document to correctly scale its token estimate; defaults to
        "en". Non-sensitive metadata, and the caller has already passed a doc-level ACL precheck,
        so this doesn't apply its own ACL check."""
        with self._lock:
            pts, _ = self.client.scroll(self.cfg.collection, limit=1, with_payload=True,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=doc_id))]))
        return (pts[0].payload.get("lang") if pts else None) or "en"
