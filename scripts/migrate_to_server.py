#!/usr/bin/env python3
"""Data migration: embedded Qdrant → server mode (phase E, docs/SCALE_OUT.md §7).

The compose qdrant runs in server mode with a persistent volume starting from empty. This script
moves every point from the bare-metal embedded store (~/rag_real/qdrant) into the server.
The sidecar (~/rag_real/sidecar/*.json) is a separate piece of data, moved independently with
`cp -a` on the host (see §7).

Four pitfalls that must all be covered (missing any one means it's only a "fake-complete" migration):
  1. Collection creation reuses Store.ensure_collection — it creates dense+sparse plus all 7
     payload indexes in one go. Hand-copy one and miss it, and server-side ACL fail-closed breaks
     silently (acl_tenant/acl_allow etc. have no index -> filtering silently goes wrong).
  2. scroll must pass with_vectors=True — by default vectors are not returned, so omitting this
     migrates "point with no vector" and retrieval breaks completely without any error.
  3. Named vectors (dense+sparse) are passed through as-is; an image-only chunk has no sparse
     vector, and that absence is preserved rather than backfilled with an empty one.
  4. Migration only counts as complete once dst.count == src.count; point ids use deterministic
     uuid5, so the migration is idempotent and safe to re-run.

This script only performs the first layer of verification (count). The other three layers
(vectors actually present / sidecar coverage / ACL fail-closed) live in SCALE_OUT §7 and are
run separately.

Run (a slim environment is enough — plain qdrant-client, no GPU needed):
  # start the server first: docker compose --env-file .env.compose up -d qdrant
  python scripts/migrate_to_server.py \
      --src ~/rag_real/qdrant --dst http://localhost:6333 --collection real --dense-dim 1024
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qdrant_client import QdrantClient, models   # noqa: E402

from embedder.config import EmbedConfig           # noqa: E402
from embedder.store import Store                  # noqa: E402

BATCH = 256


def migrate(src_path: str, dst_url: str, collection: str, dense_dim: int) -> None:
    src_path = os.path.expanduser(src_path)
    if not os.path.isdir(src_path):
        raise SystemExit(f"Source embedded store directory does not exist: {src_path} (should point at ~/rag_real/qdrant)")

    # 1) Acquire the exclusive lock on the source store first -- fail BEFORE causing any side
    #    effects on dst (otherwise dst ends up with an empty collection already created before the
    #    error is raised, leaving a "dst has a collection but nothing migrated" intermediate state,
    #    plus a bare RuntimeError with no guidance on what to do).
    print(f"[1/4] Opening source embedded store {src_path} (exclusive lock; all custodian serve/index "
          f"processes must be stopped during migration)…", flush=True)
    try:
        src = QdrantClient(path=src_path)
    except RuntimeError as e:
        raise SystemExit(f"Source store is locked by another process (usually custodian serve/index still "
                          f"running); stop everything accessing {src_path} before migrating. Original error: {e}")

    # 1b) Verify the source collection exists and is **non-empty** -- before causing any side effects
    #     on dst. Otherwise, if the source has no such collection / 0 points, the next step would
    #     still create an empty collection on dst, which triggers exactly the failure mode custodian's
    #     /readyz collection_missing guard is meant to catch (collection exists but has 0 points ->
    #     readyz reports ready -> nginx routes traffic to an empty store -> everything looks fine while
    #     queries return nothing).
    if not src.collection_exists(collection):
        raise SystemExit(f"Source store has no collection {collection!r} (check --src {src_path} / --collection); dst was not modified.")
    src_count = src.count(collection).count
    if src_count == 0:
        raise SystemExit(f"Source collection {collection!r} is empty (0 points); refusing to migrate an empty store (would create an empty collection on dst and mislead /readyz into reporting ready).")

    # 2) Create the collection on dst -- reuse Store.ensure_collection (dense/sparse + 7 payload
    #    indexes), never hand-copy this.
    dst_cfg = EmbedConfig(qdrant_url=dst_url, collection=collection, dense_dim=dense_dim)
    print(f"[2/4] Creating collection {collection!r} on the server (reusing ensure_collection: "
          f"dense/sparse + 7 payload indexes)…", flush=True)
    Store(dst_cfg).ensure_collection()   # idempotent: skips if it already exists with matching dims; loudly errors on a dim mismatch
    dst = QdrantClient(url=dst_url)

    # 3) scroll through every point (with_vectors=True is required) and upsert as-is
    print("[3/4] scroll + upsert (with_vectors=True; named dense+sparse vectors passed through as-is)…", flush=True)
    offset = None
    migrated = 0
    while True:
        points, offset = src.scroll(
            collection, limit=BATCH, with_payload=True, with_vectors=True, offset=offset)
        if not points:
            break
        dst.upsert(collection, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        migrated += len(points)
        print(f"      migrated {migrated} …", flush=True)
        if offset is None:
            break

    # 4) count verification (first layer; the other three layers -- vectors/sidecar/ACL -- are in SCALE_OUT §7)
    sc = src.count(collection).count
    dc = dst.count(collection).count
    print(f"[4/4] count verification: src={sc}  dst={dc}", flush=True)
    if sc != dc or dc == 0:      # dc==0 also counts as failure (an empty migration is meaningless and would mislead /readyz into reporting ready)
        raise SystemExit(f"Count verification failed (src={sc}, dst={dc}); migration incomplete or migrated an empty store -- safe to re-run (idempotent).")
    # Note: this exit message must not imply "vectors have been verified" -- only the point count was
    # checked; degraded vectors or truncated payloads can still leave the counts equal (a fake-complete migration).
    print(f"OK point migration complete ({dc} points); WARNING only point count was verified, vectors/ACL "
          f"were NOT verified -- you must still run the three SCALE_OUT §7 verification layers "
          f"(vectors actually present with dense==dim / sparse non-empty, sidecar coverage, ACL fail-closed) "
          f"+ cp -a sidecar/.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedded Qdrant -> server migration (phase E)")
    ap.add_argument("--src", default="~/rag_real/qdrant", help="source embedded store directory (default ~/rag_real/qdrant)")
    ap.add_argument("--dst", default="http://localhost:6333", help="destination server URL")
    ap.add_argument("--collection", default="real")
    ap.add_argument("--dense-dim", type=int, default=1024)
    args = ap.parse_args()
    migrate(args.src, args.dst, args.collection, args.dense_dim)


if __name__ == "__main__":
    main()
