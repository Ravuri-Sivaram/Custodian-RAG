"""Indexing time: consumes chunker's ChunkResult plus the raw elements, vectorizes them, and writes
to Qdrant + the sidecar.

Routing (mirrors chunker's image_only flag):
  - image_only, a pure image -> an image vector (dense), **skips sparse** (a pure image has no
    searchable text; recall relies on the shared text/image embedding space);
  - text/table/an image or chart with text -> a text dense vector + BM25 sparse.
ACL: `acl_split` splits chunk.acl into 4 filterable payload fields (fail-closed).
Sidecar: stores elements+sections+banners+acl_index per doc_id -- required for assemble_big's
small-to-big assembly at query time (Qdrant only stores chunk vectors; the raw elements/sections
never go into the index)."""
from __future__ import annotations

import itertools
import json
import os
import sys
import uuid
from dataclasses import asdict

from qdrant_client import models

from .acl import acl_split
from .config import SIDECAR_VERSION, EmbedConfig
from .dense import Dense
from .remote import make_dense
from .sparse import doc_sparse
from .store import Store


def _batched(items: list, n: int):
    """Yield successive lists of up to n items each, preserving order. A local helper so this
    file has no new runtime dependency for something itertools.batched (3.12+) would give
    natively -- this project's floor is Python 3.10 (see pyproject.toml)."""
    it = iter(items)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch


def _payload(ch, acl_fields: dict) -> dict:
    """The payload written to Qdrant: display/citation/filter fields, plus what retrieve needs to
    reconstruct a hit chunk, plus ACL."""
    return {
        "chunk_id": ch.chunk_id, "doc_id": ch.doc_id, "kind": ch.kind, "text": ch.text,
        "content_raw": ch.content_raw, "breadcrumb": ch.breadcrumb, "section_path": ch.section_path,
        "section_id": ch.section_id, "section_anchor": ch.section_anchor,
        "page_start": ch.page_start, "page_end": ch.page_end, "source_indices": ch.source_indices,
        "flags": ch.flags, "lang": ch.lang, "doc_type": ch.doc_type,
        "image_path": ch.image_path, "doc_meta": ch.doc_meta,
        "acl": ch.acl,                       # The raw acl is also stored: needed for the exit-side re-check (iron rule #5) / debugging
        **acl_fields,                        # acl_unset/acl_tenant/acl_allow/acl_visibility (filterable)
    }


class Embedder:
    def __init__(self, cfg: EmbedConfig | None = None, store: Store | None = None, dense: Dense | None = None):
        # store/dense can be shared: the embedded Qdrant client only allows one client per path
        # (embed and retrieve within the same process must share it), and reusing dense avoids
        # loading the 8B model twice (saves ~2 minutes plus GPU memory). Server-mode Qdrant has no
        # such single-client constraint in production.
        self.cfg = cfg or EmbedConfig()
        self.dense = dense or make_dense(self.cfg)   # Factory: cfg.inference_url empty = local (default), non-empty = remote inference service
        self.store = store or Store(self.cfg)
        self.store.ensure_collection()
        os.makedirs(self.cfg.sidecar_dir, exist_ok=True)

    @staticmethod
    def _pid(chunk_id: str) -> str:
        # A Qdrant point id must be a uint64 or a UUID; chunk_id is a string -> deterministic UUID5
        # (re-running with the same id produces the same point, so upsert overwrites it -- idempotent)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def index_document(self, doc_id: str, elements: list, chunk_result, image_root: str) -> dict:
        """elements: the raw list[Element] (idx-aligned); chunk_result: a ChunkResult;
        image_root: the MinerU output root directory (used to join absolute image_path values,
        i.e. parsed/<doc>/).

        **Contract: doc_id must be globally unique and stable.** point_id = uuid5(chunk_id), and
        chunk_id = f"{doc_id}#{n}"; re-indexing the same doc_id is an idempotent overwrite (a
        feature), but **reusing the same doc_id for different content silently overwrites the
        previous document's vectors + payload + ACL + sidecar** (seal review #3). Multiple ingest
        sources must each guarantee their own doc_id namespace doesn't collide with another's
        (e.g. by prefixing with a source tag).

        Failure surface (per an earlier review fix): encoding and preparing the sidecar tmp file
        are both **pure preparation** -- a failure there has no side effects (the old index is
        still intact and usable); only the closing delete -> upsert -> replace sequence is on the
        order of milliseconds, and a failure inside that window leaves the doc out of the index
        (the old vectors are already deleted, the new data never landed) -- this raises after a
        loud warning, and re-running this method is required to recover. This replaces the old
        implementation's much longer, minutes-scale exposure window ("delete first, then encode
        chunk by chunk").

        Encoding is batched (cfg.embed_batch_size, default 16): text chunks and image chunks are
        each grouped into fixed-size batches and encoded with one dense.encode_* call per batch,
        instead of the previous one call per chunk. A document can have thousands of chunks, and
        each encode_* call is a GPU forward pass with its own fixed overhead -- batching amortizes
        that overhead across many chunks per pass instead of paying it per chunk, which is a
        meaningful indexing-throughput win with no change in the vectors produced (encode_text/
        encode_image already accept and process a list; this only changes how many chunks go into
        each list). Per-chunk output (id, vector, payload, and the n_img/n_txt/n_skip counters) is
        unchanged -- only the grouping of calls."""
        text_chunks, image_items = [], []
        n_skip = 0
        for ch in chunk_result.chunks:
            if "image_only" in (ch.flags or []):
                if not ch.image_path:                          # A pure image with no path -> can't be vectorized
                    n_skip += 1; continue
                p = os.path.join(image_root, ch.image_path)
                if not os.path.exists(p):                       # Path no longer valid (post-sanitization / files moved) -> skip, don't silently pretend it worked
                    n_skip += 1; continue
                image_items.append((ch, p))
            else:
                text_chunks.append(ch)

        batch_size = max(1, int(self.cfg.embed_batch_size))
        points: list = []
        n_txt = n_img = 0
        for batch in _batched(text_chunks, batch_size):
            dvecs = self.dense.encode_text([c.text for c in batch])
            for ch, dvec in zip(batch, dvecs):
                svec = doc_sparse(ch.text, self.cfg.stopwords)
                vec: dict = {"dense": dvec.tolist()}
                if svec is not None:
                    vec["sparse"] = svec
                points.append(models.PointStruct(id=self._pid(ch.chunk_id), vector=vec, payload=_payload(ch, acl_split(ch.acl))))
                n_txt += 1
        for batch in _batched(image_items, batch_size):
            dvecs = self.dense.encode_image([p for _, p in batch])
            for (ch, _), dvec in zip(batch, dvecs):
                vec = {"dense": dvec.tolist()}
                points.append(models.PointStruct(id=self._pid(ch.chunk_id), vector=vec, payload=_payload(ch, acl_split(ch.acl))))
                n_img += 1

        # The sidecar tmp file is also prepared before delete (json.dump+fsync is where a disk-full
        # or similar failure would mainly happen); after delete, all that's left is an atomic rename.
        tmp, path = self._prepare_sidecar(doc_id, elements, chunk_result)
        deleted = False
        try:
            # The old data is still deleted before upsert: if re-indexing produces fewer chunks
            # than before, the old higher-numbered points would never get overwritten -- deleting
            # first avoids orphan vectors (same motivation as review item #4, unchanged).
            self.store.delete_by_doc(doc_id)
            deleted = True
            if points:
                self.store.upsert(points)
            os.replace(tmp, path)          # An atomic same-filesystem rename: the new vectors and the new sidecar land together, as one generation (seal review #9)
        except Exception as e:
            try:
                os.remove(tmp)             # Clean up the prepared artifact; a cleanup failure must never mask the real exception
            except OSError:
                pass
            if deleted:                    # The old vectors are gone and the new data never fully landed -- raising silently here would let "skipped" hide the fact that the doc just fell out of the index
                print(f"[embedder] FATAL: doc {doc_id} has fallen out of the index (old vectors deleted, "
                      f"new vectors/sidecar never landed): {type(e).__name__}: {e} -- must re-run index_document to recover.",
                      file=sys.stderr, flush=True)
            raise
        return {"images": n_img, "texts": n_txt, "skipped": n_skip, "indexed": len(points)}

    def _prepare_sidecar(self, doc_id: str, elements: list, chunk_result) -> tuple[str, str]:
        """Writes the full sidecar to a .tmp file and fsyncs it, returning (tmp, path). **Does not
        write the real file**: the caller does the closing os.replace once the index write has
        succeeded. Splitting preparation from the closing step is half of what narrows
        index_document's failure surface -- a write failure now only ever happens before delete
        (no side effects yet); after delete, all that's left is a same-filesystem atomic rename
        (seal review #9: this prevents a half-written, truncated JSON file from permanently
        breaking small-to-big assembly)."""
        data = {
            "version": SIDECAR_VERSION,        # Checked on the read side: a mismatch is refused with a message to rebuild (guards against a silent bug from schema drift)
            "elements": [asdict(e) for e in elements],
            "sections": [asdict(s) for s in chunk_result.sections],
            "banners": list(chunk_result.banners),
            # acl_index: {element_idx: acl} -- assemble_big uses this to guarantee small-to-big never pulls material across an ACL boundary
            "acl_index": {str(k): v for k, v in chunk_result.acl_index().items()},
        }
        path = os.path.join(self.cfg.sidecar_dir, f"{doc_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        return tmp, path

    def _write_sidecar(self, doc_id: str, elements: list, chunk_result) -> None:
        tmp, path = self._prepare_sidecar(doc_id, elements, chunk_result)
        os.replace(tmp, path)

    def delete_document(self, doc_id: str) -> None:
        """Deletes one document: its Qdrant points and its sidecar are removed together, so
        deleting only one side never leaves an orphan behind (an orphan vector is still
        retrievable; an orphan sidecar just wastes disk)."""
        self.store.delete_by_doc(doc_id)
        path = os.path.join(self.cfg.sidecar_dir, f"{doc_id}.json")
        if os.path.exists(path):
            os.remove(path)
