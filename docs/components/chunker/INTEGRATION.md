# Integration guide

## 1. Connecting into a RAG indexing pipeline

```
parse ──(adapter)──► Element[] ──► Chunker.chunk ──► Chunk[] ─► embed(chunk.text) ─► vector store
                                                  └► Section[] ─► section store (used at retrieval time)
```

**At index time** (including document-level metadata + access policy):
```python
from chunker import Chunker, extract_doc_meta
from chunker.adapters.mineru import from_mineru

def index_document(doc_id, content_list, layout, doc_type, lang, src_path, acl):
    elements = from_mineru(content_list, layout)
    doc_meta = extract_doc_meta(src_path, doc_type=doc_type, lang=lang,   # from the file's core.xml/PDF info
                                source="...", domain="...")               #  + manifest/source-system overrides
    # acl comes from the source system (SharePoint/Drive/S3 permissions, tenant mapping, ...), not inferred from content
    result = Chunker().chunk(elements, doc_id=doc_id, doc_type=doc_type, lang=lang,
                             doc_meta=doc_meta, acl=acl)
    for c in result.chunks:
        # Text chunks embed their text; image/chart chunks (especially pure images carrying the `image_only` flag)
        # should instead go through "image vectorization" — use a VL embedder to embed the cropped image at
        # image_path (relative to the MinerU output root, needs to be joined into an absolute path), not embed(c.text).
        # For image_only chunks, the text field is only a placeholder (n_tokens=0); the sparse path is useless for it,
        # it can only be recalled via the dense image vector.
        vec = embed(c.text)
        vector_store.add(id=c.chunk_id, vector=vec, payload={
            "doc_id": c.doc_id, "kind": c.kind, "text": c.text,
            "breadcrumb": c.breadcrumb, "section_id": c.section_id,
            "section_anchor": c.section_anchor, "source_indices": c.source_indices,
            "content_raw": c.content_raw, "page_start": c.page_start, "flags": c.flags,
            "doc_meta": c.doc_meta, "acl": c.acl,         # ← document-level metadata + access policy
            "image_path": c.image_path,                   # ← reference to the image/chart's cropped image (for VL image vectorization; None for text/table)
            "lang": c.lang, "page_end": c.page_end,       # ← required reading for assemble_big at query time (token estimation / slide windowing)
            "doc_type": c.doc_type,                       # ← assemble_big uses this to pick the query-time token budget
        })
    section_store.put(doc_id, [vars(s) for s in result.sections])   # the section tree
    element_store.put(doc_id, [vars(e) for e in elements])          # the raw elements (for assembling big blocks)
```

> You can also store just `section_anchor` + the raw elements and compute the big block at query time — all `assemble_big` needs is `sections + elements`.

**At query time (small-to-big), permission is the first hard filter**:
```python
def retrieve(query, user, top_k=8):
    # ① Hard ACL pre-filter (at the vector-store filter layer, fail-closed) — a user can never get an unauthorized chunk
    acl_filter = (
        "acl.unset != true AND "                                  # documents with permissions not yet wired up are denied by default
        "(acl.tenant == :tenant) AND "                            # tenant isolation
        "(acl.allow ANY IN :principals OR acl.visibility == 'public')"
    )
    hits = vector_store.search(embed(query), top_k, filter=acl_filter,
                               params={"tenant": user.tenant, "principals": user.groups + [user.id]})
    out = []
    for h in dedup_by_section(hits):
        chunk = rebuild_chunk(h.payload)                          # payload -> Chunk (including doc_meta/acl)
        secs  = {s["sec_id"]: rebuild_section(s) for s in section_store.get(h.doc_id)}
        els   = [rebuild_element(e) for e in element_store.get(h.doc_id)]
        # ② small-to-big must be ACL-aware: big-block assembly pulls material from the raw elements by idx range,
        #    which crosses chunk boundaries. If a particular section was tightened individually (invariant 3),
        #    pulling material purely by idx would drag the plaintext of an unauthorized sibling section into big.text.
        #    Build an {idx: acl} map (from all this document's chunk payloads' source_indices->acl) and pass it to assemble_big:
        acl_index = {i: cp["acl"] for cp in chunk_store.get(h.doc_id)   # every chunk payload for this doc
                     for i in cp["source_indices"]}
        from chunker import assemble_big
        big = assemble_big(chunk, secs, els, acl_index=acl_index)  # default: only pulls elements with the same ACL as the hit chunk
        # To instead "pull everything the caller is allowed to see (across different but visible ACLs)", pass
        # admit= to reuse the same predicate as the hard filter:
        #   big = assemble_big(chunk, secs, els, acl_index=acl_index,
        #                      admit=lambda acl: acl_admits(acl, user))
        out.append({"title": chunk.doc_meta.get("title"), "source": chunk.doc_meta.get("source"),
                    "breadcrumb": big.breadcrumb, "context": big.text, "hit": chunk.text,
                    "context_acl": big.acl})                       # big.acl: for a second check at the exit point
    return out
```

## 6. Document-level metadata and access control (permissions)

Both are **document-level attributes** (consistent across every chunk of the same document), **extracted at ingest time**, **stamped onto every chunk** by the chunker, and carried through to the vector store in the payload. **The chunker only stamps them on, it doesn't extract them or evaluate permissions** (it's format-agnostic and doesn't know a file's title/source/permissions).

- **`doc_meta` (convenience information)**: `extract_doc_meta(path, **overrides)` extracts from office `docProps/core.xml` (title/author/created/modified) and PDF `/Info`, merged with `source/doc_type/domain/lang` from the manifest/source system (overrides take priority, stripping empty values and junk like "admin"). Used for payload **filtering** ("2024 English-language financial reports") + **citation provenance** (title/source/date). Optionally fold just `title` into the embed text for disambiguation, keep **everything else payload-only** (folding everything in would dilute the embedding).
- **`acl` (a security boundary, an entirely different kind of thing)**: given by the **source system** (document-library ACLs / tenant mapping) at ingest time, and stamped onto `chunk.acl` by the chunker. **Invariants:**
  1. **Hard pre-filter**: permission is enforced at the vector store's **filter layer** (as shown above), not via re-ranking, and definitely not by "asking the LLM not to mention it." An unauthorized chunk is **simply never retrieved**.
  2. **Fail-closed**: if `acl` isn't passed → defaults to `RESTRICTED_ACL` (`unset=True`, empty `allow`) — a document whose permissions haven't been wired up **denies everyone by default**, never accidentally goes public.
  3. **Independently overridable per chunk**: `acl` is deep-copied onto each chunk, so a sensitive section can be tightened individually after chunking (`chunk.acl = stricter`) without affecting other chunks in the same document.
  4. **Big-block assembly must be ACL-aware (it's not "same document = safe")**: `assemble_big` pulls material from the raw elements by idx range, which **crosses chunk boundaries**. "Same document" ≠ "same ACL" — if invariant 3's per-chunk tightening was used, pulling material purely by idx would drag the plaintext of an unauthorized sibling section into `big.text`. **You must pass `acl_index`** (`{idx: acl}`, see the retrieve example above / `ChunkResult.acl_index()`): by default it only pulls elements with the **same ACL** as the hit chunk (fail-closed: unknown idx are excluded), or pass `admit=` to use the caller's own visibility predicate. The returned `BigBlock.acl` carries back the effective acl for a second check at the exit point; **if `acl_index` isn't passed, it's legacy unprotected mode (assumes the whole document has a single ACL), and `BigBlock.acl=None` explicitly marks it as "not access-checked."** The convenience wrapper `Chunker.assemble_big(hit, result, els)` already builds the acl_index from `result` automatically, and is safe by default.
  5. **Exit-point invariant**: any **text returned to a user** (a hit / big-block context / doc_meta) should, at the point it leaves the system, be mappable back to an acl and re-checked by the caller — don't trust that "the upstream filter already handled it." Both small-to-big (big.text) and `deny` (query filter) have historically been bypass routes around the filter.

> The ACL schema is a free-form dict (`{visibility, allow:[principals], tenant, classification}`), fill it in to match your own permission model; the chunker doesn't interpret the fields, it just faithfully stamps them on + defaults fail-closed.
> **`deny` (a blocklist) does not take effect in the hard-filter example above** — that example filter only reads `unset/tenant/allow/visibility`. To "exclude a group," just **remove it from `allow`**; if you genuinely need blocklist semantics, you must add `AND NOT (acl.deny ANY IN :principals)` to the filter yourself — otherwise a denied group will still be retrievable as long as it's still in allow (or matches public). **A security field that's "set but has no effect" is the most toxic kind of contract failure** — don't assume the chunker or the example filter enforces deny for you.

## 2. Writing a new parser adapter

If the parse stage is swapped for something else (Docling / Unstructured / a custom one), all it needs to produce is `Element[]`:

```python
from chunker.types import Element

def from_yourparser(parsed) -> list[Element]:
    out = []
    for i, item in enumerate(parsed.blocks):
        out.append(Element(
            idx=i,
            kind=map_kind(item.type),          # -> text|table|image|chart|list|header|footer|page_number
            text=item.text,
            text_level=item.heading_level,      # the heading level given by the parser (None if not available)
            page=item.page,
            caption=item.caption, table_body=item.html, asset_content=item.vlm_desc,
            merge_prev=item.continues_prev,     # continues across a page break (False if not available)
        ))
    return out
```
**Key fields**: `kind`, `text`, `text_level` (the primary leveling signal), `caption`/`table_body` (assets), `merge_prev` (page continuation). The rest can be left at their defaults. The core and `assemble_big` are reused entirely as-is, no changes needed.

**Before writing a new adapter, check whether the format is actually a good fit for this model** (see [`ARCHITECTURE.md` §Format and parser scope](ARCHITECTURE.md) for details):

- **PDF (MinerU)** ✅ verified (77 documents) — `from_mineru` is exactly this; scanned PDFs go through OCR (40 documents, 13 languages).
- **Word (.docx)** ✅ verified (~50 documents) — **MinerU's native office backend is recommended** (`scripts/parse_office.py`→content_list→`from_mineru`); there's also a zero-dependency fallback `adapters/docx.py` (paragraph style→`text_level`, with a bold/formatting fallback for the ~64% of documents that have no heading style). The notion of "page" is weak, so `page` is approximate.
- **PPT (.pptx)** ✅ verified — MinerU's native office backend is recommended; there's also a fallback `adapters/pptx.py` (slide = page, title→heading, bullets→`text`, images→`image`).
- **Excel (.xlsx)** ✅ adapted — goes through the **separate path** `table_chunker.py` (TableChunker, Approach A, verified on 51 documents, 100% cell coverage): splits by sheet/row group + uses column headers as context; the heading-tree doesn't apply, but it **outputs the same Chunk schema**, so it can be consumed by the same embedder/retriever. See [`../../methodology/MULTIFORMAT_IMPL.md`](../../methodology/MULTIFORMAT_IMPL.md).

## 3. Configuration

- **Token budgets**: `Chunker(target=800, min_tokens=200, max_tokens=1500)` overrides everything uniformly; or leave it unset to use the built-in `BUDGETS` per `doc_type` (`Chunker(budgets={"my_type": (200,700,1200)})` to extend them).
- **doc_type routing (recommended)**: for numbered types (academic/standards documents), a per-type budget is enough; for **unnumbered types, set expectations to "generic parent-child"** and don't expect precise hierarchy. `doc_type` doesn't need to be passed for it to run (it falls back to the default budget + `text_level`).
- **lang**: `"ch"`/`"en"` affects token estimation; be sure to pass `"ch"` for Chinese.
- **One chunk per page**: `Chunker(page_grouped={"slides_tutorial","my_slide_type"})`.

## 4. Relationship to the prototype harness

`../scripts/chunk_document.py` + `retrieve_big.py` are experimental prototypes (tied to a MinerU directory and a manifest); this component started out as a decoupled, installable version of them (at the time, byte-for-byte parity, 77/77). **Since then, fixes across rounds R1–R3 (reset-aware leveling / repeated-banner guard / content recovery / aside_text removal) have diverged from the prototype — the component is now the source of truth, the prototype was never updated, and is kept only as historical reference.**

## 5. Testing

```bash
python -m pytest -q          # 42 unit tests (test_core + test_table; uses its own fixtures, no external data needed; includes docx adapter cases)
python examples/run_mineru.py            # fixture demo
python examples/run_mineru.py <mineru_dir>   # run against a real document
```
