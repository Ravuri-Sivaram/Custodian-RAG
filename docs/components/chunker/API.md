# API reference

`from chunker import Chunker, assemble_big, Element, Chunk, Section, ChunkResult, BigBlock`

## `Chunker`

```python
Chunker(target=None, min_tokens=None, max_tokens=None, budgets=None, page_grouped=None)
```
- `target/min_tokens/max_tokens`: if `target` is passed, this `(min,target,max)` token budget is used to override all doc_types; if not passed, the built-in `BUDGETS` are used per `doc_type` (`DEFAULT_BUDGET=(200,800,1500)`).
- `budgets`: `{doc_type: (min,target,max)}`, overrides/extends the built-in budgets.
- `page_grouped`: `set[doc_type]`, these types are chunked as "one chunk per page" (default `{"slides_tutorial"}`).

### `Chunker.chunk(elements, *, doc_id, doc_type=None, lang="en", doc_meta=None, acl=None) -> ChunkResult`
Splits `Element[]` into chunks + a section tree.
- `elements`: `list[Element]` (produced by the parse adapter).
- `doc_id`: the document id (used as the prefix for chunk_id / sec_id).
- `doc_type`: optional, used to select the budget + for `law`-specific handling; if not passed, uses the default budget.
- `lang`: `"en"` or `"ch"` (affects token estimation: en≈chars/4, ch≈chars/1.7).
- `doc_meta`: document-level metadata dict, **stamped onto every `chunk.doc_meta`** (for payload filtering + citation). Obtain it with `extract_doc_meta`.
- `acl`: access policy dict, **stamped onto every `chunk.acl`** (security; deep-copied, can be overridden per chunk). **If not passed → fail-closed `RESTRICTED_ACL`**. Retrieval must hard-filter on it — see INTEGRATION §6.

### `extract_doc_meta(path, **overrides) -> dict` (an ingest helper, not core)
Extracts title/author/created/modified + format from office `docProps/core.xml` / PDF `/Info`; `**overrides` (source/doc_type/domain/lang) take priority; strips empty values and junk values like "admin".

### `Chunker.assemble_big(hit_chunk, result, elements, target=None, min_tokens=None, max_tokens=None, admit=None) -> BigBlock`
Query-time small-to-big. Budget defaults inherit whatever was set at construction time / the doc_type. **Automatically builds an `acl_index` from `result` and passes it along → ACL-safe by default** (only pulls elements with the same ACL as the hit chunk; sibling sections that were tightened won't get pulled into big.text). `admit=acl->bool` can be used to substitute a custom visibility predicate. Equivalent to the module-level `assemble_big(hit, result.sections_by_id(), elements, acl_index=result.acl_index(), ...)`.

## `assemble_big(hit_chunk, sections_by_id, elements, target=800, min_tokens=200, max_tokens=1500, banners=None, acl_index=None, admit=None) -> BigBlock`
The pure-function version. `sections_by_id`: `{sec_id: Section}` (use `result.sections_by_id()`); `elements`: the same set used during chunking (idx-aligned), used to pull body text by anchor. `banners`: the document's set of banners (pass `result.banners` to avoid recomputation; recomputed internally if `None`) — ensures big blocks strip per-page banners consistently with chunking.
- **`acl_index`**: `{idx: acl}` (use `result.acl_index()`). **Security**: big-block assembly pulls material across chunk boundaries; passing this means only elements with the **same ACL** as the hit chunk are pulled (fail-closed: unknown idx are excluded), so per-chunk-tightened sibling sections' plaintext won't leak into `big.text`. If not passed → legacy unprotected mode (assumes the whole document has a single ACL), and `BigBlock.acl=None` marks it as unverified. See INTEGRATION §6, invariant 4, for details.
- **`admit`**: `acl->bool`, a custom visibility predicate (overrides the default "same-ACL equivalence class") — used for "pull everything the caller is allowed to see" (across different but visible ACLs), reusing the same predicate as the hard filter.

## Adapter: `from chunker.adapters.mineru import from_mineru, from_mineru_dir`
- `from_mineru(content_list, layout=None) -> list[Element]`: `content_list` = the parsed `*_content_list.json` (a list); `layout` = `layout.json` (a dict, only needed for `merge_prev`).
- `from_mineru_dir(doc_dir) -> list[Element]`: convenience wrapper that reads a MinerU output directory.

## Data schema (`types.py`, all dataclasses)

### `Element` (input)
| Field | Type | Description |
|---|---|---|
| `idx` | int | Reading order (starts at 0, contiguous) |
| `kind` | str | `text\|table\|image\|chart\|list\|header\|footer\|page_number` |
| `text` | str? | Body/heading text |
| `text_level` | int? | The heading-level hint given by the parser |
| `page` | int | Page number |
| `bbox` | list? | Coordinates (not used by this component, kept for provenance) |
| `list_items` | list[str]? | List items |
| `caption` / `footnote` | str? | Asset caption/footnote (normalized) |
| `table_body` | str? | Table HTML (the generation payload) |
| `asset_content` | str? | VLM-generated content for images/charts (low trust) |
| `sub_type` | str? | e.g. line chart / flowchart |
| `merge_prev` | bool | Continues the previous block (across a page break) |
| `image_path` | str? | Relative path to MinerU's cropped image (images/*.jpg); used for VL image vectorization; passed through faithfully, the core never touches I/O |

### `Chunk` (output, embeds `text`)
| Field | Description |
|---|---|
| `chunk_id` | `<doc_id>#0007` |
| `doc_id` / `kind` / `lang` | `kind ∈ text\|table\|image\|chart` |
| `text` | **This is what gets embedded** |
| `content_raw` | The asset's generation payload (table HTML / VLM content); `None` for text |
| `breadcrumb` / `section_path` | Ancestor heading chain / joined with `" > "` |
| `section_id` / `section_anchor` | The id of the section it belongs to / `[start_idx, end_idx]` (used for small-to-big) |
| `page_start` / `page_end` | Page range |
| `source_indices` | The list of source Element idx values (for provenance) |
| `n_tokens` / `trust` | Estimated token count / `high\|low` |
| `flags` | For text/image: `captionless` / `vlm_content` / `image_only` (pure image, routed through image vectorization, skips the sparse path) / `multi_page` / `merge_prev_stitched`; for table, additionally `nontabular` / `header_only` / `chart_meta` / `cols:N-M` / `sheet:X` |
| `doc_meta` | Document-level metadata dict (title/source/date/doc_type/...); for payload filtering + citation |
| `acl` | Access policy dict (security; hard filter + fail-closed `RESTRICTED_ACL` default) |
| `image_path` | Reference to the image/chart's cropped image (relative to the MinerU output root); used for VL image vectorization; `None` for text/table; **already sanitized** (rejects `../`/absolute paths/UNC paths/`://`) |
| `doc_type` | The document type (stamped at chunk time); `assemble_big` uses this to select the query-time token budget (falls back to DEFAULT if absent) |

### `Section` (the section tree)
`sec_id, doc_id, level, title, breadcrumb (including itself), start_idx, end_idx (exclusive), parent_sec_id`

### `ChunkResult`
`chunks: list[Chunk]` · `sections: list[Section]` · `banners: frozenset[str]` (the document's set of per-page banners, computed once at chunk time, reused by `assemble_big`) · method `sections_by_id() -> dict[str, Section]` · `acl_index() -> dict[int, dict]` (element idx → the acl of the chunk it belongs to; used by `assemble_big` for ACL-aware material selection; on idx conflicts, the stricter acl wins)

### `BigBlock` (a retrieval artifact)
`text, resolved_section, breadcrumb, n_tokens, climbed (how many levels it climbed and merged), anchor, note`

## Minimal example

```python
from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir

els = from_mineru_dir("parsed/mydoc")
res = Chunker(target=800).chunk(els, doc_id="mydoc", doc_type="academic_paper", lang="en")
big = Chunker(target=800).assemble_big(res.chunks[5], res, els)
print(big.n_tokens, big.breadcrumb, big.resolved_section)
```
