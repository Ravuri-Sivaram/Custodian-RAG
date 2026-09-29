# chunker

A **swappable chunking component** for the middle stage of a RAG indexing pipeline:

```
ingest → parse → [ chunker ] → embed
```

Splits a parser's output into **chunks carrying a hierarchical skeleton (breadcrumb + section tree)** for the embedding stage to embed and store, and at query time provides **small-to-big** big-block assembly. Pure Python, **zero runtime dependencies**. The core logic has been validated on **77 real documents + multiple rounds of adversarial review**, and it ships with **42** unit tests, all passing (`test_core.py` + `test_table.py`).

> **Note (this project):** the "cover-page-dense research reports in the project's original non-English language" example below is a real historical leveling
> finding (measured against `financial_research_zh`, from before this project replaced that original-language support with
> Telugu — see the top-level README), kept as genuine record and not re-measured against Telugu documents.

> **Five formats verified**: PDF (MinerU, 77 documents) / scanned PDF (OCR, 40 documents, 13 languages) / **Word (.docx), PPT (.pptx)** via MinerU's native office backend (~50 documents each; `adapters/docx.py`/`pptx.py` demoted to zero-dependency fallbacks) / **Excel (.xlsx)** via the separate `table_chunker.py` (a grid, not a document flow — the heading-tree doesn't apply, but it outputs the same Chunk schema). See [`ARCHITECTURE.md` §Format coverage](ARCHITECTURE.md) and [`../../methodology/MULTIFORMAT_IMPL.md`](../../methodology/MULTIFORMAT_IMPL.md).
> For the design motivation, trade-offs, and the conclusions of three rounds of adversarial review, see [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`../../methodology/LAZY_HEADING_TREE_DESIGN.md`](../../methodology/LAZY_HEADING_TREE_DESIGN.md) (v2).

---

## Install / run

Zero dependencies — run directly with `src/` on the path, or `pip install -e .`:

```bash
# Option A: install as a package
pip install -e .            # then `from chunker import Chunker` works from anywhere

# Option B: no install needed (the scripts already add src/ to the path)
python examples/run_mineru.py            # end-to-end demo using the bundled fixture
python examples/run_mineru.py <mineru_output_dir>   # run against real MinerU parse output
python -m pytest -q                       # unit tests (uses the bundled fixture, no external data needed)
```

## 60-second quickstart

```python
from chunker import Chunker
from chunker.adapters.mineru import from_mineru   # or from_mineru_dir("parsed_dir")

elements = from_mineru(content_list_json, layout_json)            # ① parse → Element[]
result   = Chunker().chunk(elements, doc_id="d1", doc_type="academic_paper", lang="en",
                           doc_meta={...}, acl={...})   # ② chunk it (acl defaults fail-closed; every chunk gets stamped with ACL+doc_meta for hard filtering/citation)

for c in result.chunks:                  # ③ hand off to embedding: embed text, treat the rest as metadata
    embed(c.text); store(c)              #    c.breadcrumb / c.section_id / c.section_anchor ...
store_sections(result.sections)          #    the section tree, for retrieval

big = Chunker().assemble_big(hit_chunk, result, elements)         # ④ query-time small-to-big
```

## The component contract (three seams)

| Seam | Type | Owner |
|---|---|---|
| **Input** | `list[Element]` (normalized parse units) | The parser **adapter** (`adapters/mineru.py`) |
| **Output** | `ChunkResult(chunks, sections)` | This component |
| **Retrieval helper** | `assemble_big(hit, result, elements) -> BigBlock` | This component (query time) |

→ When parsing eventually becomes its own component, all it needs to do is emit `Element[]`, or provide a new adapter (see [`docs/INTEGRATION.md`](INTEGRATION.md)) — **the core and the retrieval helper are reused entirely as-is**.

## Directory structure

```
chunker/
├── pyproject.toml                # installable (zero dependencies)
├── README.md                     # this file
├── docs/
│   ├── ARCHITECTURE.md           # design/data flow/v2 and adversarial review conclusions
│   ├── API.md                    # full API + data schema reference
│   └── INTEGRATION.md            # connecting into a pipeline / writing new parser adapters / configuration
├── src/chunker/
│   ├── types.py                  # Element / Chunk / Section / BigBlock (the stable schema)
│   ├── core.py                   # pure core: leveling · section tree · assemble_text · assets
│   ├── meta.py                   # doc-level metadata + ACL extraction (extract_doc_meta)
│   ├── table_chunker.py          # the xlsx separate path (TableChunker, grid → the same Chunk schema)
│   ├── retrieve.py               # assemble_big (small-to-big)
│   └── adapters/{mineru,docx,pptx}.py   # parser adapters (docx/pptx are zero-dependency fallbacks)
├── examples/run_mineru.py + fixtures/   # demo with a bundled fixture (can also run against a real directory)
└── tests/{test_core,test_table}.py      # 42 unit tests (using the fixture)
```

## What it does (in one sentence)

`text_level` (given free by the parser) as the primary signal + **decimal-point-numbering refinement** (`2.1` for hierarchy) + **reset-aware bare-integer promotion** (only treats it as an outline when document-wide numbering is monotonic; a cyclic restart means it's treated as a list and promotion is abandoned) + bullet guard + **repeated-banner guard** (removes per-page banners) → builds the section tree with a monotonic stack → asset atomization + **content recovery** (equations/footnotes/references go into the body, margin watermarks are removed) → every chunk gets `section_anchor` attached → at query time, `assemble_big` expands to an enclosing region based on real token counts (**climbing to merge ancestors when too small / merging adjacent sibling sections**, with banners consistently removed).

## Honest positioning

The quality of leveling **depends on how clean the parser's `text_level` is, not on doc_type**: when the parser gets it right (clean academic papers/standards documents) → precise hierarchy, accurate breadcrumb; when the parser flattens things but the numbering is monotonic (deep-dive reports) → reset-aware promotion recovers the chapter structure; when the parser flattens things and the numbering is cyclic (weekly reports, cover-page-dense research reports in the project's original non-English language) → it honestly degrades to a "single root + flat L2" generic size-based chunking, **don't expect chapter nesting**. Three rounds of adversarial review (0 refuted), format coverage, and boundaries are detailed in [`docs/ARCHITECTURE.md` §7 / §Format / §Scope of applicability](ARCHITECTURE.md).
