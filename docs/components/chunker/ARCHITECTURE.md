# Architecture / Design

> **Note (this project):** §7's adversarial-review findings and the "research reports in the project's original non-English language" leveling tier below were
> measured against a real dataset in that original language (`financial_research_zh`), from before this project replaced
> that original-language support with Telugu (see the top-level README). Kept as genuine historical findings, not re-measured
> against Telugu documents.

## 1. Position in the pipeline

```
ingest ──► parse ──► [ chunker ] ──► embed ──► (vector store)
                        │  this component
            from_mineru │                         at query time:
            (Element[]) ▼                         retrieve → hit chunk
                   Chunk[] + Section[]  ───────►  chunker.assemble_big(hit) → BigBlock → LLM
```

The component **only understands normalized `Element[]`** (it isn't tied to MinerU's specific JSON). `adapters/mineru.py` translates MinerU's `content_list.json` + `layout.json` into `Element[]` — this is the seam at the parse stage; swapping parsers only means swapping the adapter.

## 2. The ingest-time data flow (`Chunker.chunk`)

```
Element[]
  │  1. Noise filtering       drop kind ∈ {header, footer, page_number}
  │  2. Heading detection+leveling   text_level as the primary signal + numbered-section correction + year guard + TOC/overlength guard
  │  3. Monotonic stack       while top-of-stack.level >= current: pop → breadcrumb; also collect headings
  │  4. Section tree          build_sections: each section is [start_idx, next heading with level <= its own) + parent
  │  5. Asset atomization     each table/image/chart becomes its own chunk (body = generation payload; retrieval text = caption+footnote,
  │                     image/chart additionally fold in a VLM description, table additionally folds in breadcrumb+header row+row labels (capped; data cells
  │                     are not included) — with only a caption, numeric-heavy tables get crowded out of the top-k by prose; Custodian added this after real-data testing)
  │  6. Text assembly        assemble_text: consecutive text blocks within a section are greedily accumulated against a token budget (merge if under the floor, split on sentence boundaries if over the ceiling)
  │  7. Anchor attachment       every chunk is tagged with section_id + section_anchor (the section's idx range)
  ▼
ChunkResult(chunks, sections)
```

## 3. Multi-signal leveling (the core, `heading_level`)

| Signal | Usage | Notes |
|---|---|---|
| `text_level` | **Primary signal** (given free by the parser) | Baseline; on clean documents the parser already gets it right, on messy documents it gets flattened to a single level |
| Decimal-point numbering | **Refinement**: `2.1`/`3.4.2` — depth determined by the number of `.`-separated segments (`2.1`→2) | Only decimal numbering triggers a level bump — it's an unambiguous signal of real chapter/section structure |
| Bare-integer numbering (**reset-aware**) | Only promoted to L1 when the document-wide sequence of bare integers is **monotonically increasing** (a true outline); as soon as a **restart** occurs (`1…9,1…` = list usage), it's abandoned and falls back to `text_level` | **The core fix** (found by adversarial review F1): prevents cyclically-numbered list items like "1. Overseas AI:" from being mistakenly promoted to a top-level chapter |
| Bullet guard | Lines starting with `-•●○▪◦` etc. → not a heading (F3) | Prevents body-text bullet items like "- OpenAI…" from being treated as sections |
| Year guard | A leading `19xx/20xx` in the first paragraph is not treated as a number | Prevents "2026 Results" from being read as depth-1 |
| TOC / overlength guard | Dot-leader with trailing page number, or >120 characters → not a heading | Prevents table-of-contents entries/body paragraphs from polluting the tree |
| `law` SEC. | `SEC. N` → level 1 | Statutory prefix |

**Why fusion rather than betting on a single signal**: in the real corpus (77 documents / 5337 headings), only ~11% of headings have parseable numbering; the unnumbered case is the majority. So `text_level` has to be the primary signal, with numbering as a **conditional** correction — see §6/§7.
**Why reset-awareness is essential**: within the same `doc_type`, the semantics of numbering can point in opposite directions — in one weekly report, "1. Overseas AI:" is a news list item (should not be promoted), while in one deep-dive report, "1. On-device AI begins…" is a true chapter heading (should be promoted). The only structural signal that can tell them apart is **monotonicity**: a list restarts (`1…9,1…`), an outline doesn't. Neither doc_type nor keywords can reliably distinguish them, so whether the document-wide sequence of bare numbers is monotonic is what toggles promotion.

## 4. The section tree + section_anchor

Each heading opens a `Section{level, title, breadcrumb, start_idx, end_idx, parent_sec_id}`, spanning up to "the next heading with level ≤ its own." Each chunk records `section_anchor=[start, end]` (the idx range of the deepest section it belongs to). The tree doesn't inline body text — body text lives in the chunks, and the section tree is a structural index.

## 5. Query-time small-to-big (`assemble_big`)

Once a chunk is hit, it expands to a "big block" based on **actual token counts**:

```
sec = the hit's section
if tokens(sec) > max:          → open a window within sec around the hit (trim if too large)
while tokens(cur) < target:
    parent = cur.parent
    if parent is None:         → reached the top, best-effort (window over the whole document if still small)
    if tokens(parent) > max:   → open a window within parent (pulls in adjacent sibling-section content) ★ merges siblings
    else: cur = parent         → climb up and merge ancestors
return cur (or the window)
```

- **Climb ancestors when too small**: when small sections are clustered together (measured median 42 tokens), climb up level by level.
- **Merge adjacent sibling sections**: when climbing further would exceed max, instead open a window within the parent's range around the hit, which naturally pulls in content from sibling sections on either side.
- **Whole-document fallback**: if the top-level section is still small and has no parent → open a window over the whole document; a genuinely small document then correctly stays small.
- Measured: big-block median 818 tokens (≈ target), <200 tokens in 0.6% of cases (all from genuinely single-page, extremely short documents).

## 6. v2 and adversarial review (why it's designed this way now)

v1 was originally designed as "ingest keeps only a minimal numbering skeleton + lazily rebuild the hierarchy at hit time." **Adversarial review (run against 77 real documents) overturned two core assumptions:**

- **"Has numbering → zero LLM" only covers ~13%** (academic papers); unnumbered is the majority → switched to `text_level` as primary, numbering as correction.
- **"Lazy" was premature optimization**: eager full-tree construction over 77 documents takes 66.6ms total (<1ms/document), using the same algorithm as query time → switched to **eager** (built during ingest, attached to the chunk), with "lazy" reserved only for the rare edge case of huge heading counts + extreme sparsity + high update frequency.

Once settled, the essence of the design = **multi-signal-leveled eager parent-child construction + TOC removal**. Full record in [`../../methodology/LAZY_HEADING_TREE_DESIGN.md`](../../methodology/LAZY_HEADING_TREE_DESIGN.md).

## 7. Adversarial review findings: the fixture is happy-path (2026-06)

We ran the component on two documents side by side for comparison — `examples/fixtures` (13 elements, English academic paper, ideal case) versus a real research report `financial_research_zh__AP202601131816964706` (355 elements, in the project's original non-English language, dense cover page + cyclically-numbered bullets). **21 findings, 18 confirmed / 0 refuted.** Core conclusion: **on the fixture, heading numbering and `text_level` always agree, which rules out the one direction in which this leveling scheme can actually go wrong (numbering unconditionally overriding the parser)** — so the earlier claim that "numbering-based leveling is a real strength" was an artifact that only held up on clean documents, and flipped on a messy real-world report.

| Dimension | On the fixture | On the real research report (before fix) | Finding |
|---|---|---|---|
| Numbering-based leveling | ✅ Perfect (numbering ≡ text_level) | Of 25 L1s, **24 were list bullets wrongly promoted** (parser's original text_level=2); top-level precision ~12% | F1 critical |
| Hierarchy/nesting | ✅ Clean `2 > 2.1` | A true chapter, "2 Weekly Industry News," got **bisected** at idx113 by "1. Overseas AI" being wrongly promoted to L1, kicking out ~200 body elements; leaf-level items ended up promoted to sections (**head and tail inverted**) | F2 critical |
| breadcrumb | ✅ Chapter path | **51% at depth=1**; inverted into "1.Overseas AI > - OpenAI…"; cover-page label chains like "Industry Rating: Overweight" | F3 high |
| small-to-big | 4/4 degraded to doc-window (never climbed) | Hitting the cover page climbed to the heading banner s0 → **a 610-token, 7-topic grab-bag**; 43% fell back to doc-window | F1/F4 high |
| Size health | Toy-sized, which made things falsely look "normal" | 63 sections, 19%<50 tokens, 6 empty shells; 74 chunks **47%<50 tokens** | F4 partial |

**Root cause chain (single point of failure)**: `_bogus_number` only blocked years and `>40` → bare integers `1…40` all passed through → numbering correction **unconditionally** overrode the parser's `text_level` → list bullets got promoted to L1 → the stack got polluted → the tree inverted → breadcrumb/small-to-big all broke downstream as a chain reaction.

**Fix (reset-aware, see §3)**: bare integers are only promoted when the document-wide sequence is monotonic; as soon as a restart appears, fall back to the parser, with the bullet guard layered on top. This correctly identifies "cyclically-numbered weekly reports" as lists (don't promote) and "monotonically-numbered deep-dive reports" as outlines (do promote), eliminating the 24 false L1s without regressing documents with genuine numbering.

**Post-fix measurements (7 differentiated documents, 2026-06)**:

| Document | # of L1s | breadcrumb depth≤1 | orphaned content | Conclusion |
|---|---|---|---|---|
| AP4706 weekly report (target case) | 25 → **1** | 51% → **0%** | 0 | False chapters eliminated, inversion resolved |
| AP4904 deep-dive report | 4 → **4** | — | 0 | Monotonic numbering, **no regression** |
| Attention (academic) | 10 → **10** | — | 13⚠️ | Parser was already correct, **no regression** |
| law / slides / government | Unchanged | — | 0 | Numbering wasn't involved, **no regression** |
| NETFLIX 10-K | 18 → **15** | — | 0 | Numbered footnotes correctly demoted to nested L2 |

### Round-2 fix (content recovery + banner guard, adversarial review: 22 findings / 16 confirmed / 0 refuted)

② **Content recovery** (the `else` fallback in `core.py`'s emission loop): previously only text/list were emitted, and `equation` (28)/`page_footnote` (282)/`ref_text` (67)/`code` (19) were silently dropped — 407 elements lost silently across the whole corpus. After the fix, all non-asset elements carrying text go into the body. Measured result: orphaned elements in the Attention paper went from 13→**0**, and 7 equations were cleanly incorporated into chunks.
③ **Repeated-banner guard** (`_banner_texts`): if the same text appears on **≥50% of pages** and ≥3 times, it's judged to be a running banner and removed wholesale. Using page-coverage ratio (not raw frequency) avoids wrongly killing legitimate repeated headings (law document's `SALARIES AND EXPENSES`×4, slides' `Engine Sensors`×6 — both safe). Measured result: the government document's `FOR PUBLIC RELEASE` was removed, dropping L1s from 23→**4**.

### Round-3 fix (adversarial review caught 3 HIGH-severity issues on a GPT-4V paper I hadn't tested on, all now fixed)

- **A. Banner consistency at retrieval time** (`retrieve.py`): banners were only removed at chunk time, but `_gather` used the full element list → the government document had **25/38 big-blocks re-injected with banners** (up to 3 consecutive times). Fix: pass `_banner_texts` into the retrieval path as well so it filters there too → now **0/38**.
- **B. `aside_text` watermark pollution** (`core.py`'s NOISE_KINDS): the `else` fallback over-recovered margin watermarks (arXiv stamps, broker letterhead), injecting them into body text in 11/11 cases. Fix: added `aside_text` to NOISE_KINDS, so it's consistently removed at both chunk and retrieval time.
- **C. Banner-guard false positives** (`_banner_texts`): pure page-coverage ratio with no shape discrimination → wrongly deleted the GPT-4V paper's `Prompt:`/`GPT-4V:` labels (60%+ page coverage), a total of **198 speaker labels**. Fix: banners can no longer end in a colon → these 198 labels are kept as headings, the breadcrumb of 335 chunks now correctly carries speaker attribution, and **nothing is lost**.

**Round-3.1 cleanup** (4 low-severity items from the round-3 review, all landed): `NOISE_KINDS` now has a single source of truth (retrieve now imports it), `banners` is computed once on `ChunkResult` and reused (avoiding recomputation on every retrieval), `_window_within`'s growth loop counts tokens using the same filter (fixing banner/noise overcounting), and a unit test for `aside_text` at retrieval time was added. 13 unit tests → now 12 (after consolidation), all passing.

**Still unresolved (round-4 candidates)**: ① **Chapter integrity** — in AP4706, the real chapter "2 Weekly Industry News," its subsection `2.1`, and the news item "1. Overseas AI" are all `text_level=2` in the parser; once reset-aware flattens them uniformly to L2, they end up **cutting into each other** (span=1) — this needs in-document contextual leveling rather than a document-wide on/off switch. ② **Speaker-turn fragmentation** — in GPT-4V-style dialogue papers, once `Prompt:`/`GPT-4V:` (text_level=2) are kept as headings, every turn becomes its own micro-section (415 blocks, 73% <50 tokens); a possible fix is recognizing "short colon-terminated, high-frequency recurring" labels as an inline body-text prefix rather than a section heading. ③ **Footnote interleaving** — `page_footnote` is inserted mid-body-text in reading order; could add a footnote flag or move it to the end of the section. ④ **Banner colon false negatives** — the current rule "ends in a colon → not a banner" would let through genuine banners like `CONFIDENTIAL:`/`DRAFT:` (no such example in this corpus); a stronger signal would be **bounding-box position stability**.

## Scope of applicability (honest, revised version)

The quality of leveling **depends on how clean the parser's `text_level` is**, and falls into three tiers:

- **Strong (parser's text_level is already correct)**: clean academic/technical reports, English standards documents — the parser directly gets most heading levels right (e.g., for the Attention paper the parser gives 25 headings correctly as `text_level=1`), and numbering only does decimal-point refinement. Breadcrumb is accurate, zero LLM needed. This is the tier the fixture represents.
- **Medium (parser flattens headings + numbering is a real outline)**: deep-dive reports — the parser flattens all headings to a single level, but the numbering `1,2,3` is monotonic, so reset-aware promotion recovers the chapter structure. Usable.
- **Degraded (parser flattens headings + numbering is a list)**: weekly reports, cover-page-dense research reports in the project's original non-English language — numbering restarts cyclically, so reset-aware **deliberately gives up on promotion**, honestly degrading to a "single root + flat L2" generic parent-child structure (no chapter nesting); cover-page labels still end up as small noise sections (per-page banners are already removed by the round-3 guard). **Don't expect precise hierarchy in this tier — treat it as just "size-chunking with a breadcrumb attached."**

- **Not good at**: hierarchy on pure scanned documents, cross-hop synthesis (that's GraphRAG/RAPTOR's job), remote cross-references (not implemented), chapter integrity (when the same text_level mixes chapters and lists, see round-4 candidate ① in §7).
- **Recommendation**: anchor your expectations to "how clean is the parser's text_level," not to doc_type. Clean structure → precise; flattened by the parser → degrades to generic parent-child. This is a structural ceiling, not a bug.

## 8. Recent evolution (2026-06, pointers)

§1–7 covers the core chunking logic (heading-tree + small-to-big). Extensions built on top of it are covered in the corresponding docs:

- **Document-level metadata + ACL (security boundary)**: `Chunker.chunk(doc_meta=, acl=)` stamps onto every chunk, defaulting fail-closed to `RESTRICTED_ACL`; retrieval must hard-pre-filter on it; small-to-big **never pulls material across ACL boundaries** (`acl_index`/`admit`). See [`INTEGRATION.md §6`](INTEGRATION.md) + [`../../methodology/MULTIFORMAT_IMPL.md §11`](../../methodology/MULTIFORMAT_IMPL.md).
- **Multimodal**: `Chunk.image_path` (a reference to the image/chart's cropped image, already path-sanitized) + the `image_only` flag (a pure image with no text still survives, for downstream Qwen3-VL **image vectorization**, bypassing the sparse path). See [`API.md`](API.md) + MULTIFORMAT §12.
- **5-format coverage**: PDF/scanned documents → MinerU VLM; docx/pptx → MinerU's office backend (rule-based only); xlsx → a separate `table_chunker` (a grid, not a document flow — the heading-tree doesn't apply). See MULTIFORMAT.
- **table_chunker round 2**: merged-cell geometric reconstruction of multi-row headers, legacy `.xls` (xlrd), extraction of embedded chart titles/series names. See MULTIFORMAT §13.
- **doc_type on Chunk + sign-off adversarial review fixes**: same-named sibling sections no longer merged, image_path sanitization, fail-closed admission, doc_type-driven query-time budgets, `zh` alias for lang. See MULTIFORMAT §14.

**New fields in the data contract** (see API.md): `Element.image_path` · `Chunk.image_path` / `Chunk.doc_type` · `BigBlock.acl` · `ChunkResult.acl_index()`.

## Format and parser scope (honest)

By design this component is **parser-agnostic** (it only consumes normalized `Element[]`), so in principle any format can be supported by writing an adapter. **But all real-world testing to date (77 documents + three rounds of adversarial review) covers only the PDF→MinerU path.** Other formats are untested, and the underlying model's fit for them varies a lot — don't assume "it'll just work":

| Format | Fit | Notes |
|---|---|---|
| **PDF (MinerU)** | ✅ **Verified** | 77 real documents + three rounds of review. All of this component's real-world testing lives here. |
| **Word (.docx)** | ✅ **MinerU office backend (primary)** (50 documents, no VLM needed locally) | `scripts/parse_office.py` → MinerU's native office parsing (rule-based OOXML, zero models) → content_list → `from_mineru`. Fair comparison: MinerU ≈ our own adapter ≈ Tika (docx text coverage ~95%, all three tied); MinerU was chosen because the content_list schema is chunk-ready as-is (text_level/list_items/table_body/caption/chart/equation) + it unifies with the PDF/scanned-document pipeline + no JVM needed. `adapters/docx.py` has been demoted to a **zero-dependency fallback**. |
| **PPT (.pptx)** | ✅ **MinerU office backend (primary)** | Same as above, goes through `from_mineru` + `page_grouped` (slide = page). MinerU/our adapter/Tika have comparable pptx coverage (Tika edges ahead slightly on grouping). `adapters/pptx.py` has been demoted to a fallback. |
| **Excel (.xlsx)** | ✅ **Separate path** (Approach A, verified on 51 documents) | The document chunker's heading-tree **doesn't map** onto a grid → a separate `table_chunker.py` (`TableChunker`), but it **outputs the same `Chunk` schema**. Splits by sheet/blank-row table regions/row groups + **uses column headers as context** (markdown); wide-table column grouping doesn't drop columns. Cell-value coverage is **100%**, and it can handle huge tables (202k rows / 108 sheets). Limitations: legacy .xls, chart sheets, merged headers. See [`MULTIFORMAT_IMPL.md` §7](../../methodology/MULTIFORMAT_IMPL.md) for details. |
| **Scanned PDF** | ✅ **Verified** (40 documents, OCR, 13 languages) | Reuses the mineru adapter via OCR (same as a normal PDF, **zero new code**). Text-orphan rate **0**, 40/40 parsed; newspaper/book/manuscript structure comes out reasonable. MinerU's online API caps at ≤200 pages per document → beyond 200 pages requires page extraction. Quality tracks OCR accuracy (that's the parser's responsibility). |

**In one line**: this component is for **document-flow content** (prose + headings + embedded assets). **PDF / Word / PPT have each been individually verified on real corpora + adversarial review** (see [`../../methodology/MULTIFORMAT_IMPL.md`](../../methodology/MULTIFORMAT_IMPL.md) for details); **for Excel, use a different component**. **The core `core.py` remains format-agnostic and untouched** — everything format-specific lives in the adapters. When extending to a new format, be sure to run `scripts/coverage_office.py` to verify real-world fidelity (don't just trust a self-reported orphan=0 metric) and rerun adversarial review.
