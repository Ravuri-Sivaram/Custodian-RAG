# Multi-format expansion implementation log (docx / pptx / xlsx / scanned documents)

> Goal: extend the chunker — previously validated only for **PDF→MinerU** — to other document types. The core claim is that the components are **format-agnostic** (they only consume `Element[]`); adding a format means writing an adapter and rerunning adversarial validation on that format's corpus. This file is the working log for that effort (diagnose → decide → verify).
>
> Status: docx ✅ validated / pptx ✅ validated / scanned documents ⏳ Phase 2 / xlsx ⏸ decided separately (Phase 3). Date 2026-06.

---

## 0. Why this step

The chunker had previously only been tested against PDF (MinerU), and its heuristics were all tuned to MinerU's quirks (reset-aware backfilling of flattened `text_level`, banner guards, `aside_text` removal). Users asked "is this only for PDF?" — architecturally no (the `Element[]` seam), but empirically yes (only a MinerU adapter existed, only validated on PDF). This step turns that claim from "in principle it can switch formats" into "empirically verified it can switch formats."

**Key insight**: office formats are OOXML, already structured — **MinerU is not needed** — you can read them directly with python-docx/pptx/openpyxl. And docx's heading styles give a cleaner `text_level` than what MinerU infers. Scanned documents still go through MinerU (OCR). Excel is a grid, not a document flow — a genuine fork in the design.

## 1. Corpus collection (corpus_multiformat/)

Pulled roughly 50 each of docx/xlsx/pptx/scanned documents from the public web, validating each file with magic-byte checks + library open verification + sha256 dedup + content review for diversity selection. Results (details in `corpus_multiformat/MANIFEST.csv`):

| Type | pool→final | size | languages | diversity |
|---|---|---|---|---|
| docx | 75→50 | 7.7M | 7 (en34/zh·fr4/de3/es·ja2/pt1) | full structural spectrum: plain prose→99 headings/52 tables/40 images |
| xlsx | 70→53 | 70M | 3 (en49) | 1–127 sheets, 70–245k rows, charts/formulas/merged headers |
| pptx | 60→50 | 258M | 5 + RTL Arabic + 2 image-only | 6–229 slides |
| scanned | 80→50 | 494M | **13** | 1–626 pages, vertical script/Nastaliq/Fraktur/Cyrillic |

Sources: archive.org (API), government open data, GitHub (enumerated via `gh`), edu sites. All gitignored (binaries not committed). **Constraint**: the MinerU online API caps at ≤200 pages per document → for Phase 2 scanned documents need to be filtered/sampled by page count (the pool included one with 626 pages).

## 2. docx adapter (`chunker/src/chunker/adapters/docx.py`)

> ⚠️ **Now demoted to fallback** — the shipping path for docx/pptx is documented in [§9](#9-decision-correction-docxpptx-now-officially-routed-through-the-mineru-office-backend-2026-06). The in-house adapter recorded in this section and §3/§5 is still valid (zero-dependency fallback), but is **no longer the primary path**.

**Design**:
- Walks the actual reading order of `doc.element.body`'s child nodes (interleaved paragraphs `w:p` and tables `w:tbl`) — `doc.paragraphs` does not preserve table order, so the body must be walked directly.
- Heading level uses the **language-independent `style_id`** (`Heading1..9`), falling back to `style.name`/`Title` — for non-English docx files, the displayed style name is localized, but the style_id is not.
- Tables → `<table><tr><td>` HTML into `table_body`; inline images (`w:drawing`/`w:pict`) → image elements.
- `page=0` throughout (docx can reflow, with no fixed pages) → page-grouped assembly and per-page banner guards **naturally deactivate** (correct behavior: docx has no inline banners, headers/footers live in a separate part and are naturally excluded when walking the body).

**Diagnose → decide: bold-heading inference (the one genuinely new heuristic)**

The first run found that **32/50 docx files had `sec=0`** (flat). Diagnosis: these documents simply **did not use Word Heading styles** (the author manually bolded/enlarged text to serve as headings) — this is the **normal state (64%)** for real-world docx files, not an edge case. python-docx cannot see a style → `text_level=None` → the result degrades into unstructured blobs, actually giving less structure than MinerU (which can infer from font).

But docx has a signal MinerU doesn't: **direct character formatting**. Quantified: among the 32 unstyled documents there were **452 "whole paragraph bold + short" candidates** (vs. 420 styled headings across the whole corpus) — the formatting signal roughly **doubles** the recoverable structure. Samples confirmed these were genuine headings ("TAKE ACTION NOW!", "Section 1. Purpose:", German "Jahresbericht 2017/18").

**Decision**: add a conservative fallback in the **adapter** (not core, to keep it format-agnostic) — **only when a document has no Heading styles at all**, infer a paragraph as a level-1 heading if it is "entirely bold + ≤14 words + does not end in sentence-final punctuation." Kept conservative by:
- Always level-1 (not betting on font-size tiers — the adversarial review from the PDF era proved that font_rank is unreliable across documents). The goal is section boundaries and breadcrumbs, not a full hierarchy.
- The 18 documents that do have styles **do not trigger** inference (styles are trusted, not second-guessed).

**Validation (before/after)**:

| Metric | before inference | after inference |
|---|---|---|
| headings (text_level set) | 488 | **690** |
| sections | 472 | **674** |
| content orphans | 0 | **0** |

Examples of flat→structured: bylaws 0→22, legal_berkeley 0→23, lugov 0→29, Portuguese ecdc 0→22. Documents that **remained flat** (mit_esp pure prose / cngov form in the project's original non-English language [part of the earlier original-language corpus, kept as historical record — see this project's Telugu swap in the top-level README] / CV / financial statement) are ones with **genuinely no heading structure** — inference did not force anything, correctly.

## 3. pptx adapter (`chunker/src/chunker/adapters/pptx.py`)

**Design**: slide = page (`page=slide_idx`) → paired with `Chunker(page_grouped={"pptx"})`, roughly one slide per text block. Title placeholders (`PP_PLACEHOLDER.TITLE/CENTER_TITLE`) → level-1 heading; body text_frame paragraphs → text; tables → table HTML; images → image; charts → chart. Speaker notes excluded. Titles sorted first.

**Test results**: 49/50 parsed (`grinch`, a pure-image deck, → 0 elements, as expected: 58 full-page PNG slides with no text), **orphan=0**, page-grouping working.

**Known issue (not a bug, pending a Phase decision)**: `chunks/slides=2.04` is not page-grouping failure, it's **asset atomization** — every image/table/chart becomes its own chunk, so image-heavy decks (e.g. the 201-image quotes deck) naturally exceed 1. But this amplifies an old problem: **captionless images/charts → 0-token placeholder chunks** (no retrieval value). At the embed stage, 0-token chunks could be skipped; or alt-text could later be pulled for pptx images to use as captions.

## 4. Verifying the core claim

`core.py` / `retrieve.py` had **zero changes** (all 12 unit tests still pass), both docx and pptx ran end to end with zero content loss. **"Core is format-agnostic" holds** — only the adapters are format-specific. The heuristics tuned for PDF behave correctly on office formats: banner guards auto-disable because `npages<4` (docx page=0); reset-aware handling processes office heading numbering as usual; `aside_text` never appears in office formats (harmless).

## 5. Adversarial validation (workflow `w1046q938`: 27 findings / 20 confirmed / 0 refuted)

**Biggest lesson: my `orphan=0` was a false metric.** The orphan check only walks the `Element[]` the adapter **already emitted** — it is completely blind to content lost **before extraction**. It measures adapter↔chunker consistency, **not** document↔chunker fidelity. "orphan=0 proves zero loss" was **wrong**. Adversarial review using real corpora surfaced this blind spot and 3 genuine content-loss bugs:

| # | Real bug | Measured loss | Fix |
|---|---|---|---|
| **F1** | docx **text boxes** `w:txbxContent` (nested inside w:drawing/mc:AlternateContent, not a direct child of body) were never walked | 10/50 files, 390 paragraphs, eu_easa lost 232 paragraphs | `_textbox_texts` extraction + Choice/Fallback dedup |
| **F2** | pptx **grouped shapes** GROUP not recursed (`slide.shapes` doesn't descend) | 16/50 files, 352 lines, harvard lost instructor emails | `_leaf_shapes` recurses into GROUP |
| **F4** | pptx `<a:br/>` soft line breaks swallowed by run-joining | in-line line breaks | switched to using `para.text` |
| **F3** | captionless images/charts → 1617 zero-token junk chunks | chunks/slides inflated to 2.04 | core `_asset_chunk` returns None when there's no caption and no content_raw |
| +sdt | docx **block-level content controls** `w:sdt` (a body child whose tag is not w:p/w:tbl) were skipped | es_ucss form placeholder text | `_iter_blocks` descends into `w:sdtContent` |
| +A4-2 | page-grouped retrieval: a section-less slide hit expanded its window to the whole deck | slide small-to-big | `retrieve` now windows a single-page hit only within that page |

**Built a real fidelity metric**, `scripts/coverage_office.py`: word-set coverage of adapter text vs. raw OOXML (`document.xml`'s `<w:t>` / a slide's `<a:t>`, excluding header/footer furniture). This is what actually reveals content lost before extraction.

**Post-fix results (real 50+50):**

| Metric | before fix | after fix |
|---|---|---|
| docx body-text coverage | (illusory orphan=0) | **95.6%** |
| pptx body-text coverage | (illusory orphan=0) | **99.1%** |
| **text orphans (real loss)** | — | **0** (docx & pptx) |
| pptx chunks/slides | 2.04 (including 1617 junk) | **1.23** (captionless cleaned up) |
| asset orphans (captionless images, expected drop) | — | docx 115 / pptx 1926 (no retrievable text) |

The flagged cases were individually re-verified: docx tamucc "Cultural Relativism" ✅, harvard instructor email ✅. The remaining <97% of docx word coverage is almost entirely **files in scripts without inter-word spaces** — character-level verification showed this is **measurement noise** (Word splits such phrases into multiple w:t runs, so tokens don't line up; cngov's character coverage is **100%**, jpgov **99%**), not real loss. The part of the "zero changes to `core.py`" claim that still holds: reset-aware/banner/aside_text logic was unaffected; what changed was `_asset_chunk` (general-purpose, benefits PDF too) and single-page windowing in `retrieve`. All 12 unit tests pass.

**Genuine gaps that remain (documented honestly, round-2 candidates, all confirmed non-blocking)**:
- docx **footnotes/endnotes** (a separate `footnotes.xml` part) are not captured — analogous to PDF's page_footnote, could be handled similarly.
- **Inline** content controls / deeply nested SmartArt (pptx harvard still at 88%) are not fully covered.
- Refinements to bold inference (enumeration exemptions have been added; distinguishing colon-labels from field labels is left for later).

## 6. Phase 2: scanned documents (✅ validated, reuses the mineru adapter, no new code)

**40/50 documents ≤200 pages** were parsed through MinerU VLM OCR (1659 pages, balanced across 3 accounts at 553 pages/account, **40 ok / 0 fail**), going through the **existing `from_mineru` adapter** (scanned → OCR → content_list → Element[], same as an ordinary PDF, no new adapter). The 10 documents >200 pages were excluded (the MinerU online API's ≤200-pages-per-document constraint; no language coverage was lost).

**Results (40 documents, 13 languages)**:

| Metric | Value |
|---|---|
| parse success | 40/40 |
| **text orphans (real loss)** | **0** |
| asset orphans (captionless scanned images) | 919 (expected, no retrievable text) |
| heading rate | 8% of elements |
| crashes | 0 (including RTL Arabic/Urdu, vertical scripts without inter-word spaces, Odia/Tamil) |

**Why orphan=0 is trustworthy this time (no self-referential-metric trap like office)**: `from_mineru` is a **1:1 faithful mapping** of content_list (no pre-extraction logic that could drop content), so "Element↔chunk conservation" is equivalent to "content_list↔chunk conservation" = true conservation. OCR fidelity is MinerU's responsibility, outside the chunker's scope.

**Real behavior found through spot checks**:
- Newspaper scans (zh_xinminbao 28 L1 / ja_fujin 26 L1): a high L1 count is **correct** — each newspaper article has its own headline, this isn't over-segmentation.
- Clean books/annual reports: section trees are reasonable. Manuscript/index scans (ta_tdl/raggy_rastus, 0 sections): genuinely unstructured, so flat is correct.
- Chunker structure quality **faithfully reflects MinerU's OCR-level structure**: accurate OCR → accurate structure, poor OCR → degraded structure. That is the parser's job, not the chunker's.
- Corpus annotation correction: `portuguese__AlchemistPauloCoelho` is actually Arabic/Urdu (mislabeled during curation), which does not affect parsing.

**Conclusion**: the scanned-document line **needs no chunker changes**; the existing PDF path, already validated three times over, covers it directly. Zero content loss, robust across languages and OCR quality.

## 7. Phase 3: xlsx table chunker (Approach A, ✅ prototyped)

Spreadsheets are a **grid, not a document flow** — the document chunker's whole heading-tree apparatus doesn't map onto them. So a **parallel table chunker** was built, `chunker/src/chunker/table_chunker.py` (`TableChunker`), but it **outputs the same `Chunk` schema** and plugs into ingest→chunk→embed the same way.

**Approach A design**: the retrieval unit is **a row group with the column headers as context** (rendered as markdown, so a query like "federal buildings in California" can hit). 
- Each sheet is split into table regions at blank rows; the first row of a region is the header; data rows are grouped by a token budget (target 600), each block carrying its own header.
- **Wide-table column splitting**: sheets with >40 columns are sliced by column, with each slice carrying the key column (the first column) as row context → no cells are lost.
- `openpyxl(read_only, data_only)` streams values; breadcrumb=[workbook, sheet]; page=sheet.

**Results (51 .xlsx files, 593 sheets)**:

| Metric | Value |
|---|---|
| **cell value coverage** | **100.0%** (after wide-table column splitting: beataml went 49%→100%, govuk 58%→100%, boe 88%→100%) |
| chunks | 152,343 (driven by huge tables: boe's 202k rows→64k blocks, chicago police 114k rows→28k blocks) |
| large tables / multi-sheet | 202k rows / 108·127 sheets, all handled (25s) |
| degraded blocks (chart/cover sheets) | data tables 0% / chart sheets 3–8% (fragmentary source-note blocks, content not lost) |

**Clean data-table rendering** (retrieval-ready): `Sheet: Sheet1\n| Location Code | Asset Name | ... | State | Zip | Latitude |\n| --- |...\n| ... |`.

### Adversarial validation (workflow `w3thlchnn`: 23 findings / 21 confirmed / 0 refuted)

An independent **multiset audit** (counting every occurrence, 5.84 million cells) **confirmed 100% value coverage** (even cleaner than my own self-report). But it uncovered a core defect: **the self-reported "100% coverage" masked header contamination** — blindly taking `header=first row of the region` combined with pure blank-row region splitting meant that on government statistical tables (boe/census/eurostat) **about 33% of region headers were actually title rows or unit rows, not real column names**, which directly wiped out Approach A's added value relative to a plain dump. This is a recurrence of the same pattern as the office "self-metric looks perfect while the core is actually broken" trap. In addition: `_cell` was **corrupting real values** by replacing `|` with `/` (8043 cells contained real author-separator columns), `.xls` files hard-crashed the harness silently, and the colgrp flag was computed incorrectly.

**Round-2 fixes (all shipped + verified with 17 unit tests)**:

| Problem | Fix |
|---|---|
| Header blindly taken as the region's first row (~33% wrong) | Find the **real header row** (the first multi-column label-like row); **multi-row headers** forward-filled and merged; **carry-over header inheritance** across continuation bands (a continuation data band reuses the previous band's header) |
| Title/note rows above the header were dropped or mistaken for the header | Title/unit/note rows above the header are emitted as **separate standalone text blocks** (never dropped, boe's table of contents fully preserved) → header accuracy went from 33% wrong to ~20% wrong (residual errors are mostly "year column header" misjudgments) |
| `\|`→`/` corrupting real values | Switched to markdown-escaping `\|` (preserving the value), and collapsing line breaks `[\r\n]+`→space |
| Continuation bands wider than the carried header lost columns | Continuation now uses `max(carried_width, this_band_width)`; multiset deficit went from 21134 to **0** |
| `.xls` hard crash + silent drop | `chunk()` now explicitly raises `ValueError`; the harness globs `*.xls*` so failures are counted as bad results and shown |
| colgrp flag computed incorrectly | Now records the actual column span, `cols:{first}-{last}` |

**After the fix**: multiset value coverage **100%** (boe/beataml/aces all at deficit=0), 17 unit tests pass, large tables still handled fine.

**Known remaining limitations (round-3 candidates)**: ① 2 legacy `.xls` files (need xlrd<2 or conversion); ② chart/dashboard sheets still render as fragmentary tables/text blocks; ③ header detection on extremely irregular multi-row headers is still ~20% imperfect (an inherent difficulty of government tables); ④ edge cases in rendering booleans/large integers as floats.

## 9. Decision correction: docx/pptx now officially routed through the MinerU office backend (2026-06)

> ⚠️ **This section overturns the conclusion in §2/§3 that treated the in-house adapter as the primary path.** The in-house adapter is demoted to a zero-dependency fallback; the **shipping path** for docx/pptx is now **MinerU's native office backend** (`scripts/parse_office.py`).

**Trigger**: discovered that MinerU has a **native office backend** (`mineru/backend/office/`, a purely rule-based OOXML parser, **with zero VLM/ML models** — a broad search across the torch/onnx/ocr/vlm code paths turned up zero hits; `MagicModel` is just a rule-based classifier), not "converts to PDF first" as I had previously mistakenly assumed. It runs locally with no quota, purely on CPU, and outputs a content_list (the same schema as PDF/scanned documents).

**The first comparison I made, concluding "the in-house adapter is stronger (94% vs 87%)," was a measurement bug.** Adversarial review (`w2rjwb2df`, 19 confirmed / 3 refuted) nailed it: my MinerU coverage extractor **only counted `text`+`table_body`, missing `list_items`** (where MinerU puts notes/numbered lists/bullets). Counted fairly across all fields:

| Measure | In-house adapter | MinerU |
|---|---|---|
| my wrong measure (missing list_items) | 94.5% | 88% |
| **fair measure (all fields)** | 94.5% | **95.1% (now ahead)** |
| documents where "adapter leads by ≥8pct" | — | 15/50 → **0/50** |

`edu_tamucc`'s "Cultural Relativism" **was not lost by MinerU** (it was in a list_items block); the 98%→47% figure was purely a measurement artifact. A true word-level accounting actually favors MinerU (catching 157 vs. 65 more words). Real bugs were also found in the in-house adapter: `_table_html` doesn't recurse into nested tables (un_undf lost body text), and inline `w:sdt` doesn't descend (es_ucss lost Spanish form fields).

**Head-to-head with Tika (same measure, real test)**: for docx, all three (in-house/MinerU/Tika) come out **evenly matched at ~95%** text coverage; for pptx, Tika edges ahead slightly on group handling (harvard 100% vs. MinerU's 88%); for xlsx, both just give the whole table (neither does grid-based chunking).

**Why MinerU over Tika**: given coverage is even, MinerU's content_list schema is **chunk-ready out of the box** (text_level/list_items/table_body/image_caption/chart/OMML→LaTeX), it shares a **unified pipeline** with PDF + scanned documents (same `from_mineru`), and it is **pure Python with no JVM** (Tika needs a JVM + a 63MB JAR). Tika is better suited to "dump plain text straight into a search index" scenarios.

**End-to-end validation**: `parse_office.py` parsed 100/100 docx+pptx successfully (writing `_manifest.csv` for pptx's page_grouped routing), and the chunker had zero failures (50+50, 2733 headings, 3487 chunks).

**Meta-lesson (recorded honestly)**: in this expansion I **fell for a self-flattering metric three separate times** — ① the orphan=0 self-metric (exposed by office validation), ② xlsx's 100% coverage masking header contamination, ③ this time, missing list_items and judging MinerU too low. The common pattern: self-referential metrics fool you; the truth requires independent ground truth plus adversarial measurement. **This is itself the strongest argument for "do less hand-rolling, use mature libraries more, and always verify independently."**

## 10. Folding image descriptions into retrieval text (making images findable, 2026-06)

**Goal**: images are invisible in vector retrieval; fold "the best available description" into the embedding text of an image chunk so it can be retrieved (Design A: each image becomes its own chunk, with its description used as the retrieval text).

**Part 1 (core.py `_asset_chunk`/`_asset_desc`)**: for images/charts, retrieval text = caption + footnote + `asset_content`. `_asset_desc` for mermaid **extracts only node/edge labels** (`["Irrigation","Runoff"…]`), discarding scaffolding (`graph TD`/`-->`/`style X fill:#f9f`) and truncating at a word boundary at 800 characters.
**Part 2 (parse_office.py)**: for docx/pptx, alt text is pulled from `wp:docPr@descr`/`p:cNvPr@descr` into `image_caption`, with **aggressive filtering** (filenames/paths/underscore IDs/PPT boilerplate/Office's auto-generated captions like "with low confidence"/"A picture containing"), **applied only when image counts line up** (skipped otherwise to avoid mismatches).

**Adversarial review (`wmgwu4n4i`, 4 findings confirmed; verify/synth hit a session limit and didn't fully complete) → all fixed**:
- Mermaid syntax noise: 67 chunks → **0** (F2).
- Missed office auto-captions: 3/12 → **0** (part2-F1).
- **Honest number correction (F1)**: not 1049 but actually **1586** image chunks, and of those, only **47% (757)** actually have real VLM `asset_content` — the rest, in scanned documents, are sourced from caption/footnote (some of which is OCR noise like `2011141981`). Previously "VLM extraction" and "caption sourcing" had been conflated; this has been corrected.

**After validation**: 1585/1586 PDF/scanned images carry a description; office formats have 9 clean alts. **Three tiers**: Tier-0 (alt) ✅, Tier-1 (MinerU image VLM, already present for PDF/scanned documents) ✅, Tier-2 (generic captioning for plain photos) not yet wired up.

**Unverified/unresolved (session limits + design tradeoffs, pending follow-up)**: ① **alignment correctness** (is the i-th alt in the document actually the i-th image?), ② **is skipping on count-mismatch too conservative** (brooklyn has 41 real alts but was skipped because MinerU only emitted 35 — could explore reconstructing a match via img_path=sha256(b64)), ③ **retrieval precision impact** (could image descriptions cause false-positive retrievals?) — these three review dimensions did not get an independent verdict due to the session limit.

## 11. Security audit and fixes (metadata + ACL rollout, 2026-06)

> Goal: bring document-level **metadata + access control (ACL)** design into the chunk stage (the document pipeline doesn't yet wire up permissions, but the boundary needs to be built in ahead of time). Semantic layering: **`doc_meta` = convenience** (payload filtering + citation), **`acl` = security boundary** (hard pre-filtering at retrieval + fail-closed). After rollout, ran a **large security-focused workflow** as an adversarial audit.

**Design rollout**: `chunker STAMPS, ingest EXTRACTS` — `extract_doc_meta` (office core.xml / PDF Info) extracts metadata, and `Chunker.chunk(..., doc_meta, acl)` **deepcopies both onto every chunk**; `acl` defaults to **fail-closed** as `RESTRICTED_ACL` (`unset=True`, empty allow) → a document with permissions not yet wired up is denied to everyone by default, never accidentally public. `TableChunker` (xlsx's second Chunk-production path) is stamped the same way.

**Adversarial audit (workflow `wipax2btr`: 19 findings / 17 confirmed / 0 refuted / 8 confirmed real leaks)**. After merging duplicates: two root causes and one medium (the TableChunker series had already been merged by a fix task during the review, verified real by me):

| Item | severity | verdict | status |
|---|---|---|---|
| **`assemble_big` small-to-big pulling material across ACL boundaries** (S3 series + F2) | 🔴 critical | **must-fix before launch (blocking)** | ✅ fixed |
| **`deny` field silently ignored** | 🟠 high | **must-fix before launch / change the contract (blocking)** | ✅ contract changed |
| `extract_doc_meta` override has no key allowlist | 🟡 medium | recommended fix (non-blocking) | ✅ fixed |
| TableChunker fail-closed (bare `{}` bypassed the gate) | 🔴→✅ | acceptable; add a regression test | ✅ fixed during review, verified |
| deepcopy isolation / public constrained by tenant / doc_meta not a side channel | ✅ praise | correct | kept as is |

**Core conclusion (in the reviewer's own words) — "production is closed, retrieval is open"**: the chunker's two production paths (core and table) now both fail-closed and stamp `RESTRICTED_ACL`, but `assemble_big` re-fetches material **after** the hard filter, by index, from the raw elements — completely undermining the guarantee that "a chunk with no access simply can't be retrieved." Hitting one publicly visible subsection would let small-to-big pull in sibling subsections within the same range that had been individually tightened (per-chunk override, formerly "iron rule 3"), in plaintext, right alongside it. **The contract was self-contradictory**: the old "iron rule 4" claimed that "a big-block only pulls material from within the same document, so context never crosses a permission boundary" — but "same document" ≠ "same ACL," and that claim was **simply false**.

**Fixes (all shipped + verified against real corpora/unit tests)**:

| Problem | Fix |
|---|---|
| S3/F2 cross-ACL material pulling | `assemble_big` gained an `acl_index` (`{idx:acl}`, `ChunkResult.acl_index()`) and an `admit` predicate; `_gather`/`_window_within`/`toks` all now pull material ACL-aware, defaulting to an **equivalence class** (only pulling from elements with the same ACL as the hit chunk; unknown idx is **fail-closed excluded**); `BigBlock` gained an `acl` field carrying back the verification basis; `Chunker.assemble_big` auto-builds an acl_index → **the convenience path is now safe by default, with zero caller changes required** |
| deny silently ignored | Adopted review recommendation (b): §6 of the schema **removed `deny`** with an explicit warning: "the hard-filter example does not evaluate deny — to restrict, remove entries from allow; if a real denylist is needed, add your own `AND NOT deny`" |
| meta override payload confusion | `meta.py` gained an `_ACL_KEYS` denylist, rejecting attempts to copy `acl/tenant/allow/...` into doc_meta |
| contract documentation | INTEGRATION §6 removed the incorrect "iron rule 4" → replaced with "big-blocks must be ACL-aware," and added a new **iron rule 5, exit invariant** (any text returned to a user must be mappable back to an ACL for review); the retrieve example was changed to build and pass an acl_index; API.md signatures fully aligned |

**Real-corpus before/after** (`government__21-00620-INLSR`, 568 elements / 80 chunks; after individually tightening one chunk, counted how many big-blocks small-to-big pulled its unique tokens into):

| | leaking big-blocks | does small-to-big still work? |
|---|---|---|
| LEGACY (no acl_index) | **3/79** | — |
| **after fix** (auto acl_index) | **0/79** | **78/79 blocks still grow normally** (not forced back to hit-only) |

> Note: the review's "83/84" figure is "how prevalent cross-chunk material-pulling is" = the attack surface; the "3/79" here is "of one tightened chunk, how many of its neighboring windows leaked its plaintext into." The two are consistent: cross-chunk material-pulling is nearly universal, and any tightened chunk leaks into whichever few neighboring windows cover it; after the fix this drops to zero with recall essentially unchanged. **Unit tests went from 24 to 27** (added: cross-ACL exclusion, legacy-no-protection lockdown, meta allowlist).

**Two honest caveats**:
1. **The equivalence class defaults conservative**: it judges equality via `acl_index.get(idx) == hit_acl`, but `allow` is a **list** — `['a','b']` ≠ `['b','a']`. Chunks that should be equally privileged but have their list in a different order will be judged as different → sibling content gets excluded (**safe direction, at the cost of recall**). For production use cases that want to precisely "pull everything the caller has access to (across different but visible ACLs)," pass `admit=lambda acl: acl_admits(acl, user)`, reusing the exact same predicate as the hard filter.
2. **Declined to adopt one of the review's suggested fixes**: it recommended wrapping `RESTRICTED_ACL` in `MappingProxyType` to prevent tampering — but `deepcopy(mappingproxy)` raises `TypeError` on Python 3.12, which would turn the fail-closed default path into a **crash**. The existing deepcopy isolation (which the review's real testing showed defeats all five classes of pollution attacks) is already sufficient.

**Integrator dependency (outside this repo's boundary)**: `acl_index`/`admit` is the contract handed to the integrator; the actual hard-filter SQL plus the `acl_admits` predicate must be implemented where the vector store is wired up — this repo only carries the documented contract, not the query engine that enforces it. When implementing it, be sure to: ① keep the `(... OR public)` parentheses (flattening it into a plain `AND` degrades into a cross-tenant bypass); ② remember deny does not take effect automatically.

**Meta-lesson (the flip side of §9)**: §9 recorded "self-metrics fool you" — this time, the reverse happened: the independent adversarial workflow **confirmed a prediction I had made honestly beforehand** (before running the review, I had already flagged F1 and F2, the two real vulnerabilities, to the user by name, and they were both confirmed real), and my own self-defense (deepcopy isolation) held up against all five attack classes. So independent adversarial validation cuts **both ways**: it can expose self-flattering false metrics, and it can also back up an honest prediction. Another lesson: **a security field that "is written but does nothing" (deny) is the most toxic kind of contract failure** — ops gets a false positive confirmation (no error, the field made it into the payload), while the control is actually empty.

## 12. Multimodal embedding interface: passing through img_path (2026-06-25)

**Background**: the downstream dense embedder chosen is `Qwen/Qwen3-VL-Embedding-8B` (multimodal, capable of directly encoding images). This **reverses** the earlier assessment that "image semantics are a known gap that can only be addressed via VLM text descriptions" — images/charts can now be **vectorized directly from the original cropped image**, skipping the lossy chain of "image→VLM text→text embedding." The precondition is that the chunker passes through MinerU's cropped-image references (previously discarded across the whole pipeline as `img_path`).

**Per-type decisions (made after inspecting MinerU's content_list fields directly)**:

| Type | What MinerU provides | Vectorization path | Does the chunker carry image_path? |
|---|---|---|---|
| image / chart | cropped image `img_path` (images/*.jpg) | **image vectorization** | ✅ exposed |
| table | `img_path` **plus** `table_body` (HTML) | HTML text (precise, structured, preferred) | ❌ None (goes through content_raw) |
| equation | LaTeX only (`text`/`text_format`), **no img_path** | LaTeX text | ❌ no image available |

**For now, only images (image/chart) get image vectorization**; tables go through HTML and equations through LaTeX, neither carrying an image reference (to avoid downstream misuse of table/equation images).

**Full-format image coverage (B②, fixed 2026-06)**: `img_path` now covers **PDF / scanned documents / docx/pptx**, all formats. docx/pptx go through the MinerU office backend, and the original `parse_office.py` passed `image_writer=None`, so it never cropped images (content_list's `img_path` was entirely empty — tested at 0/567); **fixed by passing a `FileBasedDataWriter`, so MinerU crops the images itself, saves them to disk, and fills in img_path** (rerun tested at **1097/1116** image/chart entries carrying a path, 870 deduplicated cropped images). Key insight: MinerU **handles image↔element association internally**, sidestepping the pitfall of "aligning by OOXML media order" — media mixes masters/decorations/duplicated images (acl_gov has 20 media files but only 3 actual content images), so manual alignment would inevitably mismatch, whereas MinerU's own cropping gets exactly the right 3. Was nearly about to hand-write OOXML image extraction and alignment (a fragile, large undertaking); inspecting parse_office instead revealed it was just a disabled `image_writer` parameter. (Alt-text caption enhancement is a separate, independent line, still conservative about alignment and still skipped on count-mismatch — it doesn't affect image vectorization, since image-only content is retrieved via image vectors, not alt text.)

**Keeping image-only content alive (①, fixed after adversarial review)**: `_asset_chunk` previously returned `None` outright for an image with "no caption/footnote/VLM content" (a hygiene rule from the text-only era) — testing showed this drops **about 30%** of images (academic 38%) — and these "text-less images" are exactly the content that VL image vectorization alone can retrieve. Changed to "**having an image_path is itself enough to be retrievable**": a text-less image still produces a chunk (text=placeholder, n_tokens=0), tagged with an `image_only` flag for downstream to recognize "route through image vectors, skip the sparse path"; only a true vacuous placeholder ("no image, no text, no body") is still discarded. Empty-string `img_path` is normalized to `None` (③).

**Implementation (touches three places, pure pass-through, no I/O)**:
- `types.py`: `Element.image_path` + `Chunk.image_path` (both `str|None`, added at the **end** of the dataclass so as not to break positional argument order).
- `adapters/mineru.py`: `from_mineru` sets `image_path=el.get("img_path")` for every element — **faithful extraction** (image/chart/table all get it filled; text/equation naturally get None).
- `core.py`: `_asset_chunk` sets `image_path` **only for image and chart** (`el.kind in ("image","chart")`); table/text get None.

**Contract**: `image_path` is a path **relative to the MinerU output root**, **faithfully passed through — the chunker never touches the filesystem** (keeping the core a pure function). Resolving absolute paths and feeding them to a VL embedder is **downstream's job (embed/ingest)**.

**Validation (diagnose→fix→verify)**:
- End-to-end on real data (`parsed/academic_paper__2309.17421v2`, 136 images + 6 charts + 2 tables): at the Element level, image/chart got a path in **142/142** cases, and tables were also faithfully extracted; at the Chunk level, image/chart got a path in **107/107** cases (107<142 because entries with no caption and no content were skipped), while table and text were **all None**.
- Unit test `test_image_path_passthrough`: locks in adapter extraction (Element image+table both have it, text has None) plus core pass-through (Chunk only image has it, table/text are None). **All 28 tests pass.**

**Deferred (the Element layer already has a hook)**: image vectorization for tables/equations has not been done; if a future need for a table's visual layout arises, the Element layer has already faithfully preserved the table's `img_path` — it would only require relaxing `_asset_chunk`'s kind check and adding downstream table-image path resolution.

---

## 13. Table round-2, tier 1: header-detection rewrite (merged-header + structure-misdetect, 2026-06)

**Cause**: an adversarial red-team run against five real pain points overturned my earlier judgment that "the proportion is too low to be worth fixing" (lesson recorded in memory as `feedback_severity-not-occurrence`). Tier 1 fixes two defects that share a root cause and directly undermine the component's core selling point ("binding column names to values").

**Three mechanisms fixed** (`chunker/src/chunker/table_chunker.py`):
1. **structure-misdetect (year headers)**: `_numeric_frac` now treats 4-digit years (1900-2100) as **labels rather than numeric values** → `['Race',2010,2020]` is no longer misjudged as a data row.
2. **merged-header (multi-tier merge geometry)**: `read_only`+`values_only` loses merge geometry (a merged value only appears in the top-left cell, the rest are None). Added `_parse_merges` (lightweight parsing of mergeCell directly from the xlsx zip **without loading cells** → doesn't blow up performance on large tables) plus `_broadcast` (broadcasting the top-left label across the whole merged range) plus a column-by-column top-down join → full multi-dimensional column names, replacing the old ffill that only concatenated 2 rows.
3. **band splitting (blank rows inside the header)**: `_bands` no longer treats "a blank row covered by a row-spanning merge" as a separator → multi-row headers (hvs has 5 rows) are no longer split into header_only fragments.

**Validation (diagnose→fix→verify)**:
- 3 regression unit tests: 3-tier merged header (leaf level not lost), year header (recognized as header), blank row inside header (band not split).
- Real US Census files, before/after:
  - `retail_mrts`: column 2 name `'CV for Retail Sales'` (dimension lost) → `'CV for Retail Sales 2023Q4 (p) Total E-commerce'` (full three dimensions), 9 columns including [E-commerce+2023].
  - `hvs_vacancy`: the "United States" row's column 6.4 name was **blank** → became `'Rental Vacancy Rates First Quarter 2023'`; header_only fragments went from **6 to 2**.
- Chunker's **33 tests all pass**.

**Performance**: merge geometry is parsed lightweight from the zip's XML, without triggering `read_only=False`'s full cell load → large tables (127 sheets / 240k rows) still stream fine.

**Round-2 tier 2 (legacy .xls, fixed 2026-06)**: `.xls` previously threw an outright error = the whole document produced 0 chunks (silently unretrievable); added `_read_xls` (xlrd reader, `formatting_info=True` to get merged_cells, half-open→inclusive conversion, **reusing tier 1's header reconstruction**). On real INSEE files: `population_ensemble.xls` went from **0 to 6767** chunks, `dossier28.xls` from 0 to 17 chunks; the fact the red-team pointed to (Ambérieu commune=14022) is now retrievable.

**Round-2 tier 3 (embedded charts, fixed 2026-06)**: Approach A only read cells and dropped charts. Added `_parse_charts` (parsing `xl/charts/chartN.xml` from the zip: all `<a:t>` titles/axes plus `<c:tx><c:v>` cached series names; sheet association is derived from series formula `'Sheet'!$range`), producing one `kind='chart'` chunk per chart. On real EIA STEO data: **0 → 66 chart chunks**, and the red-team-cited `Historical spot price`/`NYMEX` series names are now retrievable. Covers both real-world encodings (Excel's cached `c:v` and openpyxl's raw `a:t`).

**Round-2 tier 4 (oversplitting) → deferred to the retrieval stage**: the red-team itself characterized this as "mitigate, don't rewrite the splitting logic" — fundamentally this is about the retrieve layer using `source_indices` to stitch back together records that were split across columns/row groups, which is a **retrieval-time contract for wiring up the vector store**, not the table_chunker's splitting logic. **The chunker-side round-2 work (tiers 1-3) is complete.**

---

## 14. Pre-freeze adversarial review + fixes (before entering the embed component, 2026-06)

Before moving on to the embed component, ran a comprehensive pre-freeze adversarial review of the entire chunker (7 dimensions × red-team plus independent verify, 47 agents / 38 confirmed). Conclusion: **3 high-severity items were must-fix; once fixed, the freeze is ready.**

**3 must-fix items (fixed + locked in with regression tests)**:
1. **Same-name sibling section merging** (core.py `section_key`): duplicate subheadings (forms/appendices/multi-entity filings, ~15% of documents) had sibling sections merged together because they had identical crumb text, anchoring incorrectly to the first section → text from unrelated sections got mixed into the same embedding (**an unrecoverable write of bad vectors into the vector store**). Fix: `section_key` switched from crumb text to `_sec_head` (a unique index per heading instance).
2. **image_path had zero sanitization** (mineru.py→core): the single file-reading entry point for the multimodal embed chain passed the value through verbatim → a polluted value (`../`/absolute/UNC/`://`) → arbitrary file read/SSRF. Fix: `_safe_rel` sanitizes it at the component boundary.
3. **admit without an acl_index lies about big.acl** (retrieve.py `_make_admit`): when admit is provided but acl_index is missing, each element is evaluated against hit_acl (crossing ACL boundaries) but stamped with a non-None acl (masquerading as verified) → a silent leak. Fix: admit now requires an acl_index, otherwise it raises.

**Should-fix items handled at the same time**:
- **doc_type added to the Chunk schema**: the Chunk previously had no doc_type → assemble_big always used DEFAULT_BUDGET (law/finance budget queries didn't take effect). Chunk now carries doc_type, stamped at chunk time, and assemble_big picks it up correctly (law=700 verified working). **Stabilizes the embed payload schema.**
- **`zh` alias for lang**: est_tokens now accepts the ISO `zh*` form (it previously only recognized `ch` — passing `zh` silently fell through to the English divisor, risking a 2.35x underestimate, for scripts without inter-word spaces, that could get truncated by Qwen3-VL).
- Documentation hygiene: API.md flags updated with `image_only`/table flags and a `doc_type` row; INTEGRATION payload updated with `lang/page_end/doc_type`; deleted the dead code `_is_title_row` (a NameError bomb referencing an undefined regex); refreshed test counts.

**Accepted tradeoffs (not blocking the freeze)**: est_tokens heuristics (to be re-calibrated against BUDGETS with an official tokenizer once Qwen3-VL is wired up); promote_bare's all-or-nothing brittleness; docx/pptx fallback doesn't fill img_path; the prototype chunk_document.py continuing to exist alongside; acl_index defense-in-depth items.

**Validation**: chunker's **38 tests all pass** (4 new additions: same-name siblings / path sanitization / admit-raise / doc_type-lang) plus real-document validation confirming the section_key change caused no breakage on an academic document (415 chunks / 300 sections / 0 missing anchors). **The chunker is frozen and ready to move to embed.** Tier 4 oversplitting and est_tokens re-calibration remain for the retrieval/embed stage.

---

## 8. To do

- **Image alt-text/caption (a genuine cross-format gap)**: MinerU/in-house/Tika all only reach the neighboring caption for embedded images (≈1%); actually extracting the semantics inside an image requires wiring up a separate vision model (Tika can extract docx's `wp:docPr@descr` alt text — this is portable).
- ~~benchmark Docling HybridChunker vs. the in-house chunker core~~ ✅ **done (2026-06, see [EVALUATION.md §8](CHUNKING_EVALUATION.md))**: Docling collapses on evidence preservation against MinerU output (missing 3–5×) → not switching; the in-house chunker's moat is engineering integration like precise source_indices provenance, **not the boundary algorithm** (on pure boundary quality, chonkie is slightly ahead, 71.6%→74.9%). Component-version retesting confirmed the R1–R3 fixes have no measurable effect on evidence preservation (their value is in breadcrumb/retrieval semantics).
- Table chunker round-2: legacy .xls support, multi-row/merged headers, chart-sheet detection and skipping, oversplit correction.
- In-house adapter (fallback only, low priority): `_table_html` recursing into nested tables, inline `w:sdt` descent.

## 7. Deliverables

- **docx/pptx parsing (primary)**: `scripts/parse_office.py` (MinerU office → `parsed_office/<id>/*_content_list.json` + `_manifest.csv`)
- **xlsx**: `chunker/src/chunker/table_chunker.py` (Approach A table chunker)
- adapter (fallback): `chunker/src/chunker/adapters/{docx,pptx}.py` + primary `adapters/mineru.py`
- harness: `scripts/review_{docx,pptx,xlsx,scanned}.py`, `coverage_office.py`
- corpus + manifests: `corpus_multiformat/{docx,xlsx,pptx,scanned}/` + `MANIFEST.csv`, `parsed_office/`, `parsed_scanned/` (all gitignored)
- **metadata + ACL (§11)**: `chunker/src/chunker/meta.py` (`extract_doc_meta` + ACL key allowlist); `core.py`/`table_chunker.py` (stamping + fail-closed `RESTRICTED_ACL`); `retrieve.py` + `types.py` (`assemble_big` ACL-aware + `BigBlock.acl` + `ChunkResult.acl_index()`); contract `chunker/docs/INTEGRATION.md §6` (iron rules 1–5), `API.md`; regression `chunker/tests/test_{core,table}.py` (ACL assertions)
