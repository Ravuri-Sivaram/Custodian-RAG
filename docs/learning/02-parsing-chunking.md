# 02 Document Parsing and Chunking

> **Note (this fork):** several stories below (the AP4706 reset-aware fix, the CJK sentence-splitting bug, the
> `est_tokens` char/token calibration) were measured against real Chinese-language documents, from before this fork
> replaced Chinese-language support with Telugu (see the top-level README's Technology stack table). They're kept as
> genuine historical findings — including the CJK sentence-splitting bug, which doesn't apply to Telugu since Telugu,
> unlike Chinese, is written with spaces between words — and have not been re-measured against Telugu documents.

> **How to read this piece**: covers custodian's parsing layer (the unified MinerU entry point + the Element seam) and chunking layer (multi-signal heading-tree reconstruction, doc_type-based budgeting, asset atomization, query-time small-to-big).
> **Interview weight: High** — chunking is the layer in RAG where it shows most clearly whether you've actually worked with real-world corpora, and this piece contains the two best stories in the whole project: reset-aware promotion of bare numbering, and the 66.6ms measurement that killed its own lazy design.
> **Prerequisite reading**: none required; for where the evaluation numbers and their basis come from, see [07 Evaluation Methodology](07-evaluation.md) and [../methodology/CHUNKING_EVALUATION.md](../methodology/CHUNKING_EVALUATION.md).

---

## 1. Conceptual foundation: what problem chunking solves in any RAG system

Before getting into custodian, let's be clear about this layer's problem on its own terms — it has nothing to do with any specific project.

**Why chunking is necessary.** Three hard constraints stack up:
1. **Embedding representations dilute as text gets longer** — cram a 30-page document into a single vector, and no specific question will retrieve it accurately;
2. **LLM context is limited and billed by the token** — if the retrieval unit is too large, most of what gets fed in is irrelevant content;
3. **A chunk is the "atom" of a retrieval system** — citation traceability, permission control, deduplication, billing, all anchor at the granularity of the chunk. Get the cut wrong, and everything downstream is wrong.

**The core tension.** Embedding wants **small, semantically pure** chunks (for accurate recall); LLM generation wants **large, complete** context (to avoid taking things out of context). Every chunking scheme, at bottom, is trying to reconcile this tension.

**The mainstream spectrum of approaches** (from simple to structure-aware):

| Approach | Idea | Representative | Weakness |
|---|---|---|---|
| Fixed window | Cut every N tokens, possibly with overlap | Early tutorials | Sentences get cut in half, semantic boundaries are random |
| Recursive rule-based splitting | Recurse down through a hierarchy of separators (paragraph→sentence→word) | LangChain / chonkie | Doesn't understand document structure, headings and body text are treated equally |
| Semantic splitting | Cut where the embedding distance between adjacent sentences jumps | semantic-chunker-type tools | Expensive, unstable, still no hierarchy |
| Structure-aware | Cut along the heading tree/layout, chunks carry hierarchy information | Docling, custom-built | Depends on parse quality; heading detection is a hard problem |
| Parent-child / small-to-big | Retrieve small chunks, feed the LLM large chunks (the parent section) | Mature systems like LlamaIndex | Needs extra structure storage and query-time assembly |

**The parsing layer has an equivalent spectrum**: plain-text extraction (PyPDF) → layout-aware parsing (MinerU / unstructured, which give element types, bounding boxes, heading levels) → VLM deep parsing (turning tables into HTML, extracting chart content). Chunking's ceiling is set by parsing's floor: if the parsing layer loses table structure, no amount of cleverness in the chunking layer can get it back.

**A perspective that's often overlooked**: the evaluation criterion for chunking isn't "how pretty the cuts look," it's **"after cutting, is the evidence for a question still sitting inside one retrievable unit"** — evidence preservation. This is custodian's starting point for evaluating chunking (§3, [../methodology/CHUNKING_EVALUATION.md](../methodology/CHUNKING_EVALUATION.md)), and it's the key to understanding all of its design trade-offs.

**The output contract of the chunking layer** (what a production-grade chunk should carry, not just text):

- **Retrieval text**: what embedding/BM25 consume, which can differ from the source text (especially true for asset chunks);
- **Generation payload**: the full content fed to the LLM after a hit (table HTML, chart data);
- **Traceability**: a back-pointer to the index of the original parsed elements, supporting citation, auditing, and evaluation alignment;
- **Structure**: a breadcrumb (the chain of ancestor headings) + the range anchor of the section it belongs to, supporting query-time context assembly;
- **Governance metadata**: document-level metadata (for filtering/citation) + access control policy (permission granularity = chunk granularity).

Use these five items as a checklist against any chunking scheme, and the difference becomes obvious — fixed windows and recursive splitting only give you the first one.

---

## 2. How Custodian does it

### 2.0 Data-flow overview

```
PDF/scanned documents/docx/pptx ──unified MinerU parsing──> content_list.json (+ layout.json)
        │
        ▼ adapter normalization (from_mineru)
   list[Element]  ←—— swap the parser = swap the adapter, the core is untouched
        │
        ▼ Chunker.chunk()
   ① three-part noise filtering (NOISE_KINDS / running banners / content reclamation)
   ② multi-signal heading-tree reconstruction (mainly text_level + reset-aware bare numbering) — eager, sub-millisecond
   ③ leaf chunking (doc_type budgeting + asset atomization + page_grouped)
   ④ stamping (chunk_id / section_anchor / doc_meta / fail-closed ACL)
        │
        ▼
   ChunkResult{chunks, sections, banners} ──> vector store (chunk) + sidecar (elements/sections/acl_index)
        │
        ▼ query time (after a hit)
   assemble_big: small-to-big, three states (climb / window / fallback), reads the structure ingest already built, doesn't rebuild it
```

xlsx/xls follow an independent TableChunker path (a grid is not document flow, the heading tree doesn't apply), but they output the same Chunk schema.

### 2.1 Unified parsing: MinerU + the Element seam

custodian collapses every format down to a single parser, MinerU: PDF, scanned-document OCR, and docx/pptx (MinerU's native office backend) all produce the same `content_list` schema. The adapter layer has exactly one responsibility — normalize the parser's output into `Element[]` ([src/chunker/types.py:22-41](../../src/chunker/types.py#L22-L41)): `idx/kind/text/text_level/caption/table_body/asset_content/merge_prev/image_path`. The core only consumes this seam — **swap the parser = swap the adapter, the core is untouched** ([src/chunker/adapters/mineru.py:45-65](../../src/chunker/adapters/mineru.py#L45-L65)).

There's an easily-overlooked detail here: the cross-page continuation flag `merge_prev` isn't in content_list — the adapter backfills it from layout.json's para_blocks using fuzzy matching on normalized text prefixes ([src/chunker/adapters/mineru.py:31-38](../../src/chunker/adapters/mineru.py#L31-L38)). Why not align by bbox instead? Because content_list's and layout.json's bboxes live in **two different coordinate systems** (rendered image vs. PDF points) — this pitfall bites again during evaluation (§3.3). This fuzzy matching itself has a confirmed cross-talk defect, see §4.

The custom-built docx/pptx adapter was downgraded to a zero-dependency fallback, kept around because the pitfalls it hit have teaching value (text boxes, recursive GROUP shapes, inferring headings with no style — in the real world, 64% of docx files don't use heading styles).

### 2.2 The three-part noise filter

Parsed output is mixed with three kinds of "garbage that looks like content," each with its own targeted gate:

1. **kind blacklist**: `{header, footer, page_number, aside_text}` are dropped outright ([src/chunker/core.py:10](../../src/chunker/core.py#L10)). `aside_text` (page-margin watermarks/vertical arXiv stamps) was added in the third review round — confirmed as noise across 11/11 corpus cases.
2. **Duplicate banner guard**: text that appears on ≥50% of pages and ≥3 times is judged a running banner and removed entirely ([src/chunker/core.py:115-133](../../src/chunker/core.py#L115-L133)). Two refinements: it uses the **page proportion**, not an absolute frequency (otherwise it would wrongly kill the legitimately repeated 'SALARIES AND EXPENSES'×4 in a law document); and **anything ending in a colon doesn't count as a banner** — in dialogue-style papers, 'Prompt:' / 'GPT-4V:' recur on 60%+ of pages, but that's a speaker label, and it's content. This rule was added after 198 speaker labels were wrongly deleted in an untested GPT-4V paper.
3. **Content reclamation**: the else-branch fallback of the emission loop sweeps "has text but isn't an asset" elements — equation/code/page_footnote/ref_text and the like — back into the body text ([src/chunker/core.py:319-328](../../src/chunker/core.py#L319-L328)) — before the second review round, 407 such elements were being silently dropped.

The banner set is stored in `ChunkResult.banners`, and query-time `assemble_big` reuses the same set ([src/chunker/retrieve.py:31](../../src/chunker/retrieve.py#L31)) — **the chunk stage and the retrieval stage must strip the same things consistently**. This isn't fastidiousness: if banners are only removed at chunk time, big-block material-pulling from the raw elements will reinject them — measured on a real government document, 'FOR PUBLIC RELEASE' had contaminated 25/38 big blocks; after the fix, 0/38 (that document's false L1 headings also dropped from 23 to 4).

### 2.3 Multi-signal heading-tree reconstruction: reset-aware is the crown jewel

This is the mechanism most worth fully understanding in this piece. The problem: MinerU gives `text_level` (a hint at heading level), but it isn't always right; the numbering inside headings (`2.1`, `1.`) looks like a stronger level signal, but **the same '1.' means opposite things in different documents** — in a weekly report, '1. Overseas AI:' is a list item; in an in-depth research report, '1. Edge AI' is a real section. Neither doc_type nor keywords can distinguish the two.

custodian's leveling function ([src/chunker/core.py:158-176](../../src/chunker/core.py#L158-L176)) fuses signals in this priority order:

- **text_level as the primary signal** (given free by the parser, it's the baseline);
- **dotted numbering is unconditionally treated as a finer level**: `2.1` gets its depth from the number of '.' segments (dotted numbering almost never lies);
- **bare integer numbering is only promoted to L1 when there's "whole-document monotonicity evidence"** — this is reset-aware: before chunking, a pre-pass runs across the whole document collecting the sequence of bare numbers ([src/chunker/core.py:263-274](../../src/chunker/core.py#L263-L274)), and the `promote_bare` switch only turns on if `all(strictly increasing)`. **The key insight: lists restart (1..9, 1..), outlines don't.** The moment the sequence shows a restart, all bare numbering falls back to text_level.
- **Four guards** ([src/chunker/core.py:136-155](../../src/chunker/core.py#L136-L155)): a year (19xx/20xx) is never treated as numbering; a bullet glyph (`-•●○▪◦`) at the start is never a heading; a table-of-contents shape (dot leaders + trailing page number) or anything over 120 characters is not a heading; law documents get a special case for `SEC. N` as L1.

Once leveling is done, a monotonic stack builds the tree in a single scan ([src/chunker/core.py:277-290](../../src/chunker/core.py#L277-L290)): every body element gets a `_crumb` (its chain of ancestor headings) and a `_sec_head` (the index of the heading of its deepest containing section), and the sequence of headings is then used to build the Section tree in one pass.

**Empirical backing** (basis: 77 documents / 5,337 headings, real corpus): before the fix, on a real Chinese research report, AP4706, 24 of 25 L1s were cyclic-numbered list items wrongly promoted, the tree was inverted, and 51% of breadcrumbs were only 1 level deep; after reset-aware landed, that went 25→1, shallow breadcrumbs dropped to zero, and 7 differentiated regression documents saw zero regressions (a monotonically-numbered deep report stayed 4→4, the Attention paper stayed 10→10, and NETFLIX's 10-K correctly demoted a numbered footnote to L2).

### 2.4 The eager skeleton: the whole tree built at ingest time

The heading tree, breadcrumbs, and section_anchor are all computed in one pass at ingest time and stamped directly onto every leaf chunk ([src/chunker/core.py:331-350](../../src/chunker/core.py#L331-L350)); at query time it's **read directly, never rebuilt**. This "obvious" design is actually the product of an empirical reversal — details in §3.1 — and its design document is still called [LAZY_HEADING_TREE_DESIGN.md](../methodology/LAZY_HEADING_TREE_DESIGN.md), a name that's now a fossil.

### 2.5 Leaf chunking: token-budget typing + greedy assembly

Consecutive text blocks in the same section are greedily accumulated against a token budget ([src/chunker/core.py:208-240](../../src/chunker/core.py#L208-L240)): if a single block exceeds the max, it's first split by sentence; once the accumulation exceeds the target and it isn't a cross-page continuation (merge_prev), it's flushed; a second pass merges undersized groups forward. The budget `(min, target, max)` is looked up by doc_type ([src/chunker/core.py:22-31](../../src/chunker/core.py#L22-L31)), e.g. law=(300,700,1200), Chinese research reports=(250,550,900).

Two counterintuitive points:

- **The budget numbers themselves are actually quite close across types — the real difference between types is the layout pattern**: slides/policy use (0,9999,9999) — one slide / one whole section is one chunk; slides additionally add page to the grouping key (`PAGE_GROUPED`).
- **The grouping key must be a unique identifier for the heading instance (`_sec_head`, i.e. the heading element's idx), not the breadcrumb text** ([src/chunker/core.py:295-302](../../src/chunker/core.py#L295-L302)). For same-named sibling sections (repeated subsection headings in forms/appendices, found in about 15% of documents), grouping by text would merge body text from different sections into one chunk, wrongly anchored to the first section — **irreversibly writing wrong data into the vector store**. This was one of three high-severity issues caught by the sealing review, and the lesson generalizes: grouping keys should use instance identifiers, not repeatable display text.

Token counting uses a char/divisor heuristic (en/4.0, zh/1.7, [src/chunker/core.py:39-46](../../src/chunker/core.py#L39-L46)), not a real tokenizer — why it's acceptable to be this crude is covered in §3.4.

### 2.6 Asset atomization: separating text from content_raw

Table/image/chart elements each become their own **atomic chunk** ([src/chunker/core.py:372-402](../../src/chunker/core.py#L372-L402)), with a key field separation:

- `text`: the **retrieval text**, for embedding and BM25;
- `content_raw`: the **generation payload** (table HTML / content extracted by the VLM), only fed to the LLM after a hit.

Why separate them? Because the two have completely different optimal content. A table's retrieval text is synthesized by `_table_signal` ([src/chunker/core.py:89-112](../../src/chunker/core.py#L89-L112)): the first 2 header rows (double-layer headers are common in earnings reports) + the first non-empty cell of every other row (the row label) — **the data cells are deliberately excluded**, since raw numbers carry no retrieval semantics and only add noise. An image/chart's retrieval text is cleaned by `_asset_desc` ([src/chunker/core.py:69-81](../../src/chunker/core.py#L69-L81)): for a mermaid diagram, only node/edge labels are kept, dropping the `graph TD`/`-->` scaffolding.

This mechanism was forced into existence by a real, wrong answer (N7/N8): asked for Netflix's total revenue, the system answered with segment revenue instead — the root cause on the retrieval side was that a table chunk's `text` originally contained only a single caption sentence, with the header/row labels locked inside `content_raw` and thus unretrievable, so the table chunk got crowded out of the top-k by MD&A prose on numeric questions. After the fix, the motivating case was answered correctly, and the 72-question regression held correctness steady while faithfulness went from 0.972→1.000 (note the basis: this is the 72-question baseline, not to be mixed with the later 88-question one).

Two edge cases show real craftsmanship:

- **Image-only content isn't discarded**: an image with only an img_path and no text at all produces a placeholder chunk flagged `image_only` ([src/chunker/core.py:393-394](../../src/chunker/core.py#L393-L394)), and downstream this goes through Qwen3-VL image vectorization, skipping the sparse path. Previously, image-only content was simply dropped, losing about 30% of images (38% for academic documents) — and these are exactly the objects that only VL vectorization can recall. img_path is sanitized through `_safe_rel` (rejecting `../`, absolute paths, and URL schemes), because it's the only file-read entry point in the embed chain.
- **Phantom-chunk gating**: a table chunk survives only if `(cap | foot | body)` — at least one of caption/footnote/body — is present ([src/chunker/core.py:379-384](../../src/chunker/core.py#L379-L384)) — a breadcrumb **cannot on its own revive** a placeholder table with no caption and no table body. Without this gate, the whole store's chunk count measured 7,652→7,675, and those +23 phantom chunks shifted every subsequent chunk id in the same document, collectively misaligning the old index's gold set. This number is the direct evidence for §4's "why chunking changes must be deferred."

### 2.7 A separate path for spreadsheets: TableChunker

A grid is not document flow: xlsx has no readable-order paragraphs, so the heading tree has no way to get built, and xlsx/xls therefore follow an independent TableChunker path — but it **outputs the same Chunk schema**, so the downstream embedder doesn't fork at all. Four key mechanisms:

- **Band splitting**: each sheet is split into "bands" at blank rows, but blank rows covered by a vertically merged cell **don't count as a separator** ([src/chunker/table_chunker.py:81-101](../../src/chunker/table_chunker.py#L81-L101)) — otherwise a 5-row merged header would get chopped into header_only fragments with no column names (a real failure the red-team confirmed on the hvs_vacancy table).
- **Real header detection**: within a band, the first row that is "multi-column and label-like" is treated as the header (numeric-cell ratio <0.34, [src/chunker/table_chunker.py:66-68](../../src/chunker/table_chunker.py#L66-L68)); a 4-digit year is counted as a label, not a numeric value ([src/chunker/table_chunker.py:49-53](../../src/chunker/table_chunker.py#L49-L53)), to prevent `['Race', 2010, 2020]` from being misjudged as a data row. Any title/unit row before the header gets its own separate text chunk, and is never dropped.
- **Merged-cell geometry broadcasting**: mergeCell geometry is parsed directly from the xlsx zip (without loading any cell values, preserving read_only streaming, [src/chunker/table_chunker.py:141-178](../../src/chunker/table_chunker.py#L141-L178)), and `_broadcast` propagates the top-left label back across the whole merged region ([src/chunker/table_chunker.py:181-200](../../src/chunker/table_chunker.py#L181-L200)), then joins column-by-column, top to bottom, to reconstruct the full multi-layer column names.
- **Column-conserving splits + chart semantics**: an overly wide table is sliced by columns, with each slice carrying its own key column, so no cell is ever lost ([src/chunker/table_chunker.py:104-112](../../src/chunker/table_chunker.py#L104-L112)); embedded charts have their title + series names + axis names extracted from `xl/charts/chartN.xml` into a chart chunk ([src/chunker/table_chunker.py:231-265](../../src/chunker/table_chunker.py#L231-L265)); legacy .xls goes through xlrd, reusing the same header-reconstruction machinery ([src/chunker/table_chunker.py:213-228](../../src/chunker/table_chunker.py#L213-L228)).

Why go to so much trouble on headers? Because **binding column names to values is this component's core selling point** — a before/after comparison: a US Census table's column name went from 'CV for Retail Sales' (all dimensions lost) to 'CV for Retail Sales 2023Q4 (p) Total E-commerce' (all three dimensions intact). The story behind this, where "the red team overturned its own judgment," is in §3.5.

Measured data: a multiset audit over 5.84 million cells confirmed 100% value coverage (continuation-band deficit went from 21,134→0); INSEE's .xls went from 0 chunks (a silent hard crash) to 6,767; EIA STEO went from 0 to 66 chart chunks; a giant 202k-row / 127-sheet table was handled by streaming in 25s.

`chunk_result()` additionally synthesizes one Element per chunk and one Section per sheet ([src/chunker/table_chunker.py:311-334](../../src/chunker/table_chunker.py#L311-L334)), letting assemble_big stitch back together fragments of the same table that got scattered by row-grouping/column-splitting within a sheet — and it's precisely this field-overwriting done to "align with assemble_big's idx semantics" that planted the seed for §4's chunker#4 (loss of row-level traceability in xlsx).

### 2.8 Query-time small-to-big: three-state assembly

What gets hit is a small chunk (semantically pure, accurate recall); what gets fed to the LLM is a big-block that has "grown up" against a real token budget. `assemble_big` ([src/chunker/retrieve.py:109-171](../../src/chunker/retrieve.py#L109-L171)) reads the hit chunk's `section_anchor` and follows three states:

1. **Hit section > max** → open a window around the hit within the section (`_window_within` grows alternately outward in both directions from the hit seed, [src/chunker/retrieve.py:68-87](../../src/chunker/retrieve.py#L68-L87));
2. **Hit section < target** → climb up along `parent_sec_id`; if the parent section > max, switch to opening a window within the parent's range, naturally pulling in adjacent sibling-section content (with a `seen` set to guard against corrupted sidecar files with circular parent pointers);
3. **No section** → open a window around the hit page (to avoid pulling in the entire document for headingless multi-page documents); if the top level is still < min, fall back to opening a window over the whole document.

On real-world corpora, **climbing is the primary path, not a fallback**: the median section is only 42 tokens, and being over-large and needing a trimmed window happens only 1% of the time (measured on 77 documents); the median big-block is 818 tokens, close to the target of 800. The flag `windowed=True` marks "this is a token-constrained window, not a complete section" ([src/chunker/types.py:107-121](../../src/chunker/types.py#L107-L121)), and the embedder's retrieval layer maps it to `context_status="section_window"` ([src/embedder/retrieve.py:172-182](../../src/embedder/retrieve.py#L172-L182)) — the agent sees this and knows it can call expand on that chunk_id. Whether the context is complete or not is, from here on, no longer implicit — it's an explicit signal the agent can read and act on.

The safety dimension (big-block material-pulling reaches back into the raw elements by idx and can cross chunk boundaries — "same document ≠ same ACL") is handled by `acl_index` equivalence-class gating ([src/chunker/retrieve.py:90-106](../../src/chunker/retrieve.py#L90-L106), [src/chunker/types.py:93-104](../../src/chunker/types.py#L93-L104)) — on real corpora, measured leakage went from 3/79→0/79. The full story of this belongs to the ACL piece; for this piece, just remember: **every element pulled as material at query time is checked against an equivalence class of the same ACL as the hit chunk, and unknown idx's are fail-closed excluded** — and it's exactly this "exclude unknown idx" behavior that planted the seed for §4's chunker#0.

### 2.9 sidecar: the persistent foundation for query-time material-pulling

The vector store only holds chunks; the raw elements/sections/banners/acl_index that `assemble_big` needs are stored one JSON file per document in a sidecar ([src/embedder/embed.py:107-125](../../src/embedder/embed.py#L107-L125)), with the write side stamping a `SIDECAR_VERSION` ([src/embedder/config.py:10](../../src/embedder/config.py#L10)) and the read side doing three-way validation (missing = a transient per-document degradation; version mismatch = a systemic schema drift, loudly failing to signal a full rebuild is needed; elements that aren't densely and consistently ordered = fail-closed rejection, [src/embedder/retrieve.py:71-90](../../src/embedder/retrieve.py#L71-L90)). The point of the version number: turning "an old sidecar silently deserialized into wrong data" into "a loud failure."

---

## 3. Why it's designed this way: rejected alternatives and measured data

### 3.1 lazy → eager: killing your own clever design with 66.6ms (a methodology goldmine)

The v1 design was "ingest keeps only a minimal numbering skeleton + lazily rebuild the hierarchy on a hit" — it sounded clever: most sections are never hit, so pre-building would be wasted work. The adversarial review didn't debate the intuition — it just **measured directly**: eager full-tree construction across 77 documents took 66.6ms total, under 1ms per document, using the exact same algorithm that would otherwise run at query time. The conclusion was instantly clear: lazy loading **doesn't save anything**, and instead moves a zero-cost piece of work into the hot query path and introduces cache-invalidation issues on top. It was flipped to eager on the spot; "lazy" was demoted to an edge case only worth considering when three conditions are simultaneously met — a huge heading count, extremely sparse hits, and high-frequency updates — and this corpus satisfies 0/77 of those ([LAZY_HEADING_TREE_DESIGN.md §6](../methodology/LAZY_HEADING_TREE_DESIGN.md)).

Teaching point: **performance optimization must be measured before it's done.** Keeping "LAZY" in the document's name is deliberate — it's fossil evidence that "the design was overturned by data," and telling this story in an interview is more persuasive than describing any successful design.

### 3.2 "Betting on numbering" was overturned by 13%: where the multi-signal fusion came from

Another cornerstone of v1 was "if there's numbering → zero LLM-based leveling." Measurement: of 77 documents / 5,337 headings, only about ~11% could have their numbering parsed at all, and outside of academic documents (financial/law/policy/slides) it's almost zero — **the absence of numbering is the norm, not the exception.** Also rejected at the same time:

- **font_rank / bbox height-based leveling**: not comparable across documents — a 10-K's L1 and L2 use the same font size;
- **LLM disambiguation in the hot path**: positioned instead as an **offline, optional, doc_type-routed enhancement**, never entering the ingest/query hot path — cost predictability wins.

This converged on the scheme in §2.3: text_level as primary, numbering as correction, reset-aware as the arbiter for bare numbering. This is also the second empirical reversal recorded in [LAZY_HEADING_TREE_DESIGN.md §0](../methodology/LAZY_HEADING_TREE_DESIGN.md).

### 3.3 Why MinerU, and why not an off-the-shelf chunking library

**Parsing layer**: MinerU vs. Tika vs. a custom adapter. An initial measurement of "custom 94% vs. MinerU 87%" almost led to the wrong choice — it later turned out to be a **measurement bug** (the coverage extractor was failing to read list_items); under a fair basis, MinerU comes out ahead at 95.1%. With three-way coverage on a level playing field, MinerU wins because the content_list schema is chunk-ready as-is, its pipeline is unified with the PDF/scanned-document pipeline, and there's no JVM (Tika needs a JVM + a 63MB JAR).

**Chunking layer**: a three-way comparison done on the same evidence-preservation eval (43 documents of ground truth / 356 questions, MMDocIR-annotated) ([CHUNKING_EVALUATION.md §8](../methodology/CHUNKING_EVALUATION.md)):

| Strategy | single% (ALL basis) ↑ | missing% ↓ |
|---|--:|--:|
| **ours_exact** (with source_indices exact traceability) | **70.4** | **7.7** |
| chonkie (RecursiveChunker) | 62.1 | 21.9 |
| ours_fair (stripped of traceability, boundaries only) | 58.2 | 26.0 |
| docling (HybridChunker, consuming MinerU's output) | 47.6 | 38.6 |

The honest conclusion has three layers: ① when Docling consumes MinerU's output, its missing rate is 3-5x the custom-built one's — an immediate disqualification; ② **stripped of engineering features, comparing boundaries alone, the custom-built chunker is slightly behind chonkie** (TEXT-channel basis: 71.6% vs. 74.9%) — the boundary-cutting algorithm is not the moat; ③ the custom-built chunker's net win comes from the **engineering integration** — source_indices traceability + breadcrumbs + per-chunk ACL + small-to-big — none of which chonkie/docling have. "If all you need one day is plain-text chunking, without this set of engineering features, chonkie is the more convenient choice" — this sentence is written right into the engineering documentation, and saying it out loud in an interview, unedited, is more impressive than claiming an outright win.

A methodology story on the side: the first version of this eval came back looking terrible (72 questions unlocalized), nearly leading to the conclusion "chunking is bad" — the root cause was a coordinate-system mismatch between content_list (rendered-image coordinates) and the ground truth (PDF-point coordinates). After deriving a per-document scale factor from text pairing, unlocalized went 72→4, and under a strict threshold the metrics barely moved, proving the earlier threshold sensitivity was just a symptom of the coordinate bug. **Suspect the measurement first, then suspect the thing being measured.**

### 3.4 est_tokens: verified, then deliberately left unchanged

Is the char/divisor heuristic accurate enough? Real token counts were re-tagged on 2,927 real chunks using the actual Qwen3-VL tokenizer: for English prose char/token ≈ 3.85 vs. the current value of 4.0, an error of <4%; the deviation is concentrated in number-dense documents (earnings reports 5.08 / government documents 5.35) and Chinese research reports (1.51). A single value can't serve both clusters well, and switching to a mean would hurt the prose majority; and both directions of error already have a fallback — overestimate→smaller chunks→small-to-big compensates; underestimate→bigger chunks but still nowhere near the 32k ceiling. **Conclusion: keep the current value** (the comment at [src/chunker/core.py:39-46](../../src/chunker/core.py#L39-L46) is the record of this validation). "Decided not to change after verifying" and "never verified, so never changed" are two different things, and it's worth making that distinction explicit in an interview.

### 3.5 Falling into the same pit three times: self-referential metrics can't be trusted

Validation at the parsing/chunking layer is extremely prone to the trap of "using your own output to validate itself." This project fell into the same category of pit three times, worth its own section:

1. **The orphan=0 false metric** (docx/pptx extension): the first run of the custom-built adapter reported orphan=0 ("zero content loss"), but the orphan check only walked the Element[] the adapter had **already produced** — it was completely blind to "content lost before it was even extracted." The truth: docx text boxes (`w:txbxContent`) were never walked at all (10/50 files lost 390 paragraphs), and pptx GROUP shapes weren't recursed into (16/50 files lost 352 lines — one Harvard lecture deck even lost the instructor's email address).
2. **xlsx's 100% coverage masking header contamination**: 100% cell-value coverage is real (confirmed by a multiset audit), but taking "header = the first row of a region" blindly meant about 33% of regions' headers were actually title/unit rows rather than real column names — **the binding between column names and values was broken, and coverage as a metric simply can't measure it**. The red team ran 5 real pain points and overturned the judgment that "the header problem is low-frequency, not worth fixing"; the lesson went into the memory base: severity ≠ occurrence rate — a low-frequency defect that breaks a core selling point must be fixed.
3. **The coverage extractor missing list_items**: this undercounted MinerU's coverage by 8 percentage points, and the false comparison "custom 94% vs. MinerU 87%" almost led to the wrong parser choice; after correction, MinerU came out ahead at 95.1%.

The common fix for all three: **independent ground truth + adversarial measurement**. A dedicated metric was built specifically for the office path (word-set containment between the adapter-extracted text and the full raw OOXML XML text) — this measures **fidelity** between the document and the chunker, not **consistency** between the adapter and the chunker; after the fix, docx word coverage reached 95.6%, pptx 99.1%. The methodological takeaway is worth more than the numbers: **whenever a metric's denominator comes from the output of the very system being tested, it can only falsify, never confirm.**

---

## 4. Real-world retrospective: confirmed, but not a single one could be fixed right away

Before writing this set of learning docs, the adversarial review (2026-07) dug up **5 confirmed defects in the chunker (chunker#0-4), all deferred** — while in the same review round, generator/service/embedder/eval had 17 items landed directly (global basis: 35 suspected → 34 confirmed + 1 refuted, and after dedup, 17 fixed + 15 deferred, with this piece's 5 items being part of those 15). Why is the chunking layer special?

**Because the chunking layer's output is the index's schema.** chunk_id is numbered by an in-document sequence number (`{doc_id}#{n:04d}`), and any change that alters the number or boundaries of chunks shifts every subsequent chunk id in the same document — misaligning the old vector store, the old sidecar, and the eval's gold annotations all at once. This isn't hypothetical: in §2.6's phantom table-chunk incident, +23 chunks alone misaligned the whole store's gold set, and the gate was added on the spot to fix it. So the discipline is: **any fix that changes chunking output must be bundled with bump SIDECAR_VERSION + rebuild the index + re-run the GPU eval, landing as a single atomic action**; if the whole set couldn't be completed within the writing window, then none of it gets touched. "Confirmed but can't be fixed right now" is itself an engineering judgment, not procrastination.

The five deferred items (each independently adversarially verified as confirmed, with a minimal reproduction attached):

| # | Defect | Severity | One-line root cause |
|---|---|---|---|
| chunker#0 | On the ACL-aware path, big-blocks systematically lose all heading text | medium | Heading elements don't enter any chunk's source_indices → not in acl_index → excluded by fail-closed |
| chunker#1 | Sentence-splitting fails on unspaced CJK text; oversized Chinese paragraphs blow through the max budget | medium | The splitting regex requires whitespace after sentence-ending punctuation, and Chinese full stops have no trailing space |
| chunker#2 | merge_prev backfilling matches on a 12-character prefix, causing cross-talk between same-page, same-prefix chunks | low | Fuzzy matching has no ordered consumption and no uniqueness requirement |
| chunker#3 | A single bare-numbered list item is hard-promoted into a false L1 section | low | `all()` on an empty/single-element sequence is vacuously true, so promotion happens even with no monotonicity evidence |
| chunker#4 | xlsx's chunk_result overwrites source_indices, losing row-level traceability | low | The synthesized element's idx semantics reuses the same field as row-number traceability |

Two of the medium-severity items deserve expansion — each is its own class of teaching case:

**chunker#0 is a textbook case of "a security mechanism pointed in the right direction producing a quality side effect."** Heading elements are `continue`d past when the tree is built ([src/chunker/core.py:277-287](../../src/chunker/core.py#L277-L287)) — they only enter the Section tree, and never enter any chunk's `source_indices`, so they're not in `acl_index` ([src/chunker/types.py:93-104](../../src/chunker/types.py#L93-L104)); and `assemble_big`'s default gate fail-closed excludes unknown idx's ([src/chunker/retrieve.py:102-105](../../src/chunker/retrieve.py#L102-L105)) — so the production default path (the embedder always passes an acl_index) produces a big.text that **contains no heading lines at all**. Measured on the same document: in legacy mode, big.text reads `Compensation\nPhilosophy\nbody text…\nBenchmarks\nbody text`; in ACL mode, only two blocks of body text remain, joined directly with nothing between them. This hurts especially badly when climbing to a parent section: sibling subsections' body text gets joined together with no heading to separate them, and the LLM loses the section-boundary signal. Interestingly, `get_document` had already been patched for the very same root cause ([src/embedder/retrieve.py:212-225](../../src/embedder/retrieve.py#L212-L225), additionally including the heading for any subsection visible in its own-section context), but the assemble_big/expand path was never synced up — **the same root cause, multiple exit points; fixing one exit point doesn't mean the whole thing is fixed.** The direction is fail-closed (headings are leaking out, not leaking in), so this is a quality defect, not a security vulnerability; the fix (adding a Section's start_idx into acl_index based on the ACL of the body inside its section) would change the sidecar's semantics, requiring a bump of SIDECAR_VERSION and a full rebuild.

**chunker#1 is an ironic case of "not even tested on its home-field corpus."** The sentence-splitting regex at [src/chunker/core.py:197](../../src/chunker/core.py#L197), `(?<=[。!?.!?])\s+`, requires whitespace after punctuation — true for English, but Chinese body text has **no space** after a full stop, so re.split simply returns the whole paragraph untouched. Measured: a 3,200-character Chinese paragraph (est. 1,882 tokens) under a max=900 budget produced a single 1,882-token chunk; a control group with manually inserted spaces split normally into 3 pieces. The budget contract fails, and embedding's semantics get diluted — and Chinese research reports happen to be the largest single document type in the corpus. On the query side, big-block assembly has a `_cap` character-count hard cutoff as a fallback ([src/chunker/retrieve.py:48-65](../../src/chunker/retrieve.py#L48-L65)), but the chunk itself has no such fallback. A fix sketch already exists (zero-width splitting on full-width punctuation + a hard character-limit fallback for the overflow case), but it directly changes chunk boundaries — a textbook "bundle with a rebuild + re-run eval" deferred item.

chunker#3 also has a design-philosophy angle worth noting: reset-aware's original design intent was "**promote only with monotonicity evidence**," but `all()` being vacuously true on a single-element sequence means "promotion even with zero evidence from a single sample" — the code contradicts its own design comment. Defects often hide in "the gap between design intent and implementation boundary."

Worth mentioning in passing: one item from the same review round, related to this piece's foundation and **already landed**, is the embedder's `index_document` reordering (encoding/sidecar preparation moved earlier, delete→upsert→replace at the end, shrinking the out-of-store window from minutes to milliseconds, [src/embedder/embed.py:56-105](../../src/embedder/embed.py#L56-L105)) — it doesn't change chunking output, so it could be fixed immediately. Side by side, the criterion for "what can be fixed right away" is clear at a glance.

---

## 5. How to pitch this in an interview

**30-second version (elevator pitch)**:

> My chunking layer is structure-aware + small-to-big: MinerU uniformly parses everything into a normalized stream of elements; at ingest time, multiple signals (mainly the parser's level hints, corrected by numbering, with whole-document monotonicity arbitrating bare numbering) rebuild the heading tree, and breadcrumbs and section anchors get stamped directly onto every chunk; tables and charts are atomized, with retrieval text kept separate from the generation payload; at query time, after hitting a small chunk, the system climbs up or opens a window based on real token counts to assemble a large context. Evaluation doesn't look at how pretty the cuts are — it looks at "evidence preservation": using MMDocIR annotations to measure whether the evidence is still sitting inside one chunk, on 43 documents / 356 questions, single-chunk preservation hits 70%. I've compared against chonkie and docling: I slightly lose on the boundary algorithm against chonkie, but the engineering integration — traceability, breadcrumbs, per-chunk ACL — is a net win, and that's a conclusion I'm comfortable putting in writing.

**3-minute version (structured expansion)**:

1. **Problem definition** (20s): chunking has to reconcile the tension of "embedding wants small and pure, the LLM wants large and complete"; my evaluation criterion is evidence preservation, not chunking aesthetics.
2. **Architecture** (40s): MinerU unified parsing → the Element seam (swapping parsers only swaps the adapter) → three noise gates (kind blacklist / page-proportion banner guard / content reclamation) → the heading tree → doc_type-typed budgets → asset atomization → query-time three-state assembly. Emphasize eager: the tree is built at ingest time, query time only reads it.
3. **One deep-dive point — reset-aware** (60s): the same '1.' is a list item in a weekly report and a section in an in-depth report; neither doc_type nor keywords can tell them apart — the only reliable signal is the monotonicity of the whole document's sequence of bare numbers: lists restart, outlines don't. A real research report went from 25 false L1s to 1, with zero regressions across 7 differentiated documents. Before that, tell the lesson of fixtures being all green while the tree was actually inverted on real corpora: **fixtures are the happy path — adversarial review has to run on real documents.**
4. **Data backing** (30s): a leveling corpus of 77 documents / 5,337 headings; evidence preservation of 70.4% single-chunk on 43 documents / 356 questions (ALL basis); the full eager tree build takes 66.6ms across 77 documents — segue into the lazy→eager reversal: "measure performance designs before you build them."
5. **Honest closing** (30s): the boundary algorithm alone is slightly behind chonkie (71.6 vs. 74.9, TEXT basis); known defects — the CJK sentence-splitting bug and the ACL path losing headings — have both been confirmed with fix sketches ready, and are deferred because a chunking change is equivalent to an index-schema change, requiring rebuilding the index and re-running eval as one bundled action.

---

## 6. Rehearsing follow-up questions

**Q1: Why not just use LangChain/chonkie's chunking directly?**
Point to make: it was tested, not guessed at. The same evidence-preservation eval did a three-way comparison; honestly admit chonkie wins slightly on pure boundary quality (74.9 vs. 71.6, TEXT basis); but chonkie has no source_indices traceability, no breadcrumbs, no per-chunk ACL, no small-to-big section anchors — these are hard dependencies for what's downstream (citation/permissions/context assembly). Keyword: **the moat is engineering integration, not the boundary algorithm.**

**Q2: Why not use semantic chunking (splitting on embedding distance)?**
Point to make: semantic chunking solves "where to draw the boundary" — but measurement shows my weak point isn't boundaries (single% is only about 3pts off from chonkie), it's the retention of structural information and downstream integration; semantic chunking is expensive (an embedding call per sentence at ingest time), unstable (threshold-sensitive), and doesn't produce any hierarchy. A good closing rebuttal: the money in chunking should be spent on "evidence preservation + structural attachment," not on optimizing boundary quality from 72 to 75.

**Q3: Why not use an LLM to judge heading levels?**
Point to make: lead with data — only ~11% of headings have numbering, so absence is the norm, meaning disambiguation is genuinely needed; but putting an LLM in the ingest hot path means bringing cost and uncertainty into the hot path. The solution is layered: text_level as the free baseline signal, numbering as deterministic correction, reset-aware as zero-cost arbitration; an LLM is positioned as an **offline, optional, doc_type-routed enhancement**. Keywords: determinism, sub-millisecond, zero LLM in the hot path.

**Q4: What does reset-aware wrongly kill?**
Point to make: proactively volunteer two known boundaries — ① it's an all-or-nothing switch: a mixed document (real numbered sections coexisting with a restarting list) has only one promote_bare switch for the whole document, and the granularity is document-level, not local; ② the confirmed chunker#3: a bare number with a single-element sequence (len==1) is vacuously promoted, which contradicts the intent of "requiring monotonicity evidence" — the fix is a `len>=2` threshold, deferred because changing heading leveling changes chunking output. Being able to proactively state your own mechanism's failure modes demonstrates more skill than the mechanism itself.

**Q5: Why is small-to-big assembled at query time instead of storing the parent block directly at index time?**
Point to make: ① storing two copies (child+parent) is double the storage and consistency overhead, and the parent's boundaries also depend on budget parameters, so tuning them would require a rebuild; ② query-time assembly can open a window around the **actual hit position** (growing outward from the hit seed), which a pre-stored parent can't do; ③ measured, the assembly's raw material (sections/elements) is read from the sidecar once, the tree is already built eagerly, and the assembly itself is pure CPU work at microsecond scale. Supporting data: the median section is 42 tokens, and climbing is the primary path, meaning parent granularity isn't statically precomputable — it often has to climb multiple levels.

**Q6: How was the chunk size (target 800) decided? What if the token estimate is inaccurate?**
Point to make: budgets are looked up by doc_type, but the real difference between types is the layout pattern (slides/policy don't split whole sections); token counting uses a char heuristic, validated against the real tokenizer on 2,927 chunks: prose error <4%, number-dense documents have a larger bias but both directions of error have a fallback (overestimate→small-to-big compensates; underestimate→still nowhere near the 32k ceiling). Keyword: **verified, then decided not to change.** Also proactively admit that CJK sentence-splitting failure is a real defect (not an estimation issue — the splitter itself fails), already confirmed and pending a fix.

**Q7: The index is already built in production, and the chunking algorithm needs an upgrade — what do you do?**
Point to make: this is exactly why five confirmed defects were deferred. chunk_id is numbered by an in-document sequence; a change in chunking output = a shift in chunk id = every old index/gold/citation misaligned (cite the measured +23 phantom table chunks); so the process is: fix + bump SIDECAR_VERSION (the read side loudly fails, forcing a rebuild) + fully rebuild the index + re-run eval to confirm no regression, all four steps landing atomically. Can extend: the sidecar version check turns "silently wrong" into "loudly failed," and treats a missing file (transient, per-document) differently from a version drift (systemic).

**Q8: Why does a table get its own chunk? Why don't the numbers go into the retrieval text?**
Point to make: tell the N7 story (segment revenue mistaken for total revenue) — the table's retrieval signal originally had only a caption, with header/row labels locked inside content_raw and unretrievable; the fix is `_table_signal` extracting headers + row labels, deliberately excluding data cells (numbers have no retrieval semantics, they just add noise); then mention the phantom-chunk gate (a breadcrumb alone can't revive a placeholder table). Data: the 72-question regression held correctness steady, faithfulness went 0.972→1.000 (72-question basis).

---

## 7. Hands-on experiments

**Lab 1: A comparison of reset-aware leveling — the same batch of numbers, two different fates depending on restart vs. monotonic** (CPU, seconds-scale)

```bash
cd <custodian repo root>
python -m pytest -q tests/engine/test_core.py -k "reset or monotonic" -v
```

Two tests correspond to two different fates: `test_reset_numbering_not_over_promoted` (a bare-number sequence 1,2,1,2 restarts → '1. Overseas AI:' stays at list-item level) and `test_monotonic_numbering_still_promotes` (1,2,3 monotonic → promoted to L1). Then try it hands-on: in the restart test's fixture, change the second page's '1. Film & TV:' and '2. Gaming:' **both together** into '3. Film & TV:' and '4. Gaming:', so the whole-document bare-number sequence becomes strictly increasing (1,2,3,4), rerun, and watch that same batch of headings all flip to L1 — a direct way to feel how the single signal of "whole-document monotonicity" decides the shape of the entire tree. (Conversely, if you change only 'Film & TV' and not 'Gaming,' the sequence is 1,2,3,2, still non-monotonic, the `all()` guard won't open `promote_bare`, and the headings won't be promoted — this exactly reproduces the mechanism's intent of "any one restart in the sequence rolls everything back.") Remember to revert your changes afterward.

**Lab 2: The phantom table-chunk gate — reproduce chunk-id shifting by hand** (CPU, seconds-scale)

```bash
cd <custodian repo root>
python -m pytest -q tests/engine/test_core.py -k "placeholder_table or table_retrieval_signal" -v
```

First watch two tests pass green: a placeholder table (no caption/footnote/body) is dropped, and a normal table's retrieval signal contains its headers + row labels. Then temporarily remove the `and (cap or foot or el.table_body)` gate at [src/chunker/core.py:379](../../src/chunker/core.py#L379) and rerun — `test_placeholder_table_still_dropped` turns red: a breadcrumb alone inflates the retrieval text to non-empty, the phantom chunk comes back to life, and every subsequent chunk's id shifts. This is the minimal model for the bug where "+23 phantom chunks misaligned the whole store's gold set," and the shortest path to understanding §4's deferral discipline. Revert afterward.

**(Optional, GPU/WSL) est_tokens recalibration**: in the WSL `custodian` environment, load the Qwen3-VL tokenizer (the model at `~/models/Qwen3-VL-Embedding-8B`), compute real token counts for chunk text across several documents in the sidecar directory (`~/rag_sidecar`), scatter-plot against `chunk.n_tokens`, and group by doc_type to see the char/token ratio — reproducing the "English prose ≈3.85-4.0, earnings reports 5.08, Chinese research reports 1.51; a single value can't serve both clusters well but both directions of error have a fallback" decision to verify-then-not-change.

---

## 8. Honest boundaries

Proactively admitting these in an interview is far stronger than having them dug out of you:

- **The boundary algorithm itself is not a strong point**: stripped of engineering features, comparing boundaries alone, the custom-built chunker is at 71.6% vs. chonkie's 74.9% (TEXT-channel basis). The moat is in the engineering integration — this positioning was forced out by the three-way comparison, not false modesty.
- **Five confirmed defects still on the books, unfixed** (§4): CJK sentence-splitting failure (oversized Chinese paragraphs blow past the budget), the ACL path losing headings in big-blocks, merge_prev prefix cross-talk, a single bare number producing a false L1, xlsx losing row-level traceability. All have a minimal reproduction and a fix sketch, and the deferral follows the discipline "a change to chunking output must be bundled with a rebuild + eval" — it's not that they were never noticed.
- **The banner guard's colon rule is a heuristic**: it protects speaker labels like 'Prompt:', but it will let a real banner like 'CONFIDENTIAL:' through; bbox positional stability is a stronger signal and has already been logged as deferred.
- **The evidence-preservation eval's coverage is uneven**: only 43/77 documents (the MMDocIR subset) are annotated; **the largest single document type in the corpus, financial_research_zh, along with policy/form/tech_report, has never been evaluated for evidence preservation at all** — their chunking strategy has only been statistically profiled, never retrieval-verified. The law type's 100% comes from a single document and doesn't generalize.
- **est_tokens has a large bias on number-dense documents** (5.08 for earnings reports vs. an assumed 4.0): validated, with a fallback, and deliberately left unchanged — but it should be stated plainly that it's a heuristic.
- **promote_bare is a document-level switch**: a mixed document (real numbered sections coexisting with a restarting list) can't be arbitrated locally; the brittleness of one switch per whole document is a stated trade-off.
- One usable line: "What I'm most confident about in this layer isn't any single algorithm — it's that **every number can be traced to its basis, and every known defect has a minimal reproduction** — including the ones that don't reflect well on me."

---

*Every anchor in this piece was verified against the actual code as of 2026-07-07 (after the adversarial review's fixes had landed); the evaluation numbers' bases: 77 documents / 5,337 headings (leveling corpus), 43 documents / 356 questions (evidence preservation), 72 questions (the N8 regression baseline, do not mix with the 88-question one).*
