# Design document — a structure-aware chunking system

> *Archived document: migrated as-is from the engine-prototype repo, kept unchanged. Where the text refers
> to `../analysis/CHUNKING_STRATEGY.md` and `EVALUATION.md`, it refers to the old repo's layout (not carried over during migration, so demoted to plain text to avoid pointing at a 404). The current version lives at
> [docs/components/chunker/](../components/chunker/) and [CHUNKING_EVALUATION.md](../methodology/CHUNKING_EVALUATION.md).*

> **Note (this project):** §7's "prior-language /1.7" token-estimation figure describes the original prototype's support
> for the project's original non-English language, before this project replaced it with Telugu (see the top-level README). Kept as historical record, not re-measured
> for Telugu.

> Purpose: turn the PDF → retrievable-chunk process into an engineering system that is **reusable, auditable, and tunable**.
> This document covers "why it's designed this way"; implementation details are in [IMPLEMENTATION.md](engine-prototype-IMPLEMENTATION.md); process and rationale are in [PROCESS_LOG.md](PROCESS_LOG.md); strategy rationale is in `../analysis/CHUNKING_STRATEGY.md`.

---

## 1. Goals and non-goals

**Goals**
- Make full use of the structure the parser (MinerU VLM) has already delivered, splitting **deterministically** (boundaries do not depend on an LLM's decision).
- **Assemble upward** the parser's output "fragments" into self-contained, context-carrying chunks.
- Every chunk is **traceable** (can be mapped back to its original elements), for auditing and regression.
- Handle documents **differently by type**, rather than one-size-fits-all.

**Non-goals**
- Not doing parsing (that's MinerU's job). Not doing embedding/retrieval (chunks are the input; downstream is a separate discussion).
- Not chasing an "optimal" chunk size — size is a tunable knob, dependent on the retrieval task.

---

## 2. Overall architecture and data flow

```
PDF corpus ──select_sample.py──> corpus/<type>/        (stratified sampling + categorization)
        ──parse_batch.py────> parsed/<doc_id>/       (MinerU VLM, load-balanced across 3 accounts)
                               ├─ *_content_list.json   semantic units (table HTML/captions already assembled)
                               └─ layout.json           pdf_info[]: noise/merge_prev/score/index
        ──analyze_chunks.py─> analysis/               (statistics → strategy rationale)
        ──chunk_document.py─> chunks/<doc_id>.jsonl   (leaf chunks)
                               chunks/<doc_id>.parents.jsonl  (section aggregation, small-to-big)
```

**Division of labor between the two input files** (a key design premise, derived from real testing):
- `content_list.json` = **already-assembled semantic units**: a table's `table_body` (HTML) + `table_caption` + `table_footnote` are bundled into a single object; images/charts work the same way. Array order ≈ reading order.
- `layout.json.pdf_info[page]` = **relational signals**: `discarded_blocks` (noise), `para_blocks[].merge_prev` (cross-page continuation), `lines[].spans[].score` (OCR confidence), `index` (reading order).

> The online VLM API **does not produce the local version's `block_list.json`** (no `is_discarded`/`mergeConnections`). Equivalent information lives in `layout.json`. See PROCESS_LOG §Parsing for details.

---

## 3. Chunk data model (schema)

leaf chunk (`chunks/<doc_id>.jsonl`, one per line):

| field | meaning | design rationale |
|---|---|---|
| `chunk_id` | `<doc_id>#0007` | stable and unique, easy to reference/dedup |
| `doc_id` / `doc_type` / `language` | source and type | retrieval filtering + type-specific strategy |
| `kind` | `text\|table\|chart\|image` | downstream handles assets and text differently |
| `text` | **the text fed to the retrieval vector** | for text = assembled body; for assets = caption+footnote (+ an optional summary) |
| `content_raw` | asset generation payload (table HTML / VLM content) | **separates retrieval from generation**: embed the summary, feed the LLM the original |
| `breadcrumb` / `section_path` | heading breadcrumb | provides context + citability, at zero LLM cost |
| `parent_id` | the section parent it belongs to | small-to-big: a leaf hit fetches the parent |
| `page_start`/`page_end`/`bbox` | location | citation, re-ranking, manual verification |
| `n_tokens` | estimated token count | budget control and monitoring |
| `trust` | `high\|low` | content reconstructed by VLM/from low-confidence OCR is marked low |
| `flags` | `captionless`/`vlm_content`/`cross_page_merged` | surfaces risk for downstream decisions |
| `source_indices` | list of original content_list indices | **auditability**: every chunk can be traced back to its original elements |

parent chunk (`*.parents.jsonl`): `parent_id, section_path, child_ids[], text (aggregated), pages[], n_tokens`.

---

## 4. The seven-step pipeline (design and rationale for each step)

| Step | What it does | Signal source | Rationale (data) |
|---|---|---|---|
| 1 noise filtering | drop `discarded_blocks` + type∈{header,footer,page_number} | layout + content_list | noise rate 0→0.37, heaviest in government/financial_zh |
| 2 cross-page stitching | a `merge_prev=true` block gets stitched back onto the previous one (**text-prefix matching** to align to content_list; bbox-IoU fails entirely due to differing coordinate systems, see §7) | layout | academic 4.2 / form 13.8 / research 7.8 occurrences per document |
| 3 domain-specific tree building | text_level as primary signal, numbering determines depth, law uses a `SEC.` regex; TOC entries and overlong "pseudo-headings" are removed | content_list | text_level is unreliable in both directions; law's num% is 0.02 |
| 4 asset special-casing | tables/images/charts each become an atomic chunk; caption→retrieval, body→generation; tagged captionless/vlm_content | content_list | table HTML is reliable at 0.86-1.0; image caption swings 0.04-0.88 |
| 5 text assembly | greedily accumulate blocks within the same section by token budget (**tunable strategy**, §5) | — | 11/14 categories have a median <75 tokens, must be merged upward |
| 6 metadata attachment | breadcrumb/page/bbox/token/trust/flags/source_indices | all | most of retrieval quality comes from the metadata |
| 7 parent-child | leaf→section parent, aggregating the parent's text | — | the highest-ROI upgrade for document types that have a hierarchy |

---

## 5. The one core tunable strategy: `assemble_text()` (Step 5)

This is the **only place in the whole system with real tradeoffs**, deliberately isolated into a pure function with signature `assemble_text(blocks, lang, cfg)`, `cfg=(min, target, max)`.

**Current strategy (default implementation)**:
- Greedily accumulates consecutive text blocks within the same section up to `target`;
- Blocks with `merge_prev=true` are **unconditionally** merged into the current group (cross-page continuation takes priority over the budget);
- A single block exceeding `max` is split at a sentence boundary;
- A trailing group that falls short of `min` is **merged backward into the previous group in the same section** (never merged across sections).

**The tradeoff (why this choice, and its cost)**:
- A smaller `target` → precise retrieval but easily loses context (mitigated by parent-child).
- A larger `target` → more context but dilutes the embedding and eats into the generation window.
- Not merging across sections → respects structure and stays interpretable; the cost is that **same-name/single-paragraph sections leave behind small chunks** (financial_zh measured a median of 124 tokens) — this is intentional, compensated for at the parent level.

> **This is the knob left for you to tune**: change the three-value tuple in `CONFIG[doc_type]`, or change `assemble_text`'s merging logic (e.g. allowing merges across adjacent sections, or switching to semantic-similarity-based splitting), and the retrieval granularity changes while the other six steps are unaffected.

**Per-type budget `CONFIG` (min/target/max, in tokens)**: see the top of `scripts/chunk_document.py`; slides/policy are set to "never split a section" (one page/one section = one chunk), everything else follows the playbook in `CHUNKING_STRATEGY.md` §4.

---

## 6. Key design decisions (decision record)

1. **content_list as the spine, layout as a supplementary signal**, rather than picking one or the other. Rationale: content_list already has tables/captions assembled (rebuilding them yourself is error-prone), while layout has unique merge/discarded/score information. Cost: requires bbox-IoU alignment between the two (already implemented, threshold 0.5).
2. **Splitting is deterministic, no LLM call**. Rationale: after structural cleanup the boundaries are already clear; LLM-based splitting is expensive, not reproducible, and redundant with structure that already exists. The LLM is reserved only for "asset summarization/caption filling" (an interface is reserved, off by default).
3. **Asset atomization + separating retrieval from generation** (multi-vector). Rationale: table HTML embeds poorly, but generation needs the original text.
4. **Domain-specific hierarchy rules**. Rationale: testing showed each domain has different numbering conventions (academic `1.2` / law `SEC.`+`(a)(1)` / financial reports unnumbered).
5. **source_indices preserved throughout**. Rationale: auditability is a hard requirement — any chunk must be traceable back to its original elements for regression comparison.
6. **Load-balancing across 3 accounts greedily by "page count"**, not by file count. Rationale: the quota is pages/day, so page count is the real cost.

---

## 7. Known limitations (recorded honestly, for review and improvement)

- **Parents are grouped by breadcrumb text**: same-name sections (both the body and an appendix titled "Weekly Market Review") get merged into the same parent, giving the parent's `pages` a discontinuous range. Improvement: fold the first occurrence's page or section sequence number into parent_id.
- **merge_prev alignment**: the original bbox-IoU approach had **zero hits throughout**, because content_list (rendered coordinates) and layout (PDF points) use different coordinate systems (adversarial review F1); switched to **text-prefix matching** (after the fix, 112 chunks across 17 documents hit). Blocks with no text at all may still be missed.
- **Provenance degrades on long-block splitting**: when a single block over `max` is split into multiple pieces by `_sentence_split`, each piece's `source_indices` reuses the same original block index, so it can only be traced back to the original block, not distinguish between pieces (review finding F5). Character-range annotations would be needed for piece-level provenance.
- **Deep legal clauses** are only kept together via "blocks merged upward," without explicitly building an `(a)(1)(A)` tree; precise retrieval by clause would require a second-level parse of inline enumeration markers.
- **Tokens are estimated by character count** (English /4, prior non-English language /1.7), not a real tokenizer. Adequate for monitoring; billing scenarios would need a real tokenizer.
- **VLM-derived chart/image content values are not trustworthy** (already flagged `vlm_content`), but are not automatically verified; downstream should not index them as fact.
- **The news category is multiple concatenated documents**: currently handled as an ordinary document (1399 chunks); ideally it should be split at document boundaries first.

---

## 8. Validation (against ground truth)

Not relying solely on self-sampled checks — used MMDocIR's annotated questions (43 overlapping documents / 356 questions) to quantify "evidence preservation," detailed in `EVALUATION.md`.
**After the coordinate fix**: evidence kept in a **single chunk 70.1%** (identical under a stricter threshold → not threshold-dependent), split 22.5%, missing 7.4%, **asset channel-kind match 92.4% (any) / 83.2% (true-asset, strict)**; answer-substring soft recall 16.3% (reference only, short words/paraphrases give a high false-negative rate).
This lines up self-consistently with strategy difficulty (academic/financial_report/government do relatively well).
This evaluation is **the objective yardstick for tuning knobs**: after changing `CONFIG`/`assemble_text`, watch `single%↑ / split%↓ / missing%≈0`.
**The correct reading of split**: research/brochure/slides have high split because "image + discussion text" is deliberately split into separate chunks by asset atomization — this is expected, and parent-child retrieval re-gathers them; it measures "does the leaf level need to fetch the parent," not "evidence lost."
**Coverage limitation**: ground truth covers only 43/77 documents; financial_research_zh/policy/form/tech_report were never validated this way; law's single=100% comes from a single document only.

## 9. Extension points

- New document type: add a budget to `CONFIG` plus a domain regex in `heading_level()` (see IMPLEMENTATION §Extending).
- Enabling LLM enhancement: wire into `_mk_asset_chunk` for "one-sentence table summary / caption for uncaptioned images," writing into `text` and tagging `trust=low`.
- Wiring up a real tokenizer: replace `est_tokens()`.
