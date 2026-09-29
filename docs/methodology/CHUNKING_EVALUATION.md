# Evaluation document — checking chunking quality against ground truth

> No self-grading: use the annotated questions that already ship with the dataset to check whether "chunking keeps a question's evidence inside a retrievable unit."
> Script: `scripts/eval_chunks.py`; results: `analysis/eval_report.json` + `analysis/eval_by_doctype.csv`.
> **The numbers in this version are after the coordinate-bug fix** (the earlier version was contaminated by coordinate mismatches — see §2 and [PROCESS_LOG stage 9](../archive/PROCESS_LOG.md)).

---

## 1. Why evaluate this way

What chunking actually controls is: **after splitting, is a question's evidence still sitting together in a coherent chunk** (not chopped up, not dropped as noise). So we don't evaluate embedding/retrieval (that's downstream) — we only evaluate **evidence preservation**, which is the part chunking is responsible for and which the annotations can directly check.

**Ground truth**: `mmdocir/MMDocIR_annotations.jsonl`. Each question carries an `answer`, `page_id`, `type` (channel: Figure/Table/Chart/Pure-text), and `layout_mapping[]` (the evidence's page + bbox + page_size).
Overlap with our sample = **43 documents / 356 questions** (only mmdocir has annotations; financial_research_zh/policy/form/tech_report etc. are **not covered by ground truth** — see §6).

---

## 2. Method (coordinate handling is the key to correctness)

**Bridge = `source_indices`**: evidence bbox → matched to a content_list element index → the chunk containing that index. This is exactly the field the chunker preserves for auditability.

**Coordinate systems (after the fix)**: measurement showed that `content_list`'s bboxes are in **rendered-image coordinates**, while `layout.json`/the ground truth are in **PDF point coordinates**, related by `content_list = layout × (sx, sy)` (same origin, a constant within a given document, but sx≠sy and differs across documents).
- Old bug: comparing GT (points) directly against content_list (rendered pixels), or normalizing content_list bboxes using layout's page_size → systematic mismatch, misjudging many questions that were actually locatable as unlocalized.
- Fix: for each document, derive the median `(sx, sy)` via **text pairing** (matching content_list elements to layout para_blocks with identical text), then scale the GT bbox into content_list space before computing coverage. **Effect: unlocalized dropped from 72 to 4.**

**Evidence-element criterion (bidirectional coverage)**: an element counts as evidence when `intersection/evidence_area ≥ 0.3` **or** `intersection/element_area ≥ 0.5`.
**`type` is a string**, `"['Figure']"`, requiring `ast.literal_eval` to parse.

---

## 3. Verdicts and metrics

Verdicts separate "measurement couldn't determine this" from "chunking didn't do well":

| verdict | meaning | counted in the quality denominator? |
|---|---|---|
| `out_of_range` | evidence page beyond `PAGE_CAP=50` | no (measurement limit) |
| `unlocalized_zero` | in range but **zero element overlap** (parsing-layer content missing or layout mismatch) | no (closer to a parsing gap) |
| `unlocalized_threshold` | some overlap but below threshold | no (threshold edge case, 0 after the fix) |
| `missing` | element located but **not retained by any chunk** (dropped as noise or a title) | yes |
| `single` | evidence elements all fall in **one** chunk | yes (good) |
| `split` | evidence elements span **multiple** chunks | yes (fragmentation risk) |

Also: **asset channel-kind matching** is measured two ways — `any` (counted as a hit if the hit set contains the expected kind at all, so a mixed channel can be rescued by text) vs. **`TRUE-asset`** (a genuine asset question must hit a table/chart/image chunk), the latter being the stricter, more honest measure.

---

## 4. Results (after the coordinate fix; 43 of 77 documents have ground truth)

Overall (localized=311 / 356; excluded: out_of_range 41, unlocalized 4):

| Metric | Value |
|---|--:|
| evidence kept in a **single** chunk | **70.1%** |
| ↑ under a stricter threshold (0.5/0.7) | **70.1%** (**identical to the default → not threshold-dependent**; the old version's threshold sensitivity was itself a symptom of the coordinate bug) |
| evidence **split** across chunks | 22.5% |
| evidence **missing**, dropped | 7.4% |
| **asset channel-kind match (any)** (n=118) | 92.4% |
| **asset channel-kind match (TRUE-asset, strict)** (n=95) | **83.2%** |
| answer-substring soft recall (n=288, reference only) | 16.3% |

By document type (**the distinct_docs column guards against a single document masquerading as a type-level trend**):

| doc_type | # docs | # questions | single% | split% | missing% | notes |
|---|--:|--:|--:|--:|--:|---|
| academic_paper | 7 | 30 | 86.7 | 6.7 | 6.7 | |
| financial_report_en | 7 | 37 | 73.0 | 27.0 | 0 | |
| government | 5 | 13 | 69.2 | 7.7 | 23.1 | |
| slides_tutorial | 4 | 23 | 60.9 | 39.1 | 0 | |
| brochure | 4 | 21 | 42.9 | 52.4 | 4.8 | image-heavy, high split |
| guidebook | 4 | 21 | 42.9 | 57.1 | 0 | image-heavy, high split |
| research_report | 4 | 21 | 33.3 | 66.7 | 0 | highest split |
| admin_industry | 2 | 6 | 50.0 | 33.3 | 16.7 | n≤2, reference only |
| news | 1 | 136 | 81.6 | 6.6 | 11.8 | **a single concatenated document, do not extrapolate** |
| law | 1 | 3 | 100.0 | 0 | 0 | **a single document, do not extrapolate; also does not cover F2's long-SEC-body scenario** |

---

## 5. How to read these numbers (with nuance)

- **The results line up self-consistently with strategy difficulty**: academic 87%, financial_report 73%, government 69% do relatively well on single-chunk; image-dominated brochure/guidebook/research_report have high split.
- **High split ≠ bad chunking**: for research/brochure/slides, the evidence is often **an image plus discussion text**, deliberately split by asset atomization into an image chunk and a text chunk; **parent-child retrieval re-gathers them under the same section parent**, so recall isn't lost. Split measures "does the leaf level need to fetch the parent," not "was evidence lost."
- **asset match any 92.4% vs TRUE 83.2%**: about 9pt of the gap comes from mixed-channel questions being rescued by a text chunk; **treat 83.2% as the real measure** of asset atomization's effectiveness.
- **Single-document rows (news/law) should not be extrapolated**: the news document is one concatenated file contributing 136 questions, and law contributes only 3 — this is noted in the table.

---

## 6. Limitations (recorded honestly)

- **Ground-truth coverage bias**: only 43/77 documents (the mmdocir subset) are annotated; **financial_research_zh (12 documents), which is the largest single group, plus policy/form/tech_report, are entirely unevaluated for evidence preservation** — their strategies have only been profiled statistically, never checked against retrieval.
- **PAGE_CAP=50** → 41 questions are out_of_range; long documents get truncated at the head, so **conclusions about "cross-page/deep-hierarchy" behavior cannot be drawn from the truncated sample** (see [PROCESS_LOG stage 9 / sampling limitations](../archive/PROCESS_LOG.md)).
- **The 16.3% answer-substring recall is not trustworthy**: answers are mostly short words/numbers/paraphrases, so substring matching has a high false-negative rate → **reference only, not used as a chunking metric**; a rigorous measure would need embedding- or LLM-based semantic judgment.
- **law's single=100% comes from a single mmdocir document**, and does not cover the F2 (long SEC body text) scenario found in the pdf_corpus bills; law's real preservation rate needs additional ground truth of that kind.
- **scale derivation depends on text pairing**: documents with no text/pure scans cannot derive a scale → those questions are counted as unlocalized_zero.

---

## 7. Reproducing

```bash
# In-house: the finalized component version (source of truth), regenerate chunks/ from the component package
python scripts/gen_chunks_component.py
python scripts/eval_chunks.py        # -> analysis/eval_report.json + eval_by_doctype.csv
# Note: scripts/chunk_document.py is the prototype version, has diverged from the component and hasn't tracked R1-R3 (see INTEGRATION §4); historical reference only
```
Determinism. Compare `single%↑ / split%↓ / missing%≈0` — **this is the objective yardstick for tuning the knobs.** The script also reports strict-threshold single% as a sensitivity self-check.

---

## 8. Three-way comparison: in-house vs. chonkie vs. docling (finalized component version, 2026-06)

**The question being asked**: should the chunking layer also use a mature off-the-shelf solution (parsing is already settled as MinerU, not hand-rolled)? Using the **same evidence-preservation eval, the same 43 ground-truth documents**, compared the in-house component, chonkie's `RecursiveChunker(recipe=markdown)`, and Docling's `HybridChunker` side by side. Script `scripts/compare_chunkers.py` → `analysis/compare_report.json`.

**Two bridge variants (key to separating attribution)**:
- **`ours_exact`**: uses the in-house `source_indices` **precise** bridge (the chunk knows exactly which elements it came from) — this is how the in-house approach behaves in **real RAG use**.
- **`ours_fair` / `chonkie` / `docling`**: all use the **same text-substring bridge** — stripped of source_indices, comparing **only the quality of the split boundaries themselves**.

**TEXT-channel (fairest comparison, pure-text evidence, localized=215)**:

| Strategy | single%↑ | split% | missing%↓ |
|---|--:|--:|--:|
| **ours_exact** | **79.5** | 10.2 | **10.2** |
| chonkie | 74.9 | 11.2 | 14.0 |
| ours_fair | 71.6 | 10.7 | 17.7 |
| docling | 56.3 | 10.2 | 33.5 |

**ALL evidence (including table/image/chart, localized=311)**:

| Strategy | single%↑ | split% | missing%↓ |
|---|--:|--:|--:|
| **ours_exact** | **70.4** | 21.9 | **7.7** |
| chonkie | 62.1 | 16.1 | 21.9 |
| ours_fair | 58.2 | 15.8 | 26.0 |
| docling | 47.6 | 13.8 | 38.6 |

**Conclusions (balanced, with an opinion)**:
1. **Docling's HybridChunker should not be used (against MinerU output)**: it has the highest missing rate across the board (text 33.5% / all 38.6%, 3–5x that of ours_exact). Its tokenizer-aware plus semantic merging over MinerU-exported markdown drops or fails to locate a large amount of evidence.
2. **The in-house moat is engineering features, not smarter boundary-drawing**: `ours_exact` wins across the board thanks to precise source_indices provenance; but with that stripped out, `ours_fair`'s pure boundary comparison **actually loses slightly to chonkie** (71.6% vs. 74.9%). Honestly — judged purely on "how smart the split boundaries are," chonkie's recursive splitting is a bit better than the in-house approach. The in-house net win comes from the engineering integration of source_indices + heading-tree breadcrumbs + per-chunk ACL + small-to-big — neither chonkie nor docling has any of that.
3. **The R1–R3 fixes are "invisible" on this metric**: the component version and the prototype version split text two genuinely different ways (verified different chunk counts/boundaries, `identical=False`), yet evidence preservation converges almost completely (text_only unchanged, ALL moves only +0.6% single). This means that what reset-aware leveling / banner guards / aside_text removal actually improve is **breadcrumb accuracy and retrieval-semantic cleanliness**, not evidence aggregation — this metric can't detect them; measuring that requires breadcrumb-correctness or retrieval-relevance metrics instead.

**Final judgment**: the in-house approach **is worth keeping**, but its moat should be repositioned as **engineering integration, not a smarter boundary algorithm**. **If all you ever need is plain-text splitting without this engineering package, chonkie is the more convenient choice** (comparable or slightly better boundary quality, zero maintenance). The prototype baseline is archived at `analysis/compare_report_proto.json`.
