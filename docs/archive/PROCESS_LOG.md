# Process log — from problem to strategy to implementation (audit trail)

> *Archived document: migrated as-is from the engine-prototype repo, kept unchanged. Where the text refers
> to `../analysis/CHUNKING_STRATEGY.md` and `EVALUATION.md`, it refers to the old repo's layout; those two documents were not carried over during the migration; the body text is unchanged, only these references have been demoted to plain text so they don't point at a 404.*

> Records the **inputs, decisions, findings, and corrections** of each stage in chronological order, so the whole chain of conclusions is reusable and auditable.
> Date: 2026-06-21. Model-side execution and user decision points are both marked.

---

## Stage 0 — The conceptual question (motivation)

The user asked: "Once the parser outputs structured JSON, how should chunking be done? Should it use an LLM?"
**Conclusion**: chunking doesn't need an LLM — the parser has already delivered semantic boundaries, so LLM-based chunking is paying twice for the same work and isn't reproducible; an LLM should only be used for "enhancement" (table summarization, filling in captions), not for "splitting." This became the first principle behind every subsequent design decision.

---

## Stage 1 — First look at real JSON (single document)

Subject: a locally-run MinerU output for a brokerage research report (`AP2026...Satellite Communications`).
**Findings**:
- `content_list.json` is a chunking-friendly view; `block_list.json` contains `is_discarded`, `mergeConnections`, and stable `id`s — a goldmine for chunking.
- text_level is unreliable in both directions: real headings get flattened, non-headings get promoted; TOC entries get mixed in.
- tables = HTML (rowspan/colspan), charts = VLM-estimated values (`~5`), images = VLM mermaid (clearly hallucinated in places).
**Correction point**: initially assumed `content_list.json` was "the" JSON, but testing showed `block_list.json` is the one carrying relationship signals → this established the "two files, two jobs" division.

---

## Stage 2 — Dataset inventory and sampling design

Data source `knowledge-base/datasets`, 4 collections / 880 PDFs / 2.9GB.
**Findings/decisions**:
- `omnidocbench` has no PDFs (annotations only) → skipped.
- `pdf_corpus_v1/manifest.jsonl` is the cross-collection master manifest (its `benchmark_pdf` field, 312 entries, points at mmdocir) → deduped, yielding 14 real type categories.
- Hard constraint = MinerU's 1000 pages/day/account → **exhaustive coverage is impossible**, so stratified sampling was required (what chunking needs is type×layout diversity, not raw volume).
- Each manifest carries `doc_type/layout_tags/page_count/language` → sampling can be done with real justification.

**User decision point ①**: provided **3 account tokens**, asked for load-balanced distribution across them; coverage chosen as **breadth-first, balanced across all types**.
→ Settled on **77 documents / 1867 effective pages** (`PAGE_CAP=50`), ~620 pages per key across the 3 keys.

**Correction points**:
- Worried that mmdocir's on-disk filenames (a mix of descriptive names and hash names) might not line up with the annotation `doc_name` → tested and found a **313/313 exact match**, the mapping was trivial.
- `pypdf` wasn't installed → installed it for mmdocir page-count statistics (cached to `config/`).

---

## Stage 3 — Learning the MinerU online API

Read https://mineru.net/apiManage/docs.
**Confirmed schema**:
- `POST /api/v4/file-urls/batch` → `{data:{batch_id, file_urls[]}}` (the url order corresponds to the files).
- `PUT <file_url>` binary, **must not set Content-Type**; parsing starts **automatically** once the upload completes.
- `GET /api/v4/extract-results/batch/{batch_id}` → `extract_result[]{file_name,state,full_zip_url}`.
- Limits: ≤50 files per batch, ≤200MB/≤200 pages per file, 1000 pages/day/account, URLs valid for 24h.

---

## Stage 4 — Repo setup and client

- Created `chunk-test-repo`, with `.gitignore` written **before** `.env` (protecting the 3 tokens).
- `select_sample.py` (stratified sampling + categorization + 3-key balancing), `mineru_client.py`, `parse_batch.py`.
- **Sampling result**: 77 documents / 1867 pages, all 14 categories met their targets; key loads A623/B622/C622, files 26/25/26.

---

## Stage 5 — Parsing (smoke test → full run)

**Smoke test** (1 law document, end to end) passed, and exposed a **key schema difference**:
- the online VLM output has **no `block_list.json`** (only the local version does). The richer structure instead lives in `layout.json.pdf_info`:
  - `discarded_blocks` (noise, equivalent to is_discarded)
  - `para_blocks[].merge_prev` (cross-page continuation, finer-grained than mergeConnections)
  - `lines[].spans[].score` (OCR confidence, a new usable signal)
→ the analyzer and chunker's signal sources were adjusted accordingly.

**Full run**: 3 accounts, 6 batches (grouped by key×language), run in parallel; **77/77 all done**, exit code 0.

---

## Stage 6 — Cross-document analysis

`analyze_chunks.py` computes per-document metrics, aggregated across the 14 categories.
**Core quantitative findings** (details in `../analysis/CHUNKING_STRATEGY.md` §1 table):
- Noise rate 0→0.37; `discarded` tracks the type-level noise rate almost exactly → the discarded judgment is trustworthy.
- Numbering-recoverability rate: academic 0.64 vs. law 0.02 vs. financial 0.04 → hierarchy recovery must be handled per domain.
- 11/14 categories have a body-block median < 75 tokens → upward merging is required.
- Table-to-HTML conversion is reliable at 0.86-1.0; table/figure caption coverage swings wildly between 0.04-0.89 → captions cannot be assumed to exist.

**Three anomaly checks (engineering discipline: don't just trust the aggregate)**:
1. **law's num%=0.02 is misleading**: its chapters are `SEC. N` (letter-first, doesn't match a digit regex), and the deeper `(a)(1)(A)(i)` enumeration markers are hidden at the start of body lines → the chunker added `LAW_SEC_RE` and relies on upward merging to keep the clause tree intact.
2. **form**: fillable fields are empty-value HTML cells, and it's the labels that have values → added a special case that preserves the whole table as an asset.
3. **news outlier** (1335 headings / 381 images): `news_combined.pdf` is multiple concatenated documents → flagged as a special case.

---

## Stage 7 — Chunker implementation and validation

`chunk_document.py` implements a 7-step pipeline (noise removal/stitching/domain-specific tree building/asset special-casing/strategy assembly/metadata/parent-child).
**Output**: 7695 leaf chunks (77 documents).
**Passed review** (via spot checks):
- law's `SEC. 2` keeps the whole (a)(1)(2)(A)(B)(C) clause tree together in a single 707-token chunk, not split mid-clause.
- financial report tables: `text`=caption+source, `content_raw`=HTML, correctly separated.
- brochure: 21/21 captionless images all correctly tagged `captionless`+`vlm_content`.
- parent "Weekly Market Review" aggregates 12 child blocks / 816 tokens, leaf↔parent linkage is bidirectionally consistent.
**Recorded limitation**: parents are grouped by breadcrumb text → sections with the same name get merged across pages (see DESIGN §7).

---

## Stage 8 — Ground-truth evaluation (an auditable closed loop)

Motivation: don't grade your own homework — check chunking against the dataset's own built-in annotations.
Subject: `mmdocir/MMDocIR_annotations.jsonl`, overlapping the sample at **43 documents / 356 questions** (carrying page+bbox+channel+answer).
Method: `source_indices` as the bridge, evidence bbox → content_list element → chunk; see `EVALUATION.md` for details.

**Two measurement bugs (handled via "diagnose → fix → verify", not treated as conclusions on their own)**:
1. **Channel matching at 0%**: the annotation's `type` field is the string `"['Figure']"`, not a real list, so membership checks were always false → fixed with `ast.literal_eval` parsing. Diagnostic evidence: DIAG2 produced zero output for asset questions.
2. **missing inflated to 55.6%**: a single-element coverage threshold of ≥0.5 was too strict, and it didn't distinguish "measurement couldn't localize this" from "chunking lost it" → switched to a bidirectional coverage criterion (intersection/evidence≥0.3 or intersection/element≥0.5) to collect the evidence-element set, and added `unlocalized`/`out_of_range` exclusion categories.
   - Also diagnosed page alignment: offset 0 on clean documents gave 0.7-0.79 coverage, confirming correctness; on full-page-visual documents, "a non-zero offset winning" turned out to be noise, so **hacking a per-document offset was rejected as a workaround**.

**Post-fix results**: channel matching 0%→**79.4%**, missing 55.6%→**8.2%**; evidence kept in a single chunk **65.8%**, split 25.9%.
This lines up self-consistently with strategy difficulty (law 100%/government 92%/academic 87% best on single-chunk).
**Key interpretation**: high-split cases in research/brochure/slides are "image + discussion text" deliberately separated by asset atomization; parent-child retrieval re-gathers them → split measures "needs to re-fetch the parent," not "evidence lost."

---

## Stage 9 — Adversarial multi-agent review + fixes (the most important quality check)

The user asked to "review the results and output using adversarial agent review." Orchestrated a workflow: 4 independent skeptics (code/evaluation/strategy/sampling) working in parallel, only looking for problems → each finding sent to an independent verifier for reverse-checking. **33 agents, 29 findings → 14 confirmed / 11 partial / 4 refuted.**

**Confirmed and fixed real bugs (diagnose → fix → verify):**
- **F1 fatal**: Step 2's cross-page stitching bbox-IoU had **zero hits throughout** — content_list (rendered coordinates ~823×936) and layout (PDF points, 612×792) use different coordinate systems. Diagnosis: measured 261 merge_prev entries, merge_flag hit 0; the same heading's bbox in both sources had a constant ratio (x1.63/y1.26), proving it was a scaling relationship. **Fix**: switched to text-prefix matching → 112 chunks across 17 documents now hit (verified >0).
- **F2 high**: law's `LAW_SEC_RE` returned before the length check, misjudging `SEC. 2102. (a)(1)...long body text` as a heading and discarding it. Diagnosis: publ158 lost 16 body blocks. **Fix**: moved the length/TOC guard earlier → 16/16 recovered (verified).
- **F6 high**: the eval script had the same underlying coordinate bug (normalizing content_list bboxes using layout's page_size). **Fix**: per-document text pairing to derive the scale (sx,sy), transforming GT into content_list space. **Effect verified**: unlocalized 72→4, single 65.8%→70.1%, and still 70.1% under a stricter threshold (**the threshold sensitivity itself was a symptom of the coordinate bug — fixing it made the result stable**).
- **F3 high**: financial_zh TOC entries (ending in spaces + page numbers) were polluting the breadcrumb. **Fix**: added TOC_TAIL_RE → 0/622 (verified).

**Confirmed and honestly corrected in the documentation:** government's "table/chart caption 0.11" was actually table caption (government has 0 charts); academic's image-caption mean of 0.88 was actually a weighted 0.22; "77 documents" was actually 43 evaluated + added a distinct_docs column; news/law were each labeled "not to be extrapolated" as single-document rows; small samples (n≤2) were flagged with a warning; "chunking doesn't need an LLM" was downgraded to a design assumption (no A/B test was done); asset matching was supplemented with the strict true-asset figure of 83.2%.

**Post-fix core numbers (coordinate-fixed version):** evidence in a single chunk 70.1% / split 22.5% / missing 7.4% / asset (true) 83.2%.

**4 findings judged refuted (the opposing side can be wrong too, don't take it at face value)**: "the strategy ignores eval" (eval was done afterward and was already flagged as 11.2% unreliable), "news's extreme value is disrupting the iron rules" (already flagged as an outlier), overgeneralizing across languages for financial (the document never claimed that), "the 11.2% figure being treated as positive evidence" (it was already explicitly flagged as unreliable).

See `analysis/eval_report.json` and the adversarial review artifacts for details.

---

## Decision summary (traceable)

| # | Decision | Rationale | Alternative / cost |
|---|---|---|---|
| D1 | Chunking doesn't use an LLM | Structure already contains boundaries, reproducible | LLM chunking: expensive/not reproducible |
| D2 | content_list as the spine + layout as signal | Each carries information the other lacks | Using only one loses information; requires bbox alignment |
| D3 | Stratified sampling rather than exhaustive coverage | Quota constraint + prioritizing diversity | Exhaustive: exceeds quota, diminishing returns |
| D4 | 3 accounts balanced by page count | Quota is measured in pages/day | By file count: doesn't reflect real cost |
| D5 | Domain-specific hierarchy rules | Numbering conventions differ by domain | A uniform regex: misses a lot in law/financial documents |
| D6 | source_indices preserved throughout | A hard requirement for auditability | — |
| D7 | Strategy isolated into `assemble_text` | The one place with real tradeoffs, easy to tune | — |

---

## Reproduction entry point

See [IMPLEMENTATION.md §2](engine-prototype-IMPLEMENTATION.md). All scripts are deterministic and resumable.
