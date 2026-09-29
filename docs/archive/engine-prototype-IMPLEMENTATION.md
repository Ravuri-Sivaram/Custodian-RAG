# Implementation document — running, reproducing, extending, auditing

> *Archived document: migrated as-is from the engine-prototype repo, kept unchanged. The scripts and paths described
> belong to the old repo's layout; the current version lives at [docs/components/chunker/](../components/chunker/).*

> An operations manual aimed at "clone it and run it, modify it, inspect it." Design rationale is in [DESIGN.md](engine-prototype-DESIGN.md).

---

## 1. Environment

- Windows + Python 3.12 (as tested); `pip install -r requirements.txt` (requests / pypdf / python-dotenv).
- MinerU online API tokens: copy `.env.example` to `.env` and fill in `MINERU_TOKEN_A/B/C` (multi-account load balancing). `.env` is already excluded by `.gitignore` — **do not commit it**.

---

## 2. One-command reproduction (end to end)

```bash
python scripts/select_sample.py     # ① sample → corpus/ + sample_manifest.csv
python scripts/parse_batch.py       # ② MinerU parsing → parsed/        (idempotent, resumable)
python scripts/analyze_chunks.py    # ③ statistics → analysis/per_doc_stats.csv + by_type.json
python scripts/chunk_document.py    # ④ chunking → chunks/*.jsonl + _summary.csv
python scripts/eval_chunks.py       # ⑤ ground-truth evaluation → analysis/eval_report.json + eval_by_doctype.csv
```

Smoke test (verify the single-document pipeline + API schema first, before running the full set): `python scripts/smoke_test.py`.

**Idempotency**: `parse_batch.py` skips documents in `parsed/<doc_id>/` that already have a content_list; the other scripts overwrite on rerun. `select_sample.py` has no randomness, same input gives the same output (page counts are cached in `config/mmdocir_pagecount.json`).

---

## 3. Module-by-module notes (scripts/)

| File | Responsibility | Key functions/data | Output |
|---|---|---|---|
| `select_sample.py` | reads the 3 manifests, normalizes into 14 categories, stratified sampling (evenly spread across page-count buckets), greedily balances across 3 keys by page count, copies into `corpus/<type>/` | `TARGETS` (per-category document count), `spread_pick`, `assign_keys` | `sample_manifest.csv` |
| `mineru_client.py` | MinerU v4 client | `create_batch`/`upload` (PUT with no Content-Type)/`poll_batch`/`download_and_extract` | — |
| `parse_batch.py` | submits in batches grouped by (key, language), uploads concurrently, polls, downloads and extracts concurrently | `POLL_SECS`/`MAX_POLL_MIN` | `parsed/`, `parse_results.csv`, `batches.json` |
| `analyze_chunks.py` | cross-document statistics (noise/hierarchy/tables-images/merge/OCR confidence), aggregated by type | `analyze_doc`, `NOISE_TYPES`/`NUM_RE`/`TOC_RE` | `analysis/per_doc_stats.csv`, `by_type.json` |
| `chunk_document.py` | the 7-step chunking pipeline | `CONFIG`, `heading_level`, `assemble_text` (the tunable knob), `chunk_doc` | `chunks/*.jsonl`, `*.parents.jsonl`, `_summary.csv` |
| `eval_chunks.py` | checks evidence preservation using MMDocIR annotations (bridged via `source_indices`) | `region_elements` (bidirectional coverage), `parse_listish`, verdict classification | `analysis/eval_report.json`, `eval_by_doctype.csv` |

---

## 4. Tunable knobs (the ones changed most often)

1. **chunk granularity** → `CONFIG[doc_type] = (min, target, max)` at the top of `chunk_document.py`.
   - For finer-grained (favoring fact retrieval): lower `target`/`max`. For coarser-grained (favoring synthesis): raise them.
2. **assembly logic** → `assemble_text()`. Currently does not merge across sections; if cross-adjacent-section merging or a semantic-splitting approach is wanted, only this one pure function needs to change, and the other six steps stay untouched.
3. **sample scope/composition** → `select_sample.py`'s `TARGETS`, `PAGE_CAP`, `KEYS`.
4. **parsing parameters** → `mineru_client.create_batch`'s `model_version` (default vlm), `enable_formula/table`, `language`.
5. **polling cadence** → `parse_batch.py`'s `POLL_SECS` (15s), `MAX_POLL_MIN` (45).

---

## 5. How to extend with a new document type

1. `select_sample.py`: map the raw doc_type to your normalized name in `MMDOCIR_DOMAIN_MAP` or `PDFCORPUS_TYPE_MAP`, and give it a quota in `TARGETS`.
2. `chunk_document.py`:
   - add a budget triple to `CONFIG`;
   - if its numbering scheme is unusual (like law), add a domain regex in `heading_level()`;
   - if it chunks per page (like slides), add the type to `PAGE_GROUPED`.
3. Rerun step ④ and check whether `chunks/_summary.csv`'s `ch/doc`, `median_text_tok`, and `captionless` look reasonable.

---

## 6. How to audit the output (auditability)

**A. Tracing a single chunk back to the source**
Every chunk carries `source_indices` (indices into the original `content_list`). Compare:
```bash
python -c "import json; d=[json.loads(l) for l in open('chunks/law__PLAW-118publ38.jsonl',encoding='utf-8')]; \
c=d[1]; print(c['source_indices']); print(c['text'][:400])"
```
Take the `source_indices` and check the corresponding entries in `parsed/<doc_id>/*content_list.json` for consistency — whether anything was mis-split or mis-merged.

**B. Spotting anomalies from the summary metrics**
`chunks/_summary.csv` lists `leaf_chunks / text / table / image / captionless / median_text_tok` per document. Check for:
- an unusually large `ch/doc` (e.g. news at 1399) → possibly concatenated documents;
- `median_text_tok` < `CONFIG.min` showing up widely → the assembly/merge threshold needs adjusting;
- a high `captionless` rate → a large retrieval blind spot for that category's assets, consider enabling VLM caption generation.

**C. Manual spot-checking** (already run during development, commands are reusable)
- Law: check whether `section_path` is `SEC. N`, and whether clause `(a)(1)(A)` stays together as a whole in one chunk;
- Tables: check whether `text` = caption+footnote and `content_raw` = HTML;
- Parents: check whether `parents.jsonl`'s `child_ids` is bidirectionally consistent with the leaf's `parent_id`.

**D. Regression**
After rerunning step ④, diff `chunks/_summary.csv`; unrelated metrics should show zero change (determinism). After changing `CONFIG`/`assemble_text`, only the target type's chunk count/token numbers should change.

---

## 7. Quotas and limits (operational notes)

- Single file ≤200MB / ≤200 pages; ≤50 files per batch; **1000 pages/day/account**.
- Upload URLs are valid for 24h; parsing starts automatically once upload completes, no separate submit call is needed.
- On hitting rate limits/failures: `parse_results.csv` records `state` (done/failed/timeout) and `err`; rerunning `parse_batch.py` for failed documents resumes automatically (idempotent).
- First-run cost: 77 documents / 1867 effective pages, ~620 pages per account across 3 accounts, completed within a single day.
