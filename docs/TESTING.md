# Custodian Testing Documentation

> **Note (this fork):** several results below (§3's cross-language table-ranking runs) were measured against real Chinese-language
> financial reports, from before this fork replaced Chinese-language support with Telugu (see the top-level README's Technology
> stack table). They're kept as genuine historical findings and have not been re-measured against Telugu documents.

> Two gates: CPU unit tests (run on every change; one pytest suite must be fully green, baseline counts in §1) / GPU smoke tests + ACL regression (run before going to production).
> All numbers are real results from running WSL `custodian` (on an RTX 4090), not estimates.

## 1. CPU unit tests (one pytest suite for the whole repo, ~20 seconds, touches no Qdrant/GPU/network)

```bash
conda activate custodian && cd custodian && python -m pytest -q
```

> **§1 of this document is the single authoritative source for the whole repo's test baseline counts.** Every other document links back here rather than restating the numbers — an earlier stale baseline had been copied into seven documents, and once the suite grew nobody kept them in sync, so all seven were wrong together.
>
> **Measured** (2026-07-22, WSL `custodian` / RTX 4090): `264 collected` → **`259 passed, 5 skipped`**.
> Breakdown: product surface **85 items** (6 files, in `tests/`) + engine surface **179 items** (13 files, in `tests/engine/`, of which 5 are auto-skipped for lack of a GPU / Qdrant server — see each file's skip reason).

The product surface is in `tests/`; the engine surface is in `tests/engine/` (after folding into this repo, it runs together with the same pytest suite, including the embedder's `test_acl.py` ACL predicate unit tests). The CPU CI gate = this one pytest suite passing fully green.

**Where CI runs**: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), triggered on every push/PR. Two tiers: ① install only `[dev]` (py3.10 + 3.12) — runs exactly the command this document promises contributors; ② additionally install CPU torch (py3.12) — this makes the 5 assertions in `test_remote.py` that compare torch↔numpy truncated dimensions for equivalence actually run instead of being skipped (that's the floor for "building locally / querying remotely doesn't get misaligned"). The GPU gate does not run in CI — see §3.

| File | Coverage |
|---|---|
| test_sessions.py | Same session → same set; cross-session isolation; bounded LRU eviction; touch refresh |
| test_service.py | healthz; wiring for the six endpoints; **no_identity fails closed**; bad_arg goes through structured handling, not 422; API key gate (healthz is exempt / wrong key is rejected); **per-session dedup isolation + no header means no dedup**; ask (citation mapping / include_contexts / empty query / llm_unconfigured / ask_failed doesn't leak internal details) |
| test_adapter.py | Parameter forwarding mapping for the six tools; backend-down → structured backend_unavailable (hint points to custodian serve); mapping for 401/5xx/non-JSON; session header presence; instructions share the same source as the engine |
| test_review_fixes.py | Regression tests for adversarial-review P1 fixes: no_identity hint points to CUSTODIAN_TENANT; indexer rejects restricted+empty allow; factory exceptions degrade to ask_failed; doc_id URL encoding / empty arg rejected locally / 3xx handled structurally; .env inline comments/quotes/int-with-a-name errors/override paths get expanduser; **the adapter's and `mcp_stdio`'s six-tool docstrings are verbatim-identical** (a repo-internal structural assertion: the two sides' docstrings are equal + `_INSTRUCTIONS` is single-sourced from `toolcore`) |
| test_smart.py | smart-ask (D9): numeric questions only trigger the table-leg retry on refusal (select-the-better-answer) / non-numeric doesn't trigger / explicit kind is respected / the on/off switch / refusal hints |
| test_team.py | Multi-identity (D10): keys parsing fails closed / 401 / identity flows to the engine per request / cross-user session isolation / stats admin gating / non-loopback startup guard / name uniqueness + `|` forbidden / keys new doesn't throw raw errors / observability is crash-safe / structured failures count as errors / logs never contain the key + truncation is toggleable |

The single source of truth for the tool-semantics primitives (already_returned/omitted_budget/budgets including assets/no leaking existence without access, etc.) is `src/custodian/toolcore.py`, covered by the tool-semantics tests under `tests/engine/` (run together with the product-surface tests in the same pytest suite).

The product surface's six test files: sessions / service / adapter / review_fixes / smart / team. Baseline counts are at the top of this section.

## 2. Engine-surface tests (already folded into this repo's `tests/engine/`)

Since the engine was folded into this repo, it is no longer a separate repository with its own gate: the engine surface and the product surface run together with the same command (`python -m pytest -q`); baseline counts are in §1. Contract-drift tests (adapter vs. `mcp_stdio` docstrings being verbatim-identical, `_INSTRUCTIONS` being single-sourced from `toolcore`) are now **repo-internal structural** tests, no longer requiring a cross-repo exec.

```bash
python -m pytest tests/engine -q               # run only the engine surface (can run standalone; usually run merged with the product surface)
```

## 3. GPU smoke tests (real index at ~/rag_real, 77 documents / 7652 chunks)

> Besides this section's smoke tests, the pre-production gate also includes `eval/acl_regression.py` (WSL+4090, end-to-end 0 leakage); since it needs a GPU, it doesn't run in CPU CI, and like the GPU smoke tests below, it's a manual pre-production gate.

Steps and measured results (all passed):

| # | Step | Measured Result |
|---|---|---|
| 1 | `python -m custodian serve` (background) | Starts in seconds; log confirms the index was opened exclusively, collection=real tenant=demo |
| 2 | GET /healthz | `{"status":"ok",…,"tenant_bound":true}` |
| 3 | GET /v1/documents | 77 documents, coverage across 14 doc_types |
| 4 | POST /v1/retrieve "Netflix 2015 revenue" (first query) | **19.1s** (including dense model load), top1 hit is NETFLIX_2015_10K, status=ok |
| 5 | POST /v1/ask "What was Netflix total revenue in 2015?" | status=ok, grounded answer + 2 citations (chunk_id/page number correct), finish_reason=stop; when top_k=5 fails to retrieve the total revenue figure, the model honestly says "insufficient information" (grounding works correctly, no fabrication) |
| 6 | Repeat retrieve in the same session (X-Custodian-Session: smoke1) | All 3 results are `already_returned`, already_n=3 |
| 7 | New session smoke2, same query | `section_window×2 + deduped`, already_n=0 (**isolation confirmed**) |
| 8 | MCP adapter (live daemon): list/retrieve/outline | 77 docs / retrieve ok / outline shows 88 sections |
| 9 | CLI: `python -m custodian ask "What content does the library have on DDoS attack protection?"` | Grounded Chinese-language answer + sources (title/page/section/chunk_id) |

Reproduction commands are in the git history and in [IMPLEMENTATION.md](IMPLEMENTATION.md) §6.

**Follow-up evidence after N3 (ask retrieval filtering) landed** (2026-07-03, same real index; prompted by a user's real-world test of "Netflix 2015 revenue" against the closed pipeline resulting in a refusal):

| Phrasing | Result |
|---|---|
| Chinese, default parameters (user's original attempt) | Honest refusal (prose crowds out the table; the number is in the table on p.16, confirmed present in the library) |
| Chinese/English + top_k 12 + rerank (no kind) | Still refuses |
| Chinese + `--kind table` + top_k 15 (no rerank) | ⚠ **Wrong answer**: segment revenue 4,180,339 was mistaken for total revenue (filed as TODO P2) |
| **English + `--kind table --rerank`** | ✅ **$6,779,511 thousand, citing p.16 Selected Financial Data** |
| Chinese + `--kind table --rerank` | Honest refusal (explicitly states it only saw segment data) — cross-language table-ranking gap remains |

Conclusion: for numeric/table questions, recommend `--kind table --rerank` plus document-language keywords; cross-language enhancement is filed as TODO P2.

**Numeric-range wrong-answer fix (2026-07-03, full diagnose → fix → verify cycle)**:

- **Root-cause chain** (both had to be missing for the wrong answer to occur): ① the prompt had no numeric-range constraint; ② **the range evidence wasn't in the prompt** — the segment information was only in the section_path metadata, and the table chunk's body text didn't contain a single word like "Domestic Streaming" (confirmed: adding only constraint ① still produced the wrong answer; only once the evidence was also added did it take effect).
- **Fix** (engine generator): added a narrowly-targeted numeric-range constraint to SYSTEM (distinct from the sentence-level tightening in R3 that was reverted) + merged the context's source line into the section breadcrumb (`Title § FORM 10-K > Domestic Streaming Segment`).
- **Verified in three passes**: ① reproduced case B: no longer answers wrong, explicitly labels it as "partial business data only" and declines to extrapolate; ② re-ran case C: total revenue still answered correctly ($6,779,511, p.16), no collateral damage; ③ **before/after comparison with the same DeepSeek judge** (72 questions, retrieval/index/judge all identical): faithfulness 0.972 → **1.000** (+0.028), correctness 0.847 → **0.847** (±0). Zero regression.
  (Baseline scoring artifacts: eval/baseline_single_prescope*.json, gitignored)

**Table-chunk retrieval-text enhancement (2026-07-03, diagnose → implement → rebuild → regress → decide, engine commit `2bd97a5`)**:

- **Root cause** (confirmed at the source-code level): a table chunk's retrievable signal was only a single sentence made of caption+footnote (the body only went into content_raw and was not part of embed/sparse indexing; the breadcrumb wasn't spliced in either) — English queries suffered the same disadvantage; cross-language was just an amplifier.
- **Implementation**: `_table_signal` (the first 2 header rows + the first non-empty cell of each row, data cells excluded) + breadcrumb splicing in; **an existence gate** (cap|foot|body): without the gate, the full index went 7652 → 7675 (+23 ghost chunks, which would shift chunk ids and invalidate gold/old citations); with the gate, it came back to exactly **7652** (hard evidence of id stability). Chunker tests went from 43 → 44.
- **Rebuild**: full rebuild of both ~/rag_real (77 documents / 7652) and ~/rag_eval_big (15 documents / 1409).
- **Regression** (72 questions, same judge): correctness **held steady at 0.847**; citation recall +0.007; faithfulness 0.986 (−1 question, a borderline judgment on inferring an extreme value from a chart, a known hard category); retrieval recall 0.854 → 0.833 (−2.1pp).
- **Question-by-question diagnosis of the drop**: it all came from 3 questions with double gold answers in the multi_intra category, each of which lost one **redundant** gold answer (3×0.5/72=2.1pp accounts for all of it); **all 3 questions were still answered correctly under the new index**; 0 questions went up in score — because gold answers were sampled from prose chunks, questions that benefit from tables are near-zero in the gold set (a measurement blind spot, already filed as TODO).
- **Motivating case (decisive)**: the original Chinese phrasing + `--kind table` (no rerank) went from wrong/refused to **directly answering $6,779,511 thousand correctly** (citing the p.18 consolidated results table).
- **Verdict: kept.** Judged by degree of damage: the question categories unlocked were previously wrong/refused (high damage), the displaced hits were redundant recall that didn't affect the answer (zero damage), and correctness held steady.

**Adding table questions to gold + new 88-question baseline (2026-07-03, engine commit `2ccda93`)**:

- gen_gold_tables.py generated 16 targeted table questions (mixed Chinese/English; programmatic QC caught and rejected 2 hallucinated questions); gold went from 72 → 88, **a break in comparability** (the historical 72-question aggregate numbers can no longer be directly compared).
- **New authoritative baseline** (88 questions, DeepSeek judge, closed-pipeline default parameters): retrieval recall 0.818 / MRR 0.627 / citation recall 0.767 / faithfulness 0.977 / correctness 0.818.
- **Split**: prose, 72 questions (retrieval 0.833 / correctness 0.861 / faithfulness 0.972) vs. **table, 16 questions (retrieval 0.750 / correctness 0.625 / faithfulness 1.000)** — table questions get 75% retrieval hit rate even without a kind filter (before the enhancement this category was nearly unreachable, and there's no "before" number since the old index has already been rebuilt over — left honestly blank); the perfect faithfulness score comes from the 4 questions with a retrieval miss all honestly refusing, with zero fabrication (the numeric-range constraint holds up on this new question category).
- **The 6 errors on table questions**: 4 are retrieval misses (refusal judged wrong — retrieval-side headroom); 2 are cases where retrieval succeeded but the table reading was wrong (misaligned rows/columns in a large table — generation-side headroom). This is the symmetric yardstick for the next round of table-oriented work.

**smart-ask launch record (2026-07-03/04, shape settled over four rounds of 88-question experiments, design covered in DESIGN D9)**:

| Version | Table-16 correctness | Prose-72 correctness | Faithfulness (whole set) | Verdict |
|---|---|---|---|---|
| Baseline (no smart) | 0.625 | 0.861 | 0.977 | — |
| ① Pre-emptive table leg | **0.875** | **0.792** ❌ | 0.966 | Rejected: the numbers the leg brought in collaterally damaged 5 prose questions that were previously correct |
| ② Failure-driven, rerank_top_n=30 | 0.750 | 0.847 | 1.000 | An illusion: the leg was too shallow (the five-year table sat at coarse rank 31–50, so it didn't fit in the reranking pool), and the flagship hand-crafted case still failed |
| ③ Failure-driven, top_n=50, unconditional adoption | 0.625 | 0.833 | **0.932** ❌ | Rejected: partial answers carried the wrong "X was not provided" claim of missing information (X was actually in the context) |
| ④ **Failure-driven + select-the-better-answer (final version)** | 0.688 | 0.833 | **0.977** | **Adopted** |

Attribution for the final version (rows tagged retried/retry_kept): among the 81 questions that never triggered the leg, only 2 flipped compared to the baseline when paired (both are known-unstable questions that flip-flopped across all five rounds of experiments = noise floor of ±2 questions); of the 4 questions on the path that ended up not being adopted, correctness exactly matches the baseline (zero loss); of the 3 questions on the path that was adopted, 1 flipped from ✗ to ✓. **The flagship hand-crafted case (a multi-value, multi-year question type not covered by the exam) went from a refusal to fully correct under default parameters**: the Chinese "net profit by year" question got all five years right (auto=table_leg_retry), and the Chinese "total revenue" question was also answered correctly.

Methodological takeaways: ① intelligence built into default behavior must only act on the failure path — any "help" on the success path is a risk (the lesson of ①); ② the reranking pool depth must be ≥ the worst coarse rank of the correct chunk (the lesson of ②); ③ when a retry turns "total refusal" into "partial answer," an incorrect claim about the missing part is a new failure surface, and the selection threshold must catch it (the lesson of ③); ④ single-pass LLM eval has a noise floor of ±2 questions, so comparisons at this granularity must be attributed pairwise (the retried tag is now built into run_eval).

## 3b. Team service-surface field testing (multi-identity / observability / load testing / drills)

**CPU unit tests 46 → 59** (test_team.py additions: keys parsing fails closed / 401 / identity flows to the engine per request / cross-user session isolation (forging the same session id) / stats admin gating / non-loopback startup guard / logs never contain the key + truncation + toggleable / name uniqueness and `|` forbidden / keys new doesn't throw raw errors / observability is crash-safe / structured failures count as errors).

**Multi-identity live demo** (real index, alice=demo/admin, bob=other): no key / forged key → 401; alice sees 77 documents; **bob (tenant "other") sees 0 documents, retrieval is empty** (engine ACL fails closed; identity is only "who is asking"); bob reading stats → 403, alice → 200.

**Concurrency load test** (bench.py, 4090/77 documents): retrieval p50 scales roughly linearly with concurrency (1 concurrent client 408ms → 10 concurrent clients 3.07s), a throughput ceiling of ~3.2 req/s, **zero errors**; under mixed load, while an ask is in flight retrieval p50 is still 352ms (the LLM segment doesn't hold the lock, confirming D8 in practice). Full table in [OPERATIONS §4](OPERATIONS.md). Capacity conclusion: usable experience for ≤10 concurrently active users.

**Backup/restore drill**: stop service → backup (27MB/3s) → restore to a standby directory → standby instance on port 8788 → all 77 documents visible + retrieval ok, hot-model RTO 33s (cold start adds model lazy-load time on top, honestly noted in OPERATIONS).

## 4. Adversarial review (P1, complete)

Three perspectives (security/correctness · concurrency/contract · documentation) × 2 independent counter-reviewers per finding, 39 agents. Result: **2 confirmed + 13 self-verified as genuine (verification agents ran out of budget, so the main line cross-checked each against the source code) → all fixed and pinned with regression tests** (test_review_fixes.py, 11 items); 3 were disproven and filed away. Full list and disposition in [COMPONENT_NOTES.md §Adversarial Review P1](COMPONENT_NOTES.md).

**Team service-surface security review (T5)**: identity/session/observability/operations, four perspectives × 2 counter-reviewers per finding, 36 agents. **0 confirmed security vulnerabilities** — the verification that did complete confirmed the core ACL boundary is solid (the identity name is just a display label plus a session-dedup prefix; the real boundary is the engine's User built by `_current_user`; a cross-tenant dedup collision fails in the safe direction, not a privilege escalation). ⚠ Honest caveat: most verification agents hit the server's rate limit before finishing, **so this doesn't rely on voting — everything was individually cross-checked**. After verification, a batch of **robustness/documentation/observability-completeness** defects were fixed and pinned with regression tests (relevant items in test_team.py): observability skipping a record on a crash → wrapped in try/finally; structured failures (200) not counted in errors → now judged by business status; query-truncation logic coupled to the wrong layer → pushed down into the obs layer; name uniqueness + `|` forbidden; keys new throwing raw errors → reuses load_keys validation; backup missing .env / missing mkdir; RTO labeled hot/cold; several documentation field inconsistencies. Regression: the whole repo's single pytest suite is fully green.

## 5. Not Covered (Honest List)

- **Cold-start RTO** has not been measured (the drill was done with a hot model; cold start adds model lazy-load time, 20s–2min, an estimate already noted in OPERATIONS);
- **>10 concurrent / long-running load testing** has not been done (the known throughput ceiling is ~3.2 req/s; larger scale moves to the v2 Qdrant server mode);
- A full real run of `custodian index` (~/rag_real was already built by the indexing scripts under `scripts/`; indexer.py is its parameterized version, and so far only logic review + verifying the lock-conflict-message path has been done; this will be measured for real the next time the corpus is rebuilt);
- MCP adapter end-to-end inside a real Claude Code session (you need to connect it once in the app yourself; the tool logic is already covered by adapter×live-daemon smoke testing).
