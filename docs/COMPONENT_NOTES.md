# Known Decisions and History for the Engine Components (post-merge, these were once cross-repo review records)

> **Historical note**: N1–N8 and the P1 review below were originally cross-repo records of "objections to / fixes for / deliberate non-fixes of" an independent engine repo.
> The engine has since been folded into the single custodian repo (`src/{chunker,embedder,generator,custodian}`), so that two-repo boundary no longer exists.
> These entries are kept as a **history of the engine's internal evolution and its rationale**; wording that referred to "the other repo / contract drift" has been corrected to reflect the current single-repo reality.

> Convention: **objections, fixes, and deliberate non-fixes** for engine components are recorded here, each with motivation + action + verification.
> The corresponding changes have matching commits in this repo's git history.

> **Note (this project):** the non-English-phrased anecdotes below (the motivating regression case, the cross-language table-ranking
> tests) predate this project's replacement of the project's original language support with Telugu (see the top-level README). Kept as
> genuine historical record, not re-measured against Telugu.

## Fixed

### N1: MCP-server tool semantics were coupled to the stdio transport — split out toolcore

- **Objection**: the six tools' validation/structured-result construction/deduplication/budgeting/error mapping (a core asset refined over five rounds of review, R1–R5)
  all lived inside the stdio server, bound to MCPServer. If Custodian's HTTP endpoints copied this logic, the contract would inevitably drift
  (the same already_returned semantics implemented in two places, one gets fixed and the other forgotten).
- **Action**: a pure move-and-split into `toolcore.py` (transport-agnostic, pure stdlib, retriever/user injected as dependencies),
  with the stdio side (now `src/custodian/mcp_stdio.py`) keeping its stdio bindings plus an explicit re-export. **Zero logic changes.**
  - **Verification**: the original test_tools.py passed all 22 items green with **not a single line changed**, plus embedder test_store's 7 items all green (both now live under `tests/engine/`).
- **No longer applicable after the merge**: the original stdio server used to insert its own directory into sys.path (so that toolcore could be resolved when imported externally).
  Under the single-repo src-layout (`pip install -e .`, import names unchanged), that self-insertion into the path is no longer needed and has been removed accordingly — n/a.

### N2: `_RETURNED_KEYS` deduplication at process scope breaks across multiple sessions — Custodian makes good on per-session isolation (a pitfall the design flagged in advance)

- **Objection**: the stdio server's own comment on this already said "before switching to a multi-session HTTP/SSE transport, this must become per-(session, user)
  isolated." This isn't a newly discovered bug so much as Custodian being exactly that "moment of switching transports," where **it must be made good on, or it's a real leak**:
  a segment session A already retrieved could get mislabeled already_returned for session B (empty body), even though B never actually received it.
- **Action**: Custodian's `sessions.py` (a bounded-LRU SessionRegistry) + an opt-in `X-Custodian-Session` header;
  toolcore's `returned_keys` was already parameterized, so toolcore itself needed zero changes.
- **Verification**: CPU tests (same-session dedup / cross-session isolation / no header means no dedup) + GPU smoke testing (smoke1 all
  came back already_returned, smoke2 was fully isolated).

### N7: Numeric-range wrong answer (a segment figure extrapolated as the total) — fixed in the engine generator (2026-07-03)

- **Objection**: the one confirmed case of "a confidently wrong answer" in real testing (segment revenue mistaken for total revenue, carrying a real citation, extremely hard to notice). Diagnosis found **two root causes**:
  ① the prompt had no numeric-range constraint; ② the range evidence (the section path) was never even in the prompt — the table's body contained not a single word naming the segment,
  so the model had no way to judge. Fixing only ① was measured to have no effect; only once both ① and ② were fixed did it work.
- **Action**: added a narrowly-targeted numeric-range constraint to SYSTEM (as opposed to the blanket tightening in R3 that was reverted); merged the context's source line into the
  section_path breadcrumb (as a side benefit, this also improved traceability quality across the board).
- **Verification**: all three checks passed — the wrong-answer case now reports "range noted, declines to extrapolate"; correct cases showed no collateral damage;
  a before/after comparison with the same DeepSeek judge on 72 questions showed faithfulness 0.972→1.000, correctness unchanged. See custodian TESTING §3 for detail.

### N8: Table-chunk retrieval-text enhancement — fixed in the engine chunker (2026-07-03, `2bd97a5`)

- **Objection**: a table chunk's only retrievable signal was a single caption sentence — the table's actual semantics (column headers/row labels) were entirely locked away in content_raw
  and unretrievable — numeric questions about a table got crowded out of the top-k by prose (the retrieval-side root cause behind the N3/N7 cases).
- **Action**: `_table_signal` (headers + row labels, data cells excluded) spliced into the retrievable text alongside the breadcrumb;
  an **existence gate** to prevent ghost chunks (without the gate this would add +23 chunks and shift chunk ids — caught and fixed on the spot during the rebuild, detail in
  TESTING §3). Both indexes were fully rebuilt.
- **Verification**: 44 chunker tests; on the 72-question regression, correctness held steady, 3 questions shifted due to redundant gold answers (all still answered correctly),
  and the motivating case was answered correctly directly in its original non-English phrasing. Verdict: kept (from a damage-severity view, unlocking a wrong-answer category outweighs the redundant-recall displacement).
- **Related finding**: the gold set had no table questions — an evaluation blind spot (filed as TODO: targeted gold-question generation).

## Deliberately Not Fixed (Recorded, Open to Reconsideration)

### N3: `Generator.answer` didn't support retrieval filters — **actually landed on 2026-07-02** (status changed from not-fixed to fixed)

The original verdict was "cover it with the agentic path for now, TODO." Real usage from a user exposed the actual pain point: for questions like "Netflix 2015 revenue" where
**the number is buried in a table**, ordinary phrasing gets the table chunk crowded out of the top-k by MD&A prose, and neither top_k nor rerank could save it,
while a `kind=table` filter hits it directly (the number is in the p.16 Selected Financial Data table, confirmed present in the library).
**Action**: the engine's Generator.answer gained optional doc_ids/doc_type/kind/strategy parameters — **passed only when needed**
(unset parameters don't appear in the call at all), so the older narrow-signature retriever (used in unit-test mocks/smoke tests) and existing call sites were unaffected;
eval only uses the top_k/rerank keywords (confirmed at run_eval.py:80), so it wasn't touched and the 72-question evaluation wasn't re-run.
`/v1/ask` and `custodian ask --kind/--doc-type/--doc-id/--strategy` pass these through.
**Verification** (as measured at the time, before the single-repo pytest unification, counted separately by repo):
generator side, 17 tests (+1 for the pass-through/narrow-signature compatibility), product side, 38 tests (+2). Real-index confirmation (reported honestly):
English "total revenue" phrasing + `--kind table --rerank` → **correctly answers $6,779,511 thousand, citing the p.16 summary table**;
pure non-English phrasing with kind=table + rerank → still fails to retrieve the p.16 table but **honestly refuses** (explicitly states it only sees segment data);
⚠ non-English + kind=table + top_k 15 **without rerank** once mistakenly answered the segment revenue (4,180,339) as the company's total revenue —
the remaining gap for cross-language numeric questions and usage guidance are in the TODO (P2) and TESTING §3. **Recommended approach for numeric questions:
`--kind table --rerank` plus keywords in the document's own language (e.g. "total revenues" for an English financial report)**.

### N4: index_real.py has hardcoded paths/ACLs — leave the script alone, productize as `custodian index` instead

The engine's index_real.py is kept as a historical script (it's what built the current ~/rag_real); Custodian's indexer.py parameterizes
corpus/dest/collection/ACL fully. **Reason for not touching the original script**: it's a reproducible record of "how the index was built at the time."

### N5: stdio-direct mode is kept, not retired

When the daemon isn't running, `custodian mcp --direct` (stdio direct-connect, no daemon) is still available (at the cost of re-paying the model load every session).
It cannot open the same index at the same time as the `custodian serve` daemon (single-client lock) — already documented. **Reason for not retiring it**:
it's the only consumption path that doesn't depend on the daemon, kept as a fallback path. (After the single-repo merge, all three entry points share the same toolcore:
`custodian serve` HTTP daemon / `custodian mcp` stdio→HTTP adapter / `custodian mcp --direct` stdio direct-connect.)

### N6: `OpenAICompatibleLLM` has no retry logic

Occasional DeepSeek 5xx/timeout errors turn /v1/ask directly into ask_failed (retriable=true, the client can retry).
**Reason for not fixing this**: putting the retry on the client side is semantically clearer (answers aren't idempotent, so a server-side automatic retry would just double the latency and cost);
if measured failure rates turn out to be high, add a bounded retry then — filed as TODO (P3).

## Adversarial Review P1 (2026-07-02) Results and Disposition

Three perspectives (security/correctness · concurrency/contract · documentation) × 2 counter-reviewers per finding. 39 agents; some verification subtasks ran out of session budget,
and the 13 unverified findings were cross-checked one by one against the source code by the main line (all confirmed genuine). **All fixes were pinned with regression tests** (tests/test_review_fixes.py).

**Confirmed (by review)**
- C1 indexer: building an index with `restricted` and an empty `allow` succeeded, but no identity could then retrieve anything, with zero warning → the indexing entry point now explicitly rejects this;
- C2 the no_identity hint told the user to set `RAG_TENANT` when Custodian only reads `CUSTODIAN_TENANT` (following the hint still leaves everything empty — a misleading dead loop)
  → the service's binding layer now translates the contract text into the actual CUSTODIAN_* config name. (After the single-repo merge the namespace is unified to `CUSTODIAN_*`,
  with `RAG_*` kept only as a single-generation DEPRECATED alias, so the root cause of this error is gone.)

**Self-verified as genuine and fixed** (the ones verification agents didn't finish checking): the adapter's doc_id had no URL encoding (`#`/`/` would get truncated/misroute) → wrapped in quote;
an empty doc_id hit a different route (307) → now rejected locally as bad_arg, plus a 3xx branch added to `_call`; under concurrency, /v1/ask's shared LLM's
last_finish_reason cross-contaminated between requests → Generator changed to per-thread (threading.local, no concurrency window within one thread);
the Generator factory only caught ValueError (a missing openai package, etc. threw a raw 500) → now falls back to ask_failed; .env inline comments/quotes weren't stripped, plus
int() with no guard crashed startup → fixed via _parse_env_value/_int_env; CUSTODIAN_QDRANT_PATH/SIDECAR_DIR overrides weren't expanduser'd
(a literal "./~" opened an empty index) → expanduser added; the CLI's --url flag skipped .env loading (losing the API key, causing permanent 401s) → _client now always loads env;
3 places where the adapter's and stdio's docstrings differed → aligned verbatim plus a same-text regression test (after the single-repo merge this contract is now backed by a **structural test**:
adapter vs. mcp_stdio docstrings equal + `_INSTRUCTIONS` single-sourced from toolcore); the stdio server's list_documents docstring was missing coverage → added;
the mcp-server documentation didn't mention the toolcore layering or the daemon/lock mutual exclusion → added (this is what N5 claims was implemented);
API.md's meta was missing requested_k/rerank, and documents was missing retriable → added; .env.example was missing 4 actually-read variables → added.

**Refuted, not fixed (recorded)**: the API key comparison isn't constant-time (timing can't be measured over plain HTTP on a LAN, and anyone who could measure it is already inside the trust boundary);
X-Custodian-Session collision griefing (under the single-identity model this is the same boundary, and there are stronger attack vectors available with no marginal capability gained); /v1/ask citations lack an
untrusted marker (the only consumer, the CLI, only prints the metadata, with no downstream LLM path; citation raw text isn't returned by default).

## Observed (No Action Taken)

- FastAPI's `on_event` is deprecated → Custodian uses lifespan directly instead (this is our own code, not a component concern).
- toolcore's delivery budget historically read the `RAG_MAX_CONTEXT_TOKENS` environment variable; after the single-repo namespace unification, it now reads
  `CUSTODIAN_MAX_CONTEXT_TOKENS` (`RAG_*` kept as a DEPRECATED alias for one generation), realized as that env var at create_app time.
  Turning it into a proper function parameter would be cleaner, but that would change toolcore's signature, and the benefit doesn't outweigh the churn — left as is.
