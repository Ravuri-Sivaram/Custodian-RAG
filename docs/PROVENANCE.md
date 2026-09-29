# PROVENANCE — Provenance of the Folded-In Engine

This repo's RAG engine (`src/chunker` / `src/embedder` / `src/generator` / `src/custodian/toolcore.py` /
`src/custodian/mcp_stdio.py`), its evaluation harness (`eval/`), and its component/methodology documentation (`docs/components/`, `docs/methodology/`,
`docs/archive/`) are a **clean copy** from the engine repo `chunk-test-repo`:

- **Source commit**: `chunk-test-repo@55bf854` (`docs(OVERVIEW): remove the 'personal' framing; 72→88 evaluation; add pointers to the Custodian productization`)
- **Method**: a clean copy (not a git subtree/filter-repo) — the engine's file-by-file git history **stays in chunk-test-repo**; this repo carries none of its blame.
- **After the copy, the engine repo was archived read-only** and is no longer developed.

## History = Removing a Boundary

Historically, custodian was the engine repo's **thin product shell**, consuming the sibling repo chunk-test-repo at runtime via a path dependency, constrained by the hard rule
**D12 "zero changes to the engine"**, which split the system across two repos. This migration **actively lifts D12**, folding the engine in completely, so custodian becomes a
self-contained, single-repo, complete RAG system. The old `bootstrap`/`load_toolcore` (the version guard from commit `7fbf709`)/`CUSTODIAN_ENGINE`/
`RAG_*` namespace have all been removed; see the individual commits for details.

## Files Carrying Load-Bearing Fixes (Detailed Blame Lives in the Engine Repo)

These files carry load-bearing correctness/security fixes; to trace the line-by-line rationale, check `chunk-test-repo`'s history:

- `src/embedder/acl.py` — the fail-closed ACL predicates (`acl_admits`/`acl_split`).
- `src/embedder/store.py` / `src/embedder/retrieve.py` — the RRF-fusion push-down fix for the embedded Qdrant's "drops top-level should" fail-open bug;
  asset chunks are exempted from section deduplication.
- `src/generator/generate.py` — feeding `content_raw` back to the LLM when an asset is hit (the table/numeric grounding fix).
- `src/generator/signals.py` — the single source of truth shared by smart-ask and `eval --smart-tables` (prevents drift).

## Things Not Migrated

- **Large, regenerable data** (~3.2GB: `corpus/ parsed/ chunks*/ …` plus the built index `~/rag_real`) — left outside the repo; can be rebuilt from the source PDFs via MinerU, pointed to by config (`CUSTODIAN_CORPUS_DIR` / `CUSTODIAN_INDEX_DIR`).
- **Private eval artifacts** (`gold*.jsonl` / `results_*.json` / `verdicts.json` / `baseline_*.json` / `_judge/` / `_units/`) — contain research-report excerpts, gitignored, rebuilt by just running the scripts.
- The engine repo's `index_real.py` / `index_demo.py` (already superseded, productized as `src/custodian/indexer.py`).
- Each engine package's `examples/`, `scratchpad/` verification/demo scripts (`run_mineru.py` / `verify_dense.py` / `smoke_deepseek.py` /
  `diag_acl.py` / `e2e*.py`, etc.) and `mcp_server/AGENTIC_{REVIEW_LOG,TODO}.md` — artifacts from the engine era, not migrated. Passages in `docs/components/`
  referencing them are original text from the engine era; to reproduce the corresponding verification, check the engine repo, or look at the corresponding unit tests under `tests/engine/`.

## One Deliberate Default-Value Change (stdio-direct)

The engine's stdio server fell back to `EmbedConfig`'s defaults when no env was set (`~/qdrant_data` / collection `rag_chunks`); after folding in, `custodian mcp --direct`
now falls back to custodian's own configuration defaults (`~/rag_real` / collection `real`, matching the daemon). **If an old deployment relied on the `~/qdrant_data`/`rag_chunks` defaults,
you need to explicitly set `CUSTODIAN_INDEX_DIR`/`CUSTODIAN_COLLECTION` (or use the `RAG_*` alias).**
