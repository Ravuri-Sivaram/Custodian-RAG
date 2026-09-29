# docs/components — Component deep-dive docs (original text from the engine era)

Design/API/evaluation docs for the `chunker` / `embedder` / `generator` / MCP components. When the engine was folded into this repo, these docs were **kept largely as-is** (only cross-repo dead links and namespacing were fixed).

> ⚠ **References from the engine era were not carried over during the migration**: these docs were written while the components were still standalone packages, and the verification/demo scripts they mention — `examples/run_mineru.py`,
> `scratchpad/verify_dense.py`, `scratchpad/diag_acl.py`, `verify_seal4`, `examples/smoke_deepseek.py`, `e2e*.py`,
> and so on — **were not migrated into this repo** (they're artifacts of the engine era). The current runnable equivalents are:
> - Component unit tests → `tests/engine/` (chunker's `test_core`/`test_table`, embedder's `test_acl`/`test_sparse`/`test_store`/`test_retrieve`, generator's `test_generate`/`test_prompt`, MCP's `test_tools`)
> - Index building → `custodian index`; parsing → `scripts/` (see `scripts/README.md`); end-to-end eval → `eval/` (see `eval/README.md`)
> - To reproduce the results of the specific verification scripts mentioned in the text, check the engine repo `chunk-test-repo` (see [../PROVENANCE.md](../PROVENANCE.md) for provenance).

Document index: `chunker/{README,API,ARCHITECTURE,INTEGRATION}` · `embedder/{README,DESIGN,EVALUATION}` ·
`generator/{README,DESIGN}` · `mcp-server.md`. See [../methodology/](../methodology/) for the design lineage.
