# tests/ -- unified test tree

Now that the engine has been folded into custodian, what used to be "each repo runs its own
pytest" has collapsed into **one pytest suite** (for the count, see
[docs/TESTING.md §1](../docs/TESTING.md)).

## Layout

- `tests/*.py` -- product layer (service / smart / team / sessions / adapter / review_fixes),
  pure CPU (fake retriever + MockLLM, doesn't touch Qdrant/GPU/network).
- `tests/engine/*.py` -- engine-component unit tests, migrated in from the engine repo (the
  original `sys.path.insert` is gone, replaced by the installed package):
  - **chunker**: `test_core` / `test_table` (reads MinerU samples from `tests/engine/fixtures/`)
  - **embedder**: `test_acl` / `test_sparse` / `test_store` / `test_retrieve`
  - **generator**: `test_generate` / `test_prompt`
  - **MCP**: `test_tools` (via `custodian.mcp_stdio`, i.e. the folded-in stdio transport)
- `tests/engine/fixtures/` -- MinerU samples for the chunker unit tests (`sample_content_list.json`
  / `sample_layout.json`).

## Green-light gates (two tiers -- see invariant #1)

- **CPU CI gate** = `pytest` (includes the **ACL-predicate-level** assertions in embedder's
  `test_acl.py`). Must run after any import/store/namespace change.
- **GPU pre-release gate** = `python eval/acl_regression.py` (WSL + 4090; 44+ end-to-end "RRF
  fusion exit has zero leakage" assertions, **hard-depends on GPU**, not part of CPU CI).

## Running

`pip install -e '.[dev]'`, then `pytest -q` (the custodian environment already has every
dependency). No drift in the contract is enforced by
`tests/test_review_fixes.py::test_transports_contract_no_drift` (the six tools' docstrings are
word-for-word identical across the HTTP adapter and stdio transports, and `_INSTRUCTIONS` comes
from the single source of truth, `toolcore`).
