# scripts/ —— corpus building + tools (offline/batch)

Auxiliary scripts that produce MinerU parsing artifacts consumable by `custodian index`. **Offline tools, not runtime components**.

## Parsing pipeline

```
External dataset       scripts/select_sample.py        custodian parse                 custodian index
knowledge-base/  ──▶  sample + balance across 3 accounts ──▶  MinerU batch parsing + download ──▶  build index
  datasets/           write sample_manifest.csv          parsed/<doc_id>/
```

- **`select_sample.py`** — one-off corpus construction: stratified sampling from an external dataset (env `CUSTODIAN_KB_ROOT`), copied into `corpus/<type>/`,
  balanced across 3 MinerU accounts by effective page count, writes `sample_manifest.csv`. Page-count cache at `config/mmdocir_pagecount.json`.
- **`parse_office.py`** — docx/pptx/xlsx go through the MinerU source-code backend (env `MINERU_REPO` points at the MinerU source repo),
  parses `corpus_multiformat/` → `parsed_office/`.
- **`bench.py`** — small retrieval benchmarking tool.

> **Batch PDF parsing has been productized as the first-class command [`custodian parse`](../src/custodian/parser.py)** (the former `parse_batch.py` + `mineru_client.py` have been merged into it):
> ```bash
> custodian parse [--manifest sample_manifest.csv] [--dest $CUSTODIAN_CORPUS_DIR] [--corpus-root <repo root>]
> ```
> Reads `MINERU_TOKEN_A/B/C` (custodian/.env), batches calls to the MinerU v4 API by (account, language), uploads concurrently, polls, downloads and extracts to
> `<dest>/<doc_id>/`, and can resume (already-parsed docs are skipped). Output defaults to `CUSTODIAN_CORPUS_DIR`, falling back to repo root `parsed/`.

## Dependencies

`pip install -e '.[parse]'` (pypdf / requests / python-dotenv). Office parsing additionally requires the MinerU source repo.
Large artifacts such as `corpus/ parsed/ parsed_office/` are all gitignored (regenerable, kept out of the repo).
