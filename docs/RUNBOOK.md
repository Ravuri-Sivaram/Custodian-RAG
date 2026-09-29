# RUNBOOK —— Local Startup and Operations Quick Reference

custodian runs on **WSL Ubuntu + conda env `custodian`** (already includes the GPU stack: torch cu128 / transformers / qdrant-client / jieba / mcp).
The commands below all assume you have already run `conda activate custodian` and `cd`'d into the custodian repo root. For operational details, see [OPERATIONS.md](OPERATIONS.md).

## 0. One-time setup

```bash
conda activate custodian
cd <custodian repo root>                     # local clone path
pip install -e '.[dev]'              # src-layout editable install (the engine is folded in, import name unchanged)
cp .env.example .env                 # fill in DEEPSEEK_API_KEY; for the sample library use CUSTODIAN_TENANT=demo
```

> The GPU models (Qwen3-VL-Embedding-8B / reranker) are downloaded via modelscope to `~/models`, and lazily loaded on the first `retrieve` call (about 1-2 minutes).
> For non-4090 machines / models stored elsewhere: set `CUSTODIAN_GPU_NAME` (a substring of the torch device 0 card name; leave empty to skip validation), `CUSTODIAN_DENSE_MODEL_PATH` / `CUSTODIAN_RERANK_MODEL_PATH` (must contain the official `scripts/`; if missing, `custodian serve` will fail with a clear error on startup).

## 1. Start / stop the daemon (holds the GPU exclusively + embedded Qdrant, runs persistently)

```bash
# Method A: systemd (starts on boot, recommended; this is what the rag MCP connects to)
sudo systemctl start custodian          # stop: stop | restart: restart | logs: journalctl -u custodian -f
sudo systemctl status custodian

# Method B: manual foreground (for debugging)
python -m custodian serve               # http://127.0.0.1:8787
```

## 2. Verify

```bash
python -m custodian health              # {status:ok, collection:real, tenant_bound:true, llm_model:..., identity_mode:...}
```

## 3. Everyday use

```bash
# Closed-pipeline Q&A (retrieval → grounding → DeepSeek → with citations)
python -m custodian ask "What content is in the library about X?"
python -m custodian ask "What was net profit in 2021?" --kind table       # for numeric questions, retrieve only from table blocks
python -m custodian ask "..." --rerank --strategy sparse         # reranking / pure keyword routing

# Claude Code agentic: the rag tool in .mcp.json = custodian's mcp thin adapter (connects instantly to the warm backend)
# Fallback when there's no daemon running: it loads the GPU model itself, so the first query is slow
python -m custodian mcp --direct
```

## 4. Building the library (adding new documents; requires stopping the daemon first — the embedded Qdrant has a single-client lock)

```bash
sudo systemctl stop custodian
# corpus = the directory of MinerU-parsed output (one <doc_type>__<name>/ per document, containing content_list.json + layout.json)
python -m custodian index --corpus <parsed_dir> --dest ~/rag_real
sudo systemctl start custodian
```

> To generate parsed/ from PDFs: `custodian parse --manifest sample_manifest.csv --dest <CUSTODIAN_CORPUS_DIR>`
> (batch parsing via MinerU, requires `MINERU_TOKEN_*`; the manifest is generated with `scripts/select_sample.py` — see [../scripts/README.md](../scripts/README.md) for details).

## 5. Running eval (Tier 1: DeepSeek self-judging, reproducible within the repo; requires stopping the daemon to avoid GPU contention)

```bash
sudo systemctl stop custodian
CUSTODIAN_EVAL_SRC=~/rag_eval_big CUSTODIAN_EVAL_COLLECTION=evalbig \
  python eval/run_eval.py --mode single --judge deepseek --smart-tables --gold eval/gold.jsonl
sudo systemctl start custodian
```

> Knobs: `--mode single|agentic|decompose|both` `--top-k 6` `--rerank` `--rounds 2` `--limit N` (for smoke tests).
> Tier 2 authoritative evaluation (dual-Claude, cross-vendor judging) is not reproducible within the repo, and requires Claude Code multi-agent orchestration — see [../eval/README.md](../eval/README.md) for details.

## 6. Tests / gates

```bash
pytest -q                            # CPU gate: product-surface + engine-surface tests, run together in one pass (see docs/TESTING.md §1 for the baseline)
python eval/acl_regression.py        # GPU pre-release gate: zero ACL leakage (requires stopping the daemon)
```

## 7. Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `ask` returns `llm_unconfigured` for everything | `.env` is missing `DEEPSEEK_API_KEY` | Add the key, then `restart` |
| Retrieval returns nothing / `no_identity` | `CUSTODIAN_TENANT` not set (fail-closed) | For the sample library, set `CUSTODIAN_TENANT=demo` |
| `index` reports "already in use" | The daemon is holding the lock | Run `systemctl stop custodian` first |
| `rag` in Claude Code won't connect | The daemon isn't running | `systemctl start custodian`; or `custodian mcp --direct` |
| eval OOMs / hangs | The daemon and eval are competing for the GPU | `stop custodian` before running eval |
