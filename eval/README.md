# eval — End-to-end / agentic RAG evaluation loop (Batch 7)

Answers one question: **is this RAG actually getting answers right, is it hallucinating, is retrieval pulling in what it should, and does ACL isolation hold up?**
Unlike [`eval/component_retrieval`](component_retrieval) (which only tests the **retrieval component**'s MRR/Recall),
this directory tests the **whole chain** (retrieve → generate → cite → ACL), and does it **fully automatically**:
synthesize gold, run the system, let an LLM judge — zero manual labeling.

## Migration notes (after folding the engine into the single custodian repo)

- **Three separate evaluation axes, kept in separate directories so they don't get mixed up**: this directory `eval/` = five end-to-end
  metrics (the crown jewel); `eval/component_retrieval/` = retrieval component (BM25/BGE/RRF, MRR/Recall);
  `eval/component_chunking/` = chunking evidence preservation (vs MMDocIR, requires bringing your own external
  annotations `MMDocIR_annotations.jsonl`).
- **Environment variables unified as `CUSTODIAN_EVAL_*`** (`CUSTODIAN_EVAL_SRC/COLLECTION/GEN_MODEL/JUDGE_MODEL`); `RAG_EVAL_*` is
  kept as one deprecated alias generation. `.env` is automatically read from the custodian repo root's `.env` (the same
  one the daemon uses). ⚠ `CUSTODIAN_EVAL_GEN_MODEL` **must match the production `CUSTODIAN_LLM_MODEL`**.
- **Stop the live daemon before running GPU eval**: `sudo systemctl stop custodian` (avoids contending with the
  daemon for GPU rerank and causing OOM); the embedded Qdrant single-client lock is worked around by `_common.copy_demo`
  copytree-ing into a temp library (doesn't contend for the lock, doesn't pollute the original library).
- **Two tiers of reproducibility**: **Tier1** (reproducible within the repo) = `run_eval --judge deepseek` self-judging
  (same vendor, useful for trends); **Tier2** (authoritative, **not reproducible within the repo**) = dual-Claude
  cross-vendor judging, requires Claude Code multi-agent orchestration to produce `verdicts.json` (steps 3/6, not
  checked into the repo).

## Scripts

| Script | What it does | Dependencies |
|---|---|---|
| `gen_gold.py` | **7.0 (fast version)** samples chunks from the demo index, has DeepSeek generate a (question, golden answer) pair per chunk → `gold.jsonl` | CPU + DeepSeek |
| `run_eval.py` | **7.A/7.B** runs the system to produce four metrics + single-hop vs agentic dual-layer attribution (`--judge none` emits rows for a Claude judge) | GPU + DeepSeek |
| `acl_regression.py` | **7.C** builds a 2-tenant synthetic library, asserts **zero recall** of restricted content for cross-tenant/no-permission/unset identities | GPU |
| `index_eval_corpus.py` | Expand corpus: 15 documents from parsed/ → a larger evaluation library `~/rag_eval_big` | GPU |
| `dump_chunks.py` | Scan library and sample → `_units/` (input for Claude sub-agents to construct gold) | CPU |
| `dump_judge_units.py` | `results_*.json` → `_judge/` (input for the Claude judge agent) | CPU |
| `aggregate.py` | Merges programmatic metrics + Claude verdicts, splits by hop + dual-layer attribution | CPU |

> See the "debias full pipeline" section below for the complete flow (both gold and judging use Claude sub-agents; DeepSeek is only the system under test).

## Four metrics (run_eval)

- **Retrieval recall@k / MRR** (component layer): was the golden chunk retrieved, and at what rank — retrieval only, not generation.
- **Citation recall**: did the answer **actually cite** the golden chunk (`[cite:n]` → chunk_id) — grounded evidence.
- **Faithfulness** (LLM-judge): is every claim in the answer supported by the context — guards against hallucination. A reasonable refusal ("insufficient information") counts as faithful.
- **Correctness** (LLM-judge): does the answer match the golden answer factually.

## Dual-layer attribution (7.B, `--mode both`)

Separates "credit belonging to the retrieval component" from "credit belonging to the agent's multi-hop cleanup":

- **single**: closed-pipeline single hop (retrieve once → generate) = the component-layer baseline.
- **agentic**: DeepSeek judges whether the context is sufficient; if not, it **rewrites the query and searches again**, accumulating context before generating = simulated agent.
- **Δ = agentic − single**: a positive `correctness Δ` means the agent's multi-hop genuinely closed a gap left by single-hop retrieval; a rise in `avg_rounds` = the extra retrieval cost spent.

> This column is the core evidence for agentic RAG: if Δ≈0, single-hop retrieval was already good enough and agent multi-hop is wasted effort; only a clearly positive Δ proves agent-driven search is worth it.

## How to run

```bash
conda activate custodian
python eval/gen_gold.py --per-doc 6        # generate gold.jsonl (~24 entries)
python eval/run_eval.py --mode both        # four metrics + dual-layer attribution; results land in results_single.json / results_agentic.json
python eval/acl_regression.py              # ACL isolation regression; exit code 0 = no leakage
```

Optional: `--top-k 6`, `--rerank` (enable cross-encoder reranking, +16G VRAM), `--rounds 2` (agentic max rounds), `--limit N` (smoke test).

## Debias full pipeline (cross-vendor judge + expanded corpus + multi-hop) — recommended

`run_eval --judge deepseek` is DeepSeek self-judging, which carries same-vendor circular bias. **To get numbers you can
stand behind**, use this path instead: both gold and judging use **Claude sub-agents** (different vendor, frontier
model, data never leaves the Claude Code trust boundary), DeepSeek is **only the system under test**.

```bash
python eval/index_eval_corpus.py                                      # 1. Expand corpus: 15 docs from parsed/ -> ~/rag_eval_big
CUSTODIAN_EVAL_SRC=~/rag_eval_big CUSTODIAN_EVAL_COLLECTION=evalbig python eval/dump_chunks.py   # 2. Sample -> _units/ (for the gold agent)
#  3. Orchestrator (Claude) starts a Workflow: sub-agents read _units/ and construct gold (single-hop + single-doc multi-hop + cross-document) -> eval/gold.jsonl
CUSTODIAN_EVAL_SRC=~/rag_eval_big CUSTODIAN_EVAL_COLLECTION=evalbig python eval/run_eval.py --mode both --judge none   # 4. Run the system to produce rows
python eval/dump_judge_units.py                                       # 5. rows -> _judge/ (for the judge agent)
#  6. Orchestrator starts a Workflow: 2 independent Claude judge passes decide faithfulness/correctness -> eval/verdicts.json
python eval/aggregate.py                                              # 7. Merge + split by hop + dual-layer attribution
```

> The Workflow scripts for steps 3/6 are not in the repo (the orchestrator generates them on demand as needed); everything else is checked in. This is multi-agent orchestration within Claude Code — a different environment will need to bring its own cross-vendor judge.

## Authoritative run (2026-06-30, **after R5 methodology correction** / 15 docs / gold 72 [single-hop 38 / single-doc multi-hop 29 / cross-document 5] / full-context Claude judge)

> ⚠ **R5 self-review correction (important)**: the earlier reported "faithfulness 0.83 / 17% unsupported claims" was
> an **eval bug** — `dump_judge_units` truncated the context fed to the judge to 5000 characters (actual median was
> 16k), so the judge only saw about 40% of the passages → claims grounded in later passages were wrongly marked
> unfaithful. After removing the truncation and having the judge re-judge with the **full context**, **actual
> faithfulness ≈ 1.0**. The table below shows the corrected values, reproducible via `aggregate.py` (which includes
> fingerprint verification and paired attribution).

| Metric | single | agentic | decompose |
|---|---|---|---|
| **Faithfulness** | **1.000** | **1.000** | 0.972 |
| Correctness | 0.847 | 0.750 | 0.831 |
| Retrieval recall | 0.854 | 0.840 | 0.852 |
| Full recall | 0.792 | 0.792 | 0.792 |
| Citation recall | 0.750 | 0.729 | 0.771 |
| Avg. rounds | 1.00 | 1.22 | 1.78 |

Per-hop correctness (AND): single-hop 0.97/0.87/0.92; single-doc multi-hop 0.83/0.72/0.82; cross-document multi-hop 0.00/0.00/0.20 (n=5).
Dual-layer attribution (paired, common judged question set): single→agentic Δ**−0.097** (n=72); single→decompose Δ**−0.014** (n=71).
> Full-context **dual-judge AND** (pass1+pass2, all 216; decompose missing 1 verdict, judged 71/72). Reproducible via `aggregate.py`, with fingerprint verification.

**Three findings after the correction:**

1. **Faithfulness ≈ 1.0, the grounding contract is nearly watertight.** The earlier "17% hallucination" was a
   CTX_CAP truncation artifact, not real hallucination — **this overturns finding B1's conclusion that "17% unsupported
   claims is an open item."** The system would rather decline to answer than make something up, a much stronger safety
   property than previously believed. **Lesson: a bug in the evaluation pipeline itself can manufacture a "finding" out of thin air.**
2. **Correctness bottleneck = numbers in tables (now largely mitigated) + cross-document synthesis (still hard).**
   Table numbers were fixed by ③ up to single-hop 0.97; the remaining hard problem is multi_cross (0.00, stuck on
   synthesis rather than retrieval, n=5 is small).
3. **agentic/decompose net negative (now paired, common question set, cleaner):** single beats agentic on every hop
   (Δ−0.083); decompose only has a slight edge on cross-document (correctness 0.20 vs 0, retrieval 0.37 vs 0.20),
   overall Δ−0.014. **The closed pipeline should be the default.**

## ③ Table/numeric grounding fix (2026-06-30, see git log for the commit)

**Diagnosis** (traced question-by-question, overturning the initial hypothesis that "content_raw wasn't being fed
in"): of 21 correctness failures, 14 were "retrieved it but still got it wrong," and the real root cause was
**(a)** `search_with_context`'s per-section deduplication was **folding chart/table asset chunks into their prose
siblings**; **(b)** `assemble_big`'s big-block assembly uses `_gather`, which only takes `el.text/caption` and
**does not include the asset's `asset_content`**. The numbers inside charts/tables (living in `chunk.content_raw`)
were both dropped by dedup and excluded from assembly — "retrieved" effectively meant "not retrieved."

**Two fixes**: in `embedder/retrieve.py`, asset chunks (chart/table) no longer participate in per-section dedup
(they get their own key by chunk_id); in `generator/generate.py`, when an asset is hit, `content_raw` is added back
into what's fed to the LLM (the big-block was missing it).

**Effect** (single, Claude single-pass judge, apples-to-apples vs the pre-fix pass1):

| | Before fix | After fix |
|---|---|---|
| Correctness | 0.704 | **0.819 (+0.115)** |
| └ single-hop | — | 0.97 |
| └ multi_intra | — | 0.76 |
| Retrieval recall (agentic) | 0.764 | 0.840 |

Targeted before/after: all 4 known table questions (photoresist market share / NAND spot price / FLAN-T5 comparison
score / token consumption) went from "insufficient information" to the correct numeric answer.

**Faithfulness prompt attempt = negative result, and the problem didn't actually exist**: we once tightened the
grounding prompt to try to close the "17% unsupported claims" gap; this backfired in practice (−0.12 faithfulness /
−0.04 correctness) and was reverted. **After the R5 self-review: that 17% was itself a CTX_CAP truncation artifact**
(the judge never saw the full context) — actual faithfulness ≈ 1.0, **there was no gap to fix at all.** Double
lesson: ① confirm a problem actually exists before touching anything (the prompt attempt was wasted effort);
② a bug in the evaluation pipeline itself can manufacture a false "finding" out of thin air.

## decompose multi-hop experiment (B2/B3, 2026-06-30) — negative result, kept for the record

To address weak cross-doc retrieval (0.20) and weak multi_intra, we added `--mode decompose`: **split into
sub-questions → retrieve each → union → synthesize** (distinct from agentic's "rewrite and replace the query"
approach). Three-way comparison (after R5 correction, full-context single-pass judge, 72 questions, reproducible via
`aggregate.py`):

| Correctness | single | agentic | decompose |
|---|---|---|---|
| single-hop (38) | **0.97** | 0.87 | 0.92 |
| multi_intra (29) | **0.83** | 0.72 | 0.82 |
| multi_cross (5) | 0.00 | 0.00 | **0.20** |
| **Overall** | **0.847** | 0.750 | 0.831 |
| Avg. retrieval rounds | 1.0 | 1.22 | 1.78 |

**Conclusion: the single-hop closed pipeline is both the best overall and the cheapest** (Δ vs agentic −0.097, vs
decompose −0.014, paired). agentic is ≤ single on every hop (more retrieval = more distractor dilution); decompose
only has a slight edge on cross-doc (correctness 0.20 vs 0, retrieval 0.37 vs 0.20), and is still overall worse than
single. Cross-doc is stuck on **synthesis** (even getting chunks from both documents doesn't produce a comparison),
not retrieval. **On this workload (mostly single-hop, table-dense), agent orchestration is a net negative; the closed
pipeline should be the default.** (cross-doc n=5 is small, treat absolute values as reference only.)

## gold expansion: added table questions (2026-07-03, gold 72 → 88)

**Motivation**: the original 72 questions were all sampled from prose chunks (`gen_gold.py` explicitly SKIPped
tables at the time — back then table chunk text was caption-only, so no question could be formed). In practice this
meant: the table-retrieval text enhancement (chunker `2bd97a5`) that fixed the real question category of "numbers
buried in a table" from wrong to correct only showed up as cost (3 gold questions shifted around as noise) and no
benefit on a pure-prose gold set — **the exam was lopsided, and table-facing changes had no symmetric metric to
validate against.**

**Method** (`gen_gold_tables.py`): targeted sampling of chunks with kind=table and content_raw≥120 characters (full
coverage of all 15 documents, ≤2 per document); DeepSeek generates questions from the table body HTML that "must be
answered using values inside the table"; **programmatic QC: every number in the answer must appear verbatim in the
table body**, otherwise the question is discarded (this caught 2 hallucinated questions during the actual run).
Produced 16 questions (mixed-language, non-English/English) appended to gold.jsonl; the original 72 questions are backed up as
`gold_backup_prose72.jsonl`, and the table questions are kept separately as `gold_tables.jsonl` (tagged
`asset:"table"` so they can be split out for separate statistics).

⚠ **Baseline discontinuity**: aggregate numbers from the 88-question set onward are **not directly comparable** to
the historical 72-question numbers; see the new baseline in docs/TESTING.md §3.

## Honest warnings (read before trusting any number)

1. **Judge choice + full context**: the table above uses **Claude sub-agent judging, fed the full context**
   (cross-vendor debiasing). **Lesson (R5)**: earlier, the judge was fed context truncated to 5000 characters →
   faithfulness was artificially pushed down to 0.83, giving the false impression of "17% hallucination"; with full
   context it's ≈ 1.0. The faithfulness judge **must see every passage the generator actually used.**
   `run_eval --judge deepseek` is the fast version with DeepSeek self-judging (same-vendor circular bias), useful
   only for relative trends.
2. **Gold is generated by Claude**: single-hop questions are sourced from a single chunk, so the golden answer is
   unambiguous and citation can be judged programmatically; multi-hop questions are constructed by Claude across
   chunks/documents. There are few cross-document multi-hop samples (n=5), so treat absolute values as reference only.
3. **Corpus of 15 documents** (5 English papers + 4 English earnings reports + 6 non-English research reports, table-dense
   to stress-test the numeric bottleneck); conclusions describe "performance on this particular corpus."
4. **gold/results/intermediate artifacts are not checked into the repo**: they contain document excerpts (which may
   include private data from research reports) and can be regenerated, so they're gitignored; scripts are checked in
   and rebuild everything on a single run.
