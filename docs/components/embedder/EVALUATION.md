# embedder retrieval quality evaluation

> **Note (this project):** measured against the original jieba sparse tokenizer for the project's prior non-English language and the original English/prior-language `est_tokens`
> divisors. This project later replaced that prior-language support with Telugu (see [DESIGN.md](DESIGN.md)'s note at the top); the
> prior-language-specific numbers below (e.g. the chars/token divisor for "the prior non-English research-report category") describe the old implementation and
> have not been re-measured for Telugu — the new Telugu `est_tokens` divisor is an unverified placeholder pending a real
> measurement run of `eval/component_retrieval/retok.py` against Telugu documents.

> 2026-06-28. This document records the "quality sign-off" evaluation for the embedder component: process, methodology, results, and decisions.
> See [`eval/component_retrieval/`](../../../eval/component_retrieval/) for reproducible scripts; see [DESIGN.md](DESIGN.md) for design background.

## 1. Purpose

The chunker has a data-driven chunking evaluation (`eval_chunks.py` + MMDocIR ground-truth F1). Until now, the embedder had only completed **structural/security sign-off** (the adversarial review caught 13 bugs), but lacked **data-driven validation of retrieval quality**. This evaluation fills that gap, building a retrieval benchmark from real corpora, and pins down 3 open questions with data:

1. **Sparse model choice**: BM25 vs BGE-M3-sparse — which should go into the hybrid.
2. **RRF weighting**: the optimal ratio for fusing dense and sparse + whether hybrid genuinely beats either route alone.
3. **est_tokens recalibration**: should chunker's char/token heuristic (prior non-English language 1.7, English 4.0) be recalibrated against the real Qwen3-VL tokenizer.

## 2. Methodology

### 2.1 Evaluation benchmark

- **Corpus**: 14 documents selected from `parsed/` → chunker splits them into **1067 chunks**. Covers academic_paper / law / financial_report_en / **financial_research_zh** / government, a mix of English and the project's prior non-English language (109 chunks in that prior language, filling in the chunker evaluation's blind spot for it).
- **Two query types** (the core of the design):

  | Type | Count | Construction | Golden | What it tests |
  |---|---|---|---|---|
  | **Precise-term** | 25 | Programmatically mined rare strings from the corpus with `df==1` (`Section 6.1`/`CLEF-2021`/`$1196`/long identifiers) | The single chunk containing that string | Whether sparse can match exactly |
  | **Semantic** | 31 | An agent reads a chunk and generates a natural-language question, **deliberately avoiding the original wording** (synonyms/paraphrase) | The chunk that was paraphrased | Semantic recall (recall even when phrased differently) |

- **Why both types are necessary** (to avoid evaluation bias): testing precise-term queries alone, BM25 is almost guaranteed to win (that's literally term matching); testing semantic queries alone, dense is guaranteed to win. The two types correspond to what sparse and dense are each designed for, and only together do they give a fair comparison. If a semantic query reused the original wording it would degenerate into exact matching, so the agent's prompt explicitly required "avoid proper nouns/numbers/key phrases."

### 2.2 Metrics

- **MRR** (Mean Reciprocal Rank): the mean of the reciprocal of the golden's rank — the higher, the better it's ranked.
- **Recall@10**: the fraction of cases where the golden lands in the top 10.
- Recall uses `limit=50`, finding the golden's rank within it.

### 2.3 Three-vector comparison in the same store

One Qdrant collection holds three kinds of vectors, compared in a way that stays close to the production implementation:

| Vector | Model | Similarity |
|---|---|---|
| `dense` | Qwen3-VL-Embedding-8B (MRL 1024) | COSINE |
| `bm25` | Our own `doc_sparse` (jieba+regex) | Qdrant `Modifier.IDF` (computes BM25) |
| `bgem3` | BGE-M3 `lexical_weights` | dot product (weights already encode importance) |

`#1` only compares sparse routes against each other (dense is the same route in every comparison, so it doesn't affect the sparse comparison), but dense is listed alongside as a reference point.

## 3. Process

### 3.1 Pipeline

```
build_corpus_chunks.py  → chunks.jsonl (1067) + queries_exact.jsonl (25, programmatic)
select_semantic.py      → candidates.jsonl (47 candidates)
[workflow: 4 agents]    → queries_semantic.jsonl (31, agent-generated, avoiding original wording)
index_eval.py           → indexes 1067 chunks into Qdrant (dense+bm25+bgem3)
eval.py                 → #1 single-route recall + #2 RRF weight sweep
retok.py                → #3 tokenizer recalibration (a separate pipeline: all 2927 chunks through the Qwen3-VL tokenizer to measure char/token)
```

### 3.2 Engineering snags along the way (diagnostic log)

While running indexing, we repeatedly hit apparent hangs, and we **fully worked through diagnose → fix → verify each time, rather than blindly re-running**:

- **Symptom**: the process appeared hung (GPU util 0%, CPU time not increasing, stuck at `wchan=pipe_write`); subsequent re-runs then showed `exit 9`, then `RuntimeError: ... freeze_support()`.
- **Diagnostic approach**: repeated sampling of `nvidia-smi util` + `ps -o time` (confirmed it was blocked, not computing) → `cat /proc/<pid>/wchan` (finding the wait point) → `pgrep -P` (child processes, found defunct + resource_tracker entries) → reading logs (found **the dense progress bar appeared twice** = the script had actually run twice).
- **Root cause**: BGE-M3 detected two GPUs (4090+5070) and auto-started a multiprocessing pool; the spawned child process **re-imported the script as `__main__`** (the script had no `if __name__` guard) → the entire script ran again + nested spawning → multiple child processes contending for the stdout pipe → the `pipe_write` hang.
- **Fix**: ① moved all logic into `main()` + `if __name__=='__main__'`; ② pinned BGE-M3 to `devices='cuda:0'` (a single GPU won't start a pool).
- **A side lesson**: never use `python ... | grep | tail` for long-running tasks (`tail` waits for EOF and doesn't consume the pipe, so the buffer fills and `print` hangs) — instead `python -u > file` and read the file directly.

(Preserved as repository memory `reference_bge-m3-multiprocess-reentry`.)

## 4. Results

### 4.1 #1 single-route recall (MRR / Recall@10)

| route | precise-term (25) | semantic (31) | overall (56) |
|---|---|---|---|
| **bm25**  | **0.738** / 0.96 | 0.210 / 0.45 | 0.446 / 0.68 |
| **bgem3** | 0.584 / 0.88 | 0.347 / 0.61 | 0.453 / 0.73 |
| **dense** | 0.149 / 0.24 | **0.794** / 0.94 | 0.506 / 0.62 |

- **Precise-term**: bm25 ≫ bgem3 ≫ dense. BM25 **clearly wins** over BGE-M3 on sparse's home turf (numbers/model numbers/amounts matched exactly, 0.738 vs 0.584); dense is very weak on exact numbers (0.149), as expected.
- **Semantic**: dense ≫ bgem3 > bm25. Semantics is dense's job (0.794 dominates both sparse routes); BGE-M3's lexical weighting picks up a bit of semantics (0.347 > bm25's 0.210), but nowhere near dense.
- **Overall**: dense is highest (0.506); the two sparse routes are nearly tied (0.453 vs 0.446).

### 4.2 #2 RRF weighting (dense + bm25, client-side weighted sweep)

| w_dense | overall MRR | precise-term | semantic | |
|---|---|---|---|---|
| 0.00 | 0.449 | 0.738 | 0.217 | pure sparse |
| 0.25 | 0.496 | 0.771 | 0.275 | |
| **0.40** | **0.541** | 0.781 | 0.348 | ← peak |
| 0.50 | 0.499 | 0.627 | 0.396 | equal weight |
| 0.60 | 0.450 | 0.481 | 0.425 | |
| 0.75 | 0.431 | 0.384 | 0.469 | |
| 1.00 | 0.510 | 0.158 | 0.794 | pure dense |

- **Hybrid beats any single route**: the peak 0.541 (at w_dense≈0.4) > pure dense 0.510 > pure sparse 0.449. Dense handling semantics and sparse handling precise terms, complementing each other, holds up.
- The optimum is around w_dense≈0.4 (sparse weighted slightly heavier), but this **strongly depends on the query's exact:semantic ratio** (≈1:1 in this set); equal weighting (0.50) at 0.499 is already close to the peak.

> Note 1: #1's single-route MRR (dense 0.506 / bm25 0.446) differs by ±0.004 from #2's endpoints (w=1.0 giving 0.510 / w=0.0 giving 0.449) — #1 computes MRR directly against the top-50 list, while #2 rebuilds the ranking via client-side RRF scoring (k0=60), and the tie-breaking on reordering accounts for the difference; this is expected.
> Note 2: #2's weight sweep is a **client-side reimplementation of weighted RRF** (k0=60) in `eval.py`, used to explore the optimal ratio; production's store uses **Qdrant's native `FusionQuery.RRF`** (equal-weighted, server-side) — the two behave identically at the equal-weight point, and the weighted points are only used for exploration.

### 4.3 #3 est_tokens recalibration (real Qwen3-VL tokenizer, 2927 real chunks)

> Note: #3 uses **all 2927 chunks** (a broader scope than #1/#2's 14-document/1067-chunk subset), so the corpus scope differs between them. Script: `retok.py`.

| | measured char/token | current heuristic | deviation |
|---|---|---|---|
| Prior non-English language (431) | 1.513 | 1.7 | token count underestimated by 11% |
| English (2496) | 4.340 | 4.0 | token count overestimated by 8% |

By document type (char/token): academic **3.83** / law **3.87** / financial_report_en **5.08** / government **5.35** / financial_research_zh 1.51.

- **The real dividing line is "prose vs. number-dense," not "the prior non-English language vs. English"**: English prose (academic/law) ≈ 3.85, almost exactly the current value of 4.0 (error < 4%); the deviation comes entirely from number/table-dense documents (financial reports/government 5.0+).
- A single coefficient can't serve both clusters, and switching to the overall mean (English 4.34) would actually hurt the most common case, prose documents.
- Both directions of error are absorbed downstream: overestimating tokens → chunks come out a bit small → small-to-big compensates at query time (nothing is lost); underestimating → chunks come out a bit large, but still well under Qwen3-VL's 32k ceiling. The 32k ceiling only protects dense; production's sparse route is **BM25** (jieba tokenization has no window limit, so it's unaffected by oversized chunks — the evaluation's BGE-M3 512-token window doesn't apply in production).

### 4.4 #4 reranking (Qwen3-VL-Reranker-8B, re-ranking the hybrid recall's top-50)

Second-stage cross-encoder re-ranking, using the same 56 queries (script `eval_rerank.py`):

| | precise-term MRR | semantic MRR | overall MRR | R@5 |
|---|---|---|---|---|
| hybrid (recall) | 0.544 | 0.584 | 0.566 | 0.82 |
| **+rerank** | **0.924** | **0.821** | **0.867** | **0.93** |
| improvement | **+70%** | +41% | **+53%** | +0.11 |

- **The cross-encoder reranking is a big quality gain**: overall MRR goes from 0.566→0.867 (the correct chunk's average rank improves from ~1.8th to ~1.15th place). The mechanism: it feeds the whole `(query, chunk)` pair into the model and sees **token-by-token interaction**, which gives it a far higher precision ceiling than a bi-encoder (dense encodes each side separately and just measures distance between them, seeing no interaction at all).
- **Being honest about a caveat**: the semantic column's +41% may be somewhat optimistic (the semantic queries' same-source bias, see §6, also affects the cross-encoder); but **the precise-term column's +70% is clean** (programmatically mined, no same-source bias), and that gain is real.
- Note: the hybrid baseline of 0.566 used here comes from **Qdrant's native RRF** top-50 (a different basis than §4.2's client-side reimplemented RRF); the rerank comparison uses the same baseline throughout, so it's a fair comparison.

## 5. Conclusions and decisions

| Item | Decision | Rationale |
|---|---|---|
| **Sparse model choice** | **BM25** | **Main reason**: the two sparse routes are **roughly tied overall** (0.453 vs 0.446), and BM25 is **zero model/zero GPU/zero maintenance** versus BGE-M3's 2.3GB+GPU. **Bonus**: inside the hybrid, dense already covers semantics (making BGE-M3's semantic advantage redundant), so sparse only needs to cover precise terms, which is exactly BM25's home turf (0.738 vs 0.584) |
| **RRF weighting** | **Keep Qdrant's standard RRF (equal weight)** | Hybrid does genuinely beat either single route; equal weight (0.499) is already close to the peak (0.541) — the 0.04 gain isn't worth the client-side fusion complexity, and the optimal weight of 0.4 depends on the query distribution, which will drift. Principle: **don't set the sparse weight too low** |
| **est_tokens** | **Don't recalibrate (keep the current value)** | Accurate on prose, biased on number-dense documents, but absorbed by small-to-big + the 32k ceiling; a single coefficient can't serve both clusters, and switching to the mean would actually hurt prose |
| **rerank** | **Adopted (off by default, opt in as needed)** | Cross-encoder reranking takes MRR from 0.566→0.867 (a clean +70% on precise-term queries); but it costs +16GB of VRAM and +several seconds per query — `Retriever.search(rerank=False)` is off by default and must be enabled explicitly, with the quality/cost trade-off left to the caller depending on the scenario |

## 6. Limitations (stated honestly)

- **A synthetic evaluation set**: precise-term queries are programmatically mined and semantic queries are agent-generated, not a real user query distribution.
- **Small scale**: 56 queries / 14 documents — the trend is credible, but absolute values are for reference only.
- **A single golden per semantic query**: each semantic query is only labeled with 1 golden chunk, so other genuinely relevant chunks may get counted as "wrong" — but this is equally strict across all three routes, so **the comparison itself stays fair**.
- **Same-source bias in semantic queries**: semantic queries were generated by an agent reading the source chunk and working backward, so the golden and the query share a source (even though the wording avoids the original terms, the semantic structure still closely tracks the source chunk), which **systematically overestimates dense's semantic recall** — 0.794 should be read as an upper bound, not a realistic expectation for production.
- **RRF's optimal weight depends on the query distribution**: this set is ≈1:1 exact:semantic; a different real-world distribution would give a different optimal w.

## 7. Reproduction

```bash
cd eval/component_retrieval
python build_corpus_chunks.py && python select_semantic.py   # semantic queries need agent generation, see the README
python index_eval.py && python eval.py    # #1 single-route recall + #2 RRF weighting
python retok.py                           # #3 tokenizer recalibration (independent, no index needed)
```

Requires the WSL `custodian` environment (GPU=4090) + Qwen3-VL + BGE-M3 models + real data in `parsed/`. See [`eval/component_retrieval/README.md`](../../../eval/component_retrieval/README.md) for details.
