# 07 Evaluation Methodology — This Project's Biggest Differentiator

> **Reading guide for this chapter**
> This chapter covers Custodian's evaluation loop: synthesize gold data → run the real system → LLM judges → attribution, along with the debiasing design behind it, the five-metric layering, the reproducibility tiering, and the complete real-world story of "the evaluation pipeline itself can have bugs."
> **Interview weight: the highest of the whole set.** "How do you evaluate RAG" is a must-ask question, and most candidates only have a one-word answer ("ragas") — this chapter gives you a whole framework that can hold up against 20 minutes of continuous follow-up questions.
> Suggested prior reading: [01 RAG Overview](01-rag-overview.md), [05 Generation and Grounding](05-generation-grounding.md); engineering details in [eval/README.md](../../eval/README.md) and [docs/TESTING.md §3](../TESTING.md).
>
> **Note (this project):** the gold-question findings that name "non-English-language research reports" (eval#3's coverage skew, the
> 15-document corpus composition in §8) describe the earlier evaluation corpus in the project's original non-English language,
> from before this project replaced that language support with Telugu (see the top-level README). Kept as genuine historical
> findings, not re-measured against Telugu documents.

---

## 1. Conceptual foundation: why "evaluating a RAG system" is harder than "building one"

Any RAG system can be up and running within a week, but answering "is it actually any good" can stall a team for a whole quarter. The difficulty has three layers:

**Layer one: there's no exam.** Traditional ML has a labeled test set; your own document library doesn't. Public benchmarks (HotpotQA, MS MARCO, RAGBench) test on someone else's corpus distribution — your users are asking "what was Netflix's total revenue in 2015," not Wikipedia trivia. So the first thing you have to do is write your own exam, and each of the three ways to do that has a fatal flaw:

| Approach | Advantage | Fatal flaw |
|---|---|---|
| Human annotation | High quality | Expensive, slow — unaffordable for an individual or a small team |
| Sampling from real query logs | Realistic distribution | A cold-start system has no logs; answers still need human labeling |
| LLM-synthesized gold | Fully automatic, batchable | **The exam itself may be wrong** (question-generation hallucination), and there's homology bias |

There's an inevitable follow-up question at this layer: "Doesn't the industry already have a bunch of off-the-shelf benchmarks?" Yes, and it's worth memorizing them in four layers (as of 2026-07) — but once you look closely, none of them can substitute for writing your own exam:

| Layer | Representatives | What it can standardize |
|---|---|---|
| Component layer | MTEB (already on v2 as of 2026, scores not comparable to v1), BEIR, MS MARCO | Closest to "standardized": for embedding / reranker selection, the leaderboard alone will do — it only tests the component, independent of your corpus |
| End-to-end layer | TREC RAG track (NIST, since 2024; the 2026 edition has already gone agent-first), CRAG, RGB, RAGBench, RAGTruth (strictly speaking a hallucination-annotated corpus) | Standardizes **methodology and tooling**; the corpus is still the official one — you're testing "your pipeline on someone else's corpus" |
| Framework layer | ragas (the de facto standard for RAG-specific evaluation), TruLens, ARES, DeepEval | Standardizes **the metric protocol**, but you bring your own data — the very existence of this layer of frameworks is proof that "there is no standardized dataset" |
| Agentic layer (new in 2025–26) | BrowseComp / BrowseComp-ZH / BrowseComp-Plus, DeepResearch Bench | What's being evaluated shifts from a single-turn pipeline to a multi-step search agent; Plus uses a fixed corpus to decouple "the retriever is strong" from "the agent is strong" |

The TREC 2026 entry is worth expanding on a bit (three new terms so you don't get caught cold reading them): that edition switched to an official corpus called ClimbMix, released a companion toolkit called RAGDoll, and scores using the nugget method — breaking the reference answer down into factual nuggets and scoring by coverage, converting a subjective score into a countable fact. That's the same approach as this chapter's citation_recall.

On the non-English-language academic-benchmark side there's also CRUD-RAG (the leading such benchmark, which builds its question set around four scenarios: create, read, update, delete), SuperCLUE-RAG (asking the same question twice — once without documents, once with — to measure the RAG uplift), and the OpenCompass leaderboard.

Why can't these be unified? **RAG's output is a function of "pipeline × corpus,"** and a public benchmark can only bring its own corpus along. LLM benchmarks like MMLU can be unified because the thing being tested is self-contained; RAG isn't. So the consensus the industry has converged on has two layers: **use the public leaderboards for component selection, but build your own golden set for system acceptance** — writing your own exam isn't a fallback for "lacking a benchmark," it's the correct answer in its own right. (A related discipline, by the way: MTEB v2 and v1 scores not being comparable is the same issue as the "72→88 scope discontinuity" discussed under Alternative E in §3 — when the exam changes, you redraw the baseline; even leaderboards aren't exempt from this.)

**Layer two: an end-to-end score can't localize the problem.** RAG is a multi-stage pipeline (parse → chunk → retrieve → assemble → generate → cite), and "accuracy 0.8" doesn't tell you which of the remaining 0.2 is broken at which stage. The industry's answer is layered metrics: the retrieval layer uses programmatic metrics like Recall/MRR, and the generation layer uses faithfulness (whether the answer is supported by the context) and correctness (whether the answer matches the facts) — the latter two have no programmatic algorithm and can only be judged by asking an LLM (LLM-as-judge), which is the approach that both ragas and TruLens wrap.

**Layer three (the one most easily overlooked): the measuring instrument itself can break.** An LLM judge introduces three new failure surfaces:

1. **Same-vendor self-preference bias**: using GPT to generate answers and then GPT to score them, and the model systematically favors output in its own house style. There's a lot of academic evidence for this, but in practice almost nobody checks whether "the gold generator, the system under test, and the judge are all the same vendor's model."
2. **Judge input bias**: if the context the judge sees differs from what the generator actually saw (truncated, summarized, reconstructed), the faithfulness score isn't measuring the system anymore — it's measuring this bias.
3. **Exam skew**: if the question-type distribution of the exam doesn't line up symmetrically with what a change is supposed to benefit, eval will systematically vote against a correct change (the benefit is invisible while the cost is fully exposed).

Custodian's evaluation subsystem is built around three questions: where does the exam come from, how does the judge get debiased, and how do we discover when the instrument itself is broken. Its most valuable output isn't the handful of scores it produces — it's a real case study: **a truncation bug in the evaluation pipeline itself manufactured the false conclusion that "the system hallucinates 17% of the time" out of thin air, and the team wasted a round of fixes chasing that false conclusion.** That story is in Section 4.

---

## 2. How Custodian does it

### 2.0 Data flow overview

```
                    ┌── Tier1 (reproducible in-repo, fast) ──────────────┐
                    │  run_eval --judge deepseek: DeepSeek self-judges,   │
                    │  same-vendor circular bias, trends only             │
                    └───────────────────────────────────────────────────┘
 gen_gold.py ──┐
 gen_gold_tables.py ─┤→ gold.jsonl (88 questions)
 dump_chunks.py → _units/ → Claude sub-agents author multi-hop gold ──┘
                    ┌── Tier2 (authoritative, dual cross-vendor Claude judges) ─┐
                    │  run_eval --judge none → rows (answer+ctx_text)          │
                    │  dump_judge_units.py → _judge/judge_NN.json               │
                    │  2 independent Claude judge passes → verdicts.json        │
                    │  aggregate.py: fingerprint check + dual-judge AND + paired attribution │
                    └───────────────────────────────────────────────────┘
 acl_regression.py: an independent security regression axis (65 assertions, exit code 0 = production gate)
```

### 2.1 Debiasing triangle: gold and judge use Claude, DeepSeek is only ever the system under test

There are three LLM roles in the evaluation: the question author, the system under test, and the judge. If all three are the same vendor, that's "self-question, self-answer, self-score." Custodian's split:

- **The system under test** = production DeepSeek. Key detail: GEN_MODEL defaults back to the production `CUSTODIAN_LLM_MODEL` ([eval/_common.py:44](../../eval/_common.py#L44)) — "testing the actual deployed configuration" isn't a slogan, it's the code's default value.
- **The question author and judge** = Claude (a cross-vendor frontier model). `run_eval --judge none` only computes programmatic metrics, and writes the rows (answer + the full ctx_text fed to the LLM + golden) to disk ([eval/run_eval.py:225-226](../../eval/run_eval.py#L225)); [eval/dump_judge_units.py](../../eval/dump_judge_units.py) slices the rows into judging-unit files of 8 items each, which Claude Code orchestrates through 2 independent Claude judge passes to produce verdicts.json; [eval/aggregate.py](../../eval/aggregate.py) takes the **dual-judge AND** (only true if both passes judge it true).
- Gold works the same way: [eval/dump_chunks.py](../../eval/dump_chunks.py) produces `_units/` unit files that Claude sub-agents use to author gold (single-hop / single-document multi-hop / cross-document, three categories) — the original chunk text only ever enters the unit files, never the orchestrator's context — private report data never leaves the Claude Code trust boundary and is never sent to a third-party API.

**Reproducibility is honestly split into two tiers** (see the migration note in [eval/README.md](../../eval/README.md)): Tier1 = `--judge deepseek`, one command reproducible in-repo, but a same-vendor self-judge only shows trends; Tier2 = dual-Claude cross-vendor AND, an authoritative number, but the orchestration workflow is deliberately kept out of the repo (the README explicitly states steps 3/6 are generated by the orchestrator on demand). We don't pretend "the authoritative numbers are reproducible by everyone" — that itself is part of the methodology.

### 2.2 Five-metric layering: three programmatic metrics with zero judge involvement + two judge-based metrics that keep each other honest

Programmatic metrics are computed directly by [eval/run_eval.py:179-181](../../eval/run_eval.py#L179) and [:200-211](../../eval/run_eval.py#L200), free of judge cost and suitable for frequent regression. `agg()` writes 5 raw fields, and the rollup consolidates retrieval_recall / retrieval_full into one "retrieval recall" item while listing avg_rounds separately as a cost metric — the headline externally is **three programmatic metrics (retrieval recall / MRR / citation recall) + two judge metrics = five metrics**, the same scope used in README/OVERVIEW. The table below is the 5 raw fields:

| Metric | What it measures | How it's computed |
|---|---|---|
| retrieval_recall | Retrieval layer: was what should have been fetched, fetched | Fraction of golden chunks recalled (proportional for multi-gold cases) |
| retrieval_full | Retrieval layer: was everything recalled | golden set ⊆ recalled set |
| MRR | Retrieval layer: is it ranked near the top | Reciprocal rank of the best hit |
| citation_recall | **Is grounding actually real** | Fraction of `[cite:n]` markers in the answer, mapped back through [Generator._parse_citations](../../src/generator/generate.py#L117) to chunk_id, that hit the golden set |
| avg_rounds | Cost | Average number of retrieval rounds, read paired with Δ correctness |

citation_recall deserves its own explanation: it turns "is the answer actually built on evidence" into a **countable structural fact** — instead of asking an LLM "do you think this is grounded," it counts citation markers. Citation markers use `[cite:n]` rather than a bare `[n]` (the sole `CITE_RE` in [src/generator/prompt.py:15](../../src/generator/prompt.py#L15)), because retrieved body text often contains bare footnote/reference numbers like `[n]`, and if the same marker shape were shared, the LLM would copy it verbatim and the parser would mis-map it into a fake citation.

The two judge metrics are deliberately made to point in **opposite directions** ([eval/run_eval.py:29-37](../../eval/run_eval.py#L29)):

- FAITH_SYS defines: a reasonable refusal ("insufficient information") = faithful=**true**;
- CORRECT_SYS defines: a refusal, when the reference answer actually has substantive content = correct=**false**.

A system can't game the faithfulness leaderboard by refusing everything (correctness would collapse), and can't game the correctness leaderboard by fabricating (faithfulness would collapse) — the two metrics act like a pair of pincers, squeezing out the real safety property of "refuse rather than fabricate." In the 88-question baseline, the fact that the 16 table questions had a perfect faithfulness score of 1.000 is built exactly out of: 4 questions with retrieval misses **all honestly refused, with zero fabrication** (scope: 88 questions, DeepSeek judge, [docs/TESTING.md §3](../TESTING.md)).

### 2.3 Synthesizing gold: single-chunk question generation ⇒ unambiguous golden data

The core tradeoff in [eval/gen_gold.py](../../eval/gen_gold.py): scroll through the entire payload set from the index (pure CPU, no encoding needed), group by doc, use `pick_evenly` for evenly-spaced sampling (deterministic, no random seed, covering the beginning/middle/end of a document, [gen_gold.py:33-39](../../eval/gen_gold.py#L33)), and feed each chunk to DeepSeek's JSON mode to generate a (question, answer) pair. **Because the question is derived from a single chunk ⇒ golden_chunk_id is unambiguous by construction ⇒ retrieval/citation recall can be judged programmatically, with zero human annotation.**

Both the costs and the mitigations are stated openly:

- Cost ①: questions skew toward single-hop factual types → multi-hop gold is constructed by Claude sub-agents crossing chunks/documents to fill the gap (the `_units/` doc_NN / type_N units).
- Cost ②: homology bias (the question's phrasing overlaps with the golden chunk's vocabulary, making retrieval artificially easy) → the SYS prompt mandates "no reference-style phrasing like 'according to this passage,' not so broad the passage can't fully answer it, and phrased like a real user's question" ([gen_gold.py:25-30](../../eval/gen_gold.py#L25)); the README's "honest warning" section tells whoever reads the numbers directly that the gold is generated.
- Filtering: SKIP_KINDS skips asset blocks like table/image/figure, and MIN_CHARS=150 filters out fragmentary chunks ([gen_gold.py:21-22](../../eval/gen_gold.py#L21)).

**The programmatic QC gate for table gold** is the most interesting mechanism in this whole line. [eval/gen_gold_tables.py](../../eval/gen_gold_tables.py) generates questions "that must be answered from the table's own values" for blocks where kind=table and content_raw≥120 characters; `answer_grounded` performs programmatic acceptance ([eval/gen_gold_tables.py:48-53](../../eval/gen_gold_tables.py#L48)): at least one ≥2-digit numeric value in the answer (after comma/whitespace normalization) must appear verbatim in the table body, or the whole item is discarded ([:99-101](../../eval/gen_gold_tables.py#L99)) — **"if the question author made up the number, it doesn't get into the exam."** In practice this caught 2 question-generation hallucinations. An LLM question-author reading a large table can misread rows and columns or even invent numbers; the biggest risk in synthesized gold is that "the exam itself is wrong," and exact numeric matching is the cheapest mechanical gate for it (its blind spot is discussed in the deferred item eval#2 in Section 4).

### 2.4 Two-layer attribution: the paired design behind single vs. agentic vs. decompose

This is the experimental apparatus for "is agentic RAG actually worth it." Three modes ([eval/run_eval.py:78-161](../../eval/run_eval.py#L78)):

- **single**: closed-pipeline single-hop, runs through the production Generator — the component-layer baseline;
- **agentic**: DeepSeek judges whether the context is sufficient (SUFFICIENCY_SYS); if not, it **rewrites the query and searches again as a replacement**, accumulating context before generating;
- **decompose**: first breaks the question into 1–4 sub-questions (DECOMPOSE_SYS, a cross-document comparison must have one sub-question per object), retrieves for each independently and takes the **union** (capped at 14 to avoid overflow) before synthesizing — distinct from agentic's "replacing the rewritten query can narrow the scope and lose a hop."

Attribution is done **paired** in [eval/aggregate.py:114-128](../../eval/aggregate.py#L114): the subtraction only happens on the "common set of questions judged by both modes under the dual judge" — you can only subtract when the denominators match, a discipline learned the hard way (see Section 4). It's broken down by hop (single / multi_intra / multi_cross). There's also an easy-to-miss detail of scope: agentic/decompose's union_ids get duplicates appended round after round, and must be **order-preserving deduplicated** ([run_eval.py:54-60](../../eval/run_eval.py#L54)'s `_dedup`) before MRR/best_rank are on the same scope as single's ordered hits.

Results (scope: 72 questions, Tier2 dual-Claude judge, paired): single→agentic Δ**−0.097** (n=72), single→decompose Δ**−0.014** (n=71); agentic is ≤ single at **every hop level** (more retrieval = more distracting blocks diluting the signal); decompose only weakly edges ahead on cross-document questions (correctness 0.20 vs. 0, retrieval 0.37 vs. 0.20, but n=5 is small). Conclusion: **on this workload, the closed pipeline should be the default — agent orchestration is a net negative, and we publish negative results too.** (This Δ's magnitude has a confirmed measurement flaw, see eval#0 in Section 4 — honestly flagging it is also part of the conclusion.)

### 2.5 Fingerprint gate + loud missing-verdict handling: the discipline of aligning results with judgments

In the Tier2 flow, results and verdicts are produced in two separate steps, and their alignment depends on row order — if results is re-run without re-judging, row i could end up as a different question entirely, and scores would come out completely mismatched **with no error whatsoever**. Custodian's answer is to make the alignment mechanically enforced:

- [dump_judge_units.py:44](../../eval/dump_judge_units.py#L44) writes `_judge/fingerprint.json`, whose contents are `{id: sha1(query)[:12]}`;
- [aggregate.py:67-71](../../eval/aggregate.py#L67) compares the query fingerprint of results row by row before consuming verdicts, and any mismatch triggers `SystemExit` directly, **refusing to output numbers**.

Handling of missing verdicts is likewise loud rather than silent: [aggregate.py:45-50](../../eval/aggregate.py#L45)'s `andflag` rule states that if either pass didn't judge or a field is missing → None (excluded from the denominator), never silently treating `bool(None)=False` as a judgment of false; n_judged is printed for every mode/hop, and an empty judgment is marked n/a rather than mixed in as nan. Every conclusion depends on eval being correct — **alignment has to be mechanically enforced, it can't rely on human diligence.**

### 2.6 The faithfulness judge's full-context contract: the judge's input = the exact text the generator's input was

run_eval stores the exact **original** user message fed to the LLM (the numbered passage blocks assembled by PromptBuilder + the question) directly into rows.ctx_text ([run_eval.py:103](../../eval/run_eval.py#L103): `next(m.content for m in ans.raw_messages if m.role == "user")`), and dump_judge_units hands it to the judge unchanged — what the judge sees and what the generator saw are **byte-for-byte identical**, zero reconstruction bias. [dump_judge_units.py:21](../../eval/dump_judge_units.py#L21)'s `CTX_CAP = 200000` is effectively "no truncation" (ctx_text's median is ~16k), only guarding against pathologically long inputs.

Why this one behavior is the most important piece of methodology in the entire subsystem — the 0.83 false-conclusion story in Section 4 provides bloody proof.

### 2.7 The three-layer adversarial design of the ACL regression: 65 assertions

[eval/acl_regression.py](../../eval/acl_regression.py) is an independent security-evaluation axis. The demo library is entirely public, so it can't test isolation — the script instead builds a synthetic library with the real Chunker+Embedder, freshly creating 2 tenants × 4 documents, with **each document carrying a unique sentinel string in its original text** ([acl_regression.py:29-42](../../acl_regression.py#L29)), and 65 assertions across five sections, each layer built against "faking a pass":

1. **The recall isolation matrix** (5 identities × 4 sentinel exact-match queries, [:82-90](../../eval/acl_regression.py#L82)): using the original text itself as a query means that even a literal exact match must return 0 recall — proving ACL hard filtering happens **before relevance ranking**, rather than "it happened not to be found" luck.
2. **Positive authorization** ([:93-98](../../eval/acl_regression.py#L93)): an authorized identity must be able to find it, guarding against "refuse everything to fake a pass" — a system that denies every request could otherwise pass any isolation test.
3. **get_document direct-read fail-closed**: unauthorized → PermissionError.
4. **expand across ACL boundaries**: taking docA's real chunk_id and calling expand with an unauthorized identity → None.
5. **The exit-gate-disabled check** ([:123-138](../../eval/acl_regression.py#L123)): monkeypatch [store.acl_admits](../../src/embedder/acl.py#L29) to always return True, **and after disabling the exit-level double-check, the whole matrix must still rerun with 0 leaks** — proving that pushdown filtering within RRF fusion prefetch itself blocks unauthorized access, rather than relying on the exit gate as a backstop that masks a regression. "The embedded QdrantLocal's fusion drops the top-level `should` filter" is a real pitfall we once hit — defense in depth can mask an inner-layer regression, so a regression test needs to be able to **disable the outer layer to prove the inner layer.**

Exit code 0 = no leaks, usable directly as a pre-production gate. (Historical documentation in REVIEW_PLAN.md records "44 items" — that was the count before the exit-gate-disabled check was added; the current code has 65 items.) The ACL design itself is covered in [04 ACL and Security](04-acl-security.md).

### 2.8 Two engineering details: lock avoidance and eval/product sharing a common source

**copy_demo**: an embedded Qdrant is exclusively locked to a single client — if a live daemon process/MCP server is already holding the lock on ~/rag_demo, an evaluation script that opens the same directory directly will crash with "already accessed by another instance." [_common.copy_demo](../../eval/_common.py#L49) unifies this by always doing a copytree to a temp directory first before opening it — it doesn't fight for the lock, doesn't pollute the original library, and eval can be run at any time without requiring the user to first tear down their production environment.

**--smart-tables shares a source with production**: eval's failure-driven table supplementary retrieval ([run_eval.py:78-104](../../eval/run_eval.py#L78)) reuses the exact same signal functions as production smart-ask ([src/custodian/service.py:336-353](../../src/custodian/service.py#L336)) — [generator/signals.py](../../src/generator/signals.py)'s `looks_numeric` / `is_refusal` / `DEFAULT_TABLE_LEG` is a single source, preventing the wordlist from drifting between the two places. **eval and production must test the exact same strategy with the exact same parameters, or eval numbers have no predictive power for production**; a code comment nails down "the front-loaded-leg version has already been rejected by the 88-question measurement, don't change it back."

---

## 3. Why this design: rejected alternatives

**Alternative A: use ragas / an off-the-shelf evaluation framework directly.** Rejected for three reasons: ① ragas defaults to a single-model judge, which doesn't solve same-vendor circular bias; ② it can't test this project's own custom assets (citation_recall depends on our own structured `[cite:n]` citation protocol, ACL isolation depends on our own permission model); ③ private report data would have to be sent out to a third-party API. We kept its underlying idea (dual-judge faithfulness/correctness metrics) and built our own pipeline. An updated status check (as of 2026-07): ragas remains the de facto standard for RAG-specific evaluation — its metric library has grown to 30+ (8 of which are RAG-specific); broader LLM evaluation frameworks (like DeepEval) have already overtaken it by GitHub stars, but for RAG scenarios specifically it's still the most widely cited. That's exactly why "why not use it" is a must-ask question — see follow-ups Q1/Q9.

**Alternative B: same-vendor self-judging (DeepSeek doing everything).** Not fully rejected, but downgraded and kept as **Tier1** ([run_eval.py:225-226](../../eval/run_eval.py#L225)'s `--judge deepseek`): fine for fast regression checks on relative trends, not fine for asserting absolute numbers. This tiering is itself a design decision — more honest than either "pretend circular bias doesn't exist" or "always run the expensive cross-vendor dual judge."

**Alternative C: human annotation / real query logs.** Too expensive / an individual-scale system has no accumulated logs, both outweighed by "fully automatic + programmatically judgeable." The costs (single-hop bias, homology bias) are managed with mitigations plus explicit warnings, rather than being pretended away.

**Alternative D: making the table supplementary retrieval a front-loaded leg (attached to every question).** Rejected by a four-round, 88-question experiment (scope: 88 questions, DeepSeek judge, [TESTING.md §3](../TESTING.md) smart-ask log):

| Version | Table (16) | Prose (72) | Faithfulness | Verdict |
|---|---|---|---|---|
| Baseline | 0.625 | 0.861 | 0.977 | — |
| ① Front-loaded leg | 0.875 | **0.792** | 0.966 | Rejected: similar-looking values collaterally damaged 5 originally-correct prose questions |
| ② Failure-driven, rerank_top_n=30 | 0.750 | 0.847 | 1.000 | Illusory: the correct block was ranked 31–50 in coarse ranking, the reranking pool couldn't hold it, the leg was effectively a no-op |
| ③ top_n=50, unconditional adoption of the retry | 0.625 | 0.833 | **0.932** | Rejected: partial answers smuggled in a false "X was not provided" claim of missing information (X was actually in the context) |
| ④ Failure-driven + best-of adoption | 0.688 | 0.833 | 0.977 | **Adopted**: zero loss on the discarded-retry path across 4 questions, one question flipped ✗→✓ on the adopted-retry path |

Four pieces of methodology settled out of this (each one from a round of rejection): intelligence should only act on failure paths; the reranking pool's depth must be ≥ the correct block's worst rank in coarse ranking; "turning a total refusal into a partial answer" introduces a new failure surface; under a noise floor of ±2 questions, any horizontal comparison must use paired attribution (the retried/retry_kept markers on rows exist exactly for this).

**Alternative E: mixing old and new gold together.** After gold went from 72→88, a hard **scope discontinuity** was enforced: the original 72-question set was backed up once as `gold_backup_prose72.jsonl` ([gen_gold_tables.py:110-111](../../eval/gen_gold_tables.py#L110) only backs it up once), table questions carry an `asset:"table"` tag so they can be split out for separate statistics, and TESTING established a new baseline. Once the exam changes, you redraw the baseline — the 88-question 0.818 and the 72-question 0.847 **cannot be subtracted from each other** — they don't even share a judge (the former is DeepSeek Tier1, the latter Claude Tier2). This discipline looks trivial but is the bedrock of "the numbers are trustworthy": any cross-scope subtraction manufactures a false conclusion.

Why add table questions at all? Because we'd been bitten by "exam skew creates a measurement blind spot": a chunker enhancement for table-block retrieval text (chunker `2bd97a5`) fixed a real question type — "the number is buried inside a table" — from wrong to correct, but on the 72-question all-prose gold set it only showed up as a cost (retrieval recall −2.1pp, 3 questions' redundant gold shifted) and no benefit (0 questions improved) — per-question diagnosis confirmed the 3-question shift was entirely redundant gold on double-gold questions being displaced, with the answer still correct in every case, and only then was the decision made to keep the enhancement, followed by adding 16 table questions so the exam would be symmetric with the change. **The acceptance ruler must cover the question type a change is meant to benefit, or eval will systematically vote against a correct change.**

---

## 4. Real-world retrospective

### 4.1 The 0.83 false conclusion: a bug in the evaluation pipeline itself manufactured "17% hallucination" out of nothing (R5.H1)

This is the single most worth-retelling story in the whole project. The full chain:

1. **The illusion**: an authoritative run reported a faithfulness score of 0.83, i.e. "17% of claims lack support," and it was listed as an open item, B1. The problem looked real and stubborn.
2. **Wasted effort chasing the false conclusion**: the team tightened the grounding prompt (blanket sentence-level constraint: "EVERY sentence MUST be supported"), and measurement showed it **backfired** — faithfulness −0.12, correctness −0.04, and it was reverted (trace left in the [src/generator/prompt.py:40-42](../../src/generator/prompt.py#L40) comment).
3. **Adversarial review of the instrument itself**: the R5 review turned the eval methodology itself into a target. It found that dump_judge_units had a `CTX_CAP=5000` **head truncation**, while ctx_text's measured median was 16k — the judge was only seeing about 40% of the passages.
4. **Confirmation**: re-reviewing all 16 unfaithful verdicts one by one, 12 of them (75%) had their cited evidence located exactly in the part that had been truncated away — claims where "the evidence is in the later part of the text" were being systematically misjudged as unfaithful.
5. **Fix and re-judgment**: the truncation was removed (CTX_CAP set to 200000, purely as a guard against pathological length), and both judges re-judged all 216 items (72 questions × 3 modes) against full context: true faithfulness was **≈1.0** (single/agentic 1.000, decompose 0.972). The B1 conclusion was overturned — the system was refusing rather than fabricating far more consistently than had been believed; the earlier prompt tightening had been fixing **a problem that didn't exist.**

Two hard rules settled out of this: ① confirm a problem is real before you start fixing it; ② the faithfulness judge must see every passage the generator actually used — this is exactly where the "ctx_text stores the raw user message verbatim" contract in Section 2.6 comes from. One-line interview summary: **before you fix the system, suspect the measuring instrument first.**

### 4.2 Attribution numbers couldn't be reproduced: the paired refactor of aggregate (R5.H2/M1)

The cross-document/decompose numbers in the README couldn't be reproduced with the aggregate.py committed in the repo: the old version hard-coded only the single/agentic modes, read stale verdicts left over on disk (agentic was missing 8 cross-document judgments), and the two-layer attribution subtracted using **different denominators** directly — the cross-doc case printed nan, and the conclusion was hanging in the air. After the refactor: parameterized MODES, a paired common question set, loud missing-verdict handling, and a sha1 fingerprint gate (Section 2.5), Δ−0.097/−0.014 can now be reproduced with one click of aggregate.py; a subsequent pass 2 with dual-judge AND re-reviewed all 216 items, and the numbers matched the single-judge run (agentic 0.764→0.750, slightly stricter). Lesson: **every published number has to be replayable from the committed code + data.**

### 4.3 The .env timing bug: silent configuration failure is more dangerous than a crash

Various eval scripts depend on module-level constants in _common (EVAL_SRC/GEN_MODEL/JUDGE_MODEL) read from `CUSTODIAN_EVAL_*`; an overall adversarial review found that the call to load `.env` was written **after** those constants were defined — any evaluation library/model configured in `.env` silently fell back to the default, and the scripts ran with the wrong library and model, producing numbers that "looked normal." Fix: load_env() was moved to before the constants ([_common.py:35-36](../../eval/_common.py#L35) carries a "review fix" comment as a trace), with setdefault semantics ensuring explicit environment variables still take priority. This kind of bug runs to completion successfully, but tests the wrong thing — evaluation infrastructure must have zero tolerance for it.

### 4.4 This round's adversarial review: 1 fixed, 3 confirmed but deferred

Before writing this set of documentation, another round of adversarial review was done on the eval subsystem (every finding first assigned to an independent verifier who tried to refute it — only what survived refutation counted as confirmed). The result: 4 confirmed items, and how they were triaged is itself a teaching point — **"confirmed but can't be fixed right away" is an engineering judgment, not procrastination**:

**Already fixed, eval#1: dump_chunks lock-avoidance alignment** (fixes_applied.md #17). Symptom: dump_chunks.py opened the original library directly without copying it, inconsistent with the copy_demo strategy used by every other eval script — _common.py claims "unified copytree" but had this exception; the script would crash if the target library was locked by a daemon, and while it ran it would hold an exclusive lock in the opposite direction, blocking the daemon. Root cause: a pure oversight. Fix: switched to going through a copy_demo temporary copy, the same as gen_gold ([dump_chunks.py:42-44](../../eval/dump_chunks.py#L42) current state). Why it could be fixed right away: CPU-only scroll, doesn't change any evaluation output content, and the copy cost has already been shown acceptable by gen_gold_tables.

**Deferred, eval#0 (medium): the three-way comparison is systematically unfavorable to agentic/decompose.** The agentic/decompose context assembly bypasses the production Generator, taking only `ctx.text or hit.text` ([run_eval.py:115/148](../../eval/run_eval.py#L115)), missing two already-shipped fixes: ③ the asset content_raw supplement ([generate.py:75-80](../../src/generator/generate.py#L75)) and the § section_path breadcrumb ([generate.py:85-89](../../src/generator/generate.py#L85)). Consequence: under the 88-question scope (16 table questions), a table block's hit.text has only the caption + a retrieval signal, with the data in content_raw — the agentic path "recalls it but can't answer it" — **the Δ(agentic−single) is overstated as a failure of agent orchestration, when part of it is actually the evaluation implementation missing two fixes**; the verifier even confirmed that a table block with empty text (asset_no_prose) gets skipped entirely, worse than initially claimed. Why deferred: the fix (extracting a common `build_context_entry` for reuse in three places) is itself a small, CPU-testable change, but **after fixing it, we'd have to re-run the GPU-based 88-question three-way comparison and update the already-published Δ conclusion** — a change that alters numbers can't be mixed into the same round as writing documentation. Documentation treatment: wherever Δ−0.097 is cited, this flaw is honestly flagged (including in this chapter).

**Deferred, eval#2 (medium): the table gold QC gate guards against "fabricated numbers" but not "misassigned numbers."** [answer_grounded](../../eval/gen_gold_tables.py#L48) only requires that any ≥2-digit numeric value in the answer appears in the table body; when the question author misreads rows/columns (attributing segment A's 2023 value to segment B's 2024), the resulting wrong number will **also necessarily still be in the table**, and it will pass the gate all the same. At the small sample size of 16 questions, 1–2 bad gold items amounts to a systematic 6–12 percentage-point underestimate of correctness. Why deferred: it only affects future gold regeneration, and regenerating gold means another scope discontinuity; the correct order is to first use "answer read-back" (independently calling an LLM again to answer from the table and comparing the numbers) to **retroactively audit the existing 16 questions**, turning a hypothetical contamination into a measured conclusion, and only then decide whether to regenerate.

**Deferred, eval#3 (low): gen_gold_tables' cap fills up in doc_id alphabetical order.** In [gen_gold_tables.py:79-85](../../eval/gen_gold_tables.py#L79), the loop breaks as soon as `len(rows)>=cap`; with 30 candidates competing for 16 slots, English-named documents that sort earlier alphabetically fill up the slots first — the verifier measured this directly: of 6 research reports in the project's original non-English language, only 1 made it into the exam, so the README's claim of "full coverage across all 15 documents" is actually false (that was the accidental outcome of one particular run). A cross-language table weakness (an already-known residual gap) might end up not appearing on the exam at all by pure chance. The fix (round-robin sampling: take 1 question from each document first, then a second round for a 2nd question) is likewise bound to a gold regeneration, and will land together with eval#2.

Note the common thread in eval#2/#3: **these are bugs in the "exam generator," not bugs in "the system"** — and this class of issue is exactly an extension of the lesson from Section 4.1: in synthesized evaluation, every step of question-writing, scoring, and aggregation is itself a measuring instrument, and every one of them has to survive adversarial review.

---

## 5. How to talk about this in an interview

### 30-second version (elevator pitch)

"I didn't use ragas — instead I built a debiased evaluation loop of my own: gold generation and judging use Claude, and the system under test uses production DeepSeek — three roles, three different vendors, eliminating the circular bias of an LLM grading itself. Metrics are layered into two tiers: retrieval recall/MRR/citation recall are programmatic, zero judge cost; faithfulness and correctness use a dual judge taking the AND, and the two metrics point in opposite directions — a reasonable refusal counts as faithful but counts as wrong, so the system can't game either single leaderboard. This pipeline caught a very telling bug: the judge's context was truncated to 40%, manufacturing a false '17% hallucination' conclusion out of thin air, and we'd wasted a round of prompt fixes chasing it — so my core point is that the evaluation pipeline itself is a system that also needs to be evaluated."

### 3-minute version (structured expansion)

1. **Where the exam comes from** (40s): single-chunk question generation makes golden_chunk_id unambiguous, so retrieval/citation recall can be judged programmatically with zero human annotation; the cost is single-hop bias and homology bias, mitigated by having Claude sub-agents author multi-hop questions and using a prompt rule that forbids referential phrasing. Table questions have a programmatic QC gate: the answer's numeric values must appear verbatim in the table body, which in practice caught 2 question-generation hallucinations — "if the question author made up the number, it doesn't get into the exam."
2. **How the judge gets debiased** (40s): three different-vendor roles + dual-judge AND + one contract that matters most of all — the context the judge sees is byte-for-byte identical to the user message actually fed to the generator's LLM, no reconstruction, no summarizing. Reproducibility is honestly tiered: Tier1 same-vendor self-judging shows trends, Tier2 cross-vendor dual judge asserts the numbers.
3. **Data point one: the false-conclusion story** (50s): CTX_CAP=5000 head truncation → the judge only saw the middle ~40% of passages → faithfulness falsely reported as 0.83 → the false conclusion drove a prompt tightening that backfired by −0.12 → adversarial review found the truncation, and re-checking all 16 unfaithful verdicts found 75% hit the pattern "the cited passage got truncated away" → re-judging all 216 items on full context, true faithfulness ≈1.0. Suspect the measuring instrument before you fix the system.
4. **Data point two: daring to publish negative results** (30s): the paired two-layer attribution measured the value of agent orchestration, single→agentic Δ−0.097, single→decompose Δ−0.014 (on the common set of 72 questions judged both ways) — "closed pipeline should be the default for this workload" is a data-driven conclusion, not a stance. At the same time, proactively acknowledging this Δ has a confirmed measurement flaw (the agentic path is missing two production-side fixes, so the Δ magnitude is likely overstated) — already queued for a re-run.
5. **Wrap-up** (20s): there's also an independent security-evaluation axis — the ACL regression's 65 assertions, including monkeypatching to disable the exit-level double-check and still requiring 0 leaks, proving that pushdown filtering itself blocks unauthorized access rather than relying on the backstop to fake a pass. Evaluation, security regression, and product decisions (the four-round smart-ask experiment) all share this same infrastructure.

---

## 6. Anticipated follow-up questions

**Q1: Why not use ragas?**
Key point: it's not that I haven't heard of it — there are three concrete reasons: a single-model judge has no answer to same-vendor bias; it can't test our own custom assets like citation_recall/ACL; and private data can't be sent to a third party. Add "I kept the metric ideas of faithfulness/correctness — what I replaced was the implementation." Keywords: self-preference bias, structured citation protocol, data boundary.

**Q2: The gold is LLM-generated — what if the exam itself is wrong?**
Key point: layered acknowledgment plus layered defense. Prose questions: single-chunk generation + prompt constraints + an explicit README warning; table questions: answer_grounded's exact numeric gate, which in practice caught 2 bad items. Then **proactively** mention the gate's blind spot: it guards against fabrication but not misassignment (a misread row/column number is still technically "in the table") — already confirmed, the fix is answer read-back, bound to the next gold regeneration. Being able to volunteer your own QC's blind spots is a level above just saying "we have QC."

**Q3: Faithfulness is ≈1.0 — isn't the judge just being too lenient?**
Key point: three counter-arguments. ① Dual-judge AND can only make it stricter, never more lenient; ② the exact same judge setup once produced a 0.83 during the CTX_CAP bug period, proving it's capable of returning "unfaithful"; ③ correctness during that same period was only around 0.85, and the two metrics point in opposite directions — if the judge were generally lenient, both would be inflated. Also mention the mechanism: the closed pipeline returns "insufficient information" deterministically on zero recall without going through the LLM at all, and counting a refusal as faithful is simply the metric definition.

**Q4: If Δ−0.097 says agent orchestration is useless, could your agentic implementation just be too weak?**
Key point: first acknowledge — this is partly true and already quantified (eval#0: the agentic path is missing the content_raw supplement and the breadcrumb, systematically unfavorable on table questions, so the Δ magnitude is overstated). Then hold the line — the direction very likely won't flip: back in the 72-question all-prose gold era (where asset impact was small), agentic was already ≤ single at every hop, and the mechanistic explanation (more retrieval rounds = more distracting blocks diluting the signal, replacing a rewritten query can drop a hop) is independent of this flaw; decompose does show a genuine edge on cross-document questions, which shows the experimental apparatus can resolve fine-grained differences. Keywords: paired, common question set, pending GPU re-run.

**Q5: Why don't the 88-question baseline and the 72-question authoritative run match up?**
Key point: scope discontinuity — different exams (72 all-prose vs. 88 with 16 table questions) and different judges (Tier2 dual-Claude vs. Tier1 DeepSeek self-judge); any cross-scope subtraction manufactures a false conclusion. Demonstrate your awareness of "baseline management": the original exam was backed up, asset-tagged questions can be split out for separate statistics, and TESTING established a new baseline.

**Q6: Won't dual-judge AND artificially depress the scores?**
Key point: the measured difference is tiny (agentic single-judge 0.764 → AND 0.750); what AND buys you is "a single judge having a bad moment doesn't contaminate the conclusion"; missing verdicts are excluded from the denominator as None rather than judged false, paired with a loud n_judged print. You can extend this with a counter-question: a more expensive approach would be a three-judge majority vote, but at this scale the cost-benefit doesn't beat AND.

**Q7: Would this whole setup still work at a different company?**
Key point: split it into transferable methodology (three different-vendor roles, the judge's full-context contract, paired attribution, scope discontinuity, the fingerprint gate, "the exam should be symmetric with the change") vs. implementation tied to this project (Claude Code orchestration, the [cite:n] protocol). The methodology checklist itself is the answer — none of it depends on any specific model or framework.

**Q8: How much noise is there in a single eval run, and how do you handle it?**
Key point: measured noise floor is ±2 questions (on 88 questions that's ≈2pp) — across five rounds of experiments, 2 questions flipped back and forth repeatedly. So any horizontal comparison at this granularity always uses paired attribution: rows carry built-in retried/retry_kept markers, and smart-ask does an unaffected-question paired check on every round (81 questions vs. the baseline, only 2 flipped, exactly the two known-unstable questions). Keywords: paired denoising, failure-surface slicing rather than just reporting averages.

**Q9: Is there an authoritative, standardized benchmark for RAG evaluation? Are your five metrics your own invention?**
Key point (first half): answer in layers (as of 2026-07, see the four-layer table in §1). The component layer is closest to standardized — for embedding/reranker, check MTEB (already on v2) / BEIR; the most authoritative end-to-end benchmark is the TREC RAG track (NIST, the 2026 edition has already gone agent-first), but what it standardizes is methodology and tooling — the corpus is still official; the framework layer's ragas is the de facto standard for RAG-specific evaluation; the agentic layer has a new batch from 2025–26 (the BrowseComp family, DeepResearch Bench). Core argument in one line: **RAG performance = pipeline × corpus, a public benchmark can't answer "does it work on my corpus," and the industry consensus is exactly public leaderboards for component selection + your own golden set for acceptance.**
Key point (second half): the five metrics aren't invented from scratch — faithfulness/correctness share the same names and meanings as ragas' faithfulness / answer correctness; retrieval recall/MRR are a programmatically stricter version of its context recall/precision (we have golden_chunk_id, so we can count exactly instead of having an LLM estimate); citation recall is a reinforcement ragas doesn't have by default. Bonus point: MTEB v2 and v1 scores aren't comparable — leaderboards also observe scope discontinuity, the same discipline as our 72→88. Keywords: pipeline×corpus, public leaderboards for component selection, golden set acceptance, programmatically strengthened.

---

## 7. Hands-on experiments

Both experiments are **pure standard library, CPU is enough**, run from the repo root (works in Windows Git Bash / WSL alike; all artifacts are already gitignored — delete them when done).

### Experiment 1: reproducing the CTX_CAP truncation artifact — how the 0.83 false conclusion was manufactured

First construct one results row where "the evidence is at the tail of the context," then check whether the evidence survives in the judging-unit file:

```bash
cd eval
python - <<'PY'
import json
ctx = "PADDING. " * 2000 + " Basis: 2015 total revenue $6,779,511 thousand"
rows = [{"query": "Total revenue?", "hop": "single", "golden_answer": "$6,779,511",
         "answer": "$6,779,511 [cite:1]", "ctx_text": ctx,
         "retrieval_hit_frac": 1, "retrieval_full": True, "rank": 1,
         "citation_recall": 1, "n_citations": 1, "n_rounds": 1}]
json.dump({"agg": {}, "rows": rows}, open("results_single.json", "w", encoding="utf-8"))
PY
python dump_judge_units.py single
python - <<'PY'
import json
i = json.load(open("_judge/judge_00.json", encoding="utf-8"))["items"][0]
print(len(i["contexts"]), "Basis" in i["contexts"])
PY
```

Expected output: `8027 True` — the judge can see the basis. Then temporarily change `CTX_CAP` in [dump_judge_units.py:21](../../eval/dump_judge_units.py#L21) to 5000, and re-run the last two steps: the output becomes `5000 False` — the evidence supporting the answer has been eaten by the head truncation, and the judge will **necessarily** judge it unfaithful. This is exactly the mechanism behind R5.H1's false "17% hallucination" report: the further back the evidence sits, the more unjustly it gets penalized. Change CTX_CAP back to 200000 when you're done.

### Experiment 2: exercising the fingerprint gate — trigger "results re-run without re-judging, refuses to output numbers" yourself

```bash
cd eval
python - <<'PY'
import json, os
os.makedirs("_judge", exist_ok=True)
rows = [{"query": "Q1", "hop": "single", "golden_answer": "A", "answer": "B [cite:1]",
         "ctx_text": "C", "retrieval_hit_frac": 1.0, "retrieval_full": True, "rank": 1,
         "citation_recall": 1.0, "n_citations": 1, "n_rounds": 1}]
json.dump({"agg": {}, "rows": rows}, open("results_single.json", "w", encoding="utf-8"))
json.dump({"pass1": [{"id": "single#000", "faithful": True, "correct": True}],
           "pass2": [{"id": "single#000", "faithful": True, "correct": True}]},
          open("verdicts.json", "w", encoding="utf-8"))
json.dump({"single#000": "deadbeef0000"}, open("_judge/fingerprint.json", "w", encoding="utf-8"))
PY
python aggregate.py        # should SystemExit: "alignment failed ... refusing to output numbers"
python - <<'PY'
import json, hashlib
json.dump({"single#000": hashlib.sha1("Q1".encode()).hexdigest()[:12]},
          open("_judge/fingerprint.json", "w", encoding="utf-8"))
PY
python aggregate.py        # after fixing the fingerprint: prints the single report (n=1, judged=1, correctness 1.000)
```

The first run triggers the "refuse to output numbers" branch at [aggregate.py:71](../../eval/aggregate.py#L71) — a hands-on feel for "guarding against mismatched attribution is mechanically enforced, it doesn't rely on people remembering to be careful." When you're done, delete `results_single.json`, `verdicts.json`, and `_judge/` under `eval/`.

### GPU/WSL extra (prerequisite: WSL + conda custodian + a 4090, first run `sudo systemctl stop custodian`)

- Full ACL regression: `python eval/acl_regression.py; echo exit=$?` — all five sections, all 65 assertions PASS, exit code 0, with the fifth section live-demonstrating "0 leaks even with the exit gate disabled."
- Tier1 smoke test: `python eval/run_eval.py --mode both --judge deepseek --limit 5` (requires DEEPSEEK_API_KEY in .env; if there's no gold, first run `python eval/gen_gold.py --per-doc 6`) — prints the five metrics per question, and finishes by printing the two-layer attribution Δ; the rows contain ctx_text and the retried/retry_kept fields.

---

## 8. Honest boundaries

When asked "what are the weaknesses of this evaluation setup" in an interview, proactively handing over this list is stronger than being asked into it piece by piece:

1. **The homology bias in synthesized gold hasn't been quantified.** The question's phrasing overlapping with the golden chunk's vocabulary makes the retrieval-side scores optimistically biased; mitigations exist (banning referential phrasing in the prompt, adding multi-hop questions), but there's no controlled experiment measuring "how optimistic, exactly." Talking point: "I know the direction, not the magnitude — quantifying it would need a batch of human-paraphrased control questions, which is on the backlog."
2. **The table QC gate guards against fabrication but not misassignment** (eval#2, confirmed and pending). At the small sample size of 16 questions, 1–2 bad gold items amounts to a 6–12pp swing; the right next step is an answer read-back retroactive audit, not rushing to regenerate the exam and incur another scope discontinuity.
3. **The table question coverage is structurally skewed** (eval#3, confirmed and pending). Filling the cap in alphabetical order meant only 1 of 6 non-English-language research reports made it into the exam — a cross-language table weakness might not even be on the exam by chance — the README's "full coverage across 15 documents" was the accidental outcome of one particular run.
4. **The three-way comparison is systematically unfavorable to agentic** (eval#0, confirmed and pending a re-run). The Δ−0.097's direction has multiple independent pieces of supporting evidence, but the magnitude is overstated; this caveat must accompany the number wherever it's cited.
5. **Tier2 is not reproducible in-repo.** The dual-Claude judge depends on Claude Code multi-agent orchestration, and the workflow isn't committed to the repo; a different environment needs its own cross-vendor judge. This is the price paid for "data never leaves the trust boundary," and it's honestly labeled by tier rather than glossed over.
6. **Sample size and corpus boundary.** Cross-document multi-hop is n=5, so its absolute value is for reference only; the whole set of conclusions is "performance on this specific corpus of 15 documents (5 English papers + 4 English financial reports + 6 non-English research reports)," not something to extrapolate beyond. The single-run noise floor is ±2 questions.
7. **The judge metrics haven't been calibrated against human annotation.** Dual-judge AND guards against a single judge having a bad moment, but the agreement rate between "Claude's judgment vs. human judgment" has never been measured — strictly speaking, faithfulness ≈1.0 means "two independent Claude passes both believe there's no hallucination," not a human-verified conclusion (the closest we've come is the manual re-review of the 16 unfaithful verdicts during the CTX_CAP postmortem).
8. **There are a few known drifts between the docs and the code.** eval/README has one section header that says "four metrics" (actually 5 programmatic fields + 2 judge fields); historical review documents record 44 ACL assertions (now 65) — the code is the source of truth, and every anchor in this chapter was verified against the current code.
9. **"Negative refusal" has no dedicated test coverage.** RGB lists "refusing when the corpus genuinely has no answer" as one of RAG's four core capabilities; our gold is entirely generated from the corpus and every question has an answer, so this capability is only indirectly covered by "the closed pipeline deterministically refuses on zero recall" — **whether the system will confidently answer wrong when it retrieves a similar-but-not-actually-answering distractor block has not been tested.** The fix is clear: construct a batch of unanswerable questions (asking about entities/numbers outside the corpus), with the refusal rate as the acceptance criterion — cheap and programmatically judgeable.

---

*Series navigation: [06 Agentic and MCP](06-agentic-mcp.md) ← this chapter → [08 Service Architecture](08-service-architecture.md); the methodology story collection is in [10 Methodology Stories](10-methodology-stories.md), and the interview quick-reference is in [11 Interview Q&A](11-interview-qa.md).*
