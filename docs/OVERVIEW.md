# Custodian — Multi-Format Agentic RAG System, Overview

> The top-level entry-point document. System overview, component relationships, core design, evaluation conclusions, and current status, all in one place. New readers should start here.
> Deeper detail: [DESIGN](DESIGN.md) · [IMPLEMENTATION](IMPLEMENTATION.md) · [API](API.md) · [OPERATIONS](OPERATIONS.md) · [TESTING](TESTING.md) · [PROVENANCE](PROVENANCE.md).
> Component detail: [chunker](components/chunker/ARCHITECTURE.md) · [embedder DESIGN](components/embedder/DESIGN.md) · [generator DESIGN](components/generator/DESIGN.md) · [mcp-server](components/mcp-server.md) · [eval](../eval/README.md).
> Full review process: [methodology/REVIEW_PLAN.md](methodology/REVIEW_PLAN.md).

## 1. What This Is

Custodian is a **self-contained, single-repo, multi-format agentic RAG system**: it turns PDFs / scanned documents / docx / pptx / xlsx into a local knowledge base (running on a single 4090 machine) that **answers questions, enforces enterprise-grade ACLs, and gives traceable citations**. This isn't a chunking experiment — every key decision is backed by measurement, the whole system has been through a de-biased 88-question evaluation plus multiple rounds of adversarial review, and it has been proven out on 77 real documents. The retrieval engine and the deployment shape aimed at small teams (dual entry points + multi-identity + observability + systemd management) are now **merged into a single repository**, installed all at once with `pip install -e '.[dev]'` (src-layout, editable).

**Two consumption modes, sharing the same retrieval engine:**
- **Closed pipeline** (`generator` + DeepSeek): one question, one answer, deterministic, evaluable — **the default recommendation** (evaluation proves it's both the best and the cheapest).
- **Agentic** (three MCP entry points, sharing the same `toolcore`): the retrieval engine is exposed as MCP tools, and an agent (e.g., Claude Code) drives multi-hop retrieval interactively.

## 2. The Whole Picture: Indexing Side + Query Side + One Thread Running Through Both

```
Index side:  Files ─parse(MinerU)─► Element[] ─chunk─► Chunk[] + heading skeleton ─embed─► Qdrant + sidecar
Query side:  Question ─encode─► hybrid retrieval (dense + BM25, RRF) ─► rerank (optional) ─► hard ACL filter
              ─► query-time small-to-big (assemble_big takes the surrounding region) ─► generate (LLM + grounding + [cite:n]) ─► cited answer
Throughline: ACL fail-closed guards every step from the chunk stamp all the way to the generation exit; multimodal text and images share one embedding space; the evaluation loop quantifies every step
```

## 3. The Five Components (each: design → implementation → adversarial sign-off → evaluation; hardened over review rounds R1–R5)

| Stage | Component | What it does | Status |
|---|---|---|---|
| **parse** | MinerU client | 5 formats → unified content_list/layout | ✅ |
| **chunk** | `chunker` | Element[] → heading-tree chunking; query-time small-to-big | ✅ signed off + evaluated |
| **embed + retrieve** | `embedder` | Qwen3-VL dense + BM25 → Qdrant hybrid + hard ACL filter + optional rerank + small-to-big | ✅ signed off + evaluated |
| **generate** (closed pipeline) | `generator` | retrieve → prompt → LLM → answer + `[cite:n]` + grounding; LLM is pluggable (DeepSeek V4 Flash) | ✅ end-to-end evaluated (88 questions) |
| **consume** (agentic) | `src/custodian` (`toolcore`) | the retriever exposed as **6 MCP tools** (retrieve/list_documents/get_document/get_outline/expand/retrieve_grouped), reachable through three entry points (`custodian serve` / `custodian mcp` / `custodian mcp --direct`); the ACL identity bound at startup cannot be tampered with | ✅ tool unit tests + contract-drift tests + real-index connectivity |
| **measure** | `eval` | de-biased evaluation loop: synthesize gold → run the system → judge (Tier1 deepseek, reproducible / Tier2 dual-Claude, authoritative) → five metrics + two-layer attribution + ACL regression | ✅ see §8 |

**A single pytest suite** (product surface + engine surface, the latter under `tests/engine/`; baseline counts in [TESTING.md §1](TESTING.md)). The CI gate splits into two levels: CPU CI = pytest (including the embedder's `test_acl.py` ACL predicate tests); GPU pre-release gate = `eval/acl_regression.py` (WSL+4090, end-to-end zero leakage, does not run in CPU CI).

## 4. Highlight ①: An Eager, Cheap Skeleton + Query-Time Small-to-Big

> **Naming note**: this was originally called "Lazy Heading-Tree." An adversarial test on 77 documents / 5337 headings **disproved the "lazy" premise** — building the full tree eagerly takes just 66.6ms. The real architecture is an **eager skeleton + assembling only the big-block at query time**.

- **Eager skeleton-building at index time**: multiple signals (text_level + numbering correction + TOC stripping) reconstruct the heading hierarchy, cut leaf chunks, and attach a breadcrumb + `section_anchor` to every chunk. Sub-millisecond, deterministic, zero LLM calls.
- **Only the big-block is assembled at query time** (`assemble_big`): a leaf hit → read its attached breadcrumb and, by token budget, **take the surrounding region** (climb to the parent section if too small; open a window within the ancestor to pull in sibling sections if it would exceed the max). Material is pulled **ACL-aware** by index from the original elements. The sidecar stores the original elements/sections.
- **Self-reported return status** (`context_status`): full_section/climbed_N = a complete section, usable as-is; **section_window** = a token-limited window (incomplete, expandable); **asset_no_prose** = an asset page whose data is in content_raw; single_chunk_*/deduped/omitted_budget/already_returned each have their own semantics.

## 5. Highlight ②: End-to-End ACL Fail-Closed

Permissions run through everything: chunk stamping → embedding splits into 4 fields → hard pre-filtering at retrieval (the Qdrant filter is **pushed down into every prefetch** — a pitfall where embedded fusion drops the top-level `should` clause) → small-to-big never pulls material across ACL boundaries → a second check at the exit (rejects even when acl=None) → generation is only fed authorized context. **A document the user has no access to is "simply unretrievable," not "hidden."** Regression: all 44+ assertions in `eval/acl_regression.py` pass, including one that **disables the exit-point acl_admits check and confirms zero cross-tenant recall anyway** — proof that the prefetch push-down in RRF fusion **itself** blocks unauthorized access (not something that relies on the exit check as a backstop).

## 6. Highlight ③: Grounding for Table/Chart Numbers (the "③" fix) + Multimodal + Hybrid Retrieval

- **③ Table/numeric grounding**: evaluation found that "a number in a table is retrieved but the system still can't answer it" — the real root cause was that an asset chunk's `content_raw` (table HTML / chart data) was being collapsed by section-level deduplication, and then excluded by big-block assembly. Fix: asset chunks are exempted from section deduplication during retrieval, and the generator adds content_raw back in when an asset is hit. Measured result: 4 table questions went from "insufficient information" to answering the correct number, and single-hop correctness reached 0.97.
- **Text and images share one embedding space** (Qwen3-VL): text and images are encoded into the same vector space (description↔matching-image similarity 0.74/0.49), so image-only chunks can be retrieved by a text query.
- **Hybrid retrieval**: dense (meaning) + BM25 (words), fused with RRF; optional Qwen3-VL-Reranker for precise reranking (measured: hybrid 0.566 → after rerank 0.867); BM25 chosen for the sparse side.

## 7. Evaluation: Two Layers, Component-Level and System-Level

**Component level (chunking / retrieval, synthetic set, trends are trustworthy but absolute values are reference only):** the moat for the in-house chunker is engineering integration (source_indices/heading-tree/ACL), not the boundary-detection algorithm (ours 79.5% vs. chonkie 74.9%); hybrid+RRF beats any single method (0.541 > 0.510 > 0.449); reranking substantially improves quality (→0.867); BM25 was chosen for sparse. Evaluation splits into three tracks: `eval/` (end-to-end, five metrics, the crown jewel) · `eval/component_retrieval/` (BM25/BGE/RRF retrieval components) · `eval/component_chunking/` (chunking evidence fidelity). Details in [components/embedder/EVALUATION.md](components/embedder/EVALUATION.md) and [methodology/CHUNKING_EVALUATION.md](methodology/CHUNKING_EVALUATION.md).

**System level (end-to-end RAG, `eval/`): gold set = 88 questions (72 prose + 16 table)**, split by hop count into single 54 / multi_intra 29 / multi_cross 5, with all table questions being single-hop. Reproducibility splits into two tiers: **Tier1** (`--judge deepseek`, reproducible in-repo, same-vendor trend) vs. **Tier2** (dual-Claude authoritative, not reproducible in-repo — requires an external Claude Code orchestration run to produce `verdicts.json`).

**Tier2 authoritative numbers (dual-Claude, corrected in R5, closed pipeline single-hop, 72-question set):**

| Metric | single (closed pipeline) | agentic | decompose |
|---|---|---|---|
| **Faithfulness** | **≈1.000** | 1.000 | 0.972 |
| Correctness | 0.847 | 0.750 | 0.831 |
| Retrieval recall | 0.854 | 0.840 | 0.852 |

**Tier1 baseline (deepseek judge, 88 questions, closed pipeline single-hop):** retrieval 0.818 / MRR 0.627 / citation 0.767 / faithfulness 0.977 / correctness 0.818.

> ⚠ **Comparability break**: the table above uses the 72-question set, which predates the table-question expansion — 88 vs. 72 **cannot be directly compared**. Both are kept, each clearly labeled. By hop count, correctness (72-question basis): single-hop 0.97 / single-document multi-hop 0.83 / cross-document multi-hop 0.00 (n=5). Two-layer attribution (paired): single→agentic Δ**−0.097**, single→decompose Δ**−0.014**.

**Three hard conclusions:**
1. **Faithfulness ≈ 1.0, grounding is nearly watertight** — the system would rather answer "no relevant information" than fabricate one, and `[cite:n]` traceability is reliable.
   > ⚠ An earlier report claimed "faithfulness 0.83 / 17% ungrounded claims" — R5's self-review found that was an **eval bug** (the judge's context was being truncated, seeing only 40% of it); after feeding it the full context and re-judging, this was corrected to ≈1.0. **Lesson: a bug in the evaluation pipeline itself can manufacture a fake "conclusion" out of thin air.**
2. **Agentic / decompose are net negative** → **the closed pipeline should be the default**: agent orchestration makes each individual hop ≤ a single hop's quality (more retrieval means more dilution from distracting chunks), with only a weak advantage on cross-document questions, and overall it falls short.
3. **The remaining correctness bottleneck is purely cross-document synthesis** (0.00, n=5): the system retrieves chunks from both documents but can't synthesize a comparison from them — this is a synthesis problem, not a retrieval problem; it's a genuinely hard research problem with low marginal payoff, so it hasn't been pursued further.

## 8. Adversarial Review R1–R5 (Systematic Review of the Whole Repo, [methodology/REVIEW_PLAN.md](methodology/REVIEW_PLAN.md))

| Round | Scope | Result |
|---|---|---|
| R1 | ACL / security closure | **0 confirmed** (clean; the fusion push-down was proven to itself block unauthorized access via the "disable the exit gate" test) |
| R2 | Retrieval correctness + ③ | 7 confirmed, 6 fixed (e.g., a windowed chunk mislabeled full_section) |
| R3 | generator + citations + prompt | 9→6, 6 fixed (neutralizing passage injection, surfacing finish_reason, gating "thinking" by backend) |
| R4 | MCP tool surface | 15→7, 7 fixed (content_raw bypassing the budget + unclear degradation, section_window deduplication across calls) |
| R5 | eval methodology | 16→6 (2 HIGH), 6 fixed + re-judged (**this is what caught the faithfulness bug above**) |

**Cross-round meta-lesson: a fix in one round (③) can send ripples that surface new regressions in downstream layers (generation/MCP/eval) — only a systematic, cross-round adversarial review catches this** (R4 caught R2's aftereffects, R5 caught the eval bug). Adversarial verification also rejected roughly 40 exaggerated/false-positive findings along the way (filtering noise, keeping only real problems).

## 9. Engineering Methodology

- **Test-set-driven**: sampled 77 documents / 1867 pages to derive the strategy inductively; the evaluation loop is what settles disputes with data.
- **Diagnose → fix → verify discipline**: no blind fixes (the initial diagnosis for ③, "content_raw wasn't being fed in," was overturned by question-by-question tracing; the faithfulness bug was pinned down by directly measuring the CTX_CAP length). Every "fixed" claim comes with a reproduction plus a before/after comparison.
- **Adversarial sign-off + systematic review**: each component gets a red-team pass before sign-off; the whole stack went through 5 rounds of review, R1–R5.
- **Every decision is backed by measurement plus an honest caveat** (acknowledging same-source bias in the semantic evaluation set, a small n=5 for cross-doc, and the faithfulness bug).

## 10. Current Status + How to Use It

**Feature-complete**: all five components + three MCP entry points are built, the single pytest suite is fully green (baseline counts in [TESTING.md §1](TESTING.md)), plus an 88-question de-biased evaluation and 5 rounds of adversarial hardening; git is clean. Current version **v0.3.0**.

**In production**: the production index `~/rag_real` (`CUSTODIAN_INDEX_DIR`) holds roughly **77 real documents / 7652 chunks** (across 14 categories: papers/financial reports/research reports/regulations/government documents/manuals/slide decks/NASA technical reports/news, etc.); the evaluation index `~/rag_eval_big` (evalbig) holds roughly 15 documents / 1409 chunks. Configuration is unified through a single `.env` at the repo root (the `CUSTODIAN_*` namespace; `RAG_*`/`RAG_EVAL_*` are kept only as one deprecated alias generation), with the corpus directory at `CUSTODIAN_CORPUS_DIR` and the index directory at `CUSTODIAN_INDEX_DIR`.

**How to use it (three entry points, sharing the same `toolcore`)**:
- `custodian serve` — the HTTP long-running daemon, holding the embedded Qdrant lock plus the GPU model; the core of team multi-user support, observability, and systemd management.
- `custodian mcp` — a stdio→HTTP adapter, letting an agent (e.g., Claude Code) connect to the long-running service.
- `custodian mcp --direct` — the stdio direct-connect fallback path with no daemon.

Getting started: ① (optional) rebuild the index pointing at your corpus; ② start `custodian serve` (the first query lazy-loads the model, taking about 1–2 minutes), or configure `custodian mcp` on the agent side; ③ just ask.
- ✅ **Strengths**: single-document facts, table numbers, single-hop Q&A — trust the `[cite:n]` citations (faithfulness ≈1.0).
- ⚠ **Weaknesses**: deep synthesis/comparison across multiple documents (cross-doc 0.00) — cross-check this yourself.
- To measure the effect of any change: the `eval/` loop is the ruler (see [OPERATIONS.md](OPERATIONS.md) for operational/reproduction details, and [TESTING.md](TESTING.md) for the test gate).

**Already delivered (a former non-goal)**: back when this was a single-user personal engine, "a long-running HTTP daemon / multi-session management" was listed as **explicitly out of scope**. After merging into Custodian, these are exactly the current shape — `custodian serve`'s HTTP daemon, team multi-identity support (D10), and observability (D11) are all **complete** and are no longer non-goals.

**Still explicitly out of scope (a decision, not a debt — see [ROADMAP.md](ROADMAP.md))**: improving cross-document synthesis (a hard research problem with low marginal payoff), an MCP tool for retrieving images (image_path can't be resolved remotely).

## Document Map

```
docs/OVERVIEW.md                                 ← this document (system entry point, read this first)
docs/{DESIGN,IMPLEMENTATION,API,OPERATIONS,TESTING,ROADMAP}.md  design / implementation / interface / operations / testing / roadmap
docs/{PROVENANCE,COMPONENT_NOTES}.md             provenance (origins/decision history) + component notes
docs/methodology/{LAZY_HEADING_TREE_DESIGN,MULTIFORMAT_IMPL,CHUNKING_EVALUATION,REVIEW_PLAN}.md   methodology + review plan
docs/components/{chunker,embedder,generator}/*.md + components/mcp-server.md   component documentation
docs/archive/*                                   historical archive (original engine-repo DESIGN/IMPLEMENTATION, process logs)
eval/README.md                                   the end-to-end RAG evaluation loop (de-biased / two-layer attribution / ACL regression / Tier1·Tier2)
```
