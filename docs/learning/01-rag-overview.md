# 01 RAG Overview and Custodian Architecture

> **How to read this piece**: This is the master outline for the whole set of learning docs. It first explains why RAG exists and where naive RAG stumbles (the problem spectrum), then maps Custodian's six stages onto that spectrum one by one, and finally walks through "the complete lifeline of one request."
> **Interview weight: ★★★★★** — "Tell me about your project" is guaranteed to be the first question, and this piece is the skeleton of the answer.
> **Prerequisite reading**: None. After this piece, jump into whichever sub-piece interests you: [02 Parsing and Chunking](02-parsing-chunking.md) · [03 Retrieval](03-retrieval.md) · [04 ACL Security](04-acl-security.md) · [05 Generation and Grounding](05-generation-grounding.md) · [06 Agentic and MCP](06-agentic-mcp.md) · [07 Evaluation Methodology](07-evaluation.md) · [08 Service Architecture](08-service-architecture.md) · [09 Scaling Out](09-scale-out.md) · [10 Methodology and Stories](10-methodology-stories.md).

---

## 1. Conceptual foundation: why retrieval augmentation is needed

### 1.1 Three structural flaws of LLMs

Setting aside any specific project, every RAG system is answering the same question: **an LLM's parametric memory cannot be used as a database.** Broken down, that's three flaws:

1. **Knowledge has a cutoff date, and it can't see private data.** Your company's earnings report from last week, or internal regulations, are in no model's training set.
2. **Parametric memory is not addressable and cannot be traced back to a source.** The model "remembers" something, but you cannot ask it "which page of which document is this sentence based on" — and if it's wrong, there's no way to check.
3. **Without evidence, it tends to fabricate (hallucinate).** In enterprise scenarios, a fabricated number is far more costly than "I don't know."

RAG (Retrieval-Augmented Generation) really does one thing at its core: **replace "recalling from parameters" with "retrieve on the spot, then read on the spot"** — before answering, retrieve evidence from an external knowledge base, insert that evidence into the prompt, and let the model answer based on the evidence. This way knowledge can be updated at any time (just rebuild the index), answers can be traced (citations point back to the source text), and when there's no evidence, the system can refuse to answer (if the evidence set is empty, don't answer).

### 1.2 Naive RAG and its spectrum of problems

The most naive RAG pipeline has five steps: **chunk documents → vectorize and index → vectorize the user's question → take the top-k most similar chunks → stuff them into the prompt and generate**. Ten lines of LangChain code will get it running, but it will stumble systematically on real corpora. It's worth laying the pitfalls out as a spectrum — because **the maturity of a RAG system is precisely its coverage of this spectrum**:

| # | Problem | Symptom |
|---|------|------|
| P1 | **Parsing distortion** | PDF/scanned documents/pptx are not plain text; headers, footers, and watermarks bleed into the body text, tables turn into garbage |
| P2 | **Chunking breaks semantics** | Fixed-window chunking cuts through tables, clauses, formulas; deeper still: **retrieval wants small chunks (semantically focused), generation wants large chunks (contextually complete)** — one granularity can never satisfy both ends |
| P3 | **Single-route recall blind spots** | Pure vector retrieval loses exact strings (statute numbers, model numbers, amounts); pure keyword retrieval loses synonymous rephrasing |
| P4 | **Multimodal blind spots** | Charts and image-only pages are simply not retrievable, yet the key data in earnings reports/papers is exactly in those charts |
| P5 | **Permission blind spots** | Not everyone in an enterprise's document store can see every document; filtering after retrieval means unauthorized content has already contaminated the top-k |
| P6 | **Generation isn't faithful** | Even correct recall gets fabricated on top of; citation numbers are decorative and don't line up with the source when you click through; the indexed documents themselves may also be a channel for prompt injection |
| P7 | **Consumption pattern is monolithic** | A one-question-one-answer pipeline can't do multi-hop; conversely, is an agent that retrieves freely actually better or worse than a pipeline? Most projects have never measured it |
| P8 | **Not measurable** | You changed the chunk size, swapped the retrieval strategy — did it actually get better? Without an evaluation loop, every optimization is guesswork |

The mainstream evolutionary path can also be understood through this table:

- **Naive RAG**: the five-step pipeline, addressing none of the above — every row in the table above applies.
- **Advanced RAG**: adds machinery at both ends of the pipeline. Before retrieval (query rewriting/expansion), during retrieval (hybrid dual-route, metadata filtering), after retrieval (rerank for fine ranking, small-to-big / parent-document context expansion). Mainly targets P2/P3.
- **Modular / agentic RAG**: retrieval is no longer a fixed stage in a pipeline but is exposed as a tool, and an LLM agent decides for itself when to retrieve, what to retrieve, and whether to go multi-hop. Targets P7 — but note that "an agent is always better" is an untested intuition, and later sections will beat it down with data.
- **Evaluation side**: frameworks like ragas and TruLens give a vocabulary of metrics (faithfulness / answer relevancy / context recall), addressing P8; but there are two deep pitfalls most teams never touch: one is the **circular bias that comes from the judge and the system under test being from the same vendor**, and the other is **bugs in the evaluation pipeline itself** manufacturing false conclusions out of thin air (Custodian really hit this — see [Piece 07](07-evaluation.md)).

### 1.3 Answering a frequently asked interview question along the way: why RAG, not fine-tuning or long context?

The three are not competitors — they're answers to different problems, and it's worth being able to draw the boundary in one sentence during an interview:

- **Fine-tuning** changes a model's **behavior and style**; it's not suited for injecting frequently-updated facts (every update requires retraining, and facts still can't be traced and still get hallucinated);
- **Long context** (stuffing the entire knowledge base into the prompt) has two hard constraints: cost, and "needle-in-a-haystack" style decay — 77 documents at several million tokens means every question has to re-pay the full cost, and recall of mid-context information drops off too;
- **RAG** is "bolt-on addressable memory": update = rebuild the index, traceability = citations pointing back to chunks, permissions = filtering at the retrieval layer. The price is bringing in an entire pipeline that needs to be engineered — and that pipeline is exactly what this project is about.

**Custodian's positioning**: a self-contained, single-repo, multi-format agentic RAG system. It offers a **measurement-backed** answer to every one of P1–P8, and runs on a local 4090, on 77 real documents (7,652 chunks), as a team service with multi-identity authentication, observability, and systemd management. It's not a toy pipeline and not a stack of frameworks — every key decision comes with rejected alternatives and data.

---

## 2. How Custodian does it: a map of six stages

### 2.1 The full data flow

```
Index side:   file ─parse(MinerU)→ Element[] ─chunk→ Chunk[] + heading skeleton + ACL stamp ─embed→ Qdrant + sidecar
Query side:   question ─encode→ hybrid recall (dense+BM25, RRF, hard ACL filter) ─→ rerank (optional)
              ─→ small-to-big (assemble_big pulls the surrounding region) ─→ generate (grounding + [cite:n]) ─→ answer with citations
Consumption:  closed pipeline (HTTP /v1/ask) ⊕ agentic (MCP, 6 tools, agent-driven)
Measurement:  eval loop (synthesize gold → run the real system → cross-vendor judge → five metrics + attribution)
```

The six stages map one-to-one onto the problem spectrum:

| Stage | Component | Which pitfall it answers | One-line answer | Deep-dive piece |
|------|------|-----------|-----------|--------|
| parse | MinerU client + adapter | P1 | 5 formats unified into `Element[]`; swap parsers = swap the adapter, the core is untouched | [02](02-parsing-chunking.md) |
| chunk | `chunker` | P2 | heading-tree reconstruction + eager skeleton for small chunks; query-time small-to-big assembles large chunks | [02](02-parsing-chunking.md) |
| embed + retrieve | `embedder` | P3/P4/P5 | Qwen3-VL puts text and images in the same space + BM25, fused via RRF; ACL filter pushed down into the recall layer | [03](03-retrieval.md) / [04](04-acl-security.md) |
| generate | `generator` | P6 | grounding constraint + `[cite:n]` citation protocol + deterministic refusal on zero recall | [05](05-generation-grounding.md) |
| consume | `src/custodian` | P7 | one daemon process, two exits: HTTP closed pipeline + MCP agentic, sharing the same toolcore | [06](06-agentic-mcp.md) / [08](08-service-architecture.md) |
| measure | `eval/` | P8 | debiased evaluation loop: cross-vendor judge, five metrics, paired attribution | [07](07-evaluation.md) |

Below is a walk through the key mechanisms of each stage (details are in the individual pieces; this is only meant to establish the map).

### 2.2 parse: unifying at the Element seam

All five formats (PDF / scanned documents / docx / pptx / xlsx) go through MinerU for parsing, and the output is converted by an adapter into a unified `Element[]` — this is a deliberately designed seam: swapping parsers means writing a new adapter, and the core chunking logic doesn't change a single line ([src/chunker/adapters/mineru.py:1](../../src/chunker/adapters/mineru.py#L1)). Why MinerU instead of building one in-house or using Tika? It comes down to a three-way coverage comparison (under a fair measurement, MinerU comes out ahead at 95.1% — see Piece 02 for details, which also contains a measurement-methodology story about how "a bug in the coverage extractor itself initially flipped the conclusion").

### 2.3 chunk: eager skeleton + query-time small-to-big

This step directly answers P2's tension between "retrieval granularity vs. generation granularity," in two passes:

- **Index time**: rebuild the heading tree from multiple signals (mainly `text_level`, with numbering-based correction, [src/chunker/core.py:158](../../src/chunker/core.py#L158) `heading_level` / [core.py:179](../../src/chunker/core.py#L179) `build_sections`), and cut leaf chunks on that tree ([core.py:243](../../src/chunker/core.py#L243) `Chunker`). Every chunk carries a breadcrumb, a section anchor, and gets **stamped with an ACL** — defaulting to RESTRICTED, fail-closed ([core.py:33-36](../../src/chunker/core.py#L33)). Building the whole tree measured at 66.6ms, so it's eager rather than lazy (the early design was called "Lazy Heading-Tree," and it was overturned by measurement — the document's name is kept as fossil evidence).
- **Query time**: after hitting small chunks, `assemble_big` pulls the surrounding region against a token budget — if the section is too small it climbs up to the parent section; if it exceeds the budget it opens a window within the ancestor to pull in siblings ([src/chunker/retrieve.py:109](../../src/chunker/retrieve.py#L109)). The raw elements live in the sidecar, and material is pulled ACL-aware throughout.

In short: **small chunks at index time protect retrieval precision, large chunks at delivery time protect generation context — the two granularities are decoupled.**

### 2.4 embed + retrieve: same-space multimodal + hybrid + hard ACL filtering

- **Multimodal (P4)**: Qwen3-VL encodes text and images into the same vector space, so image-only chunks can be recalled directly by a text query (description↔corresponding-image similarity of 0.74/0.49); table chunks additionally synthesize a retrieval signal text (column headers/row labels), while the data itself is stored in `content_raw`. The indexing entry point is [src/embedder/embed.py:56](../../src/embedder/embed.py#L56) `index_document`.
- **Hybrid (P3)**: dense (semantic) + BM25 (exact string) dual-route recall, fused server-side by Qdrant with RRF — RRF only uses rank, not score, which naturally solves the two routes' incomparable score scales ([src/embedder/store.py:96](../../src/embedder/store.py#L96) `hybrid_search`). An optional cross-encoder rerank is available, and it degrades gracefully on failure rather than crashing the query.
- **ACL (P5)**: the filter is **pushed down into each prefetch route** ([store.py:118-123](../../src/embedder/store.py#L118)) — content the user has no access to never even enters the candidate set, rather than being filtered out after recall. There's a well-known trap here: embedded Qdrant, in fusion mode, drops the top-level filter's `should` clauses, and only pushing the filter down to the prefetch level actually takes effect ([store.py:103-105](../../src/embedder/store.py#L103) comment records the trace). There's a second `acl_admits` re-check at the exit as well ([src/embedder/retrieve.py:157-160](../../src/embedder/retrieve.py#L157)); the full derivation of the three gates is in [Piece 04](04-acl-security.md).
- **Query-time orchestration**: `search_with_context` chains recall → section dedup → small-to-big → exit validation, and every hit carries a `context_status` state machine (full_section / section_window / asset_no_prose…), turning "context completeness" into an explicit signal the agent can program against ([src/embedder/retrieve.py:119](../../src/embedder/retrieve.py#L119)).

### 2.5 generate: the closed pipeline and grounding

`Generator.answer` is the heart of the closed pipeline: retrieve → assemble context → prompt → LLM → resolve citations ([src/generator/generate.py:35](../../src/generator/generate.py#L35)). Three key mechanisms:

- **The `[cite:n]` citation protocol**: kept separate from any bare `[n]` tokens in the body text, to prevent footnote numbers in the retrieved body from being mistakenly picked up, and to prevent malicious chunks from forging citations; out-of-range numbers are simply discarded rather than mapped to the wrong source ([generate.py:116-128](../../src/generator/generate.py#L116)).
- **Deterministic refusal on zero recall**: when context is empty, the code directly returns "insufficient information" rather than handing the decision to answer over to the LLM ([generate.py:107-109](../../src/generator/generate.py#L107)) — the grounding floor doesn't rely on the model behaving.
- **Feeding in asset data (fix ③)**: when a table/chart is hit, `content_raw` is added into the context ([generate.py:75-80](../../src/generator/generate.py#L75)), fixing the "the numbers inside the table were recalled but couldn't be answered" problem — a real bug caught by the evaluation loop, with the full diagnostic story in [Piece 05](05-generation-grounding.md).

### 2.6 consume: one daemon process, two consumption modes

All consumption funnels into the `custodian serve` daemon — the **only** process in the system that touches the embedded Qdrant and the GPU model ([src/custodian/service.py:1-16](../../src/custodian/service.py#L1)). Why must it be a daemon: embedded Qdrant has a single-client exclusive lock and the 8B model takes 1-2 minutes to load, so "each consumer spinning up its own process" would both fight over the lock and re-pay the load cost every time.

Two consumption modes, sharing the same retrieval engine:

- **Closed pipeline** (`POST /v1/ask`, [service.py:312](../../src/custodian/service.py#L312)): retrieve→grounding→DeepSeek→answer with citations. One question, one answer, deterministic, evaluable, **the recommended default**.
- **Agentic** (6 MCP tools: retrieve / list_documents / get_document / get_outline / expand / retrieve_grouped): the retrieval engine is exposed as tools, and an agent (such as Claude Code) decides for itself when to retrieve, how to rewrite the query, and whether to go multi-hop. Tool semantics (validation/dedup/budget/error mapping/usage contract) are all collected into a single, transport-agnostic source of truth called `toolcore` ([src/custodian/toolcore.py:1-13](../../src/custodian/toolcore.py#L1)), shared by both the stdio and HTTP bindings so the contract never drifts.

MCP has three entry points: `custodian serve` (HTTP), `custodian mcp` (a thin stdio→HTTP adapter, zero GPU, millisecond startup, [src/custodian/mcp_adapter.py:1-12](../../src/custodian/mcp_adapter.py#L1)), and `custodian mcp --direct` (a stdio direct-connect fallback for when there's no daemon), with the routing in [src/custodian/cli.py:40-47](../../src/custodian/cli.py#L40).

### 2.7 Closed pipeline vs. agentic: one engine, two consumption philosophies

This isn't just "two APIs" — it's two philosophies about "who holds the control flow," and it's worth aligning them explicitly:

| Dimension | Closed pipeline (`/v1/ask`) | Agentic (MCP tool surface) |
|------|-------------------|---------------------|
| Control flow | Fixed by the system: retrieve once → generate once | Decided by the agent: when to retrieve, how many rewrite rounds, whether to expand/multi-hop |
| Determinism | High — same query, same store, reproducible behavior | Low — depends on the agent's orchestration strategy |
| Evaluability | Strong, the eval loop measures it directly | Can only be measured as a whole (the agent under test is itself a variable) |
| Anti-hallucination mechanism | grounding SYSTEM + code takeover on zero recall | toolcore `_INSTRUCTIONS` usage contract (grounding/citation anchors/when to stop, [toolcore.py:20-43](../../src/custodian/toolcore.py#L20)) — an equivalent, but relies on the agent complying |
| Interaction shape | One question, one answer — good for scripts/frontends | Multi-turn exploration — good for open-ended tasks like "help me survey this batch of documents" |
| Measured correctness | Single-hop 0.97 (72-question basis) | Net negative Δ−0.097 (same basis; magnitude pending re-evaluation, see §8) |

Two design invariants pin both modes to the same semantics: **a single source of truth for the tool contract** (toolcore, zero drift between stdio and HTTP), and **the ACL identity is authoritative on the server side** (the agent cannot change its identity via parameters). And the closed pipeline's own "intelligence" is deliberately restricted to be **failure-driven** (smart-ask: only a refusal on a numeric question triggers a supplementary table-retrieval leg) — because the moment the closed pipeline starts "helping" on the success path, it degenerates into an invisible agent, and agent orchestration has already been measured as net negative (magnitude pending re-evaluation, see §8). This "bounded intelligence" principle is expanded on in [Piece 08](08-service-architecture.md).

### 2.8 measure: the evaluation loop

`eval/` is the ruler for the whole project: synthesize gold (88 questions = 72 prose + 16 table) → run the real system → four programmatic metrics (retrieval recall / full recall / MRR / citation recall) + two judge metrics (faithfulness / correctness), with the judge split into Tier 1 (DeepSeek, reproducible) and Tier 2 (dual Claude, authoritative, cross-vendor debiased) ([eval/run_eval.py:1-16](../../eval/run_eval.py#L1)). The core discipline: **the judge must be a different vendor from the system under test**, or the numbers carry circular bias. This loop doesn't just score — it also does paired attribution (single vs. agentic vs. decompose) — the Δ−0.097 in the previous section was ruled by this loop (magnitude pending re-evaluation, see §8: the eval's agentic assembly path was missing two production fixes).

For the Tier 1 baseline, one set of numbers is worth remembering just for the order of magnitude (deepseek judge, 88 questions, closed pipeline single): retrieval recall 0.818 / MRR 0.627 / citation recall 0.767 / faithfulness 0.977 / correctness 0.818. The Tier 2 authoritative basis (dual Claude, 72 questions): faithfulness ≈1.000, correctness 0.847. **The two bases coexist and are not comparable to each other** — why there's this discontinuity in basis, and how to handle it honestly, is the centerpiece of [Piece 07](07-evaluation.md).

### 2.9 The complete lifeline of one request

The thing that best proves in an interview "this system is really yours" is being able to walk through a request without notes. Take `POST /v1/ask {"query": "What was Company X's 2023 segment revenue?"}` as an example:

1. **Authentication and identity**: X-API-Key is resolved into an identity (name+tenant+principals); keys/legacy/open modes are auto-selected; binding to a non-loopback address forces keys mode, refusing to expose the whole knowledge base naked on the LAN ([service.py:93-95](../../src/custodian/service.py#L93)).
2. **Build a Generator** (lazily, per-thread, [service.py:304-310](../../src/custodian/service.py#L304)), entering `gen.answer`.
3. **Query encoding**: a dense vector (Qwen3-VL, LRU cached) + a BM25 sparse vector. The GPU forward pass happens inside a resource lock, the LLM network call happens outside all locks — see the locking model in [src/custodian/engine.py:6-12](../../src/custodian/engine.py#L6).
4. **Hybrid recall**: two prefetch routes (each carrying an ACL filter) → server-side RRF fusion → exit-side `acl_admits` re-check ([store.py:96-129](../../src/embedder/store.py#L96)).
5. **Section dedup + small-to-big**: hit small chunks → read the raw elements from the sidecar → `assemble_big` climbs the tree/opens a window to assemble the large chunk, with each one marked with `context_status` ([embedder/retrieve.py:119](../../src/embedder/retrieve.py#L119)).
6. **Context assembly**: table hits get `content_raw` added in; the section_path breadcrumb goes into the source line (otherwise the model doesn't know a number is segment data — in practice it will mistake segment revenue for total company revenue) ([generate.py:75-89](../../src/generator/generate.py#L75)).
7. **Prompt → DeepSeek**: the SYSTEM message declares the passages as untrusted data and requires `[cite:n]`; any `[cite:n]` that already appears inside a passage is neutralized (to prevent injection via indexed documents).
8. **Citation resolution**: the `[cite:n]` markers in the answer are mapped back to sources, and out-of-range ones are discarded ([generate.py:116-128](../../src/generator/generate.py#L116)).
9. **Smart-ask failure-driven retry**: when a numeric question is refused, a retry round is attempted with a `kind=table` retrieval leg, and **only a fully-answered retry is accepted** — a partial answer can smuggle in an incorrect "X was not provided" claim, and measurement showed faithfulness dropping from 1.0 to 0.93, so the system would rather keep the honest refusal ([service.py:331-353](../../src/custodian/service.py#L331)).
10. **Response and observability**: HTTP 200 + a status field (domain errors are not expressed through HTTP status codes, for programmatic consumption by agents); request logs record the identity name, not the key.

Every stage in this lifeline has a matching deep-dive piece; get comfortable narrating it, and any "dig deeper into one layer" follow-up in an interview has somewhere to land.

---

## 3. Why it's designed this way: rejected alternatives

An architecture narrative is persuasive because of "what was considered, and why it was rejected." Custodian's key rejections (data basis annotated):

| Decision | Rejected alternative | Reason for rejection and data |
|------|------------|--------------|
| **A daemon process holding resources exclusively** | Each agent session connects via stdio directly, each spinning up its own index | Embedded Qdrant has a single-client exclusive lock — a second process errors out immediately; the 8B model takes 1-2 minutes to load, and every session would re-pay it. With the daemon + a thin MCP adapter, a session connects in milliseconds |
| **The closed pipeline as the default** | Agentic (agent-driven multi-hop) as the default | Measured paired attribution (72-question basis): single→agentic correctness Δ**−0.097**, →decompose Δ−0.014. Every agent hop underperforms single-hop (multi-round retrieval = distractor chunks diluting context), only weakly ahead on cross-document questions. **Agentic is a capability option, not the default path** — this is the whole project's most counterintuitive, and most data-backed, conclusion (note: this review round found the eval's agentic path was missing two production-side fixes, so the **magnitude** of the net negative is in question and pending re-evaluation; the direction has no evidence overturning it yet — see [Piece 06](06-agentic-mcp.md)) |
| **Eager heading-skeleton construction** | Lazy (build the tree only at query time — sounds smarter) | Measured on 77 documents, building the whole tree took only 66.6ms total — the cost "laziness" saves doesn't exist. Lesson: measure performance before optimizing |
| **Hybrid RRF** | Dense-only or sparse (BM25)-only | Component-level evaluation (synthetic set, trend credible, absolute values for reference only): hybrid 0.541 > dense 0.510 > sparse 0.449; with rerank enabled, 0.566→0.867 |
| **A custom-built chunker** | Off-the-shelf chunking libraries like chonkie | Three-way comparison: ours 79.5% vs. chonkie 74.9% (evidence-fidelity basis); and the moat isn't the boundary-cutting algorithm, it's the engineering integration — source_indices for traceability, the heading tree, per-chunk ACL stamping — none of which these libraries provide |
| **MinerU for parsing** | A custom parser / Tika | On a level playing field for coverage (MinerU comes out ahead at 95.1% under a fair basis), the content_list schema is chunk-ready as-is, all five formats share one unified pipeline, and there's no JVM to deal with |
| **Prompt grounding + code takeover on refusal** | Bringing in an NLI / fact-checking model | Simple and zero-dependency; branches with a determinate verdict (zero recall) are taken over directly by code rather than betting on the model behaving. Tier 2 measured faithfulness ≈1.000 (72-question basis) — no need yet to bring in heavyweight verification |
| **Starting with embedded Qdrant** | Going straight to a Qdrant server cluster | A gradual path: get it running with zero dependencies on a single machine first, evaluate first; only split into three layers (inference / multiple custodian replicas / qdrant server) at Stage F, because that's the point at which there's actually a need for multiple replicas. Details in [Piece 09](09-scale-out.md) |
| **HTTP 200 + a status field to express domain errors** | REST-style 4xx/5xx | The agent is a programmatic consumer; parsing natural-language error messages is unreliable. A structured status + retriable flag lets the agent know whether to retry or switch strategy |

Note the discipline around bases: the 72-question and 88-question figures in the table above **cannot be mixed** (the 88-question basis is after the table-question set was expanded; both coexist and are each labeled). This "basis discontinuity" is itself a teaching point about evaluation methodology — see [Piece 07](07-evaluation.md).

---

## 4. Real-world retrospective: an adversarial, whole-repo review before writing

Before starting to write this set of learning docs, a round of parallel deep reading + adversarial review was done across the six subsystems (6 analyst agents produced a knowledge map while simultaneously collecting 35 "suspected design issues," each dispatched to an independent verifier who **first tried to refute it**). Result: **34 confirmed / 1 refuted** (the refuted one: service#0, readyz bypassing Store._lock); of the 34 confirmed, 2 were cross-layer duplicate reports (the deploy re-review re-reported embedder#5 / service#4, already merged), leaving **32 distinct issues** after dedup. This round of review is itself the best material for reviewing the architecture:

**Fixed (17 items landed directly)**: all of them are "behavior-neutral robustness fixes." Verification evidence: `pytest tests -q` = 224 passed before the fix, 259 passed after (36 new test cases). Two are worth picking out for their teaching value:

- **Reordering the delete-old-vectors timing in index_document** (embedder#11): the original implementation "deleted the old vectors first, then encoded chunk by chunk" — encoding a large document takes minutes, and a mid-encode failure meant that document was **permanently knocked out of the store**. The fix reorders it to "encode + fully prepare the sidecar tmp file (pure preparation, no side effects on failure) → delete → upsert → atomic rename," shrinking the danger window from minutes to milliseconds ([src/embedder/embed.py:56](../../src/embedder/embed.py#L56) — the docstring records the full failure-surface analysis). Teaching point: **the failure surface of an indexing pipeline should be ordered by "after which step would a failure leave behind an unrecoverable state."**
- **Consolidating healthz information** (service#8): the unauthenticated /healthz endpoint was leaking the collection name and llm_model. Harmless in isolation, but inconsistent with the hardening on readyz — the value of a security boundary is determined by its weakest plank.

**Deferred (15 items, confirmed but not touched)**: anything that changes the chunking/retrieval/generation **output content**, or requires a GPU/compose environment to verify, was uniformly deferred with a fix sketch written down. Why "confirmed but not fixed" is the correct engineering judgment here:

- A chunking change causes chunk ids to shift (the repo has really hit this — a phantom table chunk once pushed the whole store from 7,652→7,675 chunks), invalidating all old indexes;
- A change to what retrieval delivers would distort already-published evaluation numbers — **the change must be bundled with "rebuild the index + re-run the GPU eval" as a single atomic action**, otherwise the numbers and the code drift apart.

A typical example is chunker#0 (on the ACL-aware path, big-blocks systematically lose all heading text, so the context the LLM receives is missing section-boundary signals): the root cause is clear and a fix sketch is ready, but it can only land after bumping SIDECAR_VERSION + rebuilding the index + re-running eval — details in [Piece 02](02-parsing-chunking.md). The full list of fixed/deferred items for each cluster is scattered across the real-world retrospective sections of the corresponding pieces.

---

## 5. How to pitch this in an interview

### 30-second elevator pitch

> I built Custodian, a multi-format agentic RAG system: it turns PDFs, scanned documents, and Office documents into a question-answerable, enterprise-grade permission-controlled, source-traceable local knowledge base, serving a small team on a single 4090. It has two consumption modes — HTTP closed-pipeline question answering, and an MCP tool surface (letting agents like Claude Code drive their own retrieval). Three things set it apart from a typical RAG demo: permission filtering is pushed down into the recall layer to be fail-closed; there's a cross-vendor-judge debiased evaluation loop, with measured faithfulness close to 1.0; and every key decision has a rejected alternative and data behind it — including using measurement to overturn the intuition that "agentic is always better."

### 3-minute structured version

1. **The problem** (30s): a team's document store needs to be question-answerable, but naive RAG has a whole string of pitfalls — parsing distortion, chunking that breaks semantics, single-route recall blind spots, charts you can't retrieve, permissions running wide open, generation that fabricates, and changes whose effect you can't even measure. I addressed this problem spectrum item by item.
2. **Architecture** (60s): on the index side, MinerU uniformly parses five formats, and the chunker rebuilds the heading tree, cuts small chunks, and stamps ACLs; on the query side, dense+BM25 dual-route recall fused with RRF, ACL filter pushed down into every prefetch route, small-to-big assembling large chunks to feed the LLM, and the `[cite:n]` citation protocol for traceability. The whole thing runs inside one daemon process (the embedded Qdrant's exclusive lock plus the 8B model's 1-2 minute load time dictate that it must stay resident), with two exits: HTTP closed pipeline + MCP tool surface, with tool semantics kept in a single source of truth so they never drift.
3. **Data** (45s): the production store has 77 real documents, 7,652 chunks; the evaluation set has 88 questions, with four programmatic metrics + two cross-vendor judge metrics. Faithfulness ≈1.0 — the system would rather refuse than fabricate; single-hop correctness 0.97 (72-question basis). The most counterintuitive conclusion: paired attribution shows agent-driven multi-hop scoring about 0.1 lower on correctness than the closed pipeline, which is why the closed pipeline is the default and agentic is an option.
4. **Methodology** (45s): every component went through an adversarial review before being sealed off, and the whole stack later went through five systematic rounds of re-review, which caught bugs in the evaluation pipeline itself — the judge's context had been truncated, manufacturing a false "17% hallucination" conclusion out of thin air; after re-judging with full context it was corrected to ≈1.0. That episode gave me a muscle-memory understanding of "evaluation infrastructure comes before optimization."

(Closing hook: permissions, evaluation, and scalability all have full deep-dive stories ready — whichever one the interviewer picks, there's material to follow up on.)

---

## 6. Rehearsing follow-up questions

1. **"Why not use LangChain / LlamaIndex?"**
   Answer: it's not a rejection of frameworks — the value of this project is exactly in the places a framework can't help: heading-tree reconstruction, ACL pushed down to prefetch, ACL-aware material-pulling for small-to-big, the evaluation loop — all of these have to be written right up against the storage engine and the corpus. A framework's abstractions become a liability, not an asset, under the goal of "every stage must be testable and controllable." Worth adding: the seams are all left in place (the Element adapter, a pluggable LLM, retriever duck-typing), so it's not opposed to using a framework at the periphery.

2. **"Why is the closed pipeline the default, instead of letting the agent retrieve freely?"**
   Keywords: paired attribution, Δ−0.097, distractor chunk dilution. Answer: intuitively an agent doing multiple hops should be stronger, but in a paired comparison on the same gold set, agentic correctness is net negative by about 0.1 — multiple rounds of retrieval pull in more distractor chunks, and every hop ends up less clean than a single hop. The agent has only a weak edge on cross-document questions. So the closed pipeline is the default, and agentic is reserved for scenarios that genuinely need interactive exploration. Add an honest caveat: it was later discovered that the eval's agentic path was missing two production fixes, so the magnitude needs re-evaluation — the direction has no evidence overturning it yet. Volunteering this actually earns points, not loses them.

3. **"Why RRF instead of weighted score fusion for hybrid retrieval?"**
   Keywords: incomparable score scales. Dense's cosine score and BM25's score are not on the same scale; weighting requires tuning a hyperparameter and is brittle; RRF only uses rank, has zero hyperparameters, and Qdrant supports it natively server-side. Data: hybrid 0.541 > dense 0.510 > BM25 0.449 (synthetic set, trend basis).

4. **"How are permissions handled? Why not just filter after retrieval?"**
   Keywords: fail-closed, push-down, three gates. Filtering after retrieval has two holes: unauthorized content eats up top-k slots (a leak of result quality), and once there are multiple code paths, some are bound to miss the filter. Custodian pushes the ACL filter down into every prefetch route (unauthorized content never even enters the candidate set), re-checks once more at the exit, and small-to-big's material-pulling is also gated by ACL equivalence classes. The killer piece of evidence: a regression test specifically **disables the exit gate** and retests — cross-tenant recall is still 0 — proving the first gate is effective on its own, not that safety is being faked by a fallback. Details point to [Piece 04](04-acl-security.md).

5. **"How do you prove your system doesn't fabricate?"**
   Keywords: faithfulness judge, cross-vendor, the evaluation-bug story. Answer: the faithfulness metric = whether every claim in the answer is supported by the context that was actually fed in, judged by a different-vendor model (Claude), measured at ≈1.0. The most worth telling story is that it once reported 0.83: self-review found the judge's context had been truncated, seeing only 40% of the passages — a bug in the evaluation pipeline itself manufacturing a false conclusion out of thin air. After feeding in the full context and re-judging, it was corrected. This story also answers "why are your numbers trustworthy": because the evaluation pipeline itself has also been through adversarial review.

6. **"Single 4090 — how do you scale? Where's the bottleneck?"**
   Keywords: three-layer split, throughput ceiling. Stages A–F have already been completed: the GPU forward pass split out into a standalone inference service, the application layer stripped of torch so it can `--scale custodian=N` across multiple replicas, embedded Qdrant switched to server mode, nginx fronting it. Volunteer the honest conclusion: **the throughput ceiling = a single GPU's forward pass running serially, and it doesn't rise with the number of replicas**; what multiple replicas scale is non-GPU concurrency, crash isolation, and rolling upgrades. The path to break through the ceiling is vLLM continuous batching, which already has an equivalence probe and a go/no-go gate. Details in [Piece 09](09-scale-out.md).

7. **"How did you debug the case where numbers in a table couldn't be answered?"**
   Keywords: diagnostic discipline, root-cause chain. Evaluation exposed the symptom (recall was correct but it couldn't answer) → the initial hypothesis "content_raw wasn't being fed in" was overturned question by question → the real root cause was a double loss: the asset chunk had been folded away by section dedup, and it was also excluded from big-block assembly → two fixes (exempt asset chunks from dedup + have the generator feed in content_raw) → 4 table questions went from "insufficient information" to correctly answered, and single-hop correctness reached 0.97. Full story in [Piece 05](05-generation-grounding.md).

8. **"Are 88 questions enough? The gold set is synthesized by a model — is it trustworthy?"**
   Keywords: honest caveats, layered trustworthiness. Don't dodge it: the volume really is small, and cross-document questions number only 5 (so the cross-doc 0.00 should only be read as a directional signal); the synthetic gold set has same-source bias (question wording overlaps with the golden chunk's vocabulary, which makes retrieval artificially easier), and the mitigations are: prompt-level bans on pronouns/references, multi-hop questions constructed across chunks by a different-vendor model, and programmatic QC gates added for table questions. Evaluation is positioned as a **trend ruler**, not an absolute score — the same ruler measures before-and-after the same change, and the trend is credible.

---

## 7. Hands-on experiments

### Experiment 1 (CPU, zero GPU/network): walk through the product surface using a fake layer

All product-layer tests run against a fake retriever + MockLLM — the fastest way to understand "what contract does the service layer consume":

```bash
cd projects/custodian
pip install -e '.[dev]'                     # editable install, src layout
python -m pytest tests -q --ignore=tests/engine   # product layer: HTTP service/identity/sessions/adapters
python -m pytest tests/engine/test_generate.py tests/engine/test_tools.py -q  # closed pipeline and tool-surface scaffolding
```

Then read [tests/_fakes.py](../../tests/_fakes.py): `FakeRetriever` uses duck-typing to match the method surface of the real `embedder.Retriever`, and `make_hit`/`make_res` fabricate hits and big-blocks. **Exercise**: change `make_res`'s `status` to `section_window`, run `tests/test_service.py`, and watch how toolcore passes `context_status` through to the agent — this is the touchable version of the state machine from §2.4.

### Experiment 2 (GPU/WSL, prerequisite: the WSL `custodian` environment + a 4090 + an already-built index at `~/rag_real`): walk through the real lifeline

```bash
conda activate custodian
python -m custodian health                      # confirm the daemon is up (systemd-managed)
python -m custodian ask "What's in the store about X?"   # closed pipeline: observe citations and finish_reason
```

Follow along with the ten-step lifeline in §2.9, and watch the request logs in `journalctl -u custodian -f` (note: it records the identity name, not the key). To quantify any change, the entry point to the evaluation loop is `eval/run_eval.py` (for the scenarios and pitfalls around needing to `systemctl stop custodian` first to release the embedded lock, see [Piece 07](07-evaluation.md) and [../OPERATIONS.md](../OPERATIONS.md)).

---

## 8. Honest boundaries

Proactively admitting weaknesses in an interview is far stronger than having them dug out of you. The known boundaries of this system:

1. **Cross-document synthesis is a real weak spot**: cross-doc correctness 0.00 (n=5, 72-question basis). Even with chunks from two documents in hand, it can't synthesize a comparative conclusion — this is a synthesis-capability problem, not a retrieval problem; it's judged to be a research-grade hard problem with low marginal returns, and has been **explicitly decided not to pursue**. Talking point: "I know what it can't answer, and I've written that boundary into the user documentation — cross-check the weak spots yourself."
2. **The magnitude of the agentic net negative is in question**: this review round confirmed that the eval's agentic/decompose path bypassed two fixes in the production Generator (feeding back asset content_raw, breadcrumbs), which is systematically unfavorable to agentic. The direction (not favored) has no evidence overturning it yet, but the number Δ−0.097 can no longer be cited as gospel — it's pending a re-run.
3. **Same-source bias in the synthetic gold set**: whoever wrote the questions had seen the golden chunk, so retrieval metrics are naturally optimistic. There are mitigations, but a real fix requires real user query logs — which a small-team deployment doesn't have yet.
4. **The evaluation volume is small and there are multiple bases**: the 88 vs. 72 question bases coexist and cannot be mixed; component-level evaluation's absolute values are for reference only. This is a deliberately preserved honesty, not an oversight.
5. **The throughput ceiling hasn't been broken through**: the multi-replica rework doesn't increase GPU forward-pass throughput, and the vLLM path is still sitting before its go/no-go gate.
6. **Residual error from heuristics**: est_tokens estimates token count as characters divided by a divisor, and the error is large on number-dense documents (5.08 for earnings reports vs. an assumed 4.0); both directions of error have a fallback, but it's not a precise contract.
7. **This review round still has 15 confirmed items unfixed**: all of them have a fix sketch and a deferral rationale (bound to index rebuilding/GPU eval) — this isn't uncontrolled technical debt, it's change discipline — but the sentence "confirmed but not yet fixed" has to be said out loud first, by me.

---

*Next up: [02 Document Parsing and Chunking](02-parsing-chunking.md) — the multi-signal reconstruction of the heading tree, and the methodology story of how "lazy" got its name overturned by measurement.*
