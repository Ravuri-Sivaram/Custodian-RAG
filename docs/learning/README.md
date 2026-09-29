# 00 Learning Path and Project Overview (Index)

> **What this is**: `docs/learning/` is a set of **RAG learning + interview prep** documents built around [Custodian](../../README.md). It's not an API manual or an ops guide (those live in [../OVERVIEW.md](../OVERVIEW.md) and [../OPERATIONS.md](../OPERATIONS.md)); it takes a real, actually-built agentic RAG system — one that's **been proven wrong by data over and over, and corrected each time** — and breaks it into 10 lessons, so you both learn the trade-offs at every layer of RAG thoroughly and can hold up under 20 minutes of continuous follow-up questions in an interview.
> **Audience**: engineers who want to take RAG from "I can call the library" to "I can explain why," and who plan to use this project as the centerpiece of their resume.
> **This piece is the map**: read this first, pick a reading path, then move into the main content.

---

## I. What this documentation set is, and how to read it

**The 10 main pieces (01–10) share the same structure, each with seven or eight fixed sections** — you can jump straight to whichever one you need:

| Section | What it does | Interview use |
|---|---|---|
| **1. Conceptual foundation** | What this layer solves in **any** RAG system, and why the naive approach trips up | Answers "what is this step" |
| **2. How Custodian does it** | The project's actual implementation + data flow | Answers "how did you do it" |
| **3. Why it's designed this way** | Rejected alternatives + the measurements that rejected them | Answers "why not use X" |
| **4. War-story retrospective** | Real bugs: symptom → root cause → fix → evidence | Ammunition for storytelling |
| **5. How to talk about it in an interview** | 30-second / 3-minute scripts | Memorize directly |
| **6. Anticipated follow-up questions** | How an interviewer will probe, and the keywords | Pressure resistance |
| **7. Hands-on experiments** | Two tiers of labs: CPU zero-dependency / real GPU library | Turns "can explain it" into "can do it" |
| **8. Honest boundaries** | A weaknesses list volunteered up front | Admitting a weakness proactively beats having it dug out of you |

**One rule runs through the whole set — report only true numbers, and tag every figure with its basis.** The same "faithfulness" score is a different number on different evaluation lines (the 88-question DeepSeek-judged line vs. the 72-question dual-Claude-judged line), and this documentation set **never mixes them** — every number carries its basis. Any conclusion about an agentic net negative always carries the caveat `eval#0 assembly bias, pending re-evaluation`. See [section VI](#vi-global-data-quick-reference) for the global basis quick-reference.

---

## II. Three ways to read this

### Reading path A · Systematic study (about 2 days)

Read along the "data flow," building the skeleton first, then going layer by layer, and finishing with methodology. **Be sure to do the CPU tier of the "hands-on experiments" at the end of each piece** (zero GPU, zero network, just `pip install -e '.[dev]'`) — RAG is something you only really understand once you've run it yourself.

- **Day 1 · The engine's six stages**: [01 RAG Overview](01-rag-overview.md) (the overall picture, establishing the six-stage map) → [02 Parsing and Chunking](02-parsing-chunking.md) → [03 Retrieval and Hybrid Recall](03-retrieval.md) → [04 ACL and Security](04-acl-security.md) → [05 Generation and Grounding](05-generation-grounding.md) → [06 Agentic RAG and MCP](06-agentic-mcp.md).
- **Day 2 · Measurement, service, methodology**: [07 Evaluation Methodology](07-evaluation.md) (the "ruler" for the whole set — read this one slowly) → [08 Service Architecture](08-service-architecture.md) → [09 Scaling Evolution](09-scale-out.md) → [10 Methodology and Story Collection](10-methodology-stories.md) (the wrap-up).

> Short on time? Pick one thread: the **"retrieval quality"** thread = 02 → 03 → 07; the **"enterprise readiness"** thread = 04 → 08 → 09; the **"how this project got built"** thread = 10 stands alone and can be read any time.

### Reading path B · Interview cram (one evening)

Read only the **"How to talk about it in an interview" + "Anticipated follow-up questions" + "Honest boundaries"** sections of four pieces, and don't get pulled into implementation details:

1. **[01](01-rag-overview.md)** — the skeleton. The standard answer to "introduce your project" is right there in its §5.
2. **[07](07-evaluation.md)** — differentiation. Most candidates have exactly one word ("ragas") for "how do you evaluate RAG"; this piece gives you a system that can carry 20 minutes.
3. **[10](10-methodology-stories.md)** — the story collection. 12 STAR stories you can drop straight into a behavioral interview (hardest bug / proven wrong by data / cross-layer bug / persuading someone).
4. **Pick one more piece based on the role you're targeting**: for an algorithms/retrieval role, add **[03](03-retrieval.md)**; for a backend/platform role, add **[09](09-scale-out.md)**; for a security/enterprise-facing role, add **[04](04-acl-security.md)**.

Then memorize [section IV's three-tier scripts](#iv-three-tier-project-pitch) — one evening is enough.

### Reading path C · Look up by interview question (quick-reference table)

| The interviewer asks | Primary piece | One-line hook |
|---|---|---|
| "Introduce your project" | [01](01-rag-overview.md) §5 | Multi-format agentic RAG, dual exit points, fail-closed permissions, de-biased evaluation |
| "How does chunking work / how do you handle long documents" | [02](02-parsing-chunking.md) | Multi-signal heading-tree reconstruction; query-time small-to-big; 66.6ms killed lazy loading |
| "How does vector retrieval / hybrid fusion work" | [03](03-retrieval.md) | dense+BM25 RRF; why RRF over weighted fusion; rerank 0.566→0.867 |
| "How is multi-tenant RAG permissioning done" | [04](04-acl-security.md) | ACL pushed down into every prefetch; zero recall even with the exit-point gate disabled |
| "How do you prevent hallucination / trace citations / prevent injection" | [05](05-generation-grounding.md) | The `[cite:n]` protocol; three-layer grounding defense; ③ table-numeric grounding |
| "When should agentic RAG be used" | [06](06-agentic-mcp.md) | Measured net negative — closed pipeline is the default (with the eval#0 caveat) |
| "How do you evaluate RAG / prove it isn't making things up" | [07](07-evaluation.md) | The de-biasing triangle; five metrics, layered; the evaluation pipeline's own bug conjuring up a false conclusion |
| "What steps does going from script to service take" | [08](08-service-architecture.md) | A daemon process + three entry points; the lock model; a 200+status error contract |
| "How do you scale on a single card / where's the bottleneck" | [09](09-scale-out.md) | The three-tier split; the throughput ceiling = serialized single-card GPU; kill-goes-unnoticed 50/50 |
| "Tell me about the hardest bug you've debugged" | [10](10-methodology-stories.md) §5 | The false faithfulness conclusion / ③ tables / vLLM misjudged three times |
| "Why not use LangChain" | [01](01-rag-overview.md) §6 | The value is exactly where a framework can't help (heading tree / ACL push-down / evaluation) |
| "Is 88 questions enough / can you trust an LLM-synthesized gold set" | [07](07-evaluation.md) §8 | Acknowledge the small sample; paired attribution; it's a trend ruler, not an absolute score |
| "Is there an industry-standard RAG benchmark" | [07](07-evaluation.md) §1 / follow-up Q9 | What's standardized is methodology, not the data; MTEB v2 · TREC RAG · ragas · BrowseComp, four layers (2026-07 basis) |

---

## III. Custodian in one minute

> The Lighthouse of Alexandria once guided ships past the harbor library; Custodian navigates a team's document library.

**What it is**: a self-contained, single-repo, multi-format agentic RAG system. It turns PDFs / scanned documents / docx / pptx / xlsx into a local knowledge base you can question, that carries enterprise-grade ACL, and that gives traceable citations (running on a single RTX 4090). One daemon process, **with two exit points sharing the same retrieval engine**:

```
Index side:  files ─parse (MinerU)─► Element[] ─chunk─► Chunk[] + heading skeleton ─embed─► Qdrant + sidecar
Query side:  question ─encode─► hybrid recall (dense + BM25, RRF) ─► rerank (optional) ─► ACL hard filter
              ─► query-time small-to-big (fetch the surrounding region) ─► generate (LLM + grounding + [cite:n]) ─► cited answer
Throughout:  ACL fail-closed is stamped on every chunk and carried through to the generation exit; the evaluation loop quantifies every step
```

- **Closed pipeline** (`custodian serve` → HTTP `/v1/ask`): one question, one answer, deterministic, evaluable — **the recommended default** (evaluation proves it's the best and cheapest option).
- **Agentic** (`custodian mcp` → 6 MCP tools): the retrieval engine exposed as tools, letting an agent (like Claude Code) self-drive multi-hop retrieval.

**Already in production**: a production library of roughly **77 real documents / 7,652 chunks**; hardened through 88-question de-biased evaluation plus five rounds of adversarial re-review (R1–R5); already completed the single-machine → three-tier scalable (stages A–F) multi-replica rework. Currently v0.3.0.

---

## IV. Three-tier project pitch

The three tiers go from shortest to longest, expanding as you go down; each tier flags the **deep-dive it feeds into**, so wherever the interviewer wants to go, you have ammunition for it.

### 30 seconds · resume / elevator version

> I built Custodian, a multi-format agentic RAG system: it turns PDFs, scanned documents, and Office files into a local knowledge base you can question, that carries enterprise-grade permission control and traceable citations, serving a small team on a single RTX 4090. There are two ways to consume it — an HTTP closed-pipeline Q&A path, and an MCP tool surface (for an agent like Claude Code to self-drive retrieval). What sets it apart from a typical RAG demo comes down to three things: **permission filtering pushed down into the recall layer to be fail-closed; a de-biased evaluation loop using a different-vendor judge, with measured faithfulness close to 1.0; and every key decision backed by rejected alternatives and data — including a measurement that overturned the intuition that "agentic is always better" (the direction is settled; the magnitude carries an eval#0 pending-re-evaluation caveat).**

→ Deep dive on each of the three selling points: permissions [04](04-acl-security.md) · evaluation [07](07-evaluation.md) · the agentic net negative [06](06-agentic-mcp.md).

### 3 minutes · self-introduction version (problem → architecture → data → methodology)

1. **The problem** (~30s): a team's document library needs to support Q&A, but naive RAG has a whole string of pitfalls — parsing loses fidelity, chunking breaks semantics, single-path recall has blind spots, charts and images can't be retrieved, permissions run wide open, generation makes things up, and you can't tell whether a change made things better or worse. I answer each one against this **problem taxonomy**, one by one (see [01](01-rag-overview.md)).
2. **The architecture** (~60s): on the index side, MinerU uniformly parses five formats, the chunker rebuilds a heading tree, cuts small chunks, and stamps each with ACL; on the query side, dense+BM25 dual-path RRF fusion, ACL filtering pushed down into each prefetch, small-to-big assembling large blocks to feed the LLM, and `[cite:n]` citations preserving traceability. It all runs inside one daemon process (the embedded Qdrant's exclusive lock plus the 8B model's 1–2 minute load time **make a resident process mandatory**), with two exit points sharing the exact same tool semantics with no drift (see [08](08-service-architecture.md)).
3. **The data** (~45s): a production library of 77 documents / 7,652 chunks; an 88-question evaluation set, three programmatic metrics plus two different-vendor-judged metrics (five metrics total). **Faithfulness ≈1.0 — the system would rather refuse than make something up; single-hop correctness 0.97** (72-question basis). The most counter-intuitive conclusion: paired attribution shows an agent self-driving multi-hop retrieval scores about 0.1 lower on correctness than the closed pipeline (Δ−0.097; **the magnitude carries an eval#0 assembly-bias caveat and is pending a rerun, though the direction hasn't been overturned**), which is why the closed pipeline is the default (see [07](07-evaluation.md)).
4. **The methodology** (~45s): every component goes through an adversarial review before being sealed off, plus five rounds of systemic re-review across the whole stack, which **caught a bug in the evaluation pipeline itself** — the judge's context was being truncated, conjuring up a false "17% hallucination" conclusion out of thin air, corrected to ≈1.0 after rejudging. This gave me muscle memory for "evaluation infrastructure comes before optimization" (see [10](10-methodology-stories.md)).

> Close with a hook: "Permissions, evaluation, and scaling all have complete deep-dive stories behind them — whichever direction you'd like, I can take it."

### 15 minutes · deep-dive version (whiteboard-level agenda)

When the interviewer says "tell me more," follow this line, stopping at each stop to expand and take follow-ups. The goal is to **use one real project to string together the full chain of RAG trade-offs, plus one honest self-reversal**.

| Minutes | What to cover | Which piece | The "hook fact" to always drop |
|---|---|---|---|
| 0–2 | Project positioning + the six-stage data flow (draw the diagram above on the whiteboard) | [01](01-rag-overview.md) | Two exit points share one engine; why it has to be a daemon process |
| 2–4 | Chunking: multi-signal heading-tree reconstruction + small-to-big | [02](02-parsing-chunking.md) | **66.6ms measured, killing my own lazy design**; reset-aware bare-numbering promotion |
| 4–7 | Retrieval: hybrid RRF + rerank + shared text/image space | [03](03-retrieval.md) | RRF needs zero hyperparameters using rank alone; hybrid 0.541 > dense 0.510 > BM25 0.449; rerank 0.566→0.867 |
| 7–9 | Permissions: three fail-closed gates | [04](04-acl-security.md) | **The embedded fusion dropping the top-level `should` clause, a fail-open**; zero recall even with the exit-point gate disabled |
| 9–11 | Generation: `[cite:n]` + three-layer grounding + ③ table numerics | [05](05-generation-grounding.md) | The **dual-loss root-cause chain** behind a table being retrieved but still unanswerable |
| 11–13 | Evaluation: the de-biasing triangle + the false-faithfulness-conclusion story | [07](07-evaluation.md) | **0.83's false hallucination was the evaluation pipeline's own bug**; paired attribution; basis breaks |
| 13–15 | The counter-intuitive conclusion + scaling + honest boundaries | [06](06-agentic-mcp.md) · [09](09-scale-out.md) | Agentic net negative Δ−0.097 (**with the eval#0 pending-re-evaluation caveat**); the throughput ceiling = serialized single-card GPU at ~3.2 req/s |

> The deep-dive version's killer move is **proactively admitting weaknesses**: cross-document synthesis at 0.00 (n=5), synthetic-gold same-source bias, the agentic magnitude being in question — each piece's "honest boundaries" section is exactly this confession on file. Handing this over proactively in an interview is far stronger than having it dug out of you.

---

## V. All pieces (table of contents + interview weight)

Interview weight, on three tiers: **★★★** = must-know / highest-frequency differentiator; **★★** = high-frequency bonus point; **★** = supporting / map.

| # | Piece | One-line positioning | Weight |
|---|---|---|---|
| **00** | **This piece · Learning path and overview** | The map: three reading paths + three-tier scripts + the global data quick-reference | ★ |
| **01** | [RAG Overview and Custodian Architecture](01-rag-overview.md) | The overall picture. RAG's problem taxonomy → the six stages → one request's full lifecycle; the skeleton for "introduce your project" | ★★★ |
| **02** | [Document Parsing and Chunking](02-parsing-chunking.md) | MinerU uniform parsing + multi-signal heading-tree reconstruction + query-time small-to-big; the clearest tell for whether you've actually worked with real-world documents | ★★ |
| **03** | [Vector Retrieval and Hybrid Recall](03-retrieval.md) | Shared text/image-space dense + BM25 + RRF + optional fine ranking; every decision is measured, the easiest place to go deep | ★★★ |
| **04** | [Enterprise-Grade ACL and Security Model](04-acl-security.md) | Three fail-closed gates + session isolation; "zero recall even with the exit-point gate disabled" proves one layer of the defense-in-depth is effective on its own | ★★ |
| **05** | [Generation and Grounding](05-generation-grounding.md) | The `[cite:n]` citation protocol + bidirectional injection defense + three-layer grounding defense; the ③ table-numeric diagnostic story | ★★ |
| **06** | [Agentic RAG and the MCP Tool Surface](06-agentic-mcp.md) | 6 tools / 3 entry points / one semantics; the core conclusion = agent orchestration measured as a net negative on this workload (with the eval#0 pending-re-evaluation caveat) | ★★ |
| **07** | [Evaluation Methodology](07-evaluation.md) | The de-biasing triangle + five metrics, layered + two-tier attribution; **the highest-weight piece in the whole set**, the project's biggest differentiator | ★★★ |
| **08** | [Service Architecture and Engineering](08-service-architecture.md) | A daemon process + three entry points + the lock model + identity + the error contract + probes; separates the engineer from the library-caller | ★★ |
| **09** | [Scaling Evolution: From Single Machine to Multiple Replicas](09-scale-out.md) | The monolith's three hard bindings → six stages → three tiers; the primary material for system-design interviews, with every conclusion reproducible | ★★★ |
| **10** | [Engineering Methodology and Story Collection](10-methodology-stories.md) | Four methodology threads + 12 STAR stories; the ammunition bank for behavioral interviews | ★★★ |
| **11** | [Interview Question Bank (Cram Review)](11-interview-qa.md) | The first ten pieces reorganized into interview-day ammunition: topic-clustered Q&A (three tiers per question) + a data-basis quick-reference + a reverse-question checklist | ★★★ |

---

## VI. Global data quick-reference

> **The iron rule on basis**: every number below is tagged with its source evaluation line, and **numbers from different lines cannot be compared directly.** Two main lines coexist:
> - **Tier1** = `--judge deepseek`, **88 questions** (72 prose + 16 table), reproducible in-repo, same-vendor trend only.
> - **Tier2** = dual-Claude authoritative judging, **72-question** basis (predates the table expansion), corrected by R5, not reproducible in-repo.
> 88 vs. 72 is a **basis break** — they coexist, each tagged separately, and must never be mixed.

### End-to-end five metrics (closed-pipeline single mode as the reference)

| Metric | Tier1 (88 questions · DeepSeek judge) | Tier2 (72 questions · dual-Claude judge) |
|---|---|---|
| Faithfulness | 0.977 | **≈1.000** |
| Correctness | 0.818 | 0.847 |
| Retrieval recall | 0.818 | 0.854 |
| Citation correctness | 0.767 | — (not broken out separately in Tier2) |
| MRR | 0.627 | — (not broken out separately in Tier2) |

- **Faithfulness is the flagship selling point**: the system would rather answer "no relevant information" than make something up, and `[cite:n]` keeps it traceable. ⚠ It was once reported at 0.83; R5 found this was **a bug in the evaluation pipeline itself** (the judge's context was truncated, seeing only 40%); feeding the full context and rejudging corrected it to ≈1.0 (see [07 §4.1](07-evaluation.md)).

### Correctness broken down by hop type (72-question basis)

| Hop type | Correctness | Note |
|---|---|---|
| single-hop | **0.97** | The strongest area: single-document facts, table numerics |
| single-document multi-hop (multi_intra) | 0.83 | |
| cross-document multi-hop (multi_cross) | **0.00** | ⚠ **n=5, directional signal only**; the bottleneck is synthesis, not retrieval, and this research-grade challenge hasn't been pursued further |

### Two-tier attribution (paired, 72-question basis)

| Comparison | Δ correctness | Caveat |
|---|---|---|
| single → agentic | **−0.097** | ⚠ **`eval#0` assembly bias, pending re-evaluation**: the agentic path's context is missing two production fixes (table `content_raw` / breadcrumbs), **the magnitude is in question, the direction currently has no evidence overturning it** |
| single → decompose | −0.014 | The same caveat applies |

> Citing the agentic net-negative conclusion **must** carry the eval#0 caveat — this is a hard rule of this documentation set.

### Component-level evaluation (synthetic set, **trend trustworthy, absolute values for reference only**)

| Item | Number | Basis |
|---|---|---|
| hybrid vs. single-path recall | hybrid **0.541** > dense 0.510 > BM25 0.449 | Synthetic-set trend |
| rerank gain | 0.566 → **0.867** | Synthetic set, fine-ranking toggle |
| in-house chunking vs. chonkie | ours **79.5%** vs. 74.9% | The moat = engineering integration, not the boundary algorithm |
| eager full-tree rebuild time | **66.6ms** | Measured, overturned "lazy" |
| shared text/image-space similarity | description↔matching image 0.74 / 0.49 | A pure-image chunk can be retrieved by text |

### Scale and engineering basis

| Item | Number | Basis |
|---|---|---|
| Production library `~/rag_real` | ≈ **77 documents / 7,652 chunks** | 14 categories of real documents |
| Evaluation library evalbig | ≈ 15 documents / 1,409 chunks | 5 English papers + 4 English earnings reports + 6 research reports in the project's original non-English language (historical, earlier corpus — see the top-level README on this project's original-language→Telugu swap) |
| pytest suite | **179 passed** (a historical basis, as OVERVIEW recorded it at the time; see [TESTING §1](../TESTING.md) for the current count) | A different point in time and basis: the pre-writing adversarial review's full-repo `pytest tests -q` baseline was **224/4skip** → **259 passed/5skip** (36 new test cases); spinning up a real Qdrant server gave **264/0skip** — not the same point in time/statistical basis as 179, **do not add them together** |
| Multi-replica throughput ceiling | ~**3.2 req/s** | = serialized single-card GPU forward passes; `--scale` doesn't change it |
| vLLM equivalence probe | cosine **0.99956**; 88-question top-k **87/88** agreement | A go/no-go gate, with the criterion set at the level of business impact |

### Numbers whose magnitude is in question / directional only (always tag when citing)

1. **The agentic/decompose net-negative Δ** (eval#0): multiple lines of evidence support the direction, **the magnitude is overstated**, pending a rerun.
2. **Cross-document at 0.00** (n=5): too small a sample, directional signal only, not an absolute value.
3. **Synthetic-gold same-source bias**: question wording overlaps with the golden chunk's vocabulary, biasing retrieval scores optimistic — "by how much" hasn't been quantified.
4. **The 16-question table small sample**: 1–2 wrong gold answers is already a 6–12pp swing (eval#2/#3, confirmed, pending fix).
5. **Component-level absolute values**: synthetic set, trend trustworthy, absolute values for reference only.

---

*Series navigation: this piece is the index → start with [01 RAG Overview](01-rag-overview.md); the authoritative source for evaluation basis is [07 Evaluation Methodology](07-evaluation.md); the engineering-docs entry point is [../OVERVIEW.md](../OVERVIEW.md).*
