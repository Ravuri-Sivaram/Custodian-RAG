# generator design

> RAG's generation layer (G): **LLM-agnostic scaffolding** (prompt assembly / citations / ACL exit-point checks / grounding, with a pluggable LLM interface)
> + a real backend **`OpenAICompatibleLLM`** (DeepSeek V4 Flash via OpenAI-compat, already tested end to end).

## 1. The pipeline

```
query + user
  → Retriever.search_with_context(rerank?)        # embedder: hybrid (+rerank) + ACL hard filter + small-to-big + exit-point check
  → contexts[] (each segment already authorized, with source metadata)
  → PromptBuilder.build                            # numbered citations + grounding system instruction
  → LLMClient.complete(messages)                   # pluggable (MockLLM / a real LLM)
  → parse [n] out of the answer → Citation[]        # map back to the chunk source, dropping out-of-range ones
  → Answer(text, citations, n_contexts)
```

## 2. LLM-agnostic scaffolding

- **The `LLMClient` protocol** (`llm.py`): just one method, `complete(messages: list[Message]) -> str`. Messages use the OpenAI-compatible
  `[{role, content}]` format — a local vLLM/Qwen deployment, the OpenAI API, the Claude API, all are compatible; a real backend only needs to implement this one method to plug in.
- **Dependency injection**: `Generator(retriever, llm)` doesn't import the embedder or any LLM SDK (duck typing); the core is pure stdlib with zero runtime dependencies.
- **`MockLLM`**: echoes back an answer with `[n]` citations drawn from the context's numbering, letting the whole scaffold be verified without a real model (used in unit tests).
- **`OpenAICompatibleLLM`** (the real backend): works with any OpenAI-compatible endpoint — DeepSeek / a GLM proxy / local vLLM — with **model/endpoint/thinking toggle all driven by configuration, nothing hardcoded** (switching backends = changing base_url+model). Three evaluation knobs: `model` (flash/pro), `thinking` (on/off), `reasoning_effort` (high/max); the reasoning chain is kept in `reasoning_content` separate from the answer `content` (so it doesn't pollute `[cite:n]`), with thinking off by default (a read-and-synthesize task isn't heavy reasoning, and it saves on latency/cost); the API key is read from `.env` and must never be committed. Requires `pip install openai`.

## 3. Three invariants (the design's key points)

| Invariant | How it's done | Why |
|---|---|---|
| **Grounding against hallucination** | The system prompt mandates "use only the context, say information is insufficient when there's no basis, no outside knowledge" + every claim tagged with `[cite:n]` + **numeric-scope constraints** (numbers from a subsidiary/sub-period must not be generalized to the whole); the context's source line carries a **section breadcrumb** to give the model range evidence for that constraint | Doesn't introduce an extra model, relies on the constraint instead; when the LLM drifts, it falls back to "insufficient information" instead of fabricating. Lesson learned in practice: **a constraint with no evidence is just empty text** — the subsidiary information only lived in section_path, and wasn't being fed into the prompt, so the model had no way to judge scope (see the fix writeup in custodian TESTING §3) |
| **Citation provenance** | Contexts are numbered 1-based; the answer's `[cite:n]` is parsed with a regex and mapped back to the chunk (doc/title/section/page); **`[cite:n]` is kept isolated from bare `[n]` in the body text** | Lets the answer's sources be audited (a hard requirement for enterprise RAG); the isolation prevents provenance forgery |
| **ACL at the exit point** | Only consumes what `search_with_context` returns (already hard-filtered + exit-point checked); introduces no new content | Every segment fed to the LLM is already authorized; ACL is enforced at the retrieval layer, and the generation layer doesn't undermine it |

## 4. Decisions and trade-offs

- **OpenAI-compatible messages**: the most universal shape for an LLM interface, works with both local models and every vendor's API, the cheapest way to stay pluggable (a `messages_to_dicts` helper is reused by real backends for the conversion).
- **Grounding via the prompt, not the model**: no NLI/fact-checking model introduced; simple, zero extra dependencies, at the cost of depending on the LLM actually following instructions (the hallucination rate needs to be measured once a real LLM is wired in).
- **The citation marker is `[cite:n]` rather than bare `[n]` (from adversarial review finding #1, a RAG attack surface)**: retrieved passages often already contain footnote/reference markers like `[n]`; sharing the same marker means the LLM can copy them verbatim and they get mis-mapped to the wrong source (provenance forgery), and a malicious chunk could even deliberately plant a `[1]` to steer readers toward an attacker-chosen source. `[cite:n]` almost never appears in natural text, so it stays isolated from body-text tokens.
- **Out-of-range `[cite:n]` is dropped rather than erroring**: the LLM might hallucinate a `[cite:99]`. Dropping it is safer than mapping it to the wrong source, and more stable than crashing.
- **ACL defense in depth at the exit point (the optional `acl_check`)**: by default it trusts that the retriever has already hard-filtered; but since the Generator's decoupled design could be reused with a different retriever, passing in `acl_check(acl,user)` gives a second fail-closed check on every hit, rather than assuming the retriever is always safe.
- **Degrading to the hit chunk's text when `context=None`**: if the sidecar is missing/corrupted, the answer isn't lost (the hit chunk itself is already authorized); the small-to-big context is missing, but there's still hit content.
- **A separate package (not embedder/generate.py)**: separation of concerns — embedder handles retrieval, generator handles generation. Generator connects to Retriever via dependency injection, keeping the two packages decoupled.

## 5. Status + open items

- ✅ The generation layer is complete + unit tests **16 passed** + end-to-end (the real Retriever + DeepSeek/MockLLM, full pipeline, ACL enforced throughout). All 5 categories from sign-off review fixed (citation token pollution → `[cite:n]` isolation, etc.); **6 more fixed in the R3 adversarial review** (neutralizing `[cite:n]` injection from passages, surfacing finish_reason, gating thinking per backend); **③ table/chart grounding** (content_raw is now restored on asset hits).
- ✅ **A real LLM backend**: `OpenAICompatibleLLM` (DeepSeek V4 Flash). A smoke test made real calls both ways — in-context, it gives the number with the correct `[cite:1]`; out-of-context, it answers "insufficient information" instead of making something up (`examples/smoke_deepseek.py`, no GPU/retrieval needed). Chose an API over a local LLM: DeepSeek is extremely cheap (the whole evaluation cost a few cents), has a 1M context window, and is OpenAI-compatible, and this leaves the 4090's 16GB for embedding and another 16GB for the reranker.
- ✅ **The end-to-end RAG Q&A evaluation is closed out** (`eval/`, 72 gold questions / judged by a different-vendor Claude model so the generator never grades itself): **faithfulness ≈1.0** (revised in R5 — the earlier 0.83 was an eval bug where the judge's context was being truncated by CTX_CAP), correctness (single-doc) 0.847; the closed pipeline is now settled as the default. See [OVERVIEW §7](../../OVERVIEW.md) for details.
