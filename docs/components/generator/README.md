# generator

The generation layer of RAG (RAG's **"G"**): retrieval → LLM answer synthesis, with **citation provenance + ACL at the exit point + grounding against hallucination**.
`parse → chunk → embed → retrieve → **generate**`.

## Design: LLM-agnostic + dependency injection

- **The `LLMClient` protocol**: `complete(messages) -> str` (OpenAI-compatible `messages`). Pluggable — a local Qwen/vLLM deployment, any vendor's API, Claude, all can implement this single method to plug in.
- **`Generator(retriever, llm)`**: orchestrates retrieval → prompt → LLM → citation parsing. The retriever is provided by the embedder (`Retriever`).
- The core has **zero runtime dependencies** (pure stdlib); `MockLLM` is available for offline use/when there's no real model, and `OpenAICompatibleLLM` provides a real backend over the OpenAI-compatible API (tested with DeepSeek V4 Flash).

## Three key invariants

- **Grounding against hallucination**: the prompt mandates "use only the provided context, say information is insufficient when there's no basis, no outside knowledge," and every claim is tagged with `[cite:n]` (no extra model introduced).
- **Citation provenance**: contexts are fed in numbered 1-based, and `[cite:n]` in the answer is parsed back to the chunk (doc_id/title/section/page); **out-of-range citations are dropped**, never mapped to the wrong source. The citation marker is **`[cite:n]` rather than bare `[n]`** — retrieved passages often already contain footnote/reference `[n]` markers, and sharing the marker would let the LLM copy them verbatim and get mis-mapped to the wrong source (**provenance forgery**); `[cite:n]` stays isolated from body-text tokens (an attack surface uncovered by adversarial review).
- **ACL at the exit point**: the Generator **only consumes what `retriever.search_with_context` returns** (hits already passed Qdrant's hard ACL filter, context already passed the exit-point check), and introduces no new content → every segment fed to the LLM is something the user is authorized to see. When `context=None` (the sidecar is missing/corrupted), it degrades to the hit chunk's text (a single chunk, likewise already authorized), so the answer isn't lost.

## Modules

| Module | Responsibility |
|---|---|
| `types.py` | `Message` / `Citation` / `Answer` |
| `llm.py` | The `LLMClient` protocol + `MockLLM` + `OpenAICompatibleLLM` (the real API backend: thinking gated by base_url / finish_reason surfaced) |
| `prompt.py` | `PromptBuilder` (grounding instructions + numbered citations) |
| `generate.py` | `Generator` (orchestration + citation parsing + ACL at the exit point) |

## Usage

```python
from embedder import EmbedConfig, Retriever, User
from generator import Generator, MockLLM

ret = Retriever(EmbedConfig())
gen = Generator(ret, MockLLM())          # for a real LLM: swap MockLLM for a client implementing complete(messages)
ans = gen.answer("What is TSMC's capex plan?", User(tenant="t1", principals=["g_research"]), rerank=True)
print(ans.text)                          # the answer with [n] citations
for c in ans.citations:
    print(f"[{c.marker}] {c.title} / {c.section} (doc={c.doc_id})")
```

## Status

The generation layer is complete and has passed **R3 adversarial review (6 fixes) + the ③ table/chart grounding fix** (content_raw is now restored on asset hits, see `generate.py`). Unit tests: **16 passed**
(prompt/generate, covering ACL exit-point degradation, the grounding fallback, out-of-range citations being dropped, neutralizing `[cite:n]` injection from passages, restoring short asset data). The real LLM backend
`OpenAICompatibleLLM` is already wired in; the end-to-end RAG evaluation is closed out (72 gold questions / judged by a different-vendor Claude model, **faithfulness ≈1.0**, correctness 0.847, see [OVERVIEW §7](../../OVERVIEW.md) for details).

## Open items (non-blocking)

- **Cross-document synthesis is weak**: multi_cross correctness is 0.00 (n=5) — even given chunks from both documents, it can't synthesize a comparison; this is a synthesis difficulty, not a retrieval problem, and the return on fixing it is low so it hasn't been pursued (see [OVERVIEW §7](../../OVERVIEW.md)).
- (The real LLM backend / end-to-end evaluation are already done, see "Status" above.)
