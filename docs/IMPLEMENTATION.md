# Custodian Implementation Documentation

> Design rationale is in [DESIGN.md](DESIGN.md); this document covers "how the code came to be."

## 1. Module Map

```
src/custodian/
  config.py       env → immutable CustodianConfig; a single .env (custodian/.env at the repo root) loaded once
  engine.py       LockedRetriever (thread-lock proxy) + build_retriever / build_user / build_generator
                  factories, plain in-repo imports (from chunker/embedder/generator import …)
  toolcore.py     the transport-agnostic tool-semantics layer (the six tool primitives + the single source for _INSTRUCTIONS)
  mcp_stdio.py    the stdio-direct MCP server (the no-daemon fallback for custodian mcp --direct)
  sessions.py     SessionRegistry: per-session returned_keys, bounded LRU (64 sessions)
  service.py      the FastAPI app factory create_app(cfg, retriever, user, generator_factory — all injectable)
  smart.py        the smart-ask product layer (D9): refusal detection + hints; numeric detection/leg parameters come from signals
  mcp_adapter.py  the thin MCP adapter: forwards the six tools via httpx; structured degradation; a process-level uuid session header
  indexer.py      custodian index: parses a directory with MinerU → chunker → embedder (the parameterized version of index_real.py)
  cli.py          serve / mcp / index / ask / health
  identity.py     multi-identity (D10): keys-file parsing/generation (fail-closed validation) / the Identity dataclass
  obs.py          observability (D11): Stats (per-endpoint counts + latency percentiles) + RequestLog (JSONL, truncation happens at this layer)
src/{chunker,embedder,generator}/  the three component packages (pip install -e '.[dev]', src-layout editable install)
tests/            product surface + engine surface (under tests/engine/), one shared pytest suite; _fakes.py provides FakeRetriever/make_app
```

## 2. Request Paths

**/v1/retrieve (and the other 5 retrieval endpoints)**:

```
HTTP request ─► API key middleware ─► pydantic request model (shape only, no enum validation)
  ─► X-Custodian-Session? → SessionRegistry.get(sid) / None
  ─► toolcore._retrieve_impl(LockedRetriever, bound_user, …, returned_keys)
        · validation (no_identity/empty_query/bad_arg) happens outside the lock; retrieval happens inside it
        · already_returned / omitted_budget / budgets including assets — all toolcore's own semantics
  ─► _adapt(): unifies no_identity's hint to CUSTODIAN_TENANT wording (review finding C2: if toolcore's underlying hint pointed at the old RAG_TENANT alias it would be a misleading dead loop)
  ─► the structured dict is returned as-is (HTTP 200 + status field)
```

**/v1/ask (closed pipeline)**:

```
Validation (no_identity/empty_query)
  ─► _get_generator(): builds a Generator lazily **per thread** (LockedRetriever, OpenAICompatibleLLM, acl_check=acl_admits)
        · threading.local: a shared singleton's llm.last_finish_reason would cross-contaminate between concurrent ask requests (fixed in review);
          the thread pool is bounded → the number of instances is bounded; within one thread, answer→reading finish_reason has no concurrency window
        · ValueError (missing key) → status=llm_unconfigured; other construction errors → ask_failed (never a raw 500)
  ─► first round: gen.answer(query, user, top_k, rerank, filters…) — **pure, the same path as with smart mode off**
        · retrieval segment: inside the LockedRetriever lock; the DeepSeek network call: outside the lock (D8)
        · zero recall → generator's deterministic R3.E "insufficient information" (no LLM call)
        · exception → status=ask_failed (retriable), details only go to the server-side log
  ─► smart-ask (D9, CUSTODIAN_SMART_ASK=on): if looks_numeric(query) and no explicit kind was given and
        is_refusal(first-round answer) → re-ask a round with gen.answer(…, extra_legs=[DEFAULT_TABLE_LEG])
        (kind=table top_k=5 rerank_top_n=50; the leg's hits are deduplicated by chunk_id and unioned in after the main hits),
        auto+=["table_leg_retry"]; a hard cap of 1 retry
  ─► refusal/partial refusal → smart.build_hints (≤3 items, without repeating suggestions for actions already taken automatically)
  ─► Answer → {answer, citations[] (no raw text by default, only included when include_contexts=true), n_contexts,
               finish_reason (=length means truncated by max_tokens), model, auto[], hints[]}
```

**MCP adapter**: the six tools are pure forwarding functions + `_call()` for unified error mapping
(RequestError → backend_unavailable + a `custodian serve` hint; 401 → unauthorized; 3xx → backend_unavailable
(httpx doesn't follow redirects, otherwise this would be misreported as "not JSON"); ≥400 → backend_unavailable; non-JSON → backend_unavailable).
The doc_id path parameter is always passed through `quote(…, safe="")` (so `#`, `/` no longer get truncated or misroute), and an empty doc_id is rejected locally (bad_arg).
Instructions are single-sourced from toolcore._INSTRUCTIONS; the adapter's and mcp_stdio's six-tool docstrings are verbatim-identical
(pinned by an in-repo structural regression test: adapter vs. mcp_stdio docstrings equal + _INSTRUCTIONS single-sourced).

## 3. Concurrency and Locking

- FastAPI sync endpoints run in a thread pool → they can enter the retriever concurrently.
- `LockedRetriever` wraps search_with_context / get_document / get_outline / expand / search_grouped /
  store.list_documents all under the same `threading.Lock` (both the embedded Qdrant and the GPU forward pass are treated as serialized).
- SessionRegistry has its own lock; generator is a per-thread instance (threading.local, no shared mutable state).
- toolcore's returned_keys set operations happen inside the impl (outside the retrieval lock) — two concurrent calls within a single session
  could theoretically interleave; under normal usage (one agent session calling tools serially) this isn't a practical problem, and it's logged as a TODO to watch.

## 4. Component Wiring (all in engine.py)

The components (chunker/embedder/generator) and toolcore are all packages within this same repo; after `pip install -e '.[dev]'` they're plain imports;
engine.py only does factory wiring — there's no longer a cross-repo seam.

| Wiring point | Mechanism | How a break shows up |
|---|---|---|
| chunker/embedder/generator | in-repo packages (`from chunker/embedder/generator import …`), src-layout editable install | ImportError (if `pip install -e .` hasn't been run) |
| toolcore | a same-repo module, `from custodian import toolcore` (the six tool primitives + _INSTRUCTIONS) | ImportError |
| delivery budget | create_app passes CUSTODIAN_MAX_CONTEXT_TOKENS into toolcore | — (a toolcore contract) |
| .env | a single .env (custodian/.env at the repo root; DEEPSEEK_API_KEY lives here) | ask returns llm_unconfigured |

## 5. Configuration in Practice

`config.from_env()` reads once at startup → a frozen dataclass; no hot-reload while running.
`CUSTODIAN_INDEX_DIR` is expanded and used to derive qdrant_path/sidecar_dir (each can be overridden separately).
The adapter process only uses `adapter_base_url()` (CUSTODIAN_URL) + CUSTODIAN_API_KEY; it never constructs a CustodianConfig.

## 6. How It Was Built

Bottom-up, with tests at every layer:

- **toolcore came first**: toolcore (the transport-agnostic tool-semantics layer) was split out of the MCP server, so the stdio and HTTP transports could
  share one contract, single-sourced. The original component tests passing green with zero changes was the evidence that the split introduced no regression.
- **The product layer was built up in layers**: config/engine/sessions → service (FastAPI) + mcp_adapter + indexer + cli →
  identity + obs. Each new layer got CPU unit tests added (a fake retriever + MockLLM, touching no GPU/network).
- **Two gates at every layer**: CPU unit tests (logic, product surface + engine surface in the same suite; baseline counts in [TESTING.md §1](TESTING.md)) + GPU smoke tests/load tests/drills (real index, real behavior);
  behavioral quality (smart-ask) and the service surface (multi-identity) each went through a round of adversarial review plus self-verified fixes. All numbers and records are in [TESTING.md](TESTING.md).
