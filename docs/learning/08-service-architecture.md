# 08 Servicing and Engineering Architecture

> **Reading guide for this chapter**: taking a RAG system from "a retrieval script that runs" to "a long-running service shared by multiple people" means crossing seven gates: process shape, concurrency locking, identity and authentication, session state, error contracts, observability, and health probing. This chapter covers how Custodian's service layer gets through those seven gates, and the rejected alternatives and measured data behind each decision.
> **Interview weight: high** — this is the main battleground where backend/platform roles separate "someone who wires up packages" from "an engineer"; RAG algorithms can be memorized from a book, but the tradeoffs of servicing can only be talked about by someone who's actually done it.
> **Suggested prior reading**: [../DESIGN.md](../DESIGN.md) (the original D1-D12 decisions); data scope in [07 Evaluation Methodology](07-evaluation.md).

---

## 1. Conceptual foundation: why RAG needs a "service layer"

A tutorial's RAG system is a single script: load the model → open the vector store → query → generate. It implicitly assumes three things: **a single user, a single process, and it doesn't matter how long the process lives.** All three break in production:

1. **Resource exclusivity.** Embedded vector stores (Qdrant local / Chroma persistent / LanceDB) are generally **exclusive to a single process** — a second process opening the same path either fails outright or corrupts the data. Loading the embedding model (especially a large one on GPU) takes tens of seconds to a few minutes. Who holds these resources, and how others share them, is the first question servicing has to answer.
2. **Multiple consumers.** The same retrieval stack gets consumed simultaneously by scripts (curl), the closed-pipeline Q&A, and agent tool calls (MCP). If each consumer implements its own "validation/deduplication/budgeting/error handling," it's guaranteed to drift.
3. **The error semantics of a programmatic consumer.** The caller is an agent, not a human — it needs a structured state machine to decide "retry, or switch strategy," it can't parse a natural-language error message to decide what to do next.

The mainstream design spectrum roughly falls into three tiers:

| Shape | Representative | Advantage | Cost |
|---|---|---|---|
| Every process loads its own copy (script-style) | LangChain/LlamaIndex single-machine demos | Zero ops | Exclusive locks trip over each other, models load repeatedly, no sharing |
| **Long-running daemon + thin client** | This project; Ollama-style shape | Resources load once, multiple consumers share a warm backend | You have to solve concurrency/auth/health checks yourself |
| Fully managed, split apart (vector-store server + stateless app + inference service) | Standard cloud architecture | Each layer scales independently | Operational surface multiplies; not worth it at small scale |

Custodian starts from the second tier, and moves toward the third tier as soon as scale demands it (splitting off a GPU inference layer + Qdrant server + multiple nginx replicas, see [../SCALE_OUT.md](../SCALE_OUT.md)) — this evolutionary arc is itself the most interview-valuable material in this chapter.

---

## 2. How Custodian does it

### 2.0 Data flow overview: one daemon, three entry points

```
curl / scripts ──────────────┐
custodian mcp (thin adapter, stdio↔HTTP) ──┤→ custodian serve (FastAPI daemon)
                                │     ├─ toolcore (single source of tool semantics)
custodian mcp --direct (stdio direct connect, │     ├─ Retriever (Qdrant embedded/server + GPU local/remote)
  degraded path, bypasses the daemon) ────────┘     └─ Generator (retrieval + grounding prompt + DeepSeek)
```

**D1: the daemon owns resources exclusively.** `custodian serve` is the only process in the system that opens the embedded Qdrant and loads GPU models, for two hard constraints (stated explicitly in [src/custodian/service.py:1-16](../../src/custodian/service.py#L1)'s file-level docstring):

- Embedded Qdrant has a **single-client exclusive lock** — a second process opening the same path fails outright (see the three branches and the `_lock` comment at [src/embedder/store.py:25-32](../../src/embedder/store.py#L25));
- The dense model (Qwen3-VL 8B) **takes 1-2 minutes to load** — "one process per session" would mean paying that cost again every time a new agent session starts.

The exclusive lock is only actually acquired at lifespan startup ([src/custodian/service.py:105](../../src/custodian/service.py#L105)); retriever/user/generator are all injection points — tests fully substitute fakes for them, which is the precondition for the entire service layer being regression-testable in pure CPU. Three entry points: HTTP calls directly; `custodian mcp` is a thin adapter (zero GPU dependency, millisecond startup); `custodian mcp --direct` connects to the engine directly over stdio when the daemon isn't running, as a degraded fallback path ([src/custodian/cli.py:40-47](../../src/custodian/cli.py#L40)), but it can't run at the same time as serve against the same index.

Measured smoke-test data: the adapter connects in milliseconds, the first query takes 19s (while the model warms in cache), subsequent queries take seconds (DESIGN D1).

### 2.1 toolcore: single source of tool semantics (D3)

The six tools (retrieve / list_documents / get_document / get_outline / expand / retrieve_grouped) — their input validation, structured results, cross-call deduplication, token budgeting, error mapping, and the agent usage contract `_INSTRUCTIONS` — are all collected in [src/custodian/toolcore.py:1-13](../../src/custodian/toolcore.py#L1), which is pure stdlib, doesn't import MCPServer/embedder/GPU, and injects retriever and user via duck typing. The HTTP endpoints and the two MCP bindings only handle transport binding, with zero semantic duplication.

This contract layer has three chains of detail polished by adversarial review, worth memorizing:

1. **The budget must account for asset `content_raw`** ([src/custodian/toolcore.py:105-109](../../src/custodian/toolcore.py#L105)): a table hit's prose text is nearly empty, with the data entirely in content_raw (which can run to thousands of tokens); failing to count it means the largest payload sails right past the soft cap;
2. **The section_window dedup key uses `(doc_id, resolved_section)`, not the anchor** ([src/custodian/toolcore.py:112-117](../../src/custodian/toolcore.py#L112)): the window anchor drifts with the hit's seed, so using it would mean duplicates could never be detected;
3. **returned_keys registration is deferred until after budgeting, and only registers hits that actually delivered body text** ([src/custodian/toolcore.py:151-158](../../src/custodian/toolcore.py#L151)): otherwise a block downgraded by `omitted_budget` would get mistakenly marked `already_returned` next time — even though the agent never actually received it.

There's also a layering discipline: a runtime exception for "the inference service is temporarily unavailable" needs to give the agent more precise retriable semantics, but toolcore isn't allowed to import embedder.errors (to keep it stdlib-only), so it uses a duck-typing marker attribute to branch: `getattr(e, "inference_unavailable", False)` ([src/custodian/toolcore.py:230-233](../../src/custodian/toolcore.py#L230)). **The direction of the dependency is itself a design asset — using getattr is preferable to breaking the layering.**

### 2.2 Identity: three modes + a fail-closed iron rule (D10)

- **keys mode** (for teams): `CUSTODIAN_KEYS_FILE` points at a JSON file; every request's X-API-Key gets resolved into an `Identity{name, tenant, principals, admin}`; unknown or missing keys always get a 401 without revealing "whether a key ever existed" ([src/custodian/service.py:148-164](../../src/custodian/service.py#L148));
- **legacy** (for a single person): a single `CUSTODIAN_API_KEY` threshold; **open**: neither is set, only loopback access works.

There are three levels of fail-closed, all of which **fail loudly at startup** rather than silently allowing something through at runtime:

1. If tenant isn't set → every toolcore tool entry point first checks `user.tenant`, and if it's empty, returns an empty `no_identity` result ([src/custodian/toolcore.py:215-216](../../src/custodian/toolcore.py#L215));
2. Binding to a non-loopback address without keys mode configured → `create_app` calls `SystemExit` directly ([src/custodian/service.py:93-95](../../src/custodian/service.py#L93)) — "**deployment implies authorization**," and it's never allowed to expose the whole library naked on the local network;
3. Any format error in the keys file (a key that's too short, missing name/tenant, a duplicate name, a name containing `|`, a duplicate key) → `SystemExit` ([src/custodian/identity.py:29-64](../../src/custodian/identity.py#L29)), never a silent downgrade.

Under keys mode, `_current_user` builds an engine User fresh **per request** based on the resolved identity ([src/custodian/service.py:128-133](../../src/custodian/service.py#L128)) — "who is asking" (service-layer identity) and "what they're allowed to see" (embedder ACL hard filtering) are orthogonal, separately-layered concerns (D12); identity never oversteps into ACL's decisions.

### 2.3 Session deduplication: opt-in + dual isolation by identity|session (D6)

In stdio direct-connect, "process = session," so a process-level set is sufficient ([src/custodian/mcp_stdio.py:83-85](../../src/custodian/mcp_stdio.py#L83), whose comment predicted back then: "must have per-session isolation before switching to a multi-session transport like HTTP"). The daemon is shared across multiple sessions, so:

- Deduplication is only enabled when a request carries the `X-Custodian-Session` header (**no header = no deduplication**; a one-off curl call shouldn't have cross-call state);
- `SessionRegistry` is a bounded LRU (64 sessions; eviction only loses deduplication convenience, correctness is unaffected, [src/custodian/sessions.py:16-32](../../src/custodian/sessions.py#L16));
- The registration key is `f"{identity_name}|{session_id}"` ([src/custodian/service.py:204-208](../../src/custodian/service.py#L204)) — under multiple users, even if two users happen to fake the same session id, they remain invisible to each other.

This key design **inversely derives an input-validation rule**: identity names are forbidden from containing `|` (otherwise `'a' + 'b|c'` and `'a|b' + 'c'` collide on the same key — a namespace collision) and must be unique (a duplicate name means two identities share the same deduplication namespace — cross-contamination), see [src/custodian/identity.py:51-58](../../src/custodian/identity.py#L51). The MCP adapter generates one uuid per process to serve as the session header ([src/custodian/mcp_adapter.py:26](../../src/custodian/mcp_adapter.py#L26)), so it naturally gets session semantics under stdio.

### 2.4 Error contract: domain results are always HTTP 200 + a status field (D7)

`no_identity / empty_query / bad_arg / no_access / config_error / backend_unavailable / inference_unavailable / llm_unconfigured / ask_failed` are all **domain results** — states an agent needs to decide on programmatically — and are uniformly returned as 200 + JSON `{status, retriable, hint}` ([src/custodian/toolcore.py:64-66](../../src/custodian/toolcore.py#L64)). HTTP status codes are reserved only for the transport layer: 401 for auth, 403 for non-admin access to stats, 422 for a request body that isn't valid JSON, and 5xx for crashes.

Two counterintuitive details:

- Validation of enums like mode/strategy is **deliberately kept out of the pydantic layer** (otherwise it would become a 422), left to toolcore to produce a structured `bad_arg` instead ([src/custodian/service.py:43](../../src/custodian/service.py#L43) comment + [src/custodian/toolcore.py:221-224](../../src/custodian/toolcore.py#L221));
- `_safe_doc_call`'s unified exception mapping gives **no-access and not-found the same response** (not revealing existence), maps sidecar corruption to `config_error` rather than masking it as `no_access`, and any other exception gets a uniform, generic downgrade — never leaking internal messages/stack traces to an untrusted agent ([src/custodian/toolcore.py:198-208](../../src/custodian/toolcore.py#L198)).

### 2.5 Lock model: retrieval resources locked, LLM calls unlocked (D8, now the "lock pushdown" version)

FastAPI's sync endpoints run in a thread pool and can hit the retriever concurrently. There's a complete evolutionary arc here:

1. **Original design**: a single `LockedRetriever` big lock serializing every retrieval call — retrieval was inside the lock, the LLM's network call was outside it;
2. **Discovered during the SCALE_OUT phase B review**: when the remote inference backend's HTTP retry backoff sleep happens, it **holds the big lock and blocks the entire replica** (during warmup or a rolling restart, one backoff event queues up every query);
3. **Lock pushdown to the resource classes** ([src/custodian/engine.py:6-12](../../src/custodian/engine.py#L6)): the Qdrant single-client section → `Store._lock` ([src/embedder/store.py:32](../../src/embedder/store.py#L32), the embedded client isn't thread-safe); GPU forward pass → `Dense/Reranker._fwd_lock` (overridden to a nullcontext in remote mode); query LRU → `_cache_lock`. From this point on, remote's HTTP calls and backoff sit outside every lock.

The invariant on `/v1/ask` still holds: the Qdrant/GPU section of retrieval is inside a resource lock, while the subsequent DeepSeek call — anywhere from several seconds to tens of seconds — **holds no lock at all** ([src/custodian/service.py:339](../../src/custodian/service.py#L339) comment), so one slow generation won't starve other sessions' retrieval.

The same round of adversarial review also caught a concurrency cross-contamination bug: when Generator/LLM share a singleton instance, `llm.last_finish_reason` gets polluted across requests under concurrent asks (request A ending up with request B's truncation flag). The fix wasn't to add a lock — locking would serialize the LLM's network calls, violating D8 — but rather **lazily constructing Generator per-thread** ([src/custodian/service.py:122-123](../../src/custodian/service.py#L122), [304-310](../../src/custodian/service.py#L304)): since the thread pool is bounded, the instance count is bounded, and within a single thread, the sequence answer→read finish_reason has no concurrency window. **Eliminate the sharing with the execution model, rather than locking around the sharing.**

⚠ Note: [../DESIGN.md](../DESIGN.md)'s D8 text still describes the big-lock version (the document hasn't been backfilled yet); for interviews and learning, go by the code — describe it as "the invariant + the lock-pushdown evolution."

### 2.6 smart-ask: bounded intelligence in the closed pipeline (D9)

A user asking "Netflix's net profit for each year, 2011-2015" with default parameters only got three years back — the five-year table is in the library, but ranking pushed it outside the top-k window. The knobs (kind=table/rerank) all exist, but **the user shouldn't have to understand them.** Custodian's answer is two layers ([src/custodian/service.py:331-353](../../src/custodian/service.py#L331), [src/custodian/smart.py:25-36](../../src/custodian/smart.py#L25)):

- **Layer 1, hints**: when the final answer hits a refusal pattern, generate up to 3 actionable suggestions based on this request's parameters, without repeating a suggestion for something that's already been done automatically; a normal answer produces zero noise.
- **Layer 2, failure-driven table supplementary retrieval**: the first round is entirely pure; only when `looks_numeric(query)` is true, the user didn't explicitly supply a kind, and the first round refused, does the system ask again once with a supplementary leg of `kind=table, top_k=5, rerank_top_n=50` (its hits are a **union** with the main retrieval, hard-capped at one attempt). **Best-of retry**: the retry answer only replaces the original if it's no longer a refusal, otherwise the honest first-round refusal + hints are kept, and `table_leg_retry_discarded` is left as a trace.

Three red lines: every automatic behavior leaves a trace in the response's `auto` field; `CUSTODIAN_SMART_ASK=off` turns it off with one flag for a pure experience; the numeric-detection / refusal-detection / leg parameters all come from a single source, `generator.signals` — the exam runs the exact production behavior. For why it's "failure-driven" rather than "front-loaded," see the measured verdict in §3.

### 2.7 Observability: three disciplines (D11)

Two async middlewares: `_auth` is the inner layer, `_observe` is the outer layer and uses **try/finally** — so a handler crash still gets recorded (the record carries `crashed=true`); otherwise the most severe errors would be exactly the ones invisible to observability ([src/custodian/service.py:167-197](../../src/custodian/service.py#L167)). Three disciplines:

1. **A 200 can still be a failure**: errors are counted for http≥400 **or** a structured business failure (status not in ok/empty) ([src/custodian/service.py:182-184](../../src/custodian/service.py#L182)) — looking only at HTTP codes would miss every domain error that comes back as 200;
2. **Observability must never drag down the service**: JSONL persistence runs on a background single-writer thread + a bounded queue; `write` only does `put_nowait`, and when the queue is full it drops immediately and increments a counter ([src/custodian/obs.py:112-125](../../src/custodian/obs.py#L112)). This was born out of the phase F review: originally the event loop did synchronous open/append, and one hiccup in the shared bind-mount volume's IO would freeze the entire event loop — **every endpoint on that replica, health probes included, would stall together, turning observability itself into a single point of failure for availability**;
3. **No sensitive information is persisted**: the key itself is never written to disk (only the identity name is recorded); the query is truncated to 120 characters by default, deletable via `CUSTODIAN_LOG_QUERIES=off`, and the privacy boundary is enforced at a single point before enqueueing ([src/custodian/obs.py:115-120](../../src/custodian/obs.py#L115)) — if "truncate only in the middleware" weren't a single point, every new writer added later would risk leaking it. `/v1/stats` under keys mode can only be read with an admin key (the aggregate query pattern is itself information, [src/custodian/service.py:258-268](../../src/custodian/service.py#L258)).

### 2.8 Probe system: liveness/readiness separation + starvation resistance (phase F)

`/healthz` is liveness (the process is alive, pure in-memory read); `/readyz` is readiness (checks whether downstream is reachable and the collection exists), which orchestration/healthchecks use to decide traffic routing — downstream flakiness only pulls that one replica out of traffic, without triggering a crashloop ([src/custodian/service.py:211-256](../../src/custodian/service.py#L211)).

The best SRE story here is about "starvation resistance": before rolling out multiple replicas, a full review found that sync probes **share the default 40-thread anyio thread pool** with `/v1/ask` (LLM calls taking tens of seconds) and `/v1/retrieve` (which can hang for up to ~361s when inference hangs). Under high load, the probe queues up in the pool and starves → healthcheck times out and marks it unhealthy → nginx **pulls every healthy replica that's actually working fine out of the pool** = a global outage. The higher the load, the falser the "I'm still healthy" signal becomes — pulling replicas out precisely at the moment it should least be done. The fix:

- Probes are fully made async (healthz runs directly on the event loop, never entering the thread pool);
- readyz's blocking qdrant call is offloaded to a dedicated `_PROBE_LIMITER` (8 threads, isolated from the business pool, [src/custodian/service.py:38-40](../../src/custodian/service.py#L38), [236-238](../../src/custodian/service.py#L236));
- The inference liveness check uses an AsyncClient with an explicit short timeout, `httpx.Timeout(1.5, connect=1.0)` ([src/custodian/service.py:250](../../src/custodian/service.py#L250)) — review caught that the default 3s per stage adds up to a worst case of ~9s, which exceeds the healthcheck's 5s timeout, causing the probe to misjudge itself.

Security is tightened in the same pass: probes are unauthenticated, but exceptions are only logged server-side and the response body never echoes back `str(e)` (which would otherwise leak internal qdrant/inference host:port), and a missing collection only returns `collection_missing`, never the collection's name. Graceful shutdown budgeting is internally consistent: uvicorn's `timeout_graceful_shutdown=25` < compose's `stop_grace_period` of 30s ([src/custodian/cli.py:34-37](../../src/custodian/cli.py#L34)), so before SIGKILL, in-flight requests are drained and the lifespan flushes the log queue (timeout=5.0, [src/custodian/service.py:109-115](../../src/custodian/service.py#L109)).

F-2 end-to-end measurement (3 replicas + nginx): round-robin distribution is nearly even at 6/6/7; during 50 continuous requests, `docker kill` on one replica gave **50/50 all-200, zero failures** ([../SCALE_OUT.md](../SCALE_OUT.md) F-2).

### 2.9 Two lessons about "the system doesn't error out, but it leads people down a dead end"

The service layer has two more classes of bug that never show up in the architecture diagram, yet are the best demonstration of engineering maturity — both caught by adversarial review.

**A death loop in error messages** (P1 review C2). toolcore's `no_identity` hint used to say "please set RAG_TENANT" (a name left over from the engine era), but the Custodian service only reads `CUSTODIAN_TENANT`. A user follows the hint → sets RAG_TENANT → restarts → still completely empty → the hint still tells them to set RAG_TENANT — **the error message itself manufactures an inescapable troubleshooting loop, worse than having no hint at all.** Later, the namespace was unified to CUSTODIAN_*, eliminating the error at the source ([src/custodian/toolcore.py:45-46](../../src/custodian/toolcore.py#L45)), and the service binding layer's `_adapt` was reduced to a defensive backstop ([src/custodian/service.py:139-145](../../src/custodian/service.py#L139)). Takeaway: fail-closed systems' error messages must evolve in lockstep with the configuration surface — "the action the hint points to accomplishes nothing" is a class of bug worth a dedicated audit.

**A recurrence of a configuration passthrough gap** (phase F review). During phase D, when `qdrant_url` was added, it was declared "passed through all three exit points" but forgetting to fix engine was caught by the project's own verification discipline; the phase F full review found the same kind of gap recurring on a different switch: `mcp_stdio._config`'s EmbedConfig construction was missing `inference_url`. The agentic exit had `CUSTODIAN_INFERENCE_URL` explicitly configured, yet it silently disappeared right here — and a slim (no torch) environment's first query would take the local path, `import torch` would crash directly, and toolcore's broad catch-all would then swallow it into `backend_unavailable`. After three layers of masking, all the user saw was "backend temporarily unavailable." The fix was to add the missing passthrough while also aligning the model path/gpu_name ([src/custodian/mcp_stdio.py:56-66](../../src/custodian/mcp_stdio.py#L56)), with a guard test added, `test_mcp_stdio_config_passes_inference_url` (removing the passthrough turns it red). The structural lesson: **every time a new production configuration switch is added, every single consumer of it must be enumerated, each with a "delete the passthrough and this test turns red" guard test** — "configuration silently lost" is more dangerous than "configuration wrong," because the system looks like it's working normally.

### 2.10 MCP thin adapter: a second binding with the same names, same parameters, same contract

`custodian mcp` only imports mcp + httpx + toolcore, forwarding the six tools to HTTP verbatim ([src/custodian/mcp_adapter.py:1-12](../../src/custodian/mcp_adapter.py#L1)). `_call` unifies error mapping and never throws raw errors at the agent ([src/custodian/mcp_adapter.py:42-64](../../src/custodian/mcp_adapter.py#L42)): a network error → `backend_unavailable` with a hint giving a recovery action (`custodian serve`); 401 → `unauthorized`; a 3xx → structured degradation (httpx doesn't follow redirects by default, otherwise it would falsely report "not JSON"); 4xx → `contract_mismatch`, not retriable (§4 fix); 5xx → retriable. Read timeout is 600s (the daemon's first query lazy-loads an 8B model, and the adapter mustn't time out first); the doc_id path parameter is always run through `quote(safe='')`, and an empty doc_id is rejected locally right away ([src/custodian/mcp_adapter.py:95-97](../../src/custodian/mcp_adapter.py#L95)). Tool docstrings are word-for-word identical to the stdio direct-connect ones, pinned down by a structured regression test (`test_transports_contract_no_drift`) so they can't drift.

---

## 3. Why this design: rejected alternatives

| Alternative | Reason for rejection | Data/evidence |
|---|---|---|
| Each consumer opens its own index | The embedded Qdrant single-client lock kills it outright | A second process opening the same path fails immediately (eval does a dedicated copytree just to avoid this lock) |
| stdio direct-connect as the default | Repaying the 1-2 minute model load cost every session | Kept as a `--direct` fallback; the daemon-mode adapter connects in milliseconds |
| Qdrant server mode (early on) | Ops complexity for a small single-machine team outweighed the benefit | Later delivered in SCALE_OUT phase D; and DESIGN's own top-level note that "just change the url" was an oversimplification — it actually required three branches in store, a full passthrough across every exit point, migration, and re-testing ACL for over-privileged access |
| Duplicating toolcore logic at the HTTP layer | The same semantics implemented twice is guaranteed to drift | When it was split out, the original test_tools.py passed all 22 items unchanged = pure move, no regression |
| Putting enum validation in pydantic | Would turn it into a 422, breaking the agent's 200+status state machine | Pinned down by `test_retrieve_bad_arg_structured_not_422` |
| SSO/OIDC / storing keys in a database / hot-reloading keys | Not worth bringing in an IdP at this scale; a file + a few-second restart is enough; hot reload adds a new state-consistency surface | Rotation = edit the file + systemctl restart |
| Front-loaded table supplementary leg | 88-question A/B measurement: table performance 0.625→0.875 **but 5 prose questions collaterally damaged** (0.861→0.792) | The failure-driven version: table 0.625→0.75+, prose unchanged, faithfulness 1.000 across the whole set — better overall |
| Unconditionally replacing the first round with the retry | Partial answers smuggle in a false "X was not provided" claim of missing information (X is actually in the context), faithfulness 1.0→0.93 | Switched to "best-of retry": prefer keeping the honest refusal |
| Default global rerank / default larger top_k / translating every query / a multi-round loop inside the closed pipeline | +3~5s per question / adds noise only / +1 extra LLM call / that's the MCP exit's job (agent orchestration measured as a net negative, Δ−0.097, see §8 for scope) | 88-question measurement (DESIGN D9) |
| Keeping the big lock into multi-replica v1 | Phase B review found remote's retry backoff holding the lock = the whole replica stalls | `tests/engine/test_concurrency.py` locks in "remote encode backoff happens outside search_with_context" |
| Cross-replica shared deduplication (Redis/sticky sessions) | Deduplication is a convenience, not correctness — deferred until scale demands it | Under nginx round-robin, dedup effectiveness drops to ~1/N, an accepted, stated degradation (SCALE_OUT F-3) |
| Prometheus/Grafana | Not worth operating a whole monitoring stack at this team size | JSONL can be post-processed by any tool later; a TODO to note that metrics reset on restart |

⚠ Data-scope reminder: smart-ask's 0.625/0.875/0.861 and similar numbers all come from **88-question** two-version measurements and cannot be mixed with the earlier 72-question baseline; cite them with their scope noted.

One more lesson worth recording separately about leg parameters: `rerank_top_n=30` couldn't hold a five-year table ranked 31–50 in coarse ranking (the retry became a no-op), while 50 hit it on the first try — **reranking corrects ordering, but only if the candidate is actually in the pool.**

---

## 4. Real-world retrospective: this round's adversarial review (2026-07-07)

Before writing this set of learning docs, a round of adversarial review was done on the service layer — "an analyst reads deeply → an independent verifier first tries to refute it." Whole-repo evidence: baseline before fixes, `pytest tests -q` = 224 passed / 4 skipped; after fixes, 259 passed / 5 skipped (36 new test cases). The service layer landed 4 fixes and logged 2 as deferred.

### The 4 fixes landed (symptom → root cause → fix → test)

1. **Unbounded cardinality in the stats keys (medium)**. Symptom: a long-running daemon's memory grows slowly, and the `/v1/stats` snapshot can be inflated to breaking. Root cause: the metric key used the raw URL path — `/v1/documents/abc` and `/v1/documents/xyz` are two different keys, and every new key allocates a maxlen=1000 latency deque — enumerating doc_ids (even ones that come back as an unauthenticated 404) becomes a low-rate DoS. Fix: switched to keying by **route template** (`/v1/documents/{doc_id}`), with unmatched routes merged into `/v1/_unmatched`; the key set is now naturally bounded to the number of registered routes + 1; the JSONL log's ep field keeps the original path (the debugging value belongs on disk, not in memory) ([src/custodian/service.py:185-191](../../src/custodian/service.py#L185)). Tests: `test_stats_keys_bounded_by_route_template`, `test_stats_unauthorized_requests_bounded`.
2. **healthz information tightening (low)**. Symptom: the unauthenticated `/healthz` returned the collection name and llm_model. Root cause: readyz deliberately doesn't return the collection name or `str(e)` per a review, but healthz did — the same information boundary that one review established was undermined by the neighboring endpoint. Fix: healthz was narrowed to `{status, service, version, tenant_bound, uptime_s}`, with the sensitive fields moved into the admin-gated `/v1/stats` ([src/custodian/service.py:211-222](../../src/custodian/service.py#L211), [264-267](../../src/custodian/service.py#L264)). Teaching point: **security hardening should be audited by "information boundary," not endpoint by endpoint.**
3. **Adapter 4xx mapping (low)**. Symptom: the adapter marked all non-401 4xx errors (including a 422 from version drift) as `retriable=true` and suggested "please retry later." Root cause: a contract-level error is permanent — retrying will never fix it, and the retriable semantic would drive the agent into a pointless loop and mislead troubleshooting. Fix: 4xx → `contract_mismatch` + `retriable=false` + a hint pointing at "check the adapter and service version / CUSTODIAN_URL"; only ≥500 keeps `backend_unavailable` ([src/custodian/mcp_adapter.py:52-60](../../src/custodian/mcp_adapter.py#L52)). Tests: `test_422_maps_contract_mismatch_not_retriable`, `test_404_maps_contract_mismatch_not_retriable`.
4. **A real flush timeout (low)**. Symptom: the `timeout` parameter passed to `RequestLog.flush(timeout)` was silently ignored, and the implementation did an unconditional `q.join()`. Root cause: stdlib `queue.join()` doesn't support a timeout; when the writer thread hangs on bind-mount IO (exactly the scenario this module is supposed to guard against), graceful shutdown would freeze until stop_grace_period expired and it got SIGKILLed — "don't drop logs during drain" was failing in exactly the failure mode it was meant to protect against. Fix: use an `all_tasks_done` condition variable with a deadline poll, warning and giving up if it times out; lifespan passes `timeout=5.0` (25s drain + 5s flush ≤ the 30s grace period, budget self-consistent) ([src/custodian/obs.py:92-110](../../src/custodian/obs.py#L92), [src/custodian/service.py:113](../../src/custodian/service.py#L113)). Tests: `test_reqlog_flush_timeout_returns_and_warns` (injects a hung fake write path, asserts it returns on time), `test_reqlog_flush_timeout_normal_drain`.

### The 2 items logged without changes — "confirmed but not changing it" is also an engineering judgment

- **service#0 (/readyz bypasses Store._lock and reads the embedded client directly): refuted.** This is the only service-layer report this round that a verifier **successfully refuted**, and the teaching value is exactly in the argument chain: on the surface, readyz calls `state.retriever.store.client.collection_exists(...)` directly ([src/custodian/service.py:236-238](../../src/custodian/service.py#L236)), bypassing the `Store._lock` that every business method holds — it looks like a concurrency bug. The rebuttal chain: ① the serve process has **zero write paths** (the library is built by indexer, and the docs explicitly require stopping serve first); ② `collection_exists` in QdrantLocal is a pure dict-membership read; ③ store.py's own comment states its concurrency model as "read-read safe, read-write unsafe" ([src/embedder/store.py:4-5](../../src/embedder/store.py#L4)) — together, these three points mean the race condition's trigger condition is unreachable. Conclusion: no code change, but a note was logged: "if serve ever gains an online write endpoint, this needs to switch to a locked method." **A report that looks like a concurrency bug has to be disproven by listing the process's actual read/write paths, not fixed on the gut instinct of 'no lock = danger'** — blindly adding a lock would have coupled the probe into the business lock, recreating probe starvation.
- **service#5 (create_app writing process-level env to fulfill toolcore's budget): two rounds of verification reached conflicting conclusions, logged conservatively.** `create_app` writes `cfg.max_context_tokens` into `os.environ` ([src/custodian/service.py:86](../../src/custodian/service.py#L86)), and toolcore reads the env fresh on every retrieval ([src/custodian/toolcore.py:55-61](../../src/custodian/toolcore.py#L55)) — building two apps with different budgets in the same process, one after another, would overwrite each other. One round of verification judged it confirmed (the test suite genuinely does build multiple apps in the same process), another judged it refuted (production has one app per process, so the trigger is unreachable). **The discipline for conflicting conclusions across rounds: don't take either side's word for it, handle it conservatively and log it** — the production path isn't changed (the benefit doesn't outweigh the hassle of changing the signature), but a note is left: "if the future ever has multiple apps in the same process, the budget parameter should be parameterized into toolcore (there's precedent in returned_keys' dependency injection)."

---

## 5. How to talk about this in an interview

### 30-second version

"I turned a multi-format RAG system into a daemon service: two hard constraints — an embedded vector store's single-client exclusive lock, and an 8B embedding model taking 1-2 minutes to load — forced the shape of 'the daemon owns the resources exclusively, everything else consumes it over HTTP,' with three entry points (HTTP, the MCP thin adapter, and stdio direct-connect as a fallback) sharing one toolcore tool contract to prevent drift. On the service side I built multi-identity fail-closed authentication, opt-in session deduplication, a structured 200+status error contract, a 'retrieval locked, LLM unlocked' locking model, and a health-probe system isolated from the business thread pool. Running 3 nginx replicas, killing one measurably gave 50/50 requests zero failures."

### 3-minute version (structured expansion)

1. **How the shape came about**: it's not "wanting to build microservices" — it's a resource-constraint derivation: the embedded Qdrant fails outright when a second process opens it, and the model costs 1-2 minutes to reload per process. Daemon-owns-resources + HTTP-sharing is the only shape that lets multiple agent sessions share a warm backend; stdio direct-connect is kept as a fallback rather than the default.
2. **Drift prevention**: the six tools' validation/deduplication/budgeting/error-mapping is collected in a pure-stdlib toolcore, with HTTP and the two MCP bindings doing zero semantic duplication; this contract layer was polished through five rounds of adversarial review (budgeting accounting for table content_raw, a dedup key resistant to anchor drift, registration deferred until after budgeting) — duplication is guaranteed to drift.
3. **Security model**: "deployment implies authorization" — binding non-loopback without multi-identity keys configured refuses to start; any format error in the keys file refuses to start; identity is decided authoritatively server-side, the client can't tamper with it; no-access and not-found get identical responses, not revealing existence.
4. **Concurrency**: the lock model has an evolutionary arc — a big lock (retrieval inside, LLM outside) → review found remote's retry backoff holding the big lock stalls the whole replica → lock pushed down to the resource classes; a concurrency cross-contamination bug was solved with per-thread Generator instead of a lock (a lock would violate the "LLM outside the lock" invariant).
5. **Data points**: closed-pipeline faithfulness ≈1.0, correctness 0.847 (**Tier2 dual-Claude judge, 72-question scope**); smart-ask uses failure-driven rather than front-loaded, decided by an **88-question** A/B measurement — a front-loaded leg pushed table performance 0.625→0.875 but collaterally damaged 5 prose questions, while the failure-driven version got table 0.625→0.75+, prose unchanged, and faithfulness 1.000 across the whole set. (The two sets of numbers use different scopes: the 88-question DeepSeek Tier1 closed-pipeline correctness is 0.818, and it can't be subtracted against the 72-question 0.847.)
6. **Availability**: probe starvation (sync probes sharing a 40-thread pool with business traffic — the higher the load, the falser the readiness signal) → probes made async + a dedicated 8-thread limiter; measured 3-replica kill test: 50/50 zero failures.

---

## 6. Anticipated follow-up questions

1. **"Why not go straight to Qdrant server mode from the start?"** Key point: the decision was staged — for a small single-machine team, server mode brings in a standing service to operate plus data migration, and the benefit doesn't outweigh the complexity; it was delivered in SCALE_OUT phase D once scale actually demanded it. Bonus point: proactively admit that the earlier estimate of "just change the url" was an oversimplification — it actually required three branches in store, a full configuration passthrough across every exit point, a migration script, and re-testing server-mode ACL for over-privileged access (the `:memory:` version would falsely pass).
2. **"Why do all your errors return 200? How does monitoring work?"** Key point: the consumer is an agent, so the domain state machine (no_access/bad_arg/retriable) has to be structured and separate from transport-layer semantics; correspondingly, monitoring **can't only look at the HTTP code** — the errors judgment is "http≥400 or status not in ok/empty," otherwise every failure wrapped in a 200 goes uncounted. Extend it honestly: wrapping things in 200 also blinds nginx's passive removal (`proxy_next_upstream http_5xx`) to sick replicas — already confirmed and pending a fix (deferred deploy#1: switch `inference_unavailable/backend_unavailable` to 503, with the body unchanged).
3. **"Why doesn't session deduplication use Redis? Doesn't it break across multiple replicas?"** Key point: deduplication is an opt-in token-saving convenience, **not a correctness property** — dropping to ~1/N under round-robin is a stated, accepted degradation; bringing in Redis means bringing in new shared state and a new failure domain, to be done once scale demands it (both sticky-session and shared-store paths are left open). Keywords: separating convenience properties from correctness properties, declarative degradation.
4. **"Why are identity names forbidden from containing `|`?"** Key point: this is an input-validation rule derived backward from the data structure — the dedup registration key is the string concatenation `name|sid`, and a separator character inside it would cause a namespace collision (`'a'+'b|c'` and `'a|b'+'c'` land on the same key); duplicate names have the same issue (two identities sharing the dedup namespace = cross-contamination). Being able to clearly explain "the derivability of one validation rule" is more impressive than reciting ten validation rules.
5. **"Why would the probes take healthy replicas offline?"** Key point: sync probes share the default 40-thread anyio pool with business traffic; LLM calls take tens of seconds, and a single request can retry for as long as ~361s when inference hangs — under high load, probes queue and starve → healthcheck falsely reports unhealthy → the LB pulls the replica → the surviving replicas get even more crowded → cascading collapse. Fix keywords: probe reliability tier must be higher than business traffic, async, a dedicated CapacityLimiter, an explicit short timeout (the default 3s×3 stages adds up to ~9s > the healthcheck's 5s, causing self-misjudgment).
6. **"Why does retrieval need a lock at all? How was the lock granularity decided?"** Key point: the embedded Qdrant single client and GPU forward pass aren't thread-safe and must be serialized; but the lock's scope should be bounded by "who is actually the non-thread-safe resource" — the big lock also swept in remote's retry backoff, and one backoff event would block the whole replica, so it was pushed down to three resource-class locks — Store/_fwd_lock/_cache_lock; the LLM's pure network IO is always outside every lock. In server mode, even `Store._lock` can be dropped (Qdrant server is concurrency-safe).
7. **"Could smart-ask turn into an invisible agent?"** Key point: four boundaries — failure-driven (questions already correct never trigger it, zero collateral damage), a hard cap of 1 retry, traces left in the `auto` field, and a one-flag off switch; and the retry uses best-of adoption, preferring to keep the honest refusal (a partial answer smuggling in a false claim of missing information dropped faithfulness 1.0→0.93). Data-driven verdict: both versions of the front-loaded-leg approach were run through the 88-question set and rejected by the collateral-damage data.
8. **"What do you log, and what don't you — how was that decided?"** Key point: the key itself is never persisted (only the identity name is recorded); the query is truncated to 120 characters by default and can be turned off; the privacy boundary is enforced at a single point before enqueueing (preventing any new writer from missing the truncation); an observability failure must never drag down the service (a bounded queue drops and counts when full, exposing that fact); `/v1/stats` requires an admin key (the aggregate query pattern is itself information).

---

## 7. Hands-on experiments

### Lab 1 (CPU): stand up a daemon with a fake backend and watch D6/D7 firsthand

The whole service layer can be run with an injected fake retriever, no GPU/network needed at all. Prerequisite: the repo's dev environment (`pip install -e '.[dev]'`; the repo's standard environment is WSL `conda activate custodian`, though a fully-installed Windows environment also works).

```bash
cd <repo>/tests
python -c "
import _fakes, uvicorn
app = _fakes.make_app(retriever=_fakes.FakeRetriever(
    results_factory=lambda: [_fakes.make_res(_fakes.make_hit(), ctx_text='big', anchor=[1,5])]))
uvicorn.run(app, port=8788)
" &
# 1) Call twice in the same session: the second time hits[0].context_status=already_returned and text is empty (degraded to a pointer)
curl -s -XPOST localhost:8788/v1/retrieve -H 'Content-Type: application/json' \
     -H 'X-Custodian-Session: A' -d '{"query":"q"}'
# 2) Call twice with no session header: both are full_section (dedup is opt-in)
# 3) Enum validation doesn't go through 422: HTTP 200 + status=bad_arg
curl -s -XPOST localhost:8788/v1/retrieve -H 'Content-Type: application/json' -d '{"query":"q","mode":"weird"}'
# 4) /v1/stats: that earlier bad_arg call is counted in errors (200 also counts as a failure)
curl -s localhost:8788/v1/stats
```

Read three tests for reinforcement: `test_session_dedup_and_isolation` (tests/test_service.py), `test_keys_mode_session_isolated_across_users` (tests/test_team.py, faking the same session id stays mutually invisible), and `test_observe_records_on_handler_crash` (tests/test_team.py, a handler crash still gets counted and persisted). Full regression: `pytest tests/test_service.py tests/test_sessions.py tests/test_smart.py tests/test_team.py tests/test_adapter.py -q` should be all green — a direct dividend of the "toolcore dependency injection + fully injectable app factory" layering.

### Lab 2 (CPU): the fail-closed trio — watching "a validation rule can be derived" with your own eyes

```bash
# 1) A bad keys file: a name containing '|' → SystemExit, with the error message specifically naming "the session-isolation prefix separator"
python -c "
from custodian import identity as I
import json, tempfile
p = tempfile.mktemp()
open(p, 'w').write(json.dumps({'keys': [{'key': 'x'*20, 'name': 'a|b', 'tenant': 't'}]}))
I.load_keys(p)"
# Variants: try two entries with a duplicate name; try a key shortened to 'short' — every format error produces a specifically-named SystemExit

# 2) The non-loopback guard: binding 0.0.0.0 with no keys → refuses to start
cd <repo>/tests && python -c "
from _fakes import make_app, make_cfg
make_app(cfg=make_cfg(host='0.0.0.0'))"

# 3) keys new round trip: generates pk_alice_<32hex> and only prints it once; a second call with the same name is rejected
python -m custodian keys new alice --tenant demo --file /tmp/k.json
python -m custodian keys new alice --tenant demo --file /tmp/k.json
```

### Lab 3 (optional, GPU + WSL + Docker): a real multi-replica kill test and a real dedup-degradation test

Prerequisite: compose must run inside WSL, with Docker Desktop's WSL Integration turned on for the Ubuntu distro (otherwise bind mounts will all break — see the environment prerequisites in [../SCALE_OUT.md](../SCALE_OUT.md)).

```bash
docker compose --env-file .env.compose up -d --build --scale custodian=3
for i in $(seq 1 50); do curl -s -o /dev/null -w '%{http_code}\n' -XPOST localhost:8080/v1/retrieve \
  -H "X-API-Key: <key>" -H 'X-Custodian-Session: S' -d '{"query":"net profit"}'; done &
docker kill $(docker compose ps -q custodian | head -1)
```

Expected: 50/50 all-200 (nginx's `proxy_next_upstream` + `non_idempotent` redirects in-flight POSTs to healthy replicas); the already_returned rate for that same session drops to about 1/N — a firsthand check of the "willingly accepted degradation" declared in F-3.

---

## 8. Honest boundaries

Proactively acknowledging these in an interview is much stronger than waiting to be asked into them:

1. **In-process state under multiple replicas**: SessionRegistry's dedup effectiveness drops to ~1/N, and Stats only reflects a single replica and resets on restart. Talking point: "This is a stated, tiered degradation — deduplication is a convenience, not correctness; getting it exact would need session stickiness or Redis, brought in once scale demands it, rather than introducing a new failure domain for a mere convenience today."
2. **A 200-wrapped error is invisible to the LB** (already confirmed and pending, deferred deploy#1): `inference_unavailable/backend_unavailable` come back as 200, so nginx's passive removal never triggers, and a sick replica keeps eating 1/N of the traffic. A sketch of the fix is already settled (switch these states to 503, keep the body structure unchanged), pending a real test in the compose environment before landing.
3. **readyz isn't deep enough** (deferred deploy#2): it only checks that the collection exists, not that it's non-empty. Under a collection left empty by an interrupted migration, readyz would still show green while queries come back entirely empty with no alert — exactly the "silent data corruption" category the project has itself already defined.
4. **No request-cancellation propagation** (deferred deploy#0): after nginx's 130s timeout, the client gets a 504, but the Custodian worker thread keeps retrying for up to ~361s, burning a thread; the retry chain has no wall-clock total deadline. Acknowledge it: "The timeout budgets on the two ends haven't been aligned into an explicit equality — that's the next thing to fix."
5. **The magnitude of agentic Δ−0.097 is uncertain** (deferred eval#0): eval's agentic/decompose path bypasses the production Generator, missing the table content_raw supplement and the section_path breadcrumb — systematically unfavorable to agentic — the direction of Δ is credible, but the magnitude is very likely overstated (this part is already confirmed), pending correction from a re-run. When citing this number, this caveat should come with it.
6. **The documentation lags behind the code**: DESIGN D8 still describes the big-lock version (the code has already pushed locks down); DESIGN says "/healthz is the only unauthenticated endpoint," but the code also exempts /readyz; API.md is missing the /readyz endpoint and the `inference_unavailable` status. Talking point: "The review found this batch of doc-code drift and it's already on the backfill list — this is exactly why I hold to the habit of 'the code is the source of truth + label what's actually been delivered in the docs.'"
7. **create_app's env channel** (service#5): multiple apps in the same process would overwrite each other's budgets; the production trigger is unreachable so it was conservatively logged rather than fixed, but I'll acknowledge it's using a global channel to implement injection semantics — ultimately a compromise.

---

*Previous chapter: [07 Evaluation Methodology](07-evaluation.md) · Original engineering docs: [../DESIGN.md](../DESIGN.md) / [../SCALE_OUT.md](../SCALE_OUT.md) / [../API.md](../API.md)*
