# Custodian Design Document

> Status: implemented, measured, and adversarially reviewed (evidence in [TESTING.md](TESTING.md)). Aimed at internal knowledge-base deployments for small teams.

## 1. Goals and Non-Goals

**Goal**: assemble the four components in this repo (chunker / embedder / generator / custodian) into a complete RAG service **aimed at a small team's internal knowledge base**, offering two consumption modes that share the same retrieval stack and the same contract:

1. **Closed-pipeline Q&A** (HTTP `/v1/ask`): a single question-answer round: retrieve → grounding prompt → DeepSeek → cited answer.
   Systematic evaluation (judged by an out-of-house Claude judge) has concluded this is the **default best-performing consumption mode**: faithfulness ≈ 1.0, correctness 0.847, and cheapest.
2. **Agentic RAG** (MCP): the agent drives the retrieval tools itself (when to search / how to rewrite the query / whether to hop multiple times).
   Evaluation shows agentic mode is net negative on simple questions (Δ−0.097), but for interactive deep dives / cross-document browsing it offers a shape closed-pipeline cannot — the two are **complementary**, not competing.

Service-layer coverage: multi-identity authentication (keys mode, §D10), request logging and metrics (§D11), systemd management, backup and restore.

**Non-goals** (explicitly out of scope, rationale in [ROADMAP.md](ROADMAP.md)): HTTPS/public-internet termination (LAN trust boundary + key; use a tunnel for remote access), SSO/OIDC, parsing orchestration (MinerU invocation lives in this repo's `scripts/`), a front-end UI.
(The former non-goal "horizontal scaling / multiple replicas" has since been delivered in phases A–F: splitting out the GPU inference layer, removing torch from the app, Qdrant server, and multi-replica nginx — see [SCALE_OUT.md](SCALE_OUT.md). ⚠ The D1 note below, "switching to Qdrant server is just a URL change," is a **configuration-surface-only** simplification — it actually also requires a three-way branch in the store, passing `qdrant_url` through every exit point, data migration, and re-testing ACL bypass under server mode.)

## 2. Core Architecture Decisions

### D1: The daemon owns the resources exclusively; everything else consumes over HTTP (this project's single most important decision)

**Problem**: the stdio MCP direct-connect mode (`custodian mcp --direct`, `src/custodian/mcp_stdio.py`) has two pain points confirmed in practice:
- the embedded Qdrant's **single-client exclusive lock** — a second process opening the same index fails outright (the eval harness has to `copytree` specifically to avoid this lock);
- the dense model (Qwen3-VL 8B) takes **1–2 minutes to load** — under stdio, each session is its own process, so every new Claude Code session pays that cost again.

**Decision**: `custodian serve` (FastAPI) is the **only** process in the system that touches Qdrant and the GPU; both closed-pipeline Q&A and MCP go over HTTP. Confirmed by smoke testing: the adapter connects in milliseconds, the first query takes 19s (model warming into cache), and subsequent queries are second-scale.

**Rejected alternatives**:
- *Each consumer opens its own index*: blocked outright by the single-client lock;
- *MCP connecting directly in-process to the retrieval stack (`custodian mcp --direct`)*: kept in this repo as a fallback (stdio direct-connect, no daemon, for when the daemon isn't running), but it re-pays the load cost every session, so it is not the product default;
- *Qdrant server mode (docker)*: unlocks multiple clients, but introduces the operational overhead of a long-running service plus data migration; at the current scale (single instance / small team) the benefit doesn't outweigh the complexity. As scale grows this is the natural v2 upgrade path (just swap the URL in EmbedConfig).

### D2: MCP is a thin, zero-GPU-dependency adapter

`custodian mcp` only imports mcp + httpx + toolcore (pure stdlib); its six tools forward the HTTP calls verbatim. When the daemon hasn't started, it returns a structured `backend_unavailable` (with a hint pointing to `custodian serve`) rather than throwing a raw error. The tool names / arguments / return contract are **exactly identical** to this repo's stdio direct-connect path (`custodian mcp --direct`, `src/custodian/mcp_stdio.py`) — the agent side switches between them with no perceptible difference.

### D3: A single source of truth for tool semantics (toolcore)

Validation / structured results / deduplication / budgeting / error mapping / `_INSTRUCTIONS` for both the HTTP endpoints and the MCP adapter all come from `src/custodian/toolcore.py` (its history is recorded in [COMPONENT_NOTES.md](COMPONENT_NOTES.md)). **Motivation**: this contract layer was refined over five rounds of adversarial review (already_returned / omitted_budget / budgets including assets / not leaking existence without access, etc.) — copying it a second time would inevitably drift. Now that it's folded into this repo, toolcore is an ordinary module inside the package (`import`, no longer loaded via importlib by file path).

### D4: Single self-contained repo (path-dep is retired)

The four components (chunker / embedder / generator / custodian) have now been folded into `src/` in this repo, installed editable via a src-layout (`pip install -e '.[dev]'`), with import names unchanged. **History**: custodian was originally a thin product shell that consumed a separate engine repo via a path dependency (inserting the engine's `src` into `sys.path`) — that cross-repo seam (including `CUSTODIAN_ENGINE` resolution, `bootstrap()`/version guards) **has been removed along with the repo merge**. `engine.py` is now just `LockedRetriever` + `build_*`, using ordinary package imports within this repo — no more sys.path injection or cross-repo drift surface.

### D5: Deployment is authorization — identity is the entry point of the security boundary

**Core invariant**: whoever can reach the port can see everything that identity is entitled to see, so identity must be authoritatively decided by the server and must not be tamperable by the client via parameters; if no tenant is set, toolcore fails closed (`no_identity`, empty result). The three ways identity can be bound (keys / legacy / open) and multi-identity productization are covered in **§D10**; here we only establish the invariant: bind to 127.0.0.1 by default, enforce authentication on any non-loopback bind, and `/healthz` is the only endpoint exempt from authentication. Identity only answers "who is asking" — visibility ("what can be seen") is enforced by hard ACL filtering at the embedder layer (the same fail-closed model as the stdio direct-connect path).

### D6: Deduplication is opt-in, isolated per session

On the stdio direct-connect path, `_RETURNED_KEYS` is process-scoped (valid under stdio, where a process equals a session; its own comment explicitly flagged that "per-session isolation is required before moving to HTTP with multiple sessions"). Custodian makes good on that: the `X-Custodian-Session` header → SessionRegistry (a bounded LRU of 64 sessions) fetches that session's returned_keys; **no header = no deduplication** (a one-off curl call shouldn't carry cross-call state). The MCP adapter uses one UUID per process, automatically getting session semantics. Confirmed by smoke testing: a second call within the same session is entirely `already_returned`; a new session is fully isolated.

### D7: Domain results are always HTTP 200 plus a status field

`no_identity` / `empty_query` / `bad_arg` / `no_access` / `backend_unavailable`, etc., are all **domain results** (a state machine that the agent/client needs to decide on programmatically), uniformly returned as 200 with a JSON status, consistent with the toolcore contract; HTTP status codes are reserved strictly for transport-layer semantics (401 auth, 422 request body isn't valid JSON, 5xx crash). Enum validation for fields like mode/strategy is deliberately **not done at the pydantic layer** (otherwise it would turn into a 422) — it's left to toolcore to produce a structured bad_arg.

### D8: The locking model for /v1/ask — retrieval inside the lock, the LLM call outside it

The embedded Qdrant and the GPU forward pass are not thread-safe, and FastAPI's thread pool will run sync endpoints concurrently, so every retriever call is serialized through `LockedRetriever`. The lock proxy injected into Generator is exactly this one, so `/v1/ask`'s retrieval segment holds the lock, while the subsequent multi-second-to-multi-ten-second DeepSeek network call **does not hold the lock** — it doesn't block other sessions' retrieval.

### D9: smart-ask — bounded intelligence for the closed pipeline (2026-07-03, driven by user experience)

**Problem**: a user asked, with default parameters, "Netflix net profit for each year 2011–2015" and got back only three years — the correct answer was in a five-year table, but ranking placed it outside the retrieval window. The knobs (kind/rerank/wording) all existed, but **the user shouldn't need to understand the knobs**.

**Design red line**: the closed pipeline must not turn into an invisible agent (measured evidence shows agent orchestration is net negative, −0.097); every automatic behavior response is logged (the `auto` field); `CUSTODIAN_SMART_ASK=off` gives a one-flag pure mode; any change to default behavior must pass the 88-question exam.

**Layer 1, hints**: when the answer matches a refusal or partial-refusal pattern, generate up to 3 actionable suggestions based on this request's parameters (kind=table / document-language keywords / rerank / use retrieve to check ranking yourself), without repeating suggestions for actions already taken automatically. Zero interruption for normal answers.

**Layer 2, failure-driven table-leg re-retrieval**: the first pass is **pure** (identical path to running without smart mode at all). For numeric-looking questions (`generator.signals.looks_numeric`, zero LLM calls) **that were refused or partially refused on the first pass**, re-issue one retrieval leg with `kind=table, top_k=5, rerank(top_n=50)` (a hard cap of 1 retry; the hits are **unioned** with the main retrieval — the lesson from `decompose` was that replacing narrows the result). If the user explicitly supplied `kind`, that is respected and nothing is layered on top.
**Retry selection picks the better answer**: the retry answer only replaces the first when it is **no longer a refusal** (a complete answer). A partial answer can carry an incorrect "X was not provided" claim of missing information (where X is actually in the context — measured on the 88-question set, faithfulness dropped from 1.0 to 0.93) — in that case, the first round's honest refusal plus hints is kept. Faithfulness is this system's headline metric, ranked ahead of "answering a bit more."

**Why failure-driven rather than a pre-emptive leg (decided from measurement on the 88-question set, both versions were run)**: a pre-emptive leg moved table questions 0.625 → 0.875, but **collaterally damaged 5 prose questions that were previously answered correctly** (the numbers brought in by the leg nudged the model off track / made it overly cautious), dropping prose from 0.861 to 0.792; under failure-driven behavior, "questions that were already answered correctly never trigger it," so there is zero collateral damage: table questions 0.625 → 0.75+, prose unchanged, faithfulness 1.000 across the whole set — overall better. Lesson: **intelligence built into default behavior must only act on the failure path** — any "help" applied on the success path is a risk.

**Leg parameter lesson**: rerank_top_n must be ≥ "the worst rank the correct chunk gets in the coarse pass" (measured: 30 couldn't fit the five-year table that landed at coarse rank 31–50, making the retry effectively useless; 50 hit it on the first try). Reranking corrects the ordering, but only if the candidate pool actually contains the right chunk.

**Single source of truth**: the numeric-question detection / refusal detection / leg parameters live in `src/generator/signals.py`, shared by custodian and `run_eval --smart-tables` — the exam runs exactly the production behavior.

**Rejected alternatives**: a pre-emptive supplementary leg (per the measurement above); always-on global reranking (adds 3–5s per question, minimal benefit for prose questions); a larger default top_k (measured to only add noise); translating every query (an extra LLM call per question); multi-round looping inside the closed pipeline (that's MCP's job).

## 2b. Service layer: multi-identity / observability / layering (D10–D12)

**Layering principle (D12): identity lives at the service layer, ACL lives at the retrieval layer.** After the repo merge, this is now **internal module layering within this repo** (no longer a cross-repo invariant): identity answers "who is asking" (the concern of custodian's service-layer identity module), visibility answers "what can be seen" (the concern of the embedder layer, whose multi-tenant ACL mechanism is validated by the five-user acl_regression matrix). The two are orthogonal: when the service layer does multi-identity authentication, enforcement of the hard visibility filter **converges on a single point, the embedder's ACL** — that's the test of the layering discipline: identity never crosses the line to make ACL decisions, and ACL is unaware of where a given identity came from.

### D10: Multi-identity model — API key → identity, three modes, fail-closed

- **keys mode** (team deployment, default): `CUSTODIAN_KEYS_FILE` points to a JSON file (`{"keys":[{"key","name","tenant","principals",["admin"]}]}`, gitignored, chmod 600 recommended). Every /v1/* request has its identity resolved from the X-API-Key header (name + User); **any unknown/missing key is a flat 401**; that identity's User is passed into the retrieval stack when retrieving (the embedder's hard ACL filter is what makes "what can be seen" real).
- **legacy / open modes** (single-user / local development): setting only CUSTODIAN_API_KEY = a single shared threshold key; setting neither = loopback-only, no authentication. ⚠ Both are **single-identity**: every client shares the one identity bound at startup, **this is not multi-tenancy** — multiple users must use keys mode.
- **name constraint** (fail-closed validation): an identity's name must be **unique** and **must not contain `|`** (it is the prefix of the session-dedup registry key `name|session_id`; a duplicate name or one containing the delimiter would collide namespaces); violating this refuses to start.
- **fail-closed guard**: binding `CUSTODIAN_HOST` to a non-loopback address without keys mode configured → **refuses to start** (deployment is authorization; the whole library is not allowed to be exposed to the LAN naked); a malformed keys file → refuses to start, no silent degradation.
- **session isolation**: the returned_keys registry key is `name|session_id` — different users cannot see each other even if they forge the same X-Custodian-Session header (under single-identity mode a session collision is harmless; that assumption no longer holds under multiple users).
- **rotation** = edit the keys file + `systemctl restart custodian` (a few seconds); no hot-reload (simplicity over elegance; at team scale a restart is imperceptible). Key generator: `custodian keys new <name> --tenant --principals`.
- **Rejected alternatives**: SSO/OIDC (not worth the IdP dependency at the current scale); storing keys in a database (a file plus a restart is sufficient); hot-reload (adds a state-consistency surface for little benefit).

### D11: Observability — request logs + in-memory metrics, just enough

- **Request logging**: JSONL, appended (`CUSTODIAN_LOG_DIR`, default ~/custodian_logs), one line per request: {ts, ep, user, http, ms, status, n/n_citations/auto/refusal, query (truncated, can be turned off with `CUSTODIAN_LOG_QUERIES=off`)}. `user` = the identity name under keys mode; under legacy/open single-identity mode it's always the placeholder `default`/`local` (**the key itself is never logged**). **The key itself is never written to disk**; query logging is on by default (the debugging value on a LAN outweighs the privacy risk, and this is documented, and can be turned off).
- **Metrics**: in-process memory (counters + a ring buffer of per-endpoint latencies) → `/v1/stats` (under keys mode, readable only with an admin key — even the shape of the aggregate query is itself information); resets to zero on restart (acceptable for now; persistent metrics are a TODO, to be done if needed).
- **Rejected alternatives**: Prometheus/Grafana (not worth running a monitoring stack at team scale; JSONL can be post-processed by any tool).

## 3. Naming

**Custodian (Pharos)**: the Lighthouse of Alexandria, one of the Seven Wonders of the Ancient World, standing guard beside the Library of Alexandria. "Library + navigation" is exactly what this system is: a private collection (starting from 77 multi-format documents) plus retrieval navigation. The CLI reads naturally: `custodian serve / ask / mcp / index`.

## 4. Risks and Known Limitations

| Risk | Current State | Mitigation |
|---|---|---|
| Daemon is a single point (no multiple replicas) | Acceptable at the current scale | systemd self-healing + structured degradation in the adapter + hints pointing to the recovery action |
| Drift between the adapter and stdio direct-connect contracts | After the repo merge this is a structured contract test within a single repo (adapter vs. `mcp_stdio` docstrings must be equal + `_INSTRUCTIONS` is single-sourced from toolcore) | Counted as passing only when one pytest suite is fully green (baseline counts in [TESTING.md §1](TESTING.md)); the seam is documented in COMPONENT_NOTES |
| `index` and `serve` fight over the lock | Single-client lock | The indexer catches lock errors and gives a clear message; documentation requires stopping serve first |
| Exposing the port exposes the data | Defaults to 127.0.0.1 | CUSTODIAN_API_KEY; public internet/HTTPS explicitly out of scope |
| Upstream LLM failure | /v1/ask returns ask_failed (retriable) | Details only go into server-side logs, never leaked |
