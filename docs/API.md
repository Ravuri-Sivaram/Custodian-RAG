# Custodian HTTP API Contract

Base: `http://127.0.0.1:8787` (CUSTODIAN_HOST/PORT). Requests/responses are all JSON (UTF-8).

## Authentication

Three modes (DESIGN D10): **open** (loopback only, no auth) / **legacy** (a single `CUSTODIAN_API_KEY`) /
**keys** (`CUSTODIAN_KEYS_FILE`, each key mapping to an identity with name+tenant+principals+admin). All /v1/* endpoints require the
`X-API-Key` header (except in open mode); `/healthz` is always exempt from auth. Under keys mode, the key is resolved to an identity, and its
tenant/principals are passed into the engine's ACL on every request (determining "what can be seen"); an unknown/missing key → `401`. `/v1/stats` under keys
mode requires an **admin** key (otherwise `403`).

## General Conventions

- **Domain results are always HTTP 200 plus a `status` field** (so the client can decide programmatically); HTTP codes are reserved strictly for the transport layer:
  `401` (auth failure), `403` (stats, non-admin), `422` (the request body isn't valid JSON / a field has the wrong type), `5xx` (crash).
- **The status state machine** (consistent with the engine's toolcore contract): `ok` / `empty` / `no_identity` / `empty_query` /
  `bad_arg` / `no_access` (no access and non-existence get the same response, so existence isn't leaked) / `config_error` (the sidecar needs rebuilding) /
  `backend_unavailable` (retriable) / `contract_mismatch` (specific to the MCP adapter: the daemon returned a non-401 4xx —
  a version drift or CUSTODIAN_URL pointing to the wrong place, **not retriable**) / ask-specific: `llm_unconfigured` / `ask_failed` (retriable).
- **Headers**: `X-API-Key` (optional, see above); `X-Custodian-Session` (optional; including it enables cross-call deduplication —
  a second fetch of the same segment within the same session returns `context_status=already_returned` with an empty body).
- Retrieval response bodies are always marked `trust: "untrusted"` (data, not instructions); the semantics of `hits[].context_status` are described in
  [components/mcp-server](components/mcp-server.md).

## Endpoints

### GET /healthz (no API key required)
`{status, service:"custodian", version, tenant_bound, uptime_s}`
(a minimal liveness surface, with the same information boundary as /readyz — the unauthenticated probe doesn't return collection/llm_model/identity_mode;
those fields live in the admin-gated /v1/stats.)

### GET /v1/stats (requires an admin key under keys mode)
In-process metrics: `{status, identity_mode, collection, llm_model, uptime_s, sessions, log_path,
log_write_failures, endpoints:{"<route template>":{n, errors, p50_ms, p95_ms, max_ms}}}`. Resets to zero on restart.
The endpoints keys are **route templates** (e.g. `/v1/documents/{doc_id}`, so the key cardinality is bounded); any /v1/* request that didn't match a route or was short-circuited by a 401
is folded into the fixed bucket `/v1/_unmatched`.

### GET /v1/instructions
The full agent-facing usage contract (from the same source as the MCP instructions): `{status, instructions}`

### POST /v1/ask — closed-pipeline Q&A
Request: `{query, top_k?, rerank?=false, include_contexts?=false, doc_ids?, doc_type?, kind?, strategy?}`
(the last four are retrieval filters/routing, with the same semantics as /v1/retrieve; for numeric/table questions, `kind:"table"` noticeably improves the hit rate)
Response (ok):
```json
{"status":"ok", "answer":"…an answer with [cite:n]…",
 "citations":[{"marker":1,"chunk_id":"…#0062","doc_id":"…","title":"…","section":"…","page":18,
               "text":"(only present when include_contexts=true)"}],
 "n_contexts":5, "model":"deepseek-v4-flash", "finish_reason":"stop|length|…"}
```
`finish_reason=length` means the answer was truncated by max_tokens (trailing citations may have been cut off). This value is a **snapshot from the same round** as `answer`
(if the smart-ask retry ends up being discarded, this is the first round's value, not the discarded round's); it is `null` when zero recall means the LLM was never called.

smart-ask (on by default, turn off with `CUSTODIAN_SMART_ASK=off`; design in DESIGN D9): the response also includes
`auto: ["table_leg_retry"?]` (a log of automatic actions taken — for numeric questions refused on the first round, a supplementary table-kind leg was retried) and
`hints: [...]` (only when the final answer is still a refusal/partial refusal, up to 3 actionable suggestions; an empty array for normal answers).

### POST /v1/retrieve — hybrid retrieval (+ small-to-big context)
Request: `{query, top_k?, rerank?=false, doc_ids?, doc_type?, kind?, mode?="full"|"concise",
strategy?="hybrid"|"dense"|"sparse", rerank_top_n?}`
Response: `{status, retriable, hint, warning, meta{requested_k, returned_n, deduped_n, rerank,
rerank_degraded, already_returned_n, budget_truncated, context_tokens, mode, strategy, filters}, hits[]}`;
each hit: `{n, doc_id, chunk_id, kind, title, section_path, page_start/end, anchor,
resolved_section, n_tokens, score, score_kind(rrf|cosine|bm25|rerank), context_status, trust, text,
content_raw?(table/chart), image_path?(image/chart, a locator anchor only)}`

### GET /v1/documents
`{status, retriable, hint, coverage:{doc_type: count}, documents:[{doc_id,title,…}]}`

### GET /v1/documents/{doc_id}?max_tokens=6000
Full read of an entire document (element-by-element ACL gating): `{status, doc_id, text, n_tokens, n_elements_visible, truncated, trust, warning}`

### GET /v1/documents/{doc_id}/outline
`{status, doc_id, sections:[…]}`

### POST /v1/expand
Request: `{chunk_id, target_tokens?=1500}` → `{status, chunk_id, text, anchor, resolved_section, n_tokens, climbed, trust, warning}`

### POST /v1/retrieve_grouped
Request: `{query, doc_ids(≤20), top_k?=3, rerank?=false}` → `{status, warning, groups:{doc_id:[hits]}}`

## MCP Tool ↔ Endpoint Mapping

| MCP tool (custodian mcp) | HTTP endpoint |
|---|---|
| retrieve | POST /v1/retrieve |
| list_documents | GET /v1/documents |
| get_document | GET /v1/documents/{doc_id} |
| get_outline | GET /v1/documents/{doc_id}/outline |
| expand | POST /v1/expand |
| retrieve_grouped | POST /v1/retrieve_grouped |

(MCP has no ask tool: in agentic mode, the agent synthesizes the answer itself.)
