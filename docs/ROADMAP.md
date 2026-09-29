# Custodian Roadmap

For capabilities already delivered, see the [README](../README.md) and the various design/measurement documents; this document only records **where things are headed** and **what is explicitly out of scope**.
(Motivation/negative results/measurements for already-delivered features are recorded elsewhere: retrieval intelligence in [TESTING §3](TESTING.md) + [COMPONENT_NOTES](COMPONENT_NOTES.md);
the team service surface in [DESIGN D10-D12](DESIGN.md) + [OPERATIONS](OPERATIONS.md).)

## Candidates (by priority)

- **P1 persistent stats + logrotate**: `/v1/stats` is currently in-process and resets to zero on restart; the request log is a single file. Running as a team over the long term needs
  metrics persisted to disk (or piped to Prometheus) + log rotation. Watch the actual growth rate before deciding how heavyweight a solution to build.
- **P1 key-revocation audit trail**: revocation today is edit the keys file + restart, but there's no audit trail of "who was revoked, and when." Add a
  revocation log + `custodian keys list/revoke` subcommands.
- **P2 per-key rate limiting**: there is currently no rate limiting; with a throughput ceiling of ~3.2 req/s, a single heavy user can starve everyone else. Use a
  token bucket per key, returning a structured `rate_limited` when the limit is exceeded.
- **P2 parsing orchestration**: `custodian parse <pdf|docx|xlsx…>` calling MinerU (online API tokens are already in place) → ingesting a document directly, so that "add one document" becomes a single command. Parsing already lives
  in this repo (`scripts/parse_batch.py`/`parse_office.py`/`mineru_client.py`, see [scripts/README](../scripts/README.md)); turning it into a `custodian parse` subcommand is still pending.
- **P2 table-oriented retrieval**: table questions currently score retrieval 0.750 / correctness 0.688 (88-question baseline); 4 retrieval misses + 2 large-table misreads
  form a symmetric yardstick (TESTING §3). Candidates: enhanced table-chunk embedding, cross-language query assistance.
- **P3 bounded LLM retries** (COMPONENT_NOTES N6): occasional DeepSeek 5xx errors currently surface as `ask_failed`; watch the real-world failure rate before
  deciding whether to add server-side bounded retries (trade-off: putting retries on the client side is semantically clearer).
- **P3 in-session concurrent dedup race condition** (IMPLEMENTATION §3, an observed item): concurrent tool calls within a single session can interleave returned_keys;
  under serial calls there is no real harm; if real harm surfaces, add a per-session lock.

## v2 Direction (scale-driven, not pre-built)

- ✅ **Qdrant server mode + multiple replicas + load balancing (already delivered, phases A–F, see [SCALE_OUT.md](SCALE_OUT.md))**: splitting out the GPU inference layer
  → removing torch from the app → moving the embedded Qdrant to server mode → nginx multi-replica + imperceptible `docker kill` failover. ⚠ Correction to the earlier oversimplified claim that "swapping EmbedConfig's url
  is all it takes to migrate": in practice it also required a three-way branch in `store.py` + passing `qdrant_url` through every exit point + data migration + re-testing **server-mode ACL bypass**.
  Remaining open items: session stickiness/shared dedup, swapping inference to vLLM (once GPU queuing becomes noticeable), Kubernetes.
- SSE/streaming ask; a simple web UI.

## Explicitly Out of Scope

- **HTTPS/public-internet termination**: LAN trust boundary + API key; use something like tailscale for remote access via tunnel, not TLS at the application layer.
- **SSO/OIDC**: not worth the IdP dependency at the current scale; a keys file plus restart-to-rotate is sufficient.
- **Retrieval enhancements for the cross-document synthesis problem**: evaluation shows multi_cross is a genuinely hard research problem (settled by the in-repo eval), not being pursued further.
