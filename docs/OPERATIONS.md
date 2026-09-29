# Custodian Operations Manual

> Audience: whoever operates this service (may not be the developer). All capacity/RTO numbers are real measurements, not estimates.
> For architecture and design rationale, see [DESIGN.md](DESIGN.md) (service surface = D10–D11).

## 1. Deployment Topology

**Single node (systemd, the current default):**
```
Windows login ─► Startup folder CustodianWSL.vbs (silently keeps WSL alive, prevents idle sleep)
                    └► WSL (Ubuntu) systemd ─► custodian.service (Restart=always)
                            └► custodian serve @ 127.0.0.1:8787 (exclusively holds ~/rag_real + GPU)
Consumers: custodian ask / curl / (each person's own) MCP thin adapter — all over HTTP + X-API-Key
```

**Multiple replicas (compose, phases A–F, see [SCALE_OUT.md](SCALE_OUT.md)):** three independently scalable layers — `inference` (GPU ×1, the only one touching the card), `qdrant` (server ×1, persistent volume), `custodian` (slim ×N, no torch) + `nginx` (load-balancing entry point `:8080`).
- Start: `docker compose --env-file .env.compose up -d --build --scale custodian=3`; stop: `docker compose down` (add `-v` to delete the vector volume).
- **Rolling restart (no downtime)**: `docker compose up -d --scale custodian=N` rebuilds instances one by one; nginx's `proxy_next_upstream` routes any request that hits a restarting replica over to a healthy one.
- **Single-replica failure**: the other replicas are unaffected (nginx's `proxy_next_upstream` routes around it within seconds — measured: while `docker kill`-ing one replica, 50/50 requests all returned 200).
  ⚠ Be precise about recovery semantics: `restart:unless-stopped` only auto-restarts a **genuine crash** (the process exits abnormally on its own, e.g. OOM); **a manual `docker kill` is treated by Docker as a deliberate stop and will not auto-restart** (measured: restarts=0) — after a demo/maintenance, bring it back with `docker compose up -d --scale custodian=N`.
  To make it "self-heal from any exit," change to `restart: always` (but then it will also restart deliberately-stopped containers once the daemon restarts).
- **inference failure**: the sole GPU is a single point of failure; a **transient warm-up failure** calls `os._exit(1)` → the process genuinely exits → `restart` heals it (this is not a manual kill, so it does restart); a **permanent config error** (wrong card / missing model) leaves an `err` status in `/healthz` without self-terminating (crash-looping would be pointless), and needs a human.
- **Operational boundary to know**: `--scale custodian=N` **does not raise the QPS ceiling** (the GPU forward pass is serialized on a single card); what it scales is non-GPU concurrency + crash isolation + rolling upgrades. Cross-replica state (session dedup, `/v1/stats`) lives in each replica's own process, and degrades under nginx round-robin (see SCALE_OUT §5-F F-3).

## 2. Everyday Commands

| Action | Command |
|---|---|
| Check status | `systemctl status custodian` |
| View logs (service) | `journalctl -u custodian -f` |
| View logs (requests) | `tail -f ~/custodian_logs/requests.jsonl` |
| Restart | `sudo systemctl restart custodian` (a few seconds; the model only warms up after the first retrieval) |
| Health check | `curl http://127.0.0.1:8787/healthz` (no auth required) |
| Metrics | `curl -H "X-API-Key: <admin key>" http://127.0.0.1:8787/v1/stats` |
| Stop (required before indexing/backup) | `sudo systemctl stop custodian` (the embedded Qdrant single-client lock) |

## 3. Identity and Keys (D10)

- **Issue a new identity**: `python -m custodian keys new <name> --tenant demo [--principals g_a,g_b] [--admin]`
  → the key is printed only once, hand it over securely; then `sudo systemctl restart custodian` for it to take effect.
- **Revoke**: edit `CUSTODIAN_KEYS_FILE`, delete that entry, then restart. **Rotation** = revoke + reissue.
- The keys file (default `~/custodian.keys.json`): chmod 600, **never committed to git**; a malformed file (including a duplicate name, or a name containing `|`)
  makes the service refuse to start (loud, not a silent degradation). ⚠ `keys new`'s automatic chmod 600 **only works on WSL/Linux**; if you place this file on native
  Windows, POSIX permission bits don't apply and you'll need to tighten it yourself with NTFS ACLs (this service is deployed on WSL, where the default path `~` is the WSL home, so this doesn't come up).
- **Onboarding a team member to MCP** (Claude Code): in their own `.mcp.json`, pass `CUSTODIAN_API_KEY=<their key>`
  and `CUSTODIAN_URL=http://<service host>:8787` to the adapter.
- **Exposing to the LAN**: `CUSTODIAN_HOST=0.0.0.0` — the service will **require keys mode**, otherwise it refuses to start
  (deployment is authorization; the whole library is not allowed to be exposed naked). LAN trust boundary + key; HTTPS is explicitly out of scope (use a tunnel for remote access).

## 4. Capacity (measured, 2026-07-04, 4090 / 77 documents / 7652 chunks)

| Concurrent clients | Retrieval p50 | Retrieval p95 | Throughput | Errors |
|---|---|---|---|---|
| 1 | 408ms | 528ms | 2.5 req/s | 0 |
| 2 | 659ms | 750ms | 2.9 req/s | 0 |
| 5 | 1.57s | 1.82s | 3.0 req/s | 0 |
| 10 | 3.07s | 3.26s | 3.2 req/s | 0 |
| 3 (with 20% ask) | retrieval 352ms / ask 4.1s | — | 2.3 req/s | 0 |

**Interpretation**: the retrieval segment is globally serialized (embedded Qdrant + GPU forward pass), so the throughput ceiling is ~3.2 req/s, queuing scales **linearly** with concurrency, zero errors, predictable; **ask's LLM segment does not hold the lock** — under mixed load, while an ask is in flight retrieval p50 is still 352ms (confirming the D8 design in practice).
**Capacity verdict: usable experience for ≤10 concurrently active users (p50 ≤3s); for larger scale, migrate to v2's Qdrant server mode.**
Re-run the measurement: `python scripts/bench.py --key <key> --clients 5 --n 10`.

## 5. Backup / Restore (drill record, 2026-07-04)

**Everything that needs to be backed up**: `~/rag_real` (qdrant+sidecar) + `~/custodian.keys.json` + **`.env`**
(at the repo root, `custodian/.env` — `.gitignore`'d, so git recovery won't bring it back; missing it means the service fails closed with an empty index / ask has no LLM key) + git (code/docs, single repo). Request logs (~/custodian_logs) as needed.

```bash
sudo systemctl stop custodian                       # single-client lock: must stop first
mkdir -p ~/backups                               # ~/backups may not exist on a new machine, create it first
tar czf ~/backups/custodian_$(date +%F).tar.gz \
    -C ~ rag_real custodian.keys.json \
    -C /path/to/custodian .env                      # replace with your actual custodian repo path
sudo systemctl start custodian
```

> The commands above pack the default paths (index `~/rag_real`, keys `~/custodian.keys.json`). If you've changed `CUSTODIAN_INDEX_DIR` /
> `CUSTODIAN_KEYS_FILE` / `CUSTODIAN_LOG_DIR`, replace the corresponding paths in the tar command with the actual values.

**Measured (2026-07-04)**: backup takes 27MB / 3 seconds; restore (unpack → standby directory → bring up a new instance → 77 documents visible + retrieval verified)
gives an **RTO of 33 seconds**. ⚠ That 33 seconds was measured with a **hot model** (the drill ran right after stopping the main service, while the dense model was still in GPU memory);
on a **cold start** (rebooting the machine / a new machine), the first retrieval will trigger the model's lazy load (§7), so the real RTO to "servable" is ≈ 33s + 20s–2min.
Disaster fallback: even if the backup is entirely lost, you can rebuild from the `parsed/` corpus (`custodian index`, ~15–30 minutes for 77 documents; the keys file will need to be reissued).
Recommendation: a weekly cron backup keeping 4 generations; **run a restore drill once per quarter** (following the steps above — don't let the runbook go stale on paper).

## 6. Observability (D11)

- **Request log** at `~/custodian_logs/requests.jsonl`, one line each: `{ts, ep, user, http, ms,
  query (truncated to 120 chars, can be turned off with CUSTODIAN_LOG_QUERIES=off), status/n/auto/n_citations/refusal}`.
  `user` = the identity name under keys mode (never includes the key); under legacy/open single-identity mode it's always `default`/`local` (in which case requests can't be told apart per person).
  A single file, append-only; when it grows large use logrotate or just mv it aside (the service doesn't keep a persistent open handle on it).
- **/v1/stats** (keys mode, admin-only): per-endpoint n/errors/p50/p95/max, uptime, session count, log-write-failure count.
  Resets to zero on restart (by design).
- Signals worth watching: a rising proportion of `refusal=true` on `ask` (insufficient corpus coverage or retrieval degrading), non-zero `errors`,
  non-zero `log_write_failures` (a disk problem).

## 7. Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| Service won't start, log shows "already accessed" | Another process is holding the index (manual serve or an indexing script) | Find it and stop it; always let only the systemd instance touch ~/rag_real |
| Service won't start, log shows a keys-file error | Malformed keys JSON / bad field (fail-closed by design) | Fix the file, or rebuild it with `custodian keys new`; don't try to work around it |
| All requests return 401 | Under keys mode, the key isn't in the file / the file was edited without a restart | Check the file, then `systemctl restart custodian` |
| A specific user can't see any documents | That identity's tenant doesn't match the tenant used to build the index (fail-closed) | Check the tenant in the keys file (the sample index was built with tenant=demo) |
| First retrieval takes 20s–2min | The dense model is lazy-loading | Normal; the first query after a restart is slow, then it's millisecond-scale after that |
| Service disappears after a while | WSL idle sleep (no client connected) | Confirm the CustodianWSL.vbs startup-folder entry is present; as a stopgap, open a wsl window |
| ask always returns llm_unconfigured | DEEPSEEK_API_KEY is missing/expired | Add it to `custodian/.env` + restart |
| GPU OOM | eval/indexing and the service both running rerank at the same time | `systemctl stop custodian` before load testing/eval (see TESTING) |
