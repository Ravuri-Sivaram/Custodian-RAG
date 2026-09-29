# Custodian Horizontal Scaling Evolution: Splitting Out the GPU Inference Layer → True Multi-Replica

> Status: **Phases A–F all shipped and measured (2026-07)**. F = first ran an 8-dimension multi-agent adversarial review
> against A–E (27 confirmed findings, all fixed), then rolled out true multi-replica with nginx;
> measured `docker kill` on a replica as 50/50 unnoticed, round-robin distribution 6/6/7 evenly balanced. The design was
> produced through "reconnaissance → six-dimension design → adversarial review → synthesis," then polished through
> "four-way adversarial review → editor revision"; each implementation phase was paired with an adversarial agent review +
> real run verification. **See the implementation retrospective below for an overview.**
> All `file:line` references have been checked line-by-line against the source (2026-07). v2 fixed one core implementation
> pitfall from v1 (P0-1's root cause was misdiagnosed as the service layer, when it was actually the toolcore fallback)
> plus 7 MUST-FIX items.
>
> This document turns the "horizontal scaling / multi-replica" item marked as **out of scope (v2)** in [DESIGN.md](DESIGN.md)
> into a concrete, executable **six-phase rollout plan**.
>
> **Rollout discipline**: propose before changing, each phase independently verifiable, every "fixed" claim must come with
> a reproduction command + a before/after measurement comparison.
> **Environment prerequisite (required before any verification command)**: in WSL, `conda activate custodian && cd <repo> && pip install -e '.[dev]'`.
> If the core dependencies (qdrant-client etc.) are not fully installed, running `pytest` directly will produce a
> **collection error**, not a genuine test failure.

---

## Implementation Retrospective (2026-07, A–E rolled out + committed)

> This section is an **overview after the rework was completed**. See §5 for the detailed phase design; see the "rollout
> status" block in §5-E for the gate-by-gate measured status of E.

### Architecture, before → after

```
Before (single node):                       After (three-container compose):
┌─────────────────────┐             ┌──────────┐  ┌──────────────┐  ┌──────────┐
│ custodian              │             │ custodian×N │→│ inference ×1  │  │ qdrant   │
│ + Qwen3-VL 8B×2(GPU)│    ──►      │ slim     │  │ GPU 4090     │  │ server   │
│ + embedded Qdrant (exclusive) │    │ no torch │→─────────────────→│ persistent volume │
└─────────────────────┘             └────┬─────┘  └──────────────┘  └──────────┘
  GPU/library/application deadlocked together      └── sidecar shares a :ro volume (the S3 linchpin)
```

### Six-phase progress

| Phase | Contents | Status |
|---|---|---|
| **A** | Split the GPU inference layer skeleton out: `inference_server.py` (FastAPI on :8900, `/embed`/`/rerank`/`/readyz`, background warmup + a GPU serial lock) + `remote.py` (returns full dimensions, client truncates) | ✅ Committed |
| **B** | **Push locking down** + **equivalence go/no-go**: split the big lock into per-resource locks; fixed bf16 normalization so remote↔local vectors are mathematically equivalent | ✅ Committed |
| **C** | CPU tests wired into CI (folded into B) | ✅ |
| **D** | Qdrant: **embedded → server mode** (the real switch for multi-replica): three-way branching in the store + `qdrant_url` threaded through all three call sites + ACL fail-closed re-verified | ✅ Committed |
| **E** | **Containerization**: three Dockerfiles + compose + data migration + `/readyz`; committed three-container stack brought up as-is, all healthy | ✅ Committed `93ae0ef` |
| **F** | **True multi-replica with nginx + unnoticed `docker kill` + graceful shutdown + backpressure + throughput boundaries**; plus a full adversarial review of A–E (27 confirmed) + revisions | ✅ Shipped (see §5-F) |

### Core technical changes

1. **Split out the GPU inference layer** — the pure GPU forward pass becomes an independent HTTP service (the endpoint carries no custodian business concepts whatsoever); the application only needs `inference_url` configured and no longer loads models. Returning full dimensions + client-side MRL truncation guarantees equivalence (and leaves room to swap in TEI/vLLM later).
2. **Removed torch from the core** — the core dependencies no longer include torch, isolated behind the `[gpu]` extra; the slim custodian image's `pip install .` is naturally torch-free, pinned down by a build-time assertion. **Replicas ship a 250MB slim image instead of an 8GB GPU image.**
3. **Pushed locking down** — one big `LockedRetriever` lock became per-resource locks (Store / GPU forward pass / single-flight loading / query cache each get their own lock). Fixed the bug where "the retry sleep held the big lock → the whole replica stalled."
4. **Equivalence** — `_mrl` must `.float()` before truncating dimensions + normalizing (normalizing in bf16 → norm≈1.002, which once caused cosine>1).
5. **Qdrant server + ACL** — three branches (`url > :memory: > path`); the embedded RRF fusion's hard rule about dropping the top-level `should` clause being an unauthorized-access risk is addressed by pushing ACL down into each prefetch to route around it; server mode was re-verified for fail-closed behavior.
6. **The sidecar S3 linchpin** — multi-replica small-to-big relies on a per-doc sidecar, which must share the same `:ro` volume across replicas (otherwise a replica gets a hit but reads nothing → silent degradation). Verified with both positive and negative tests.
7. **Key points of containerization** — pinning `CUDA_VISIBLE_DEVICES` to the 4090, mounting the keys file as a separate `:ro` mount rather than putting it in the RW data volume, and `/readyz` probing downstream so nginx can route traffic correctly.

### Real bugs caught along the way through diagnosis (the real payoff of the rework)

The value isn't in "how much code got written" — it's in the string of "looked correct, broke when actually run" bugs
that each phase's adversarial review + real runs forced out:

- **Root cause of pushing the lock down** = the retry sleep was inside the big lock (found in the A review)
- **bf16 normalize** left vectors with norm≠1 (found in the equivalence test)
- **A hallucinated claim in phase D**: I claimed "threaded through all three call sites" but had missed updating the engine — caught by my own verification discipline, then fixed + a guard test added
- **Pitfall ④, GPU card selection (the nastiest one)**: under `CUDA_DEVICE_ORDER=PCI_BUS_ID`, device0=5070 → the `get_device_name(0)` assertion always fails → inference never becomes ready. Confirmed by testing all three configurations, fixed by switching to `CUDA_VISIBLE_DEVICES`; also discovered that **under WSL2, `device_ids` doesn't provide hard isolation**.
- **Pitfall ⑥, scipy/einops**: present on bare metal, missing in the container — **only surfaced by actually running the committed three-container stack** — confirming the value the adversarial review placed on flagging "never run as-is" as a blind spot.
- **Pitfall ① (my own misjudgment)**: I once claimed there was "no `__main__.py`," when actually a glob search had searched the wrong directory; this has been honestly corrected.
- **Confirmed by adversarial review**: `/readyz` leaking internal network addresses (sec-high), an empty library reporting a false green status, the migration lock check being out of order, and the keys file being stored in plaintext on a shared volume — all fixed and regression-tested.

### The methodological throughline

Each phase followed `implement → dispatch an adversarial agent review → fix confirmed findings → re-verify → commit`,
paired with a "diagnose → fix → verify" discipline (rejecting verdict-style conclusions / blind patching / hallucinated
claims). **This process caught more bugs than writing it correctly the first time would have** — especially pitfall ④,
pitfall ⑥, and the hallucinated claim in phase D, all of which were things that "could never have been found without
actually running it."

---

## 0. In one sentence

Split the **only part of the system that uses the GPU** (the forward passes of Qwen3-VL-Embedding-8B + Qwen3-VL-Reranker-8B)
out into an independent inference service, making `custodian serve` **stateless, GPU-free, and not dependent on having
model files locally** → so it can run as multiple replicas. But **the real switch for multi-replica isn't splitting out
the GPU — it's migrating the embedded Qdrant to server mode** (Phase D); splitting out the GPU is a necessary but not
sufficient condition.

---

## 1. Background and Goals

### 1.1 Why split it out

`custodian serve` currently loads two 8B models into the GPU in-process (see D1). This creates a structural ceiling:
**the application layer is bound to the GPU = it cannot scale horizontally**. Running multiple replicas to handle
concurrency, rolling upgrades without downtime, and crash isolation all require the application layer to be able to
replicate without needing a GPU.

### 1.2 What gets split out / what doesn't

| | Belongs to | Reason |
|---|---|---|
| The **pure GPU forward pass** of `Dense.encode_*` / `Reranker.score` | → independent inference service | The only part that uses the GPU; splitting it out frees the application layer of the GPU |
| MRL dimension truncation / query LRU cache / writing sort order back into `Hit` / ACL / small-to-big | Stays in the application layer | Pure business logic, no GPU; keeping it here is what lets the inference service stay pure enough to swap in vLLM |
| chunker / generator / toolcore | Untouched | Never used the GPU in the first place |

**Boundary principle**: draw the line between "the pure GPU forward pass" and "business logic." The inference service's
endpoints contain **no custodian business concepts whatsoever** (no ACL, no Hit, no sidecar) — only
`texts→vectors`, `(query,docs)→scores`. This means it can be swapped out for vLLM/TEI painlessly in the future.

### 1.3 Constraints (must not be violated)

1. **Backward compatibility**: an empty `inference_url` = local mode (in-process GPU), **zero behavior change** by
   default. Regression baseline = the current CPU test suite fully passes (the count is authoritative per
   [TESTING.md §1](TESTING.md) — the single source of truth for the whole repo; at the time there were also 4 remote
   skeleton tests, which should be excluded when judging "zero breakage to local").
2. **Minimal engine changes**: achieved through dependency injection + factories (`make_dense`/`make_reranker`), without
   changing the local code path.
3. **Equivalence**: the **final retrieval vectors produced by local and remote are mathematically equivalent** — the
   same library can be built locally and queried remotely without misalignment. This is a hard constraint, structurally
   guaranteed by "return full dimensions + truncate on the client," and proven with tests (Phase B).
4. **Stay close to real production practice**: separate probes, readiness-based traffic routing, failure retries, and
   container orchestration are all done the way production systems do it, because the point of this project is to
   **learn production operations**.

### 1.4 Relationship to the existing design

- [DESIGN.md](DESIGN.md) §1, out of scope: "horizontal scaling / multi-replica (the embedded Qdrant ceiling — to be done
  in a scale-driven v2)" ← **this document puts it on the roadmap**
- [DESIGN.md](DESIGN.md) D1, rejected alternative: "Qdrant server mode… this is v2's natural upgrade path (just swap the
  url in EmbedConfig)" ← **expanded in Phase D. ⚠ Note: "just swap the url" refers only to the configuration surface**;
  in practice it also needs a three-way branch in `store.py` + threading through `engine.py` + data migration +
  **re-testing ACL under server mode (Q2)** — the latter two are the heaviest parts, and D1's wording oversimplified this
  (needs correcting per §9).
- [ROADMAP.md](ROADMAP.md) v2 direction: "multiple custodian replicas only become feasible after Qdrant goes server-mode"
  ← **delivered by Phases D→F**

These documents need to be backfilled once the rollout is complete (see §9).

---

## 2. Target Architecture

### 2.1 Target topology: three services, GPU in only one place

```
                    ┌───────────────────────────────────────────────┐
   client ──────▶   │  nginx (load balancer, Phase F)                  │
                    └──────────────┬────────────────────────────────┘
                          ┌────────┴────────┐
                          ▼                 ▼
                 ┌─────────────────┐ ┌─────────────────┐    stateless / no GPU / no torch
                 │ custodian replica #1 │ │ custodian replica #2 │ …  can scale to N replicas (unlocked in Phase D)
                 └───┬─────────┬───┘ └───┬─────────┬───┘
       /embed /rerank│         │gRPC     │         │
                     ▼         ▼         ▼         ▼
          ┌────────────────────┐   ┌────────────────────────┐
          │ inference (×1, GPU) │   │ qdrant (server mode)    │
          │ two 8B models resident, returns full dims │   │ persistent volume, shared across replicas (Phase D)│
          └────────────────────┘   └────────────────────────┘

  Library building: an offline singleton, runs local (inference_url left empty) to bypass the per-chunk HTTP tax (see D4/F4)
```

### 2.2 Three decisive trade-offs

**Trade-off one — draw the boundary between "pure GPU forward pass" and "business logic."** See §1.2. The inference
service is pure enough that its endpoints carry no business concepts → it can be swapped for vLLM.

**Trade-off two — return full dimensions + truncate on the client.** The inference service constructs its `Dense` with
`dense_dim=10**9` (`inference_server.py:33`, `_FULL_DIM`), so its `_mrl` **never truncates** and returns the model's raw,
full-dimension, normalized vectors; the client's `RemoteDense._mrl_np` (`remote.py:49`) truncates to the real
`dense_dim`. **This makes local and remote go through the same `Qwen3VLEmbedder` forward pass + a mathematically
equivalent truncation**, so equivalence is a **structural guarantee rather than an empirical bet** — this is exactly
the hard reason it's superior to switching straight to vLLM (where pooling is reimplemented and equivalence would have
to be gambled on).

**Trade-off three — a dense failure must be "loudly retryable," not swallowed by an overly broad fallback, and
definitely not silently degraded.** Inference-service warmup (1-2 minutes) and rolling replica restarts are a **normal
operational state**. The skeleton's actual current behavior (**not the "bare 500" that v1 claimed**): a dense exception
raised via `encode_query` (which is not inside retrieve's try block) bubbles up to the `except Exception` in
`toolcore.py:216`, and gets **swallowed into a generic `backend_unavailable` (HTTP 200, retriable:true), with no retry,
no backoff, losing the "inference unavailable" semantics**. Decision: the client does a bounded number of retries with
backoff to absorb transient failures, and once retries are exhausted it raises the semantic exception
`InferenceUnavailable`, which toolcore refines into `inference_unavailable`; dense **never degrades to
empty/pure-BM25** (the query vector is the primary recall signal, and silently degrading it means the user gets a
quality collapse with no visibility — worse than an outright failure). rerank keeps its existing degrade path
(`retrieve.py:105` already does `except Exception → hits[:k]`, since it's an enhancement signal and safe to degrade) —
**this asymmetry of "dense loud / rerank degrades" is intentional and must be locked in with tests (S10)**.

> Timeout semantics (the basis for F3): `inference_timeout=120` passed to `httpx.Client(timeout=...)` is a **total**
> timeout. When the process hasn't started (connection refused), httpx **immediately raises `ConnectError` without
> waiting 120s**; "waiting 120s" only applies when "connected but the server is hanging/blocked in warmup." Don't
> conflate these two scenarios (they must be covered separately when mocking in Phase A).

### 2.3 Skeleton already in place (current state, before Phase A)

| File | What's already there | Status |
|---|---|---|
| `src/embedder/inference_server.py` | FastAPI: `/embed` `/embed_image` `/rerank` + `/healthz` `/readyz`, background warmup, a GPU serial lock | Skeleton in place, has the P0-2 defect |
| `src/embedder/remote.py` | `RemoteDense`/`RemoteReranker` (inherit from the base classes, override the forward pass to go over HTTP) + `make_dense`/`make_reranker` factories | Skeleton in place, has the P0-1/P2 defects |
| `src/embedder/config.py` | `inference_url` / `inference_timeout` fields | Already added; Phase A adds the retry fields |
| `src/custodian/config.py` | `inference_url` (`CUSTODIAN_INFERENCE_URL`) | Already added |
| `src/custodian/engine.py:63` | `build_retriever` skips the local scripts check in remote mode | Already added |
| `tests/engine/test_remote.py` | 4 tests: ① factory routing ② `_load` is a no-op + `_mrl_np`↔`_mrl` equivalence (same function, `:40`) ③ encode_text + query cache (`:56`) ④ reranker contract (`:72`) | Passing |

**Assessment**: the skeleton's **core design is right** (the boundary, returning full dimensions, and the factory
switch are all sound), but the adversarial review dug up two "guaranteed to happen on every restart" operational bugs
(P0-1: no retry / P0-2: a fake "loading" status on load failure), plus an equivalence gap that's a "go/no-go for
production" issue, plus the real blocker for multi-replica (Qdrant). Each is covered below.

---

## 3. Key Decisions Table (incorporating the adversarial review)

### 3.1 Service boundary and equivalence

| # | Question | Choice | Why |
|---|---|---|---|
| D1 | Which side truncates dimensions | **Return full dimensions + truncate on the client** | Structurally guarantees equivalence; one inference service can serve libraries with different `dense_dim` values; bandwidth isn't a bottleneck within the same data center |
| D2 | `_mrl_np` silently doesn't truncate when `dense_dim ≥ full dim` | **Add a lower-bound assertion that fails loud + expose `full_dim` via `/healthz` for a startup check** | A misconfigured `dense_dim` would silently produce misaligned vectors; the assertion turns a "config mistake" into a "startup crash" |
| D3 | Who controls the rerank instruction | **The client is the sole source of truth**; the server's `/rerank` uses `q.instruction`, and falling back silently when it's missing fails loud instead | Currently the client sends an instruction but the server **ignores it** and uses its own config (`inference_server.py:116` calls `reranker.score(q.query, q.documents)` without passing `q.instruction`) — a silent dual-source footgun |
| D4 | Image encoding across processes | **Library building goes local / querying goes remote**; the base64 TODO is non-blocking | `encode_image` is only used for library building (`embed.py:72`), never called on the query path; library building is already a single-process, single-instance job that runs on the same machine as the GPU, so the cross-machine issue simply disappears |
| D5 | Is text-only querying enough for an MVP | **Yes**; a query replica only depends on `/embed`+`/rerank`+`/readyz` | The query path never calls `encode_image` |
| D6 | How the contract should evolve | **Only add fields, never rename keys**; add `full_dim`/`model_dense` fingerprints | Backward compatible + supports the equivalence assertions and detecting "connected to the wrong service" |

### 3.2 Failure modes (P0/P2)

| # | Question | Choice | Why |
|---|---|---|---|
| F1 | dense 503/timeout → swallowed by toolcore's overly broad fallback into a generic `backend_unavailable`, no retry (**P0-1**) | `_post_*` gets bounded retries + exponential backoff (only retries 503/connect/read-timeout, 4xx raises immediately); **the actual fix lands in `toolcore.py`** (refining `inference_unavailable`, see §5-A/M1); **dense does not degrade to empty** | Warmup/rolling restarts are a normal operational state; the query vector is the primary recall signal |
| F2 | `_guard` fails to check `state.err` → a permanent load failure still reports "loading" (**P0-2**) | Align `_guard` with `readyz`: check `state.err` first; **fix `/healthz` at the same time** (it also currently ignores `state.err`, see S7) | On a bad card or a missing model, ops sees "perpetually warming up" instead of "load failed," which misleads troubleshooting |
| F3 | A single 120s timeout value means even a hung server gets waited on for 120s (**P2-2**) | Split into `httpx.Timeout(connect=3, read=120, write=10, pool=5)` | A short connect timeout catches a hang quickly; a long read timeout tolerates long-text forward passes (note: connection refused already fails immediately, see the timeout semantics in §2.2) |
| F4 | Per-chunk HTTP overhead for remote library building (**P2-4**) | Library building defaults to local; if remote is insisted on, `embed.py` batches | Library building needs a GPU machine + exclusive Qdrant access and has no multi-replica requirement, so remote is pure overhead with zero benefit |
| F5 | Two separate `httpx.Client` instances each build their own connection pool (**P2-1**) | dense/reranker share one client (module-level, cached by url); **the close point is well-defined** (`create_app`'s lifespan shutdown, or `atexit`, see §5-A/M8) | Hitting the same inference service, two connection pools is pure waste |

### 3.3 Multi-replica prerequisites (the go/no-go most easily overlooked in the final plan)

| # | Question | Choice | Why |
|---|---|---|---|
| Q1 | Multiple custodian replicas won't come up | **Embedded Qdrant → server mode**. `store.py` currently has a two-way branch (`:memory:`/`path`), which needs to expand to a **three-way branch of `url`/`:memory:`/`path`** (`:memory:`'s priority must be preserved, since a lot of tests depend on it short-circuiting); add `qdrant_url` to `config.py`; **`engine.py:70-73` must explicitly thread through `qdrant_url=cfg.qdrant_url`** (otherwise it's the same pitfall as `inference_url` before: configured on the custodian side but silently dropped) | The embedded single-process file lock is mutually exclusive — multiple custodian processes can't open the same library at once; **this is the real switch for multi-replica, not splitting out the GPU**. The groundwork is already laid in the code (a comment in `store.py`'s payload index notes "no-op locally but needed for server mode") |
| Q2 | ACL semantics change under server mode | **Add new server-mode ACL tests** (not rerunning the `:memory:` version — that still goes through embedded fusion and gives zero coverage of server mode; see M4): parameterize the `Store` fixture to support `qdrant_url` → build a library against a real Qdrant server → run the same 5 fail-closed assertions as `test_acl_hard_filter` | The embedded RRF fusion once silently dropped the top-level `should` clause → unauthorized data leakage (a security hard rule); server-mode fusion has different filter semantics, and this is a matter of **security equivalence**, on the same level of importance as vector equivalence |
| Q3 | Whether to keep the big `LockedRetriever` lock | **Keep the in-process big lock for multi-replica v1** (serialized retrieval within a single replica is a known limitation); realizing "increased non-GPU concurrency" requires first auditing shared mutable state (`retrieve.py`'s query-scoped `cache={}`, sidecar reads), then evaluating loosening the lock granularity + concurrency load testing — **deferred to §5-F**, not torn out blindly | Once server-mode Qdrant is concurrency-safe and the GPU forward pass has moved out, both original reasons for the big lock are gone; but before removing it, we must confirm there's no other shared mutable state |

### 3.4 Inheritance vs. an independent client, and choosing an inference engine

| # | Question | Choice | Why |
|---|---|---|---|
| I1 | `RemoteDense(Dense)` inheritance turns the base class's internal implementation into an implicit remote contract, which is fragile | **Keep the inheritance + lock down the invariants with tests** (a light touch); extracting pure functions to remove the inheritance is deferred until it actually causes a problem | Removing the inheritance would touch local code, conflicting with "minimal engine changes"; locking invariants with tests is low-cost and fails immediately if broken |
| I2 | Can the custodian-slim image go torch-free | **The scope of the go/no-go assertion = the entire import closure**: in the slim environment, `python -c "import custodian.engine"` + constructing `RemoteDense/RemoteReranker`, then check `'torch' not in sys.modules` (not just the top level of `dense.py`/`rerank.py` — the entire `engine→Retriever→Store→qdrant_client` chain must not pull in torch) | If any link in the chain does a top-level `import torch`, the slim image crashes on import; torch should only be imported lazily inside `_load`/`_mrl` |

**Final decision on choosing an inference engine: keep the self-contained FastAPI service, don't switch to vLLM/TEI/Triton.**

| Backend | Verdict | One-line reason |
|---|---|---|
| **Self-contained FastAPI (current state)** | ✅ Keep it for now | Both sides go through the same `Qwen3VLEmbedder`, structurally guaranteeing equivalence; a single 4090 is enough; the probing/locking/equivalence patterns are all transferable to production |
| **vLLM** | Reassess once scale demands it | The only qualifying alternative (continuous batching is its real hard value); **but pooling is reimplemented, so equivalence must be measured at cosine>0.9999 before switching** |
| **TEI** | ❌ Ruled out | **Pooling is reimplemented → breaks the structural equivalence from trade-off two** (same reason vLLM is a risk); secondarily, no vision-language support, so it can't do `encode_image` either |
| **Triton** | Shelved | No out-of-the-box model for Qwen3-VL; writing a custom Python backend would be about as much work as the current `inference_server.py` — over-engineering |

**The only legitimate trigger for switching to vLLM**: query QPS rising to the point where the GPU forward pass is
visibly queuing (the `/embed` queue depth staying above 1), requiring continuous batching — not "vLLM being trendier."
When that trigger hits, migration cost is very low (the endpoint contract carries no business concepts).

> **See [VLLM_PLAN.md](VLLM_PLAN.md) for the detailed vLLM plan** (drafted 2026-07): not a replacement — coexistence
> first (a same-contract adapter + a compose profile). vLLM's official support for Qwen3-VL-Embedding pooling is already
> grounded; the equivalence/throughput go/no-go gates + the Phase 0 probe script
> `scripts/vllm_equiv_probe.py` are already in place — **it's only promoted to Plan A once it clears the gates**.

---

## 4. Gap Checklist Found by the Adversarial Review (P0/P1/P2/P3)

> The skeleton runs, but the following are its real defects, sorted by severity. **P0-1 and P0-2 are the most urgent to
> fix** — they trigger under normal operating conditions.

| # | Severity | Problem | Location | Fix | Phase |
|---|---|---|---|---|---|
| P0-1 | High ✅**Fixed in A** | dense has no retry → queries during warmup/restart windows get swallowed into a non-retryable generic `backend_unavailable` (doesn't crash outright, but transient failures aren't absorbed) | `remote.py` `_post_retry` (originally no retry); `retrieve.py:93,252` `encode_query` isn't inside the try; `toolcore.py:216/278` a broad `except` swallows the semantics | `_post_retry` retries with backoff on **any 5xx + any httpx timeout (`TimeoutException`) + `ConnectError`**, raises `InferenceUnavailable`; toolcore adds **duck-typing** before its two `except` blocks (a marker, no embedder import) to route to `inference_unavailable` | A |
| **M1** | High (concurrency) ⏳**Must fix before B** | The remote retry backoff's `time.sleep` sits inside LockedRetriever's **big retrieval lock** → the whole replica's queries serialize and stall during warmup/rolling-restart windows (a side effect of the P0-1 fix; found by the Phase A adversarial review) | `remote.py:76` sleep; `engine.py:37-39` the lock wraps the entire search; `retrieve.py:93` encode inside the lock | Restructure lock granularity: move remote-backend encode/HTTP calls out of the retrieval lock + add a fine-grained lock to `dense._query_cache` (same root cause as the big-lock split in §3.3 Q3). **Only triggers once remote is actually active, so it belongs at the start of B (before remote actually starts running), not stuffed hastily into A** | **Start of B** |
| P0-2 | High ✅**Fixed in A** | `_guard` fails to check `state.err` → a load failure still reports "loading"; `/healthz` also ignores `state.err` | `inference_server.py` `_guard`/`readyz`/`healthz` | Extract a pure `_readiness` function; `_guard`/`readyz` check `err` first; `/healthz` keeps liveness=ok but exposes an `err` field (S7: liveness≠readiness) | A |
| P1-1 | High | `_mrl_np` has no lower-bound assertion — `dense_dim ≥ full dim` silently doesn't truncate | `remote.py:49-58` | Add a `full.shape[-1] < d` fail-loud assertion; expose `full_dim` via `/healthz`, with a client-side startup assertion | B |
| P1-2 | High | **Zero test coverage** for numpy/torch renormalization equivalence (the eps values were checked and are both `1e-12`, so they're consistent; the real risk is dtype) | `remote.py:56` (`np.clip(_,1e-12,_)`, fp32) vs `dense.py:60` (`F.normalize`, default eps=1e-12) `:61` (`bf16→.float()`) | Add a GPU equivalence test with `np.allclose(atol=1e-6)`; **focus on validating the bf16↔fp32 renormalization numerical deviation** (not eps) | B |
| P1-3 | High | Inheritance turns the base class's internal implementation into an implicit remote contract, which is fragile | `remote.py:22,61` | Lock in with tests: "the entire import closure has no torch (I2)" + "`encode_query` goes over HTTP without touching `_model`" | C |
| P2-1 | Medium ✅**Fixed in A** | Two separate `httpx.Client` instances each build their own connection pool, with no close | `remote.py:29,68` | Share one client (module-level `_CLIENTS` keyed by url) + close via atexit | A |
| P2-2 | Medium ✅**Fixed in A** | A single 120s timeout value — even a hung server gets waited on for 120s | `remote.py:29,68` | Split into `httpx.Timeout(connect=3,read=120,...)` | A |
| P2-3 | Medium | Rerank instruction has two sources (the client sends one but the server ignores it) | `inference_server.py:116` | The server uses `q.instruction`, consolidating on the client as the sole source of truth | E |
| P2-4 | Medium | Per-chunk HTTP overhead for library building | `embed.py:72,75` | Default library building to local; otherwise batch | D4/documentation |
| P3 | Low | `Reranker` has no `_gpu_error` cache, asymmetric with `Dense` | `rerank.py:35-38` (bare `_assert_gpu`) vs `dense.py:25,41-47` | Align `Reranker._load` to add the same cache | F |

---

## 5. Six-Phase Rollout Plan (A→F)

> Principle: **first patch the P0 failure modes (guaranteed to occur operationally) → then add the P0 equivalence tests
> (a go/no-go for production) → then build a regression safety net → only then do multi-replica/containerization**.
> Strict ordering: replicating horizontally on top of a buggy, fragile single replica just replicates the bugs and
> amplifies the failure surface. Every "fixed" claim in each phase must come with a reproduction command + a before/after
> measurement comparison. **Every verification command first requires satisfying the "environment prerequisite" at the
> top of this document.**

### Phase A — Patch the P0 failure modes (still single-replica; fully verifiable with CPU mocks)

**Changes** (5 files):
- `embedder/errors.py` (**new, a pure exception with no httpx dependency**): `class InferenceUnavailable(RuntimeError)`.
  Placed in a neutral file so `toolcore` only needs to import a single pure exception class and doesn't touch httpx
  (keeping toolcore thin).
- `config.py` (embedder): add `inference_connect_timeout=3.0` / `inference_retries=2` / `inference_backoff=0.5`
- `remote.py`: module-level `_CLIENTS` + `_get_client` (a shared connection pool keyed by url + a split `httpx.Timeout`,
  fixing P2-1/P2-2);
  `_post_retry` (exponential backoff on **any 5xx + `httpx.TimeoutException` + `ConnectError`**, clamped to
  `max(0,retries)`, 4xx raises immediately, raises `InferenceUnavailable` once exhausted, fixing P0-1); `_post_vectors`/`score`
  switched to use it; `atexit` closes `_CLIENTS`.
- `inference_server.py`: extract a pure `_readiness(ready,err)` function; `_guard`/`readyz` check `err` first (fixing
  P0-2); `/healthz` keeps `status:ok` (liveness) but adds an exposed `error` field (S7: a config error can't be fixed by
  a restart — reporting unhealthy would just crash-loop it)
- `toolcore.py`: in `_retrieve_impl` (`:216`) and `_grouped_impl` (`:278`), change `except Exception` **to `as e` +
  duck-type routing**:
  `if getattr(e,"inference_unavailable",False): return _err("inference_unavailable", retriable=True)` (**where P0-1
  actually gets fixed**; using a marker instead of importing embedder keeps toolcore stdlib-only; `service.py` is
  unchanged — the exception never reaches that layer)
- `embed.py:45` (S4): `Dense(cfg)` → `make_dense(cfg)` (library building also goes through the factory; the default
  `inference_url=""` still means local, no behavior change; handling failures for a remote-based build belongs to
  Phase D)

**Verification** (`pytest tests/engine/test_remote.py`, 18 tests + the full `pytest tests`, 199 tests, all CPU-mocked;
first satisfy the environment prerequisite):
- Retry: the first 2 calls return 503 → the 3rd succeeds (exactly 2 retries); **502/504 transient gateway errors** also
  retry; **500** raises `InferenceUnavailable` once exhausted
- No retry: 4xx raises immediately (zero retries)
- Timeout: `ConnectTimeout` (a TimeoutException, not a ConnectError) is caught and retried; backoff is
  **exponential**, `[0.5,1.0]`; a negative retries value is clamped without crashing
- toolcore routing: mock returns 5xx continuously → both `_retrieve_impl`/`_grouped_impl` return `inference_unavailable`
  (not `backend_unavailable`); an ordinary exception still returns `backend_unavailable`
- P0-2: `_readiness(False, err)` returns `error` (not `loading`); `_readiness(True, None)` returns `ready`
- **The asymmetry (S10)**: mock reranker raises `InferenceUnavailable` → `retrieve.search` **degrades** to hybrid
  results (no loud failure, no crash); dense stays loud (as above)
- **Client sharing (P2-1)**: for the same url, `d._client is rr._client`
- **Regression**: `pytest tests`, all 199 passing (zero breakage to local; baseline per §1.3)

**Milestone**: during the inference service's restart/warmup windows, both `/v1/retrieve` and `/v1/retrieve_grouped`
receive a structured `inference_unavailable` (retriable), instead of a non-retryable generic `backend_unavailable`;
when loading fails permanently, `/readyz` and `/embed` report `error`, while `/healthz` still reports `ok` (liveness)
but exposes the `error` field.

**Phase A completion status (2026-07)**: ✅ Code + tests shipped, `pytest tests` **199 passing** (185→199, +14 remote
tests). After a four-way adversarial review, fixed M2 (the timeout base class missed `ConnectTimeout`) / M3 (non-503 5xx
codes weren't retried) / M4 (rerank degradation had only a type assertion, no behavior test) / S1 (negative retries) /
S6b (the backoff multiplier). **Remaining: M1 (high, concurrency)**: the remote retry backoff sits inside the retrieval
big lock → deferred to **the start of B** (before remote actually starts running) to fix the lock granularity; S2/S3
(test hygiene / `_get_client` has no lock) are tightly coupled with M1 and get fixed together at the start of B. **In
Phase A, the local default path doesn't trigger M1** (the factory selects local Dense, which never goes through
`_post_retry`), so it doesn't block A's claim of "zero breakage to local"; but M1 is a core availability property of
the remote path end-to-end, and must be fixed before B starts remote.

### Phase B — P0 equivalence tests (still single-replica; **requires a CUDA torch environment + a real inference service running**)

**Changes** (✅ shipped):
- `remote.py:_mrl_np` gets a lower-bound assertion (P1-1); `dense.py:_mrl` **changed to normalize in fp32** (the P1-2
  fix, see completion status); `inference_server.py:/healthz` exposes `full_dim`/`model_dense` (D2/D6); the client's
  dense_dim check falls back on the runtime lower-bound assertion in `_mrl_np`
- Added `tests/engine/test_equivalence_gpu.py` (auto-`skipif` when there's no GPU; only needs the local model to run
  both the torch and numpy truncation paths on the same real bf16, full-dimension output — doesn't start a service,
  doesn't OOM). The end-to-end script (starting a service, mixed build/query) is in the scratchpad, with the command
  recorded in the completion status.

**How to run it (copy-paste ready; requires custodian's CUDA torch, not the default CPU torch)**:
```bash
# 1) Start the inference service (uses the GPU, warms up in the background for 1-2 minutes)
conda activate custodian && python -m embedder.inference_server &   # defaults to 0.0.0.0:8900
# 2) Wait for /readyz to turn green
until curl -sf localhost:8900/readyz; do sleep 5; done
# 3) Run the equivalence tests (pointed at the real service)
CUSTODIAN_INFERENCE_URL=http://localhost:8900 pytest tests/test_equivalence_gpu.py -m gpu -v
```
- **E1, vector-level**: for the same batch of texts (original non-English language/English, long/short, special characters),
  `Dense(cfg_1024).encode_text` vs `RemoteDense.encode_text`, `np.allclose(atol=1e-6)` and `norm≈1.0`; same for images
- **E2, mixed build/query** (⭐ **the production go/no-go**): build a small library locally → switch to remote to query
  the same collection, and the top-10 doc_id set matches "built local, queried local" exactly (score difference <1e-5)
- **E3, reranker**: for the same query+docs, local `score` vs remote `/rerank` are `allclose`

**Milestone**: E1/E3 measured as equivalent + **E2 mixed build/query is end-to-end top-k consistent**. ⚠ E1's
element-wise equivalence **does not imply** E2 (top-k has to survive both HNSW approximation and RRF's **rank**-based
fusion — the score gap between near-duplicate segments can be smaller than E1's max diff and still flip the ranking);
**E2 must be tested end-to-end, and cannot be logically inferred from E1**. **If E2 fails, the entire plan does not go
to production** — misalignment from mixed build/query is silent data corruption, and the hardest kind to detect.

**Phase B completion status (2026-07; revised after the Phase B adversarial review)**: pushing the lock down (M1)
was done first (`fe04256`); **E1/E3 measured equivalent, E2 not yet validated end-to-end**:
- **E1, encode fully equivalent**: cosine=**1.0000000**, maxdiff **2.98e-08**, query cosine=1.0 (measured in a
  time-sliced run with `scripts/equiv_gpu.py`).
- **E3, rerank fully equivalent**: maxdiff **0.00**.
- **Fixed a real bug, P1-2**: `Dense._mrl` was originally normalizing in **bf16** → after converting to fp32, the norm
  was ≈**1.002**; after unifying on fp32 (`.float()` first, then truncate + normalize), the norm is **1.000**, aligned
  with remote's `_mrl_np`. **Guard test**: `test_remote.py::test_mrl_fp32_normalize_on_bf16_input` (pure CPU: a bf16
  tensor into `_mrl` → norm=1.0, plus a reverse guard that normalizing in bf16 would give >1.001; removing `.float()`
  makes it fail). **Old-library compatibility**: converting bf16→fp32 before truncation is lossless and doesn't change
  direction, so old bf16-built libraries can still be queried with COSINE; rebuilding gets numerical consistency.
- **⚠ E2 mixed build/query: a known gap, not yet validated end-to-end** (review finding M2). E1's element-wise
  equivalence (2.98e-8) **cannot** logically imply E2: the default hybrid path goes through both HNSW approximation and
  RRF's **rank**-based fusion, and near-duplicate chunks can have a cosine difference smaller than 2.98e-8 → local/remote
  ranking could flip, and RRF amplifies it. **Must be measured before production**: ~50 real queries, build the library
  locally, then encode queries with both local and remote against the same collection (hybrid), and assert the top-k
  `chunk_id` ordering matches exactly.
- **Image equivalence** (review finding S2): `encode_image` shares `_mrl` with text, so truncation equivalence holds
  for it the same way it does for text (covered by the bf16 guard); but **cross-machine consistency of the image path
  is a separate open item** (`remote.py`'s base64 TODO: the library-building machine and the querying machine looking at
  the same path could resolve to different images).
- **VRAM constraint (measured)**: inference (2×8B) uses **33GB/48GB** of the 4090; loading a local Dense as well would
  need another 16GB → double-loading **OOMs**, so the equivalence measurements were done **in separate time slices**
  (remote archives its output → stop the service → compare against local). This also confirms "the application layer is
  GPU-free": a custodian replica uses 0 GPU.
- **Making the guard wording honest** (review finding S1): the `_mrl_np` lower-bound assertion (P1-1) is a **runtime**
  fallback (only triggers on the first encode call, and only covers the case where dense_dim > full_dim); the
  `full_dim`/`model_dense` fields exposed by `healthz` currently **only serve manual troubleshooting, with no automatic
  client-side check** (a startup fail-fast client-side probe is a future item).
- **How to run it**: CPU `pytest tests` (210 passing, including remote/concurrency/bf16 guards); GPU
  `pytest tests/engine/test_equivalence_gpu.py` (custodian, auto-skipped otherwise); the end-to-end time-sliced run is
  **`python scripts/equiv_gpu.py --step remote|local`** (already checked into the repo, reproducible for audit).

### Phase C — Get the CPU-mock test matrix into CI (still single-replica; a permanent regression net)

**Changes**: expand `tests/engine/test_remote.py` into a full matrix (fully CPU-mocked, permanent in CI):
- Factory routing / `_mrl_np` truncation+renormalization / rerank contract / query LRU consistency across backends /
  Phase A's retry/timeout/degradation/asymmetry
- **The torch-free go/no-go (I2, S5)**: on the slim path, `import custodian.engine` + construct `RemoteDense`/`RemoteReranker`,
  then assert `'torch' not in sys.modules` (the entire import closure, not just the top level of the two files)
- **I1**: a regression guard that image payloads are passed as-is (not base64)

> ⚠ Precise wording (S4): it's the **tested application-layer code path** (`remote.py`/`make_dense`/`custodian.engine`)
> that doesn't import torch. **The tests themselves are allowed to use CPU torch** to cross-check `_mrl` (torch) against
> `_mrl_np` (numpy) (`test_remote.py:42` already does exactly this, and shouldn't be removed).

**Milestone**: `pytest tests/engine/test_remote.py` fully passing, and after `import remote`, `sys.modules` contains no
torch.

### Phase D — The real switch for multi-replica: Qdrant server mode (⭐ **true multi-replica is only possible after this step**)

**Changes**:
- `store.py`: the two-way branch (`:memory:`/`path`) becomes a **three-way branch of `url`/`:memory:`/`path`** (if
  `qdrant_url` is set, use `QdrantClient(url=...)`; **`:memory:`'s priority is preserved**, otherwise in-memory tests
  would fall through to the url branch and crash)
- `config.py` (embedder + custodian): add `qdrant_url` (`CUSTODIAN_QDRANT_URL`)
- **`engine.py:70-73`**: add `qdrant_url=cfg.qdrant_url` to `EmbedConfig(...)` (missing this means it's configured on
  the custodian side but silently dropped)
- **`tests/`**: parameterize the `Store` fixture to support `qdrant_url` (paving the way for Q2's server-mode ACL tests)
- Start a Qdrant server (docker `qdrant/qdrant`), and migrate the embedded library's data into it

**Verification**:
```bash
# (a) Backward compatibility: without qdrant_url set → goes through the embedded path, zero behavior change
pytest tests
# (b) The multi-replica lock unlocked (the core milestone; no need to send queries — dense's lazy load means no GPU is used):
#     dense only loads on the first query, so at custodian serve startup only Qdrant is opened. Starting two serve
#     processes is enough to verify the lock, no GPU needed.
docker run -d -p 6333:6333 qdrant/qdrant
CUSTODIAN_QDRANT_URL=http://localhost:6333 CUSTODIAN_PORT=8801 custodian serve &   # succeeds
CUSTODIAN_QDRANT_URL=http://localhost:6333 CUSTODIAN_PORT=8802 custodian serve &   # also succeeds (server mode)
#     Counter-example: with neither one setting QDRANT_URL, both pointing at the same CUSTODIAN_INDEX_DIR → the second
#     one reports a lock error when Store opens the embedded Qdrant
# (c) ⭐ Security go/no-go: new server-mode ACL tests (not a rerun of the :memory: version)
CUSTODIAN_QDRANT_URL=http://localhost:6333 pytest tests/engine/test_store.py -k acl_server
```
> ⚠ Two things to be precise about (M3/M4): ① the D milestone **stops at "both serve processes start successfully"**
> (proving the Qdrant file lock is unlocked); sending real queries requires loading models, and local double-loading =
> 4 8B models would blow past 48GB of VRAM on the 4090 — **verifying real remote two-replica request handling is
> deferred to E** (once the inference container is running and custodian has no models of its own).
> ② the ACL tests are a **new server-mode version** (a parameterized fixture pointing at a real server), not a plain
> rerun of the hardcoded `:memory:` version of `test_acl_hard_filter` — the latter still goes through embedded fusion
> and gives zero coverage of server semantics, letting things through with a false green.
> First confirm whether `custodian serve` supports `CUSTODIAN_PORT` (config has `CUSTODIAN_PORT`, ✓); if a `--port`
> flag is needed, add it as CLI support in this phase.

**Milestone**: two `custodian serve` processes connecting to the same Qdrant server both start successfully
simultaneously (the embedded case would report a lock error on the second one); the new server-mode ACL
unauthorized-access tests all pass (fail-closed behavior preserved).

**Phase D completion status (2026-07; revised after the Phase D adversarial review)**: server-mode code + regression
tests shipped, both go/no-go gates passed.
⚠ **The first version had one threading gap I missed + two tests giving a false green, both caught and fixed by the
review** (recorded honestly):
- **Code (three-way branching + threading through all three call sites)**: `store.py`'s `__init__` changed to a
  three-way branch of **url > :memory: > path** (`:memory:`'s priority preserved);
  `EmbedConfig/CustodianConfig.qdrant_url` (`CUSTODIAN_QDRANT_URL`). **Threading it through must cover every production
  call site** — review findings M1/M2/M3: the field was added but the first version **wasn't consumed at any of the
  three call sites** (query-time engine / library-building indexer / mcp_stdio agentic — I had claimed I'd updated the
  engine but had actually missed it), so the server path had never actually been connected end to end; now fixed with
  `qdrant_url=cfg.qdrant_url` added at all three of `engine.build_retriever` / `indexer.run_index` / `mcp_stdio._config`,
  **measured directly**: `build_retriever(server cfg).store.client._client == QdrantRemote`; CPU guard
  `test_review_fixes::test_qdrant_url_reaches_store_via_engine` (mocks QdrantClient, fails if the engine threading is
  removed). Backward compatibility: without qdrant_url set, it goes through the embedded path, zero breakage overall.
- **⭐ Q2, ACL unauthorized-access re-test (security go/no-go) = GO**: measured against Qdrant server v1.18.0, at
  **two levels**: ① at the pipeline level, the hybrid/dense/sparse paths all fail-closed on final delivery; ②
  **at the fusion layer (the real claim in review finding M4)**, `test_server_fusion_no_should_leak_raw` **bypasses the
  exit-point `acl_admits` check** and directly asserts that the server's raw RRF `query_points` output contains no
  unauthorized points — this is what actually proves that "the embedded mode's hard rule about dropping the top-level
  `should` clause hasn't recurred under server mode" (the ACL push-down into prefetch in store.py actually takes effect
  in server-side fusion), rather than relying on a mode-agnostic exit-point recheck as a safety net. list_documents
  (scroll) has the same kind of raw probe (S1).
- **⭐ Multi-replica lock = GO (with a caveat)**: with the embedded mode on the same path, a second client reports
  `already accessed` (single-process exclusive); with server mode, two clients can open the same library at once.
  ⚠ Review finding S2: this verifies two Stores **within the same process**, proving "the file-lock ceiling has been
  removed at the client layer," **not** real multi-replica across two independent custodian serve processes (which
  also involves sharing the sidecar, S3) — verifying real two-process replica request handling is deferred to
  **Phase E** (compose).
- **Regression tests**: `tests/engine/test_store_server.py` (skipif the server is unreachable;
  `CUSTODIAN_REQUIRE_QDRANT_SERVER=1` forces it not to silently skip = a false green, per review finding M5) — 4 tests:
  the three pipeline paths / the raw fusion probe / list_documents / multi-replica locking. The health probe was changed
  to a real connection check (`get_collections`).
- **Data migration (for a real deployment)**: migrating the existing embedded `~/rag_real` to server mode uses
  `scroll`+`upsert` (or a Qdrant snapshot); **the real library hasn't been migrated yet** (D verified the semantics
  with a small library of fake vectors); migration + real two-process custodian serve request handling is deferred to
  **Phase E** (when compose brings up inference + multiple replicas end to end).
- **Current state of `Store._lock`**: still kept (M1); once server-mode Qdrant is concurrency-safe, the rationale for
  this lock "protecting the single embedded client" is gone, and **loosening/removing it is deferred to Phase F/Q3**
  (first confirming there's no other shared mutable state). The Qdrant server runs from the v1.18.0 binary (not docker;
  Docker Desktop is deferred to Phase E's containerization).

### Phase E — Containerization / orchestration (real multi-replica request handling)

**Changes** (⚠ two parts of the plan were disproven by real measurement — see "rollout status" for the actual outcome):
- `Dockerfile.inference` (~~`nvidia/cuda` base + `.[gpu]`~~ → **base changed to
  `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`**, torch preinstalled, see pitfall ③ in the rollout status),
  `Dockerfile.custodian` (slim, **no torch installed**), `docker-compose.yml` (inference×1 with GPU passthrough + a
  Qdrant server + custodian×N)
- Also fixed P2-3 along the way (consolidating the rerank instruction on the client)
- Key compose points (pitfalls specific to Windows/WSL2):
  - ~~`CUDA_DEVICE_ORDER=PCI_BUS_ID`~~ → **`CUDA_VISIBLE_DEVICES=GPU-<4090-UUID>`** (under PCI_BUS_ID, device0=5070 →
    the app's assertion always fails → never ready; see pitfall ④ in the rollout status)
  - Models are mounted via `-v ...:/root/models:ro`, placed on WSL's ext4 (**not under `/mnt/c`** — I/O across that
    boundary is slow)
  - Healthcheck is `curl /readyz` + **`start_period: 300s`** (the two 8B models warm up serially — too short a period
    causes infinite restart loops)
  - custodian's `depends_on: inference: condition: service_healthy`
  - **`CUSTODIAN_KEYS_FILE` must be configured** (see pitfall ⑤ in the rollout status): binding to 0.0.0.0 triggers a
    fail-closed guard requiring keys mode

**Verification/milestone**:
```bash
docker compose up -d
until curl -sf localhost:8900/readyz; do sleep 10; done       # inference turns green in <5min
curl -s localhost:8787/v1/retrieve -d '{"query":"..."}'       # custodian remote request handling (this is where real two-replica remote handling is proven)
docker compose exec inference python -c "import torch; print(torch.cuda.get_device_name(0))"  # includes "4090"
docker compose exec custodian    python -c "import torch"        # should be ModuleNotFoundError (proving torch is truly gone)
```

**Rollout status (measured 2026-07-06, local Docker Desktop + WSL2 + 4090)**:

> The `pytorch/pytorch` base's 4.25GB layer once downloaded at only ~70KB/s through the proxy. First used the
> **bare-metal-inference pivot** (inference running on bare metal, with containerized custodian connecting to it via
> `host.docker.internal:8900`) to verify the architecture + the sidecar S3 linchpin; later, once the base image was fully
> pulled (through repeated retries) + `scipy/einops` were added (pitfall ⑥), the **committed three-container compose
> stack was brought up as-is with `docker compose --env-file .env.compose up -d`, running end-to-end, all healthy** —
> every gate has been measured and passed, nothing left pending ⏸️.

| Gate | Result | Evidence |
|---|---|---|
| torch-free (build) | ✅ | `Dockerfile.custodian`'s build-time assertion `find_spec('torch')`=None + closure `slim closure OK` |
| torch-free (runtime) | ✅ | inside the container, `import torch` → `ModuleNotFoundError` + `'torch' not in sys.modules` |
| GPU passthrough | ✅ | with `--gpus`, the 4090 is usable; `CUDA_VISIBLE_DEVICES=<4090UUID>` → device0=4090 (verified across all three configurations, see pitfall ④) |
| readiness-based routing | ✅ | custodian's `/readyz` checks `collection_exists(real)` via qdrant-client (the boolean underlying a GET to /collections/real/exists, **not** a plain HTTP 200) + an `httpx.get(inference/readyz).status_code==200` check — both downstream probes together; `depends_on service_healthy` |
| end-to-end remote | ✅ | containerized custodian (0 torch) → bare-metal inference → qdrant server, a real hit on `financial_research_zh__...`, `mode:full` |
| **the sidecar S3 linchpin** | ✅ **positive + negative proof** | 2 replicas sharing `/index`: 5×2 queries, `single_chunk_degraded=0`; the control group with an empty sidecar: 19 degraded (proving the test can actually detect this) |
| data migration | ✅ count / manually verified by sampling | `migrate_to_server.py`: 7652→7652 **count** (the script is reproducible); dense==1024/sparse/ACL payload spot-checked on 50 points as a **one-off manual verification within the session** (not scripted, see the pending three-layer verification in §7) |
| **inference image build** | ✅ | `pytorch/pytorch:2.8.0-cuda12.8` base (pulled via daocloud) + `.[gpu]` (adding scipy/einops) → `custodian-inference:0.3.0` (12.7GB) built successfully; `ready:true full_dim:4096` |
| **bringing up the committed compose stack as-is** | ✅ **closed the blind spot** | `docker compose --env-file .env.compose up -d` — all three containers healthy; inside the inference container, `get_device_name(0)=RTX 4090 count=1` (a real-container verification of pitfall ④); custodian's `/readyz`=ready; **the full container chain**: retrieve returns `status:ok returned_n:3 mode:full` (full_section/section_window/deduped) |

**Real bugs caught and fixed by measurement (unplanned, found by reading the code + verifying)**:
1. ~~**Pitfall ①**~~ **(corrected by the adversarial review: this was my own misjudgment, not a real bug)**: I once
   claimed `python -m custodian serve` failed because there was "no `src/custodian/__main__.py`" — actually a glob
   search had searched the wrong directory (the cwd rather than the custodian repo). `__main__.py` **does in fact
   exist**, and `python -m custodian serve` worked fine all along. Using `["custodian","serve"]` (the console script) as
   the CMD is an equally valid, more robust choice (not dependent on CWD/PYTHONPATH), but it's **not** a bug fix.
2. **Pitfall ②, `Dockerfile.inference`**: `COPY requirements-gpu.txt` (that file doesn't exist) + `python3.12`
   (not available from ubuntu22.04's base sources) → both would fail the build → switched to `.[gpu]` + the pytorch base
   image.
3. **Pitfall ③, downloading torch**: pulling cu128 from pytorch.org through the proxy was only 368KB/s (5GB≈4h) →
   switched to using the official `pytorch/pytorch` image (with torch preinstalled) as the base (pulled via a daocloud
   mirror), avoiding the download entirely. Pulling from docker hub itself also needs the daocloud mirror accelerator
   (without an accelerator, in China, it gets an EOF error).
4. **Pitfall ④, GPU card selection (the nastiest one)**: `dense.py:38` hardcodes `get_device_name(0)` and asserts it
   contains "4090". The original plan of `CUDA_DEVICE_ORDER=PCI_BUS_ID` results in **device0=5070** → the assertion
   always fails → inference never becomes ready. **Verified all three configurations**: default FASTEST_FIRST → 4090 ✅;
   PCI_BUS_ID → 5070 ✗; `CUDA_VISIBLE_DEVICES=<4090UUID>` → 4090 ✅ (and count=1). Fix = use `CUDA_VISIBLE_DEVICES` (the
   locking method the app's own error message recommends). **Also**: under WSL2, `device_ids`/`NVIDIA_VISIBLE_DEVICES=UUID`
   **don't provide hard isolation** (measured — the container could still see both cards); real isolation relies on
   `CUDA_VISIBLE_DEVICES`.
5. **Pitfall ⑤, the 0.0.0.0 auth guard**: `service.py:88` fails closed — binding to a non-loopback address
   **requires keys mode** (the legacy single-key mode is rejected). Inside a container, binding to 0.0.0.0 is required
   (Docker port forwarding needs it) → compose must configure `CUSTODIAN_KEYS_FILE`, or it crash-loops. Already added
   (review finding sec-4: the plaintext key table is mounted separately as `:ro` at `/run/custodian/keys.json`, **not**
   put in the RW data volume).
6. **Pitfall ⑥, model loading missing scipy/einops (only surfaced by a real container)**: the `[gpu]` extra originally
   only had transformers/qwen-vl-utils/torch/accelerate, but loading Qwen3-VL via transformers has a transitive
   dependency on **scipy**, and rearrange needs **einops** — the pytorch base image doesn't include them, and bare-metal
   custodian happened to have them already, so this was masked the whole time. When the real three-container stack came
   up, inference reported `error: ModuleNotFoundError: No module named 'scipy'` → unhealthy → custodian's dependency
   wasn't satisfied and it never started. Now added to `[gpu]`. **This is exactly the value confirmed by the "committed
   compose has never been run as-is" blind spot the review flagged (judged PLAUSIBLE)**: without actually running the
   three containers, scipy/einops would never have been found.

**Confirmed findings fixed by the Phase E adversarial review (2026-07-06)**:
- **sec-2 (high)**: `/readyz`'s exception handler returned `str(e)`, leaking internal network addresses
  (`qdrant:6333`/`inference:8900`) to an unauthenticated probe → changed to log server-side only, with no detail in the
  response body (+ a test asserting the 503 body contains no internal addresses).
- **compose-B**: `/readyz` discarded the return value of `collection_exists` → an empty library (a new server not yet
  migrated) would also report ready → added a `collection_missing` 503 (+ a regression test).
- **sec-3**: `/readyz`'s check of inference used `httpx timeout=3` per stage (worst case ~9s total) > the healthcheck's
  5s → explicitly changed to `httpx.Timeout(1.5, connect=1.0)`.
- **migrate-F2**: moved the source library's exclusive-lock check to **before** creating the destination collection
  (avoiding a half-built intermediate state) + turned a lock conflict into a SystemExit with guidance.
- **migrate-F1**: changed the exit message from "vector migration complete" to "only the point count was verified;
  vectors/ACL have not been checked, §7 must be run."
- **sec-4**: moved the keys table out of the RW data volume, mounted separately as `:ro`.
- **honesty**: corrected the wording of the readiness/migration evidence in this table + added the "committed compose
  brought up as-is ⏸️" row + corrected pitfall ①.

> The temporary overrides used for the pivot (`/tmp/custodian-pivot.yml` pointing inference at the host,
> `/tmp/custodian-scale.yml` clearing ports) are not checked in; the committed compose keeps the three-container shape.
> **Note**: the committed shape (the `inference:8900` service-name DNS + the `depends_on` chain + the GPU passthrough
> block) **had never been run as-is** (see the ⏸️ row in the table above) — once the network is ready, it must be
> **measured once for real** before being treated as working, not assumed to be "just needs the build, everything else
> runs as-is."

### Phase F — Rolling out real multi-replica + a full adversarial re-review and revision of A–E (the capstone)

> Status: **shipped (2026-07-07)**. F has two parts: (1) first dispatch an 8-dimension multi-agent adversarial review
> against A–E (Find + dual-lens Verify), with 27 dual-lens-confirmed findings — catching a batch of cases where
> **the docs claimed "fixed/verified" but the code hadn't actually been changed**; (2) only once the confirmed findings
> were fixed did real multi-replica with nginx get implemented (rejecting "replicating horizontally on top of a buggy
> single replica").

#### F-1 Findings caught and fixed by the adversarial review (confirmed, deduplicated by root cause)

**P1 (the linchpins of Phase F — multi-replica is guaranteed to break without these fixes):**
1. **`remote.py`'s `_post_retry` has gaps in its exception coverage for disconnects** — it only catches
   `ConnectError+TimeoutException`, missing `ReadError`/`RemoteProtocolError`/`WriteError`. And **the most typical
   disconnect from `docker kill`/restarting inference is exactly `RemoteProtocolError` (reusing a dead keep-alive
   connection) / `ReadError` (the peer sends RST)**; missing these means retries get bypassed and they're swallowed into
   a non-retryable `backend_unavailable` — breaking the "kill goes unnoticed" chain.
   Changed to `except httpx.TransportError` (covers the entire transport layer + all timeouts, and **excludes**
   4xx's `HTTPStatusError`, so 4xx still bubbles up immediately).
   Guard tests: `test_remote.py::test_retry_read_error_retried` / `test_retry_remote_protocol_error_then_exhaust`.
2. **Probes sharing the 40-thread pool with business traffic → starvation causes false unhealthy reports** —
   inference's and custodian's `/readyz`/`/healthz` are both sync def, and share anyio's default 40-thread pool with the
   GPU forward pass (`/embed`/`/rerank`), the LLM (`/v1/ask`), and `/v1/retrieve` (which can take up to ~361s when
   inference is hanging). Under high load / downstream failure, probes queue and starve in the pool →
   the healthcheck times out and reports unhealthy (the higher the load, the falser the readiness signal)
   → nginx removes healthy replicas that are actually working fine = a global outage. Fix: **probes changed to
   async def** (pure in-memory reads, so they don't go into the thread pool); custodian's `/readyz`'s blocking qdrant
   call is offloaded to a **dedicated `_PROBE_LIMITER` (8 threads, isolated from business traffic)**, and inference's
   liveness check uses `httpx.AsyncClient`. Guard tests: `test_review_fixes.py::test_readyz_inference_*` (3 branches).
3. **`mcp_stdio._config` failed to thread through `inference_url`** — the same "missed a call site" pitfall from
   Phase D recurred on the remote switch: the agentic exit point had `CUSTODIAN_INFERENCE_URL` configured but it was
   silently dropped → in the slim (torch-free) environment, the first query crashed on `import torch`, which got
   swallowed into `backend_unavailable`. Added the missing threading of `inference_url` + model paths/gpu_name.
   Guard test: `test_mcp_stdio_config_passes_inference_url`.
4. **compose's `--scale custodian=2` conflicts with the fixed published port `127.0.0.1:8787`** — the second replica
   is guaranteed to fail to start due to a port conflict, so the S3 sidecar verification had only passed falsely with a
   single replica. **Directly solved by Phase F's nginx front end** (custodian changed to `expose` instead of `publish`,
   with the entry point going through nginx on `:8080`).
5. **`start_period=300s` left zero margin + a comment that had the causality backwards** — the old comment claimed
   "setting it too short is harmless," when actually **after the period elapses, 3 consecutive probe failures mark it
   unhealthy, and `depends_on:service_healthy` aborts the entire `up`, so custodian never starts**. Changed to 600s and
   corrected the comment.

**P2:** `/rerank` on the server still ignored the client's `instruction` (SCALE_OUT §5-E once falsely claimed "fixed
P2-3 along the way," but the code was never changed) → the server now consumes `q.instruction`, with
`Reranker.score(instruction=)` falling back to config when it's None; `indexer.run_index` failed to thread through
`sidecar_dir`/model paths → the write path and read path ended up using different locations, triggering the S3 silent
degradation → when `--dest` isn't explicitly given, it now uses the same path as the config + threads through the model
paths; inference's `_warmup` treated any transient failure as a permanent error with no self-healing →
**now distinguishes a permanent config error (stays in the err state) from a transient one (bounded retries, and once
exhausted, `os._exit(1)` to let `restart` bring it back up)**; client-side read-timeout retries × no backpressure on the
server side → 3× wasted GPU work → inference added a **`BoundedSemaphore` for bounded admission, returning 503
overloaded once it's full**; migrate created the destination collection before validating the source → an empty source
library would silently pass through `/readyz` → **now validates that the source exists and is non-empty before creating
the destination**; `test_store_server` was fully skipped by default (not in CI), the P1-1 guard test was buried in a
GPU-skip file, and the `/readyz` inference-probing branch had zero test coverage → tests added + relocated.

**P3:** `_get_client` did a lockless check-then-set → added `_CLIENTS_LOCK`; `_observe` did synchronous file I/O inside
the event loop (a shared-volume stall would freeze the whole loop) → **switched to a single background writer thread +
a bounded queue** (`obs.py`); `Dockerfile.custodian` had no pip mirror configured → unified with inference via
`ARG PIP_INDEX_URL`; inference's 4 failure-mode fields had no `CUSTODIAN_*` env equivalents → added the 4 fields to
CustodianConfig + threaded them through the engine; `Reranker._load` got the `_gpu_error` cache added (symmetric with
Dense).

#### F-2 Real multi-replica (nginx + unnoticed kill + graceful shutdown)

**Changes:**
- **`deploy/nginx.conf`** (new): `upstream custodian_pool { server custodian:8787 resolve; }` +
  `resolver 127.0.0.11` — open-source nginx≥1.27.3's `server ... resolve` round-robins across **all** replica IPs
  returned by Docker's DNS, so scaling replicas up/down with `--scale` takes effect within `valid=10s` (avoiding the
  classic pitfall of "resolved once at startup and then frozen"). **`proxy_next_upstream error timeout http_50x
  non_idempotent`** = a killed replica goes unnoticed: hitting a dead replica fails to connect
  (`connect_timeout 2s` fails fast) → the request is resent to the next replica; `non_idempotent` also lets POST
  requests (retrieve/ask) retry, so even **an in-flight request on a replica that just got killed** can land on a
  healthy replica.
- **`docker-compose.yml`**: added an `nginx` service (`nginx:1.27-alpine`, the sole external entry point at
  `127.0.0.1:8080:80`); custodian changed to **`expose: 8787` instead of publishing a host port** (so `--scale`
  causes no conflicts); custodian/inference both got **`stop_grace_period: 30s`**.
- **Graceful shutdown**: `cli.py`'s uvicorn set to `timeout_graceful_shutdown=25` (< the 30s grace period → a clean
  drain before docker's SIGKILL); lifespan shutdown flushes the reqlog queue (so the last batch of logs isn't lost).
- **Backpressure/self-healing**: see F-1's P2 (inference's `BoundedSemaphore` + warmup's `os._exit`).

**How to run it:**
```bash
# ⚠ Must be run from inside WSL (the bind mount uses native /home/<you> paths) with Docker Desktop's WSL Integration
#   enabled for the Ubuntu distro, otherwise /home/<you>/models fails to resolve and inference warmup reports
#   ModuleNotFoundError (see the environment prerequisite note below).
docker compose --env-file .env.compose up -d --build --scale custodian=3   # 3 replicas + nginx
# The entry point is nginx:8080 (custodian replicas don't publish a host port)
curl -s -XPOST localhost:8080/v1/retrieve -H "X-API-Key: <key>" -d '{"query":"..."}'   # round-robins across 3 replicas
# Kill unnoticed: keep hitting 8080 continuously while killing a replica
docker kill $(docker compose ps -q custodian | head -1)
```

**F-2 end-to-end measurement (2026-07-07, 3 replicas + nginx, local Docker Desktop + WSL2 + 4090):**

| Gate | Result | Evidence |
|---|---|---|
| `--scale custodian=3` with no port conflict | ✅ | 3 replicas each `expose 8787` with no publish, all healthy; the entry point is only nginx at `127.0.0.1:8080` |
| nginx round-robin distribution | ✅ | 18 requests through :8080 → the three replicas' requests.jsonl show **6/6/7** (nearly even); all `status:ok` real hits (thanks to the shared S3 sidecar) |
| **`docker kill` goes unnoticed** | ✅ **50/50** | 50 requests sent continuously to :8080, with `docker kill custodian-2` (Exited 137) mid-stream → **50/50 all 200**, 0 failures; after the kill window, traffic redistributed to the surviving 2 replicas (34/34) |
| end-to-end real path | ✅ | nginx → custodian (0 torch) → inference (remote) + qdrant (server), `returned_n mode:full` a real hit |
| nginx's own healthcheck | ✅ (fixed one issue) | The first version's `wget localhost` reported a false unhealthy (a read-only config mount meant the entrypoint script couldn't append an ipv6 listen directive, so the container listened only on localhost→::1→refused); switched to `127.0.0.1` and it became healthy |

**Real bugs caught and fixed by measurement (unplanned):**
- **nginx healthcheck false unhealthy**: the config was mounted `:ro` → the entrypoint script
  `10-listen-on-ipv6-by-default.sh` couldn't append `listen [::]:80` → nginx only listened on ipv4 → inside the
  container, `wget http://localhost` (which resolves ipv6's `::1` first) got connection refused. The actual service was
  fine (a `curl :8080` from the host worked), a pure probe false-positive. Fixed the healthcheck to pin ipv4 with
  `127.0.0.1`. **Yet another case of "a probe falsely reporting a healthy service as unhealthy,"** the same category as
  the probe starvation issue in F-1's P1.
- **`restart: unless-stopped` does not automatically bring back a replica killed with `docker kill`** (measured:
  restarts=0/60s): Docker treats a manual `docker kill` as "intentionally stopped," and `unless-stopped` only
  self-heals from **a genuine crash** (the process exiting abnormally on its own). Honestly corrected OPERATIONS'
  wording about "automatic recovery": going unnoticed on a kill relies on nginx failover (zero client-side awareness),
  while a killed replica needs `up --scale` to bring it back; to get "self-heals on any exit," use `restart: always`.

> **Environment prerequisite (hit this repeatedly during a whole round of debugging)**: compose's bind mounts source
> from native WSL ext4 paths (`/home/<you>/models`, etc.). **compose must be run from inside a WSL terminal**, and
> Docker Desktop must have **WSL Integration enabled** for the Ubuntu distro (Settings→Resources→WSL Integration).
> Otherwise, running from Windows makes the engine resolve `/home/<you>/...` against the docker-desktop distro (which is
> empty) → the container can see some files but is missing the `scripts/` subdirectory → inference's warmup reports
> `ModuleNotFoundError: No module named 'qwen3_vl_embedding'` → `depends_on:service_healthy` hangs the entire `up`.
> Diagnostic signature: on the WSL host, `ls ~/models/.../scripts` clearly has files, but the container is missing them —
> a mount-resolution problem, not a missing model.

#### F-3 Throughput boundaries (must be spelled out clearly: scale=N ≠ N× throughput)

What multi-replica **can and cannot actually scale** — three ceilings, in order of dominance:

1. **The GPU forward pass is serial (the hardest ceiling)**: there's only one inference container, with
   `state.gpu_lock` serializing forward passes on a single GPU + `BoundedSemaphore(16)` for backpressure.
   **No matter how many custodian replicas there are, dense/rerank forward passes all queue on the same lane.**
   The ceiling on query QPS = the single GPU's forward-pass throughput, and doesn't rise with `--scale custodian`.
   This is also the only legitimate trigger for "switching to vLLM" (continuous batching, §3.4).
2. **In-process retrieval state (mostly eliminated in F)**: `Store._lock` is concurrency-safe under server-mode
   Qdrant and can be loosened (left for Q3); `Dense._query_cache`/`_cache_lock`, `_reranker_lock` are all fine-grained
   locks within a replica, not shared across replicas; `retrieve.py`'s query-scoped `cache={}` is a **local variable
   created fresh on every call** (not shared). So the retrieval path has no shared mutable state **between replicas** →
   **multi-replica genuinely scales the concurrency of the non-GPU portions** (ACL filtering, RRF fusion, sidecar reads,
   small-to-big assembly, JSON encoding/decoding).
3. **Cross-replica state (a declared degradation under nginx round-robin)**: `SessionRegistry` (dedup via
   X-Custodian-Session) and `Stats` (`/v1/stats`) are **in-process** state → under nginx round-robin, dedup
   effectiveness drops to roughly 1/N, and stats only reflect a single replica. **This is an intentionally accepted
   degradation** (dedup is an opt-in convenience for curl/agents — losing it doesn't affect correctness; stats are
   per-replica self-observation). Getting this precise would need session stickiness (`ip_hash`) or shared storage
   (Redis) — to be done once scale demands it; not introduced for now. **The sidecar's shared volume (the S3 linchpin)
   is the only cross-replica state that must stay consistent**, guaranteed by every replica mounting the same `/index`
   volume.

**Conclusion**: `--scale custodian=N` scales **the application layer's non-GPU concurrency** (retrieval orchestration,
ACL, assembly) + **crash isolation/rolling upgrades** (killing one goes unnoticed by the rest), **not** GPU
forward-pass throughput (which is pinned by the single-GPU serial path). The QPS ceiling is set by inference, not by
the number of custodian replicas.

**Open (to be done once scale demands it)**: session stickiness/shared dedup; **switching inference to vLLM (once GPU
queuing becomes noticeable) — the plan is already drafted, see [VLLM_PLAN.md](VLLM_PLAN.md); Plan B coexists +
go/no-go probes are in place, and it's only promoted to Plan A once the equivalence/throughput gates pass**; adding
`INFERENCE_API_KEY` to inference (currently sufficient via internal-network `custodian-net` isolation + not publishing
the 8900 host port); K8s (nginx+compose is currently the learning vehicle).

---

## 6. Equivalence & Security Go/No-Go Gates (must pass to proceed)

| Gate | Phase | Criterion | Consequence of failure |
|---|---|---|---|
| **Vector equivalence, E1** | B | `np.allclose(local, remote, atol=1e-6)` | Deviation somewhere in the numpy/torch path (check bf16↔fp32 first) |
| **Mixed build/query, E2** | B | Building locally and querying remotely gives the same top-10 as fully local | **Does not go to production** — silent data corruption |
| **Server-mode ACL unauthorized access (new)** | D | A parameterized fixture pointing at a real server, all 5 fail-closed assertions pass | **Doesn't migrate to server mode** — a security regression, unauthorized data leakage (rerunning the `:memory:` version would give a false green) |
| **torch-free** | C/E | In the slim environment, `import custodian.engine` pulls in no torch; `import torch` fails in the custodian container | The slim image's goal hasn't been met |

---

## 7. Risk Table

| Severity | Risk | Mitigation |
|---|---|---|
| Highest | **E2 mixed build/query hasn't been validated end-to-end**: E1's element-wise equivalence **doesn't imply** an unchanged top-k under hybrid RRF+HNSW | Must measure E2 before production (§5-B); E1/E3/bf16 already measured + guarded in CI |
| High | dense has no retry (currently): any jitter/warmup/restart → queries get swallowed into a non-retryable `backend_unavailable`, transient failures not absorbed | Phase A, fixed first (the actual fix landed in toolcore) |
| High | Embedded Qdrant's file lock: without migrating to server mode, multi-replica simply cannot start | Phase D, the real go/no-go for multi-replica |
| Medium | Server-mode fusion filter ACL semantics change → unauthorized data leakage | Phase D **added new** server-mode unauthorized-access tests (not a rerun of the embedded version) |
| Medium | `dense_dim`/`qdrant_url` have no cross-replica validation, or get dropped when threading through → silent misalignment/lost config | D2's assertion + explicit threading through `engine.py` + writing the model fingerprint into the sidecar during library building |
| Medium | custodian-slim being torch-free hasn't been validated (the entire import closure) | Phase C's I2 closure assertion |
| Medium | Multi-replica still carries the in-process big lock, so scale≠linear throughput | §5-F spells out the two ceilings (GPU serialization + the big lock) clearly; loosening the lock requires auditing shared mutable state |
| Low | The image cross-machine path TODO | Only used for library building, which happens on the same machine, so the blast radius is small; the constraint is documented clearly |

---

## 8. Decisions Made & Open Questions

**Decided** (2026-07):
- **Scope**: go all the way to real multi-replica (the full A→F set, including the Qdrant server migration +
  containerization)
- **Rollout approach**: propose a per-file diff first, then make the change once confirmed
- **Building vs. querying**: library building goes local (bypassing the per-chunk HTTP tax) / querying goes remote (D4)
- **Inheritance**: keep the inheritance + lock invariants with tests, don't refactor it away (I1)
- **Authentication**: rely on internal-network/loopback network isolation for now; `INFERENCE_API_KEY` is left as
  optional for Phase F

**Open** (to be decided when they come up):
- Whether a single docker instance of the Qdrant server is enough, or whether it needs persistent orchestration
  (single instance for now, revisit at scale)
- nginx vs. going straight to a K8s Service (nginx + compose for local learning now, K8s as a follow-up)
- Whether library building should eventually go remote too (defaults to local; if operational consistency is needed
  later, batch it in `embed.py`)
- Multiple ports for `custodian serve`: using the `CUSTODIAN_PORT` env var for now (already supported); whether to add
  a `--port` flag is still undecided

---

## 9. Documents to Backfill After Rollout

- ✅ [DESIGN.md](DESIGN.md) §1, out of scope: remove "horizontal scaling / multi-replica"; D1 **corrects "just swap the
  url"** to "swap the url on the configuration surface, plus a three-way branch in the store + data migration + a
  server-mode ACL re-test"; the test count has been consolidated to a single source of truth, [TESTING.md §1](TESTING.md)
- [ROADMAP.md](ROADMAP.md) v2 direction: move "Qdrant server / multi-replica" from candidate to delivered; also correct
  "just swap the url"
- [OPERATIONS.md](OPERATIONS.md): add inference-service operations (startup/warmup/failure degradation/retry semantics),
  multi-replica operations (rolling updates/single-replica failures/the big lock's boundaries)
- [README.md](../README.md): redraw the architecture diagram as "application layer with no GPU, multiple replicas + an
  independent inference service"
- [TESTING.md](TESTING.md): add the remote test matrix + the equivalence gates + the server-mode ACL tests

---

## Appendix: Provenance and Revisions

- **v1**: produced by a multi-agent process (2 reconnaissance agents + 3 six-dimension design agents + 1 adversarial
  review + 1 synthesis), picking apart the already-shipped skeleton item by item (P0-1…P3).
- **v2** (this document): revised through a four-way adversarial review (accuracy/completeness dependencies/executability/
  consistency) + editorial revision, incorporating 8 MUST-FIX items + 13 SHOULD-FIX items. **The most important
  correction**: v1 placed the fix for P0-1 in `service.py`, which was actually dead code — the exception was already
  being swallowed earlier by the `except Exception` in `toolcore.py:216/278` into `backend_unavailable`; the real fix
  location is toolcore (M1). Others: adding the grouped path (M2), trimming the D milestone down to "local
  double-loading proves the lock, plus the VRAM constraint" (M3), changing the ACL approach to new server-mode tests
  (M4), adding the threading through `engine.py` + the three-way branch in `store.py` (M5), the environment prerequisite
  (M6), aligning the test baseline with DESIGN (M7), and client lifecycle (M8).
