# 09 Scaling Evolution: From Single Machine to Multiple Replicas

> **How to read this piece**
> This covers how custodian evolved from a monolith — "single process bound to a GPU + an embedded vector store" — through six stages (A–F) into a three-tier, scalable architecture: "inference GPU ×1 / custodian slim ×N / qdrant server + nginx." At every step we discuss which structural bottleneck was being solved, which alternatives were rejected, and what the measurements actually said.
> **Interview weight: extremely high** — this piece is the primary material for system-design interviews (service decomposition, equivalence, failure modes, load balancing, backpressure, probe engineering, throughput ceilings — all high-frequency topics), and every conclusion here is backed by measurements you can reproduce in this repo.
> Prerequisite reading: [../SCALE_OUT.md](../SCALE_OUT.md) (the engineering doc; this piece won't repeat its operational details); evaluation methodology is in [07 Evaluation Methodology](07-evaluation.md).

---

## 1. Conceptual foundation: why horizontal scaling of a RAG service is not "just spin up more processes"

For any online service that wants to handle more concurrency, the textbook answer is "stateless application layer + load balancer + multiple replicas." But a typical self-hosted RAG service is a natural counterexample — it has at least three kinds of coupling that make "just duplicate the process" impossible:

1. **Compute-resource coupling**: embedding/rerank models are loaded inside the application process. An 8B model eats tens of GB of VRAM, which pins the replica count to the number of GPUs; worse, application logic (retrieval orchestration, ACL filtering, context assembly) doesn't need the GPU at all, yet it's stuck being unable to replicate right alongside the model.
2. **State coupling**: an embedded vector store (Qdrant local / Chroma / LanceDB and the like) opens with an exclusive file lock. A second process can't even open the store — this is the hard mutual exclusion that blocks multiple replicas, and it bites earlier than the GPU constraint does.
3. **Lifecycle coupling**: model loading needs minutes of warm-up; if one component crashes, the whole process goes down with it; upgrades require a full stop. Rolling upgrades and crash isolation are both out of reach.

The industry's solution space roughly spans three tiers:

| Dimension | Common approach |
|---|---|
| Inference-layer separation | A custom HTTP forward-pass service / TEI (text-embeddings-inference) / vLLM (continuous batching) / Triton |
| Vector store | Embedded (single process) → server mode (Qdrant/Milvus/Weaviate server, concurrent multi-client access) |
| Application layer | Made stateless + LB (nginx/Traefik/K8s Service), with readiness probes, retries, backpressure, graceful shutdown |

Once you split things apart, a whole family of "distributed-systems taxes" appears: vectors on both sides must be **mathematically equivalent** (otherwise mixed indexing/mixed querying silently drifts out of alignment); transient failures need retries, but retries need a backpressure counterpart; probes themselves can become a source of failures; timeout budgets need to be aligned across layers. Custodian's rework paid every one of these taxes one by one, and kept measured evidence for each — that's exactly its value as interview material: not "I know we should split things up," but "I've done the split, and I know where it bleeds."

---

## 2. How Custodian does it: three hard bindings → six stages → three tiers

### 2.1 Starting point: the monolith's three hard bindings

Before the rework, a single `custodian serve` process housed all of the following at once: Qwen3-VL-Embedding-8B + Reranker-8B (on the GPU), an embedded Qdrant (exclusive file lock), and all application logic (ACL/RRF/small-to-big/generation orchestration). The three hard bindings map one-to-one onto the three kinds of coupling from the previous section. Six stages, each removing one:

| Stage | What it solves | One-liner |
|---|---|---|
| **A** | Failure modes | Carve out the inference-service skeleton + client-side bounded retry/backoff, so warm-up and rolling restarts no longer swallow queries |
| **B** | Equivalence + concurrency | Push locking down (remove the big retrieval lock); measure local↔remote vector equivalence, and along the way dig up a real bf16 bug |
| **C** | Regression net | CPU-mock test matrix in CI (including an assertion that "the entire import closure stays torch-free") |
| **D** | **The real multi-replica switch** | Embedded Qdrant → server mode (the file-lock mutual exclusion is the real hard bottleneck; splitting off the GPU is necessary but not sufficient) |
| **E** | Containerization | Three-container compose (inference GPU / custodian slim / qdrant), data migration, the readiness chain |
| **F** | Real multi-replica | First run a full adversarial review over A–E (27 confirmed findings fixed), then bring up nginx; measured `docker kill` with no visible impact at 50/50 |

The post-rework topology:

```
 client ──▶ nginx :8080 (the single external entry point, dynamic DNS round-robin)
              │
      ┌───────┴────────┐
      ▼                ▼
 custodian replicas ×N (slim, no torch, ~250MB image)
      │  /embed /rerank        │  REST :6333
      ▼                        ▼
 inference ×1 (GPU 4090,     qdrant server (the only stateful
  two 8B models resident,     piece, persistent volume,
  12.7GB image)                single source of truth)
      └── replicas share a sidecar volume (read-only at query time, the lifeline of small-to-big)
```

### 2.2 Layer 1: inference :8900 — the boundary rule for "pure GPU forward pass"

The only thing in the whole system that touches the GPU is the forward pass of the two 8B models, so that's exactly what got split into an independent FastAPI service. **No custodian business concepts appear anywhere in its endpoints** — no ACL, no Hit, no sidecar, just `texts→vectors` and `(query,docs)→scores`. MRL dimension truncation, the query LRU cache, rank write-back, and small-to-big all stay in the application layer.

- The server side builds Dense with `dense_dim=10**9` (forcing local mode, to prevent recursion), so `_mrl` never truncates and always returns the model's raw, full-dimension (4096) normalized vectors: [src/embedder/inference_server.py:73](../../src/embedder/inference_server.py#L73) (`_FULL_DIM` is defined at [:34](../../src/embedder/inference_server.py#L34)).
- The client's `RemoteDense._mrl_np` truncates dimensions to the real `dense_dim` (1024) with plain numpy + L2 renorm, and carries a lower-bound assertion to fail loud (a config mismatch is never allowed to silently feed in a short vector): [src/embedder/remote.py:160-173](../../src/embedder/remote.py#L160).
- As long as `CUSTODIAN_INFERENCE_URL` is non-empty at the application layer, the factory switches to the Remote backend and this process never loads the model: [src/embedder/remote.py:203-211](../../src/embedder/remote.py#L203); in remote mode it doesn't even need the local model files: [src/custodian/engine.py:27-33](../../src/custodian/engine.py#L27).

This is the **structural equivalence guarantee of "return full dimensions + truncate client-side"**: local and remote go through the exact same `Qwen3VLEmbedder` forward-pass code plus mathematically equivalent, deterministic truncation, so a vector built with local and queried with remote is necessarily identical — equivalence is a structural property, not a gamble you verify by testing. The GPU measurements in stage B back this up: E1 encode cosine = 1.0000000, max diff 2.98e-08; E3 rerank max diff 0.00 (`scripts/equiv_gpu.py` runs the two time-sliced — 2×8B occupies 33GB/48GB, loading both at once would OOM, and this constraint itself confirms that "the application layer has to be GPU-free").

The equivalence gate also forced out a real bug: `Dense._mrl` used to normalize on a **bf16** tensor, giving norm ≈ 1.002 once converted to fp32; meanwhile the remote numpy path runs entirely in fp32, giving norm = 1.0. Under COSINE distance, the direction is still equivalent and retrieval results show nothing abnormal — eyeballing it would never catch this. The fix: call `.float()` before truncating and normalizing: [src/embedder/dense.py:65-74](../../src/embedder/dense.py#L65); the CPU-only guard tests even include a "reverse guard" (proving that normalizing on bf16 really does give >1.001, and that deleting `.float()` turns the test red): [tests/engine/test_remote.py:260](../../tests/engine/test_remote.py#L260).

### 2.3 Layer 2: custodian slim ×N — torch-free, locks pushed down, asymmetric failure handling

**Torch removed at build time.** The core dependencies don't include torch (the `[gpu]` extra is isolated). After `pip install .` in the slim image, two build-time assertions run: `find_spec('torch')` must be `None`, and after importing `custodian.engine`+`custodian.service` and constructing the two Remote backends, `'torch' not in sys.modules` — the assertion is planted against **the entire import closure** (any top-level `import torch` anywhere in engine→Retriever→Store→qdrant_client would trip it), turning contamination from "crashes on the first real query at runtime" into "fails the build": [Dockerfile.custodian:16-27](../../Dockerfile.custodian#L16). The replica image is ~250MB vs. 12.7GB for inference — since a multi-replica setup replicates the application layer, image size directly determines how fast you can scale and do rolling upgrades.

**Locking pushed down (M1).** Stage A added retry-with-backoff to the remote client to fix availability, but the adversarial review found that the backoff's `time.sleep` happened inside the old `LockedRetriever`'s big retrieval lock — during warm-up, one query holding the lock while sleeping stalls every query in the entire replica: an availability fix created a bigger availability cascade. The fix is structural: remove the big lock and push it down to per-resource locks — `Store._lock` (the embedded, single-client store isn't thread-safe), `Dense._fwd_lock` (GPU forward pass, local only), `_load_lock` (single-flight), and `_cache_lock` (an LRU split into two segments, with the encode's HTTP call + backoff kept outside the lock): [src/custodian/engine.py:6-13](../../src/custodian/engine.py#L6), [src/embedder/dense.py:29-31](../../src/embedder/dense.py#L29), [src/embedder/dense.py:89-105](../../src/embedder/dense.py#L89). The Remote backend overrides the forward/load locks to `nullcontext` (HTTP is naturally concurrent): [src/embedder/remote.py:107-108](../../src/embedder/remote.py#L107). A benign race is also explicitly accepted: two threads missing on the same query at the same time will each compute it once, with the later write winning — refusing to wrap the whole thing back in a lock just to eliminate it.

**Asymmetric failure handling: dense fails loud / rerank degrades gracefully.** This is a deliberate design, locked in by tests:

- Dense failure: once `_post_retry` exhausts its retries it raises `InferenceUnavailable` — a plain exception class carrying an `inference_unavailable=True` class attribute as a marker: [src/embedder/errors.py:11-21](../../src/embedder/errors.py#L11). The exception bubbles up into toolcore's broad `except`, which uses `getattr(e, "inference_unavailable", False)` duck-typing to route it into a structured `inference_unavailable` (retriable: true): [src/custodian/toolcore.py:229-234](../../src/custodian/toolcore.py#L229). **It's never degraded into an empty result or pure BM25** — the query vector is the primary retrieval signal, and silently degrading means the user gets results whose quality has already collapsed without any indication of it, which is worse than an outright failure.
- Rerank failure: `retrieve`'s try/except swallows it and falls back to returning `hits[:k]` from the hybrid recall, with `rerank_degraded` flagged upstream: [src/embedder/retrieve.py:113-116](../../src/embedder/retrieve.py#L113). Rerank is an enhancement signal, so degrading it is safe.

toolcore stays stdlib-only (it never imports embedder), relying on the marker rather than type checks — this is where two design constraints, "dependency boundaries" and "error semantics," intersect.

**Completeness of the retry spectrum.** `_post_retry` treats any 5xx as a transient failure to retry (502/504 are what a gateway looks like when it's pulled a backend, and hard-coding just 503 would miss them); `except httpx.TransportError` catches the entire transport-layer exception family; 4xx goes through `raise_for_status` and bubbles up immediately without retrying: [src/embedder/remote.py:67-96](../../src/embedder/remote.py#L67). Timeouts are split into connect=3s / read=120s, and dense/reranker share a locked client cache keyed by URL: [src/embedder/remote.py:40-52](../../src/embedder/remote.py#L40). Why emphasize "completeness": the stage-F review found that the first version only caught `ConnectError+TimeoutException`, while **the most typical disconnect shapes from `docker kill` are `RemoteProtocolError` (a reused keep-alive connection that's actually dead) and `ReadError` (the peer sent an RST)** — missing those two types means the request bypasses retry and gets swallowed into a `backend_unavailable` that's never retried, breaking the "kill goes unnoticed" chain right on the client side. Guard tests: [tests/engine/test_remote.py:290](../../tests/engine/test_remote.py#L290), [:300](../../tests/engine/test_remote.py#L300).

### 2.4 Layer 3: qdrant server — the real switch for multiple replicas

**The real switch for multi-replica isn't splitting off the GPU — it's the embedded Qdrant's single-process file lock** (splitting the GPU is necessary but not sufficient). `Store.__init__` builds the client from three branches by priority — `url > :memory: > path` (`:memory:` must come before `path`; a large number of tests rely on this short-circuit): [src/embedder/store.py:26-31](../../src/embedder/store.py#L26).

The teaching point here isn't "add a field" — it's that **every production exit point actually has to use that field.** The first version of `qdrant_url` missed all three of its exit points — engine (query), indexer (index building), and mcp_stdio (agentic stdio) — the server path never actually got exercised; I once claimed "I updated engine" without actually having done it, and was caught by my own verification discipline. Once fixed, all three pass-through points have guard tests (mocking `QdrantClient`; deleting the pass-through line turns them red): [src/custodian/engine.py:34-43](../../src/custodian/engine.py#L34), [src/custodian/indexer.py:59-65](../../src/custodian/indexer.py#L59), [src/custodian/mcp_stdio.py:56-66](../../src/custodian/mcp_stdio.py#L56). The same pothole later recurred verbatim with `inference_url` (mcp_stdio missed the pass-through, so `import torch` crashed on the first query in a slim environment) — "adding a config field ≠ it taking effect; every exit point must be verified as actually consuming it" is the portable lesson this rework left behind.

**Swapping storage backends means re-testing not just the data, but the security semantics too.** The embedded QdrantLocal silently drops the top-level `query_filter`'s `should` clause under RRF fusion (found by measurement), so the ACL filter has to be pushed down into every prefetch: [src/embedder/store.py:103-123](../../src/embedder/store.py#L103). After migrating to server mode, fusion filter semantics differ, so a **new server-mode privilege-escalation test had to be added** rather than re-running the old tests hard-coded to `:memory:` (those still exercise embedded fusion and give zero coverage of server semantics — a false green). `tests/engine/test_store_server.py` has a raw probe that bypasses the exit point `acl_admits` and directly asserts that server-side RRF's `query_points` raw output contains no unauthorized points — the layer you test determines what proposition the test actually proves (that prefetch push-down genuinely works, rather than being masked by a fallback at the exit). There's also a `CUSTODIAN_REQUIRE_QDRANT_SERVER=1` flag that forbids silently skipping (otherwise CI's server being unreachable and everything skipping would also be a false green).

**The data-migration script is designed to prevent an intermediate state**: it grabs an exclusive lock on the source store first, then validates that the source collection exists and is non-empty — all before it has any side effect on the destination; collection creation reuses `Store.ensure_collection` (the 7 payload indexes are never hand-copied — miss one ACL index and server-side filtering silently degrades); scroll always uses `with_vectors=True` (the default doesn't return vectors, and missing this would migrate "points with no vectors," so retrieval breaks entirely without erroring); a count check plus uuid5 idempotency makes it re-runnable; the exit message explicitly states "only point count was verified, vectors/ACL were not" rather than overclaiming: [scripts/migrate_to_server.py:44-57](../../scripts/migrate_to_server.py#L44), [:70-71](../../scripts/migrate_to_server.py#L70), [:82-89](../../scripts/migrate_to_server.py#L82). Measured: 7652→7652, counts matched.

### 2.5 The entry point: nginx load balancing and the "kill goes unnoticed" chain

nginx is the sole external entry point (`127.0.0.1:8080`); custodian was changed to `expose: 8787` rather than publishing a host port — otherwise a second replica from `--scale` would collide on the port (the sidecar validation in stage E once passed falsely because it only ran on a single replica): [docker-compose.yml:128-129](../../docker-compose.yml#L128), [:143](../../docker-compose.yml#L143).

"Docker-kill one replica, zero client-perceived impact" isn't a single switch — it's a chain, and breaking any link voids the whole promise:

1. **Dynamic DNS round-robin**: `server custodian:8787 resolve` + `resolver 127.0.0.11 valid=10s` + `zone` — open-source nginx ≥1.27.3 round-robins across all replica IPs Docker DNS returns, and scale changes take effect within 10s (avoiding the classic pitfall of "it resolved once at startup and locked in"): [deploy/nginx.conf:14-20](../../deploy/nginx.conf#L14).
2. **Cross-replica retries**: `proxy_next_upstream error timeout http_502/503/504 non_idempotent` + `tries 3` + `connect_timeout 2s` (a dead replica fails fast) — `non_idempotent` lets POSTs (retrieve/ask) also be retried, so even **in-flight** requests on a killed replica land on a healthy one; the cost (in the worst case, `ask` can trigger the LLM twice) is an explicitly accepted trade-off: [deploy/nginx.conf:36-42](../../deploy/nginx.conf#L36).
3. **The full client exception spectrum**: the `TransportError` catch-all above (the disconnect shapes from a kill are absorbed).
4. **Graceful shutdown**: `stop_grace_period: 30s` + uvicorn's `timeout_graceful_shutdown=25` (< 30s, so it drains cleanly before SIGKILL): [src/custodian/cli.py:34-37](../../src/custodian/cli.py#L34), [docker-compose.yml:84](../../docker-compose.yml#L84).

Measured (2026-07-07, 3 replicas): 18 requests round-robin to a near-even **6/6/7**; with 50 continuous requests, a `docker kill` on one replica mid-run → **50/50 all 200, 0 failures**, with traffic redistributed across the two surviving replicas (34/34). Another counter-intuitive measured fact: `restart: unless-stopped` **does not** bring back a replica that was manually killed (Docker treats it as an intentional stop, restarts=0) — kill-goes-unnoticed relies on nginx failover, not container self-healing, and OPERATIONS' wording was honestly corrected to reflect this.

Each replica writes its logs into a subdirectory named by its own container name via `$$HOSTNAME` at runtime, to prevent JSONL interleaving across replicas: [docker-compose.yml:107](../../docker-compose.yml#L107).

### 2.6 Two mechanisms that run through everything: readiness engineering and backpressure

**Readiness engineering** — this rework accumulated three measured cases of "the probe itself becomes a source of failure":

- The probe shares FastAPI's default 40-thread pool with business traffic → under high load the probe queues up and starves → "the higher the load, the falser the readiness signal gets" → the orchestrator strips out healthy replicas that are working fine = a global outage. Fix: turn the probe endpoints into `async def` (a pure in-memory read that never enters the thread pool); custodian's `/readyz` offloads its blocking Qdrant call onto a dedicated `_PROBE_LIMITER` (8 threads): [src/custodian/service.py:40](../../src/custodian/service.py#L40), [:224-256](../../src/custodian/service.py#L224).
- The inference liveness check used the default `httpx.Timeout(3)`; the worst-case cumulative time across stages was ~9s, exceeding the healthcheck's 5s budget → it misjudges itself. Fix: an explicit `Timeout(1.5, connect=1.0)`.
- The nginx container is mounted `:ro` → the entrypoint script can't append an ipv6 listen directive → inside the container `localhost` resolves to `::1` first → false unhealthy (the real service is completely fine). Fix: pin the healthcheck to `127.0.0.1`: [docker-compose.yml:147-149](../../docker-compose.yml#L147).

The three cases combine into one portable lesson: **a probe's resource path, timeout budget, and name resolution must all be isolated from business traffic and pinned down explicitly.** Beyond that, the liveness ≠ readiness split was kept clean: `/healthz` is always ok (a config error that a restart can't fix should never restart-loop, so it never reports unhealthy for that) but it exposes `error`/`full_dim` fields; `_readiness` is a pure function, and **err takes priority over loading** (a permanently failed load must report error, not "always warming up forever"); `/readyz` and each endpoint's `_guard` share this single implementation to avoid drift across three places: [src/embedder/inference_server.py:42-52](../../src/embedder/inference_server.py#L42-L52). The warmup background thread distinguishes permanent config errors (leave `err` set, don't self-kill) from transient ones (retry with backoff; once exhausted, `os._exit(1)` so a restart brings it back): [src/embedder/inference_server.py:91-112](../../src/embedder/inference_server.py#L91). On the compose side, the comment on `start_period: 600s` also corrected a wrong intuition — "setting it too short is harmless" is false: once the grace period expires, 3 failures marks it unhealthy, and `depends_on: service_healthy` blocks the entire `up` from proceeding: [docker-compose.yml:69-73](../../docker-compose.yml#L69).

**Backpressure is the counterpart to client-side retries.** If the client retries, the server must have admission control, or retries amplify overload into a cascade: with GPU forward passes serialized and unbounded queuing, a client giving up after a read-timeout leaves the request still burning GPU in the queue + the retry re-enters the queue = 3× wasted work plus a positive-feedback loop of congestion. inference uses `BoundedSemaphore(16)` to pin the total in-flight count (executing + waiting on `gpu_lock`); once full, it fails fast with an immediate 503 overloaded; the client treats 5xx as `InferenceUnavailable` and retries with backoff, which composes naturally: [src/embedder/inference_server.py:86](../../src/embedder/inference_server.py#L86), [:131-135](../../src/embedder/inference_server.py#L131). This is a simplified form of load shedding, an order of magnitude simpler than introducing a real queue/priority scheme, and it's enough.

### 2.7 The honest conclusion: the throughput ceiling = serial GPU forward passes on a single card

What multiple replicas **actually scale, and don't scale**, breaks down into three limits, in order of dominance:

1. **The hardest limit: serialized GPU forward passes.** There is one inference container; `gpu_lock` serializes the single-card forward pass ([src/embedder/inference_server.py:146](../../src/embedder/inference_server.py#L146)) plus semaphore backpressure. No matter how many custodian replicas there are, dense/rerank forward passes queue behind the same line. Measured capacity (2026-07-04, single-machine embedded form factor, RTX 4090, 7652 chunks, see [../OPERATIONS.md](../OPERATIONS.md) §4): at concurrency 1/2/5/10 → 2.5/2.9/3.0/**3.2 req/s**, p50 goes from 408ms→3.07s, queuing is linear with zero errors; under mixed load, retrieval p50 stays at 352ms while `ask` is in flight (the LLM segment doesn't hold any lock).
2. **In-process retrieval state has essentially been eliminated**: there's no shared mutable state across replicas on the retrieval path (the query-scoped `cache={}` is a local variable), so multiple replicas **genuinely scale non-GPU-segment concurrency** — ACL filtering, RRF fusion, sidecar reads, small-to-big assembly, JSON encoding/decoding.
3. **Cross-replica state is deliberately degraded**: SessionRegistry's dedup effectiveness drops to roughly 1/N, and `/v1/stats` only reflects a single replica (opt-in convenience; losing it doesn't affect correctness). The only thing that must stay consistent across replicas is the **shared sidecar volume (read-only at query time)** — a replica getting a hit but reading an empty sidecar is a silent degradation, and both the positive and negative cases were measured: 2 replicas sharing a volume gave 10 queries with 0 degradation, vs. a control group with an empty sidecar giving 19 degradations: [docker-compose.yml:93-94](../../docker-compose.yml#L93). (The honest boundary of the mount setup: in stage E, the entire `/index` is a single RW mount, and the sidecar is only *accessed* read-only at query time; splitting the sidecar into its own `:ro` mount and logs into their own `:rw` mount for defense-in-depth was the intent for stage F, but currently only exists as a comment in the compose file [docker-compose.yml:110](../../docker-compose.yml#L110) — it isn't actually implemented. The `:ro` mount that is actually in place is the plaintext key table.)

So `--scale custodian=N` scales **non-GPU concurrency + crash isolation + rolling upgrades**, **not QPS** (which is pinned to the ~3.2 req/s range by single-card serialization). The only legitimate path to break through it is vLLM continuous batching, and the only legitimate trigger is `/embed` queue depth staying above 1 — not "vLLM is trendier."

### 2.8 vLLM: a Plan B run in parallel, and equivalence downgraded from "guaranteed" to "experimental"

The equivalence of the custom FastAPI service is structural (both sides run the exact same forward-pass code); switching to vLLM means pooling is implemented by vLLM itself, so equivalence downgrades from a structural guarantee to **a claim that must be measured**. So the plan is "don't replace it — run it in parallel first": the vLLM side presents the exact same `/embed /rerank /readyz` contract (a thin adapter wraps a chat template and forwards to `/v1/embeddings`), the application layer changes nothing, and the compose profiles switch via mutual exclusion = second-scale rollback — this is where the payoff of "the endpoint carries no business concepts" actually gets cashed in. Four gates in a strict order (G0 feasibility → G1 vector equivalence → G2 mixed-index/mixed-query → G4 throughput); only clearing a gate promotes to Plan A. Details in [../VLLM_PLAN.md](../VLLM_PLAN.md).

Measured (2026-07-07, `scripts/vllm_equiv_probe.py` + `scripts/vllm_g2_topk.py`):

- **G0 GO**: vLLM 0.22.1 loads Qwen3-VL-Embedding-8B in pooling mode and produces normalized 4096-dim vectors. The key parameter is `max_model_len=8192` (the model's `max_position_embeddings=262144` would make vLLM reserve 36GB of KV cache and OOM for sure; a single embedding forward pass doesn't need a long context): [scripts/vllm_equiv_probe.py:107-111](../../scripts/vllm_equiv_probe.py#L107).
- **G1 marginal**: cosine across 8 samples ∈ [0.99956, 0.99982], with the min of 0.99956 below the 0.9999 threshold. The prime suspect is a version difference between transformers 4.57 and 5.10; the shortest text ("a") scores lowest, consistent with "short sequences amplify per-token numerical differences." **Marginal doesn't decide anything — that's for G2**, because "element-wise equivalence doesn't imply top-k stability" (HNSW's approximation plus RRF rank fusion can amplify small differences), so the criterion has to be set at the level of real business impact (recall flips), not an intermediate metric (cosine): [scripts/vllm_equiv_probe.py:147-154](../../scripts/vllm_equiv_probe.py#L147).
- **G2 basically passes**: on the real 88-question test set against the real 7652-point library, top-1 agreement was **87/88 (98.9%)**, top-10 set agreement was 97.7%, Jaccard@10 was 0.9959, and disagreements were almost entirely rank-≥6 near-duplicates swapping order. On equivalence, it's a GO; promoting to Plan A only needs to clear the G4 throughput gate (not yet run).

The process itself is also methodology material: the probe ran three times and failed all three times, and each time the surface-level conclusion was "vLLM doesn't support Qwen3-VL pooling" — the real root causes turned out to be ① `CUDA_VISIBLE_DEVICES` was given a UUID (vLLM only takes integer indices), ② KV cache OOM (missing `max_model_len`), and ③ our own `grep -v` filtered out the real traceback. All three were configuration/tooling mistakes, none was an architecture problem. That lesson was baked into the probe script itself: on init failure, print the real error verbatim and note "this doesn't necessarily mean the architecture isn't supported": [scripts/vllm_equiv_probe.py:113-115](../../scripts/vllm_equiv_probe.py#L113).

---

## 3. Why it's designed this way: rejected alternatives

| Alternative | Why it was rejected |
|---|---|
| **Switch directly to vLLM/TEI as the inference layer** | Both have to implement their own pooling, turning equivalence from a structural guarantee into a bet you verify by testing; TEI also has no vision-language support (`encode_image` isn't possible). Decision: keep the custom FastAPI service for now, and let vLLM go through Plan B's four gates (§2.8), triggered by queue depth, not by popularity |
| **Triton** | Qwen3-VL has no off-the-shelf model support; writing a custom Python backend would amount to rewriting inference_server, which is over-engineering |
| **Server-side dimension truncation** (instead of returning full dimensions) | The server would have to know each library's `dense_dim`, conflicting with "the endpoint carries no business concepts"; with multiple libraries at multiple dimensions you'd need multiple instances. Returning full dimensions lets one inference service serve libraries of different dimensions at once, and bandwidth (a full-dimension JSON) isn't a bottleneck within the same rack |
| **Degrade a failed dense lookup to BM25/empty results** | The query vector is the primary recall signal; a silent degradation = a silent collapse in result quality, which is worse than an outright failure; changed instead to a retriable, fail-loud error (`inference_unavailable`, retriable: true). Rerank is the opposite — an enhancement signal, so degrading it is safe. The asymmetry is locked in by tests |
| **Keep the big retrieval lock and hedge with multiple replicas** | Big lock + remote backoff = the whole replica stalls during warm-up; and "horizontally replicating the wrong single replica = replicating the bug and amplifying the blast radius" — stage F did a full adversarial review of A–E first (27 confirmed findings fixed) before bringing up multiple replicas, by the same logic |
| **nginx Plus / Traefik (active health checks)** | Open-source nginx having no active health checks is a known constraint; currently covered by "passive max_fails + proxy_next_upstream" for the kill scenario; the gap around not-ready-at-runtime was confirmed by review and put in the backlog (see §4 deploy#1) — an upgrade in tooling choice needs measurement, not a gut call |
| **Have index building go through remote too** | Per-chunk HTTP during index building is pure overhead (it needs a GPU machine + exclusive semantics + no multi-replica need); decision: index building stays local, querying goes remote |
| **`restart: always` for full self-healing** | After measuring that `unless-stopped` doesn't bring back a manually killed replica, we chose to honestly correct the documentation rather than blindly switch to `always` — the party responsible for kill-goes-unnoticed is nginx failover, not container self-healing |

The measured evidence is concentrated in three places: equivalence (E1 cosine=1.0000000 / bf16 norm 1.002→1.000 / G2 87/88 on the 88-question set — note the 88-question basis can't be mixed with the historical 72-question one, see [07 Evaluation Methodology](07-evaluation.md)); availability (kill-goes-unnoticed 50/50, round-robin 6/6/7); capacity (single-machine form factor caps at ~3.2 req/s, p50 3.07s at concurrency 10, linear queuing, zero errors).

---

## 4. War-story retrospective: issues confirmed by adversarial review

Before writing this set of learning docs, another deep pass plus adversarial verification round was done over the deployment subsystem (every suspected issue was first assigned an independent verifier who tried to refute it). Confirmed findings were routed by discipline: behavior-neutral robustness fixes landed immediately; anything needing a compose/GPU/load-test environment to verify was filed and deferred. **"Confirmed but can't be fixed right now" is itself an engineering judgment** — each of the five deferred items below is off-the-shelf material for a system-design interview.

### 4.1 Fixes already landed in this round (excerpted, only ones relevant to this piece)

- **Remote handshake validation**: if the server swaps models, the vector space silently drifts out of alignment (previously only a `full_dim` lower-bound assertion existed, which can't catch a same-dimension-but-different-semantic-space situation). Fix: a one-time `GET /healthz` before the first query, comparing `model_dense` against the client's configured model name and checking `full_dim ≥ dense_dim`; mismatch fails loud; warming-up/unreachable doesn't block (transient cases are left to the retry chain); single-flight: [src/embedder/remote.py:115-144](../../src/embedder/remote.py#L115).
- **`RemoteReranker.score` signature alignment**: the base class's `score(query, docs_text, instruction=None)` had been narrowed by the subclass to two parameters — a latent LSP violation (passing `instruction=` per the base class's contract would raise a TypeError), which happens to be the client-side resurrection of the old "rerank instruction dual-source footgun." Fix: add back the parameter, with `None` falling back to config, matching old behavior byte for byte: [src/embedder/remote.py:190-195](../../src/embedder/remote.py#L190).
- **`RequestLog.flush(timeout)` now actually times out**: the parameter was accepted but ignored, and `q.join()` waited unbounded — when shared-volume I/O hangs, graceful shutdown would hang out the full grace period and get SIGKILLed (exit 137 hiding the real cause). Fix: a condition variable with a polled deadline; give up with a warning on timeout; shutdown passes `timeout=5.0`: [src/custodian/obs.py:92](../../src/custodian/obs.py#L92).

Verification basis: before the fixes, the `pytest tests -q` baseline was 224 passed / 4 skipped; after, 259 passed / 5 skipped (the only new skip is a server-gated test; spinning up a real Qdrant server temporarily gave 264 passed / 0 skipped).

### 4.2 Five deferred production-readiness backlog items (deferred deploy#0–3, #6)

**deploy#0 — timeout budgets are disjointed across layers (medium).** nginx's `proxy_read_timeout 130s` ([deploy/nginx.conf:41](../../deploy/nginx.conf#L41)) only aligns with a single client read attempt (120s), missing the ×3 retry chain: custodian's worst-case retry chain runs to roughly **3×120s + backoff ≈ 361s**. When the inference forward pass hangs, the client gets a 504 at 130s, but uvicorn has no cancellation propagation into sync thread-pool tasks — the abandoned request keeps burning a thread and occupies one of inference's 16 in-flight slots, wasting ~231s of work that nobody will ever receive. Sketch of the fix: give `_post_retry` an overall wall-clock deadline (derived from attempts×read+backoff, clamped to under nginx's read timeout), with both sides' budgets written as mutually referencing equations. Deferred reason: needs WSL compose measurement to align. **Interview point: timeout budgets must be strictly decreasing from the outside in (entry point > total client retry chain > a single downstream call), and without cancellation propagation, "the client timed out" does not mean "the server stopped wasting work."**

**deploy#1 — "readyz drives traffic away" only holds true during startup (medium).** A not-ready replica at runtime still gets routed to: Docker DNS returns IPs for running containers, not filtered by health state; open-source nginx has no active health checks; and more subtly, toolcore wraps `inference_unavailable`/`backend_unavailable` as **HTTP 200** structured errors — nginx's `proxy_next_upstream http_5xx` is completely blind to it, so passive removal (max_fails) never triggers. When a replica's local downstream connection is broken (but the process is alive), it keeps eating 1/N of the traffic and returning "retriable" errors. Sketch of the fix: have the HTTP service layer return a 503 for downstream-unavailable-class errors (body unchanged, the agent contract preserved), so nginx's existing config gets failover + passive removal for free; the MCP stdio path is left untouched (there's no LB there, and the structured 200 is still the right contract). Deferred reason: the approach needs measurement to choose (503 adaptation vs. switching to Traefik) — since the downstream is shared, if every replica is equally sick, retrying is pure waste. **Interview point: when the LB's removal signal surface (HTTP status code) conflicts with the application error contract (structured 200), who yields to whom; the capability boundary between active and passive health checks.**

**deploy#2 — readyz isn't deep enough: exists ≠ non-empty (medium).** `/readyz` only checks `collection_exists` ([src/custodian/service.py:242-243](../../src/custodian/service.py#L242)); the migration script creates the collection first and fills points second, and if it's killed between the two steps, it leaves an "exists but 0 points/half a library" state — readyz is all green, nginx routes real traffic to an empty library, and every query returns `status:empty` (HTTP 200, and metrics even count `empty` as success), with zero alerts — this is exactly the "silent data corruption" category this project defines for itself. Two-tier fix: a lightweight count>0 spot check (60s TTL cache, to avoid N replicas hammering qdrant with high-frequency probes); the real fix is to rewrite migration to use a temporary collection + atomic alias switch, eliminating both the "empty library" and "half library" visible windows at once. Deferred reason: needs measurement of a migration-interruption scenario. **Interview point: how "deep" readiness should probe is a gradient (process alive < dependency reachable < data ready); a migration's atomicity should come from an alias switch, not from "verify only after it's finished."**

**deploy#3 — retries have no jitter (low).** The backoff is deterministic — `backoff*2^attempt` (0.5/1.0s): [src/embedder/remote.py:95](../../src/embedder/remote.py#L95) — so when inference has a brief outage or gets saturated, N replicas' clients resend in lockstep, repeatedly slamming the same semaphore, defeating the whole point of backoff jitter (spreading load out). Worth noting: the adversarial review **refuted the original report's amplification factor** — nginx's upstream is custodian, not inference, and since inference failures get converted to a 200 by toolcore, nginx's retry branch never triggers — so the magnitude is N×3, not N×3×3; and rejected requests don't occupy an in-flight slot either, so the cost to the server is near zero. Hence rated low. One-line fix: multiply the backoff by `random.uniform(0.5, 1.5)`, along with updating the test that pins the deterministic sequence ([tests/engine/test_remote.py:229](../../tests/engine/test_remote.py#L229)). Deferred reason: the benefit needs load-test verification. **Interview point: thundering herd and full jitter; also that "the review report itself also needs to be reviewed" — get one amplification-factor multiplier wrong and priority is off by a full grade.**

**deploy#6 — a runtime health blind spot for a hung GPU (medium).** `_readiness` is determined solely by `state.ready/err`, which warmup writes; a runtime CUDA hang (the process alive but stuck holding `gpu_lock`, never returning) doesn't change any of that state — `/readyz` stays green forever, custodian's own probe of it stays green, the healthcheck stays green, and `restart` never triggers. Requests each burn through the 120s read-timeout one by one; once the hung forward pass permanently fills all 16 in-flight slots, the semaphore's fallback makes subsequent requests fail fast with 503 — the service is effectively dead with no automatic recovery. Sketch of the fix: have the forward pass record "how long the lock has been held + timestamp of the last success," and once that exceeds a threshold, have `_readiness` switch to `stuck` and return 503; pair it with a watchdog thread that, past a higher threshold, calls `os._exit(1)` so a restart can bring it back — **this would be the only genuine automatic-recovery channel in this whole stack** (a readyz 503 can only pull traffic; a compose-level unhealthy doesn't trigger a restart). Deferred reason: needs fault injection to verify. **Interview point: liveness's semantics should cover "the process is alive but will never produce output again" (deadlock/hang); "the probe is green" and "it's actually working" are two separate claims.**

Together, the five form a ready-made "production-readiness gap list" — timeout alignment, removal signaling, readiness depth, retry storms, hang detection — when asked in an interview "what's still missing before this goes to production," being able to name the root cause, trigger condition, fix, and why it's not fixed yet for each one is an order of magnitude more credible than saying "it's all done."

---

## 5. How to talk about it in an interview

### 30-second version (elevator pitch)

> I rebuilt a single-process RAG service into a three-tier scalable architecture: the only piece touching the GPU — the model forward pass — was split into an independent inference service; the application layer was stripped of torch and turned into a stateless 250MB image you can scale to N; and the embedded vector store was migrated to server mode — the real switch for multiple replicas turns out to be the latter's single-process file lock, not the GPU. Equivalence is made a structural guarantee via "server returns full dimensions + client truncates," so vectors built locally and queried remotely are mathematically identical. nginx does dynamic DNS round-robin plus cross-replica retries; measured `docker kill` on one replica with 50 sustained requests gave zero failures. There's also an honest conclusion: multiple replicas scale non-GPU concurrency, crash isolation, and rolling upgrades — the throughput ceiling stays pinned at ~3.2 req/s by single-card serialization. The path past it is vLLM continuous batching; I built an equivalence probe for it, and the trigger condition is queue depth, not technology fashion.

### 3-minute version (structured walkthrough)

1. **Problem definition** (20s): the monolith has three hard bindings — the GPU model loaded in-process, the embedded Qdrant's exclusive file lock, and application logic tagging along for the ride. Multi-replica, rolling upgrades, and crash isolation all require untangling all three.
2. **Locating the bottleneck** (30s): the counter-intuitive part — the hard mutual exclusion for multi-replica is the embedded vector store's **single-process file lock**, and splitting off the GPU is necessary but not sufficient. So the skeleton of the six-stage plan is: fix failure modes first (retry/backoff), then lock down equivalence (a go/no-go gate), then migrate to Qdrant server (the real switch), and finally containerize + add nginx. The order is rigid: horizontally replicating the wrong single replica = replicating the bug.
3. **Two core designs** (60s): ① draw the boundary at "pure GPU forward pass vs. business logic" — the inference service's endpoints carry zero business concepts, pure enough that it could later be swapped for vLLM painlessly, and that payoff was later cashed in directly in vLLM Plan B (zero application-layer changes on swap). ② make equivalence structural rather than something you verify: the server returns full dimensions, the client truncates with numpy + renorm, and both sides run the same forward-pass code. The equivalence gate also forced out a real bug — normalizing on bf16 gives norm ≈ 1.002, which is invisible to the eye under COSINE since direction is still equivalent; after the fix, E1 cosine = 1.0000000.
4. **The availability chain** (40s): kill-goes-unnoticed is a chain — nginx's dynamic DNS (aware of a scale change within 10s), `proxy_next_upstream non_idempotent` (in-flight POSTs also retried), the client's full `TransportError` spectrum retry (the review caught the two most typical kill disconnect shapes, `RemoteProtocolError`/`ReadError`, that had been missed), and graceful shutdown with a 25s < 30s grace period. Measured: kill one of 3 replicas, 50/50 all 200. Its counterpart design: if the client retries, the server needs backpressure (a full `BoundedSemaphore` returns 503), or retries amplify overload into a cascade.
5. **Honest boundaries** (30s): the throughput ceiling = serialized single-card forward passes at ~3.2 req/s, and `--scale` doesn't change that; what it scales is non-GPU concurrency + crash isolation. Breaking through it relies on vLLM continuous batching, which I probed with four go/no-go gates: G1's cosine of 0.99956 was marginal so I didn't decide off it, and handed it to G2 to compare top-k against a real 88-question library (87/88 agreement) — the criterion is set at the level of business impact, not an intermediate metric. There are also five confirmed production-readiness backlog items (timeout alignment / removal blind spot / readiness depth / retry jitter / GPU hang detection), each with a root cause and a fix sketch.

---

## 6. Anticipated follow-up questions

**Q1: Why not just use vLLM/TEI from the start instead of building your own inference service?**
Key point: where equivalence lives. Building it yourself = both sides run the exact same official forward-pass code, so equivalence is a structural property; vLLM/TEI implement pooling themselves, turning equivalence into a bet you have to measure (and TEI also has no vision-language support yet). Also, continuous batching offers no benefit on a single 4090. Keywords: structural equivalence vs. measured equivalence, the migration trigger condition (queue depth sustained > 1), the contract has no business concepts so it's swappable later. Data point: G1 measured vLLM's min cosine at 0.99956 — even between two official transformers versions there's a small drift, which supports "pooling equivalence isn't free."

**Q2: Why doesn't multiple replicas improve QPS? Then what's the point of scaling it?**
Key point: three limits — serialized GPU forward passes is the hardest limit (~3.2 req/s on a single card, via `gpu_lock` + semaphore); no matter how many replicas, the forward pass queues behind the same line. There's no shared mutable retrieval state across replicas, so what it scales is non-GPU-segment concurrency (ACL/RRF/assembly/I/O) plus crash isolation plus rolling upgrades. Counter-intuitive but measured: throughput with 1 replica and 3 replicas is nearly identical. Be sure to also mention the state that's deliberately degraded across replicas (session dedup drops to ~1/N, stats is single-replica) and the one piece of state that must stay consistent (the shared sidecar volume, read-only at query time).

**Q3: How do you guarantee a library built locally doesn't drift when queried remotely?**
Key point: return full dimensions + truncate client-side + share the same forward pass; `_mrl_np`'s lower-bound assertion guards against config mismatches; the first-query handshake validates model identity (a server-side model swap = same dimension but a different semantic space, caught via the `model_dense` fingerprint); the bf16→fp32 normalize bug (norm 1.002) shows "mathematical equivalence" has to be pinned down at the dtype level; E1/E3 measurements plus CPU guard tests. Advanced point: element-wise equivalence doesn't imply top-k stability (HNSW+RRF amplify small differences), so mixed local/remote index-and-query needs an end-to-end comparison.

**Q4: What specific links can break in "kill a replica with no perceived impact"?**
Key point: go through the chain — DNS caching (solved by nginx's `resolve` + `valid=10s`), POSTs not retried across replicas by default (`non_idempotent`, with the trade-off that `ask` might double-bill, an explicit choice), the 60s default connect timeout being too slow (lowered to 2s), the client's exception spectrum missing disconnect shapes (RemoteProtocolError/ReadError — details in the exception inheritance tree determine whether the overall promise holds), and shutdown draining (25s < 30s). Bonus: `restart: unless-stopped` doesn't bring back a manually killed replica — self-healing and failover are two different things, and you only know that by measuring it.

**Q5: What pitfalls did your health-check design have?**
Key point: three measured cases — the probe sharing a thread pool with business traffic causing "the higher the load, the falser the readiness signal" (fixed by going async + a dedicated limiter); the liveness check's timeout budget of 9s exceeding the healthcheck's 5s and causing self-misjudgment; nginx's container resolving localhost→::1 and giving a false unhealthy. Principle: liveness ≠ readiness (a config error shouldn't crashloop); err takes priority over loading; a probe's resource path/timeout/resolution must be isolated from business traffic. Proactively add: the known blind spot is readyz staying green forever after a GPU hang (deferred), fixed via a lock-hold-duration threshold plus a watchdog self-kill.

**Q6: With server-side 503s and client-side retries together, how do you avoid a cascade?**
Key point: they're counterparts — retries must be paired with admission control. `BoundedSemaphore(16)` pins the in-flight count, returning an immediate 503 once full (a non-blocking acquire), avoiding the three sins of deep queuing (still burning GPU after a client timeout, the retry re-entering the queue, and a positive-feedback congestion loop). Known gap: backoff has no jitter, causing synchronized retry waves (deferred, magnitude N×3, fixable with a one-line full jitter).

**Q7: Besides moving the data, what else does migrating from embedded to server mode require?**
Key point: four things — the three-branch store logic, config pass-through across every production exit point (all three were missed the first time, now locked in by guard tests), the migration script defending against intermediate states (lock first / verify source non-empty / with_vectors / count check / idempotency), and **re-testing ACL semantics** (the embedded fusion dropping the top-level `should` clause — server semantics differ, so new tests were required; rerunning the old tests would be a false green). Security equivalence is on the same footing as vector equivalence.

**Q8: What's still missing before this could really go to production?**
Key point: recite the five §4.2 backlog items plus the honest boundaries directly — timeout budget alignment (361s vs. 130s), the LB being blind to 200-wrapped errors, readyz not checking for non-empty, retries having no jitter, the GPU hang blind spot; plus add K8s (nginx+compose right now is a learning vehicle), inference authentication (currently relying on network isolation), and capacity numbers across form factors that haven't been backfilled. Being able to say "why it's not fixed yet" (needs a compose/load-test/fault-injection environment) is more credible than saying "it's all fixed."

---

## 7. Hands-on experiments

### Experiment 1 (CPU-runnable): walk through the full remote failure-mode spectrum

Prerequisite: in WSL, `conda activate custodian && cd ~/projects/custodian && pip install -e '.[dev]'` (without dependencies fully installed, pytest gives a collection error, not a real failure).

```bash
pytest tests/engine/test_remote.py -v          # 27 tests, pure CPU-mock, never touches the GPU
pytest tests/engine/test_concurrency.py -v     # the 9 guard tests for lock-down (M1)
```

Expect to see: two 503s then success on the 3rd try; 502/504/500 all get retried while 4xx gets zero retries; `ConnectTimeout/ReadError/RemoteProtocolError` all go through the TransportError spectrum; the backoff sequence is exactly [0.5, 1.0] exponential; a negative `retries` gets clamped; bf16 going through `_mrl` ends up with norm=1.0 (both an fp32 guard and a reverse guard); rerank raises `InferenceUnavailable` and gets degraded, while dense fails loud (the asymmetry locked in); a slow encode's backoff doesn't block other queries (the nullcontext lock). This one file is a living document of the P0-1/P0-2/M1/bf16 fix history — worth reading line by line alongside [src/embedder/remote.py:67-96](../../src/embedder/remote.py#L67).

Then hand-play the docker-kill disconnect shapes with 10 lines of code (to feel out spectrum completeness):

```bash
python - <<'EOF'
import httpx
from embedder.config import EmbedConfig
from embedder import remote
calls = {"n": 0}
class C:  # simulate a dead keep-alive connection after docker kill: every POST disconnects
    def post(self, path, json=None):
        calls["n"] += 1
        raise httpx.RemoteProtocolError("Server disconnected")
cfg = EmbedConfig(inference_url="http://x", inference_retries=2, inference_backoff=0.01)
try:
    remote._post_retry(C(), cfg, "/embed", {})
except Exception as e:
    print(type(e).__name__, "attempts=", calls["n"], "marker=", getattr(e, "inference_unavailable", False))
EOF
# Expected output: InferenceUnavailable attempts= 3 marker= True
# —— the kill's disconnect is absorbed by retries, and once exhausted it raises a semantic exception,
#    which toolcore can recognize without importing embedder at all (duck-typing marker)
```

### Experiment 2 (needs GPU + WSL + Docker Desktop with WSL Integration enabled): experience kill-goes-unnoticed 50/50 + the throughput ceiling firsthand

Prerequisite: compose must be run **from within a WSL terminal** (the bind mount is a native WSL ext4 path); `.env.compose` needs GPU_UUID/paths/keys configured (see [../SCALE_OUT.md](../SCALE_OUT.md) §5-F for the environment prerequisites).

```bash
docker compose --env-file .env.compose up -d --build --scale custodian=3   # 3 replicas + nginx, entry point :8080
# Terminal A: 50 continuous requests
for i in $(seq 50); do curl -s -XPOST localhost:8080/v1/retrieve -H 'X-API-Key: <key>' \
  -H 'Content-Type: application/json' -d '{"query":"Netflix 2015 revenues"}' -o /dev/null -w '%{http_code}\n'; done
# Terminal B: kill one replica mid-run
docker kill $(docker compose ps -q custodian | head -1)
docker compose ps        # observe: the killed replica is Exited 137 with restarts=0 (unless-stopped won't self-heal a manual kill)
```

Expected: 50/50 all 200, 0 failures; the request distribution across the three replicas' log directories `/index/custodian_logs/<hostname>/requests.jsonl` is close to even. Then verify the throughput ceiling:

```bash
python scripts/bench.py --url http://127.0.0.1:8080 --key <key> --clients 10 --n 10
docker compose --env-file .env.compose up -d --scale custodian=1 && # scale back to 1 replica and rerun for comparison
```

Expected: throughput with 3 replicas and 1 replica is nearly identical (~3 req/s range, pinned by serialized GPU forward passes), with p50 rising linearly with concurrency and zero errors — turning §2.7's "three limits" from prose into numbers you produced yourself, and also empirical evidence for why vLLM's trigger condition is queue depth.

---

## 8. Honest boundaries

Volunteering these in an interview is far stronger than having them dragged out of you:

1. **Throughput wasn't scaled.** "Multiple replicas" sounds like it should add throughput, but the actual QPS ceiling = serialized single-card forward passes (~3.2 req/s, measured on the single-machine embedded form factor; capacity numbers for the three-container compose form factor haven't been backfilled, so extrapolating across form factors carries some risk). The line to use: "I can state precisely what this architecture does and doesn't scale, along with the trigger condition and verification plan for breaking through the ceiling."
2. **E2 mixed-build/mixed-query (top-k order consistency between building locally and querying remotely) hasn't had a dedicated end-to-end comparison.** Equivalence has a structural guarantee plus E1/E3 element-wise measurements plus a functional test across the full container chain, but SCALE_OUT's risk table still lists "E1 doesn't imply E2" as the highest risk; the vLLM side, by contrast, already did the same-method G2 (88-question comparison) first. The line to use: "The structural guarantee is why I felt comfortable deferring it, but it's a must-fix before production, and the method and scripts are already in place."
3. **Migration validation only solidified the count-check layer.** Vectors actually being present / sidecar coverage / ACL fail-closed — these three layers were verified manually in-session, but never scripted; the script's own exit message admits this. The combination of the half-library intermediate state plus readyz not checking non-empty is a confirmed silent-corruption window (deferred deploy#2).
4. **The five production-readiness backlog items are "confirmed but unfixed"** (§4.2): the timeout budget gap, the LB being blind to 200-wrapped errors, readiness depth, retries with no jitter, the GPU hang blind spot. Each has a root cause, trigger condition, fix sketch, and a reason for deferring — deferring is discipline, not laziness: the changes need a real environment to verify, and making a blind change would itself count as "blind patching" under this repo's own engineering discipline.
5. **The ceiling of a single-machine learning vehicle**: nginx+compose is not K8s (no active health checks, no autoheal, no HPA); inference has no authentication (relying on network isolation plus not publishing the port); SessionRegistry/stats degrading across replicas is an explicit acceptance; `encode_image` across machines is a base64 TODO (used only for index building, which happens on the same machine, so the blast radius is small).
6. **vLLM has only cleared the equivalence gate, not the benefit gate.** The G4 throughput comparison hasn't been run, so it's still Plan B; the root cause of G1's small drift (a transformers version difference) is the "prime suspect," not a settled conclusion. The line to use: "I won't treat 'basically passes' as 'passes' — I know exactly what that one top-1 flip out of 87/88 in G2 looks like (a near-duplicate reordering in the tail)."

---

*Every anchor in this piece has been individually verified against the 2026-07-07 workspace code. Data sources: the 88-question gold baseline (don't mix it with the historical 72-question one), the single-machine embedded capacity table (2026-07-04), and the stage-F multi-replica measurements (2026-07-07).*
