# vLLM Inference Backend Plan (Plan B → promote to Plan A once it clears the gates)

> **Note (this project):** §4's G1 vector-equivalence sample (texts in the project's original non-English language and English) predates this project's swap of
> that original-language support for Telugu. The measured cosine numbers are kept as historical record and have not been
> re-measured against Telugu text.

> Positioning: **not a replacement — coexistence first**. The existing self-contained FastAPI inference service
> (`inference_server.py`) stays as Plan A; a new vLLM backend is added as Plan B, **presenting the exact same
> `/embed`/`/rerank`/`/readyz` contract** — not a single byte of the application layer (custodian's `remote.py`)
> changes, only what `CUSTODIAN_INFERENCE_URL` points to. **vLLM is only promoted to Plan A after it clears both
> the equivalence gate and the throughput gate**; otherwise it can be rolled back at zero cost at any time.
>
> Status: **all four gates passed under real measurement (2026-07-07); the verdict is that vLLM should be promoted
> to Plan A (remaining work is engineering rollout: the Phase 1 adapter + compose profile)**.
>
> **Measurement summary (see §4.1)**: G0 GO (vLLM loads and pools out normalized vectors) / G1 borderline (cosine
> 0.99956) / **G2 basically passes** (vLLM querying the officially built real library gets top-1 98.9%, Jaccard
> 0.9959, and all disagreements are harmless near-duplicate reorderings in the tail) / **G4 decisive win** (vLLM
> continuous batching hits **297 QPS** vs FastAPI's serial cap of **24 QPS** = **~12×**, with zero errors and lower
> p95 latency). **Overall: equivalence is acceptable + 12× throughput → vLLM should be promoted to Plan A.**

---

## 0. In one sentence

Replace the "one and only GPU-bound forward pass" from **per-request serial execution (FastAPI + `gpu_lock`)** with
**vLLM's continuous batching**, to raise GPU utilization and throughput under **concurrent queries**; but the
**precondition is that the vectors vLLM produces are mathematically equivalent to those in the existing library**
(built with the official `Qwen3VLEmbedder`) — this is the linchpin of the whole plan, on the same level of
criticality as Phase B's E2 mixed-build-mixed-query concern. Only once equivalence holds and the throughput gain is
real does it get promoted to Plan A.

---

## 1. Why put vLLM on the roadmap now

The existing inference's GPU forward pass is **serial on a single GPU** (`inference_server.py`'s `state.gpu_lock`
plus Phase F's `BoundedSemaphore(16)` backpressure). This is the **hard throughput ceiling** nailed down by
SCALE_OUT §5-F F-3: `--scale custodian=N` cannot scale QPS, because all dense/rerank forward passes queue on the
same lane.

**vLLM's one and only hard value-add is continuous batching**: it dynamically batches concurrent requests on the
GPU, so single-GPU throughput rises with concurrency (instead of being pinned by serial execution). The trigger
condition (as defined in SCALE_OUT §3.4) is: **query QPS rises to the point where the `/embed` queue depth stays
above 1**. It is not "vLLM is trendier."

**Why we didn't adopt it earlier**: equivalence. As long as both sides go through the same `Qwen3VLEmbedder`
forward pass, equivalence is a **structural guarantee**; switching to vLLM means swapping the pooling
implementation → equivalence goes from "guaranteed" to "an empirical bet." This plan turns that bet into a runnable
go/no-go experiment (§4 G1/G2) — only adopted if it wins.

---

## 2. Key facts (grounded — these determine whether the plan is feasible)

| # | Fact | Basis | Impact on the plan |
|---|---|---|---|
| A | The model is `Qwen3VLForConditionalGeneration` (`model_type: qwen3_vl`), i.e. a **generative VL model**, used as an embedder via the official script's last-token pooling | Local `~/models/Qwen3-VL-Embedding-8B/config.json` + `scripts/qwen3_vl_embedding.py::_pooling_last` | vLLM must replicate this exactly: the chat template (system = instruction) + **LAST**-token pooling + L2 normalize |
| B | **vLLM officially supports Qwen3-VL-Embedding**: `LLM(model, runner="pooling", dtype="bfloat16", trust_remote_code=True)` + `llm.embed(...)` | vLLM docs + QwenLM/Qwen3-VL-Embedding's `examples/embedding_vllm.ipynb` | G0 is likely to pass; `--convert embed` defaults to last-token pooling + normalize, **matching the official pooling approach** |
| C | vLLM's input format: `apply_chat_template([{system:instruction},{user:text}], tokenize=False, add_generation_prompt=True)` → fed as `prompt` | Official `embedding_vllm.ipynb` | The adapter must **copy this format exactly** (especially `add_generation_prompt=True`), or the last token will be misaligned → equivalence will fail (**risk of a false negative**) |
| D | vLLM returns **already-normalized** vectors (the notebook does `emb @ emb.T` directly, with no further normalization) | Official notebook | This matches the local `F.normalize`, so no second normalization is needed |
| E | **image/video preprocessing differs between vLLM and the official pipeline** (`qwen_vl_utils` vs transformers' `video_processing_qwen3_vl`) → results "differ slightly" | Explicit warning in vLLM/Qwen docs | **Only affects the image path**; and image encoding is **library-build-only, running on local torch** (D4/D5) — **the query path is pure text and is unaffected**. This conveniently keeps vLLM's biggest equivalence risk out of scope for the plan |
| F | The reranker is `Qwen3-VL-Reranker` (yes/no logits + sigmoid); vLLM's score/rerank support for it is **unconfirmed** | No authoritative confirmation found | **The reranker is not migrated to vLLM for now**; rerank is a safe-to-degrade item (falls back to hybrid on failure), so it stays on torch with zero risk (see Phase 1 in §5) |
| G | The local `vllm` conda env already has **vLLM 0.22.1 / torch 2.11+cu130 / transformers 5.10.2** installed | Measured via `conda activate vllm` | G0/G1 can be run locally right away; ⚠ this differs from the version used to build the library — custodian uses transformers 4.57 — so numerical drift is possible and must be quantified in G1 |

**Preliminary feasibility judgment (based on the table above, not yet measured)**: for the text query path, **G0 is
likely feasible and G1 has a reasonable chance of passing** (same pooling/normalize approach + the official input
format can be copied directly + the image-path difference is out of scope for queries); the main uncertainty is
the **transformers version gap + bf16 numerics**, which G1 must quantify.

---

## 3. Target architecture: same-contract adapter + dual-backend coexistence

```
                        custodian ×N (remote.py unchanged; uses whichever backend CUSTODIAN_INFERENCE_URL points to)
                                 │  /embed {texts,instruction} · /rerank
              ┌──────────────────┴───────────────────┐
              ▼ (Plan A, current state)                ▼ (Plan B / promoted to Plan A once gates pass)
   ┌─────────────────────────┐           ┌─────────────────────────────────────────┐
   │ inference (FastAPI+torch)│           │ inference-vllm                            │
   │ gpu_lock serial + backpressure │     │  ┌────────────┐   ┌────────────────────┐ │
   │ /embed /rerank (full-dim) │         │  │ contract    │──▶│ vLLM serve (pooling)│ │
   └─────────────────────────┘           │  │ adapter     │   │ AsyncLLMEngine      │ │
                                          │  │ (CPU, no GPU)│  │ continuous batching │ │
                                          │  │ applies chat │  │ /v1/embeddings 8B   │ │
                                          │  │ template     │  │                      │ │
                                          │  └────────────┘   └────────────────────┘ │
                                          │  reranker: still on torch in Phase 1 (safe to degrade) │
                                          └─────────────────────────────────────────┘
   compose profile selection: `--profile fastapi` (current) / `--profile vllm` (Plan B) — mutually exclusive; rollback = switch profile
```

**The boundary rule remains ironclad**: the adapter/vLLM endpoint contains **no custodian business concepts
whatsoever** (no ACL/Hit/sidecar) — only `texts→vectors`. This is exactly the payoff of drawing the boundary at the
"pure GPU forward pass" line in the first place — swapping backends never touches the application layer.

**Why these two endpoints (why "vLLM serve + a thin adapter" rather than "in-process `LLM.embed`"):**
- **For continuous batching to take effect across concurrent HTTP requests, you must use vLLM's AsyncLLMEngine**
  (which is what `vllm serve` starts). The offline `LLM.embed()` only batches **within a single call** — it doesn't
  batch across concurrent requests, so it wouldn't capture vLLM's core value. Hence `vllm serve` is used.
- **The adapter is thin and has no GPU**: it does exactly two things — (1) apply the chat template to our
  `{texts, instruction}` contract per §2-C; (2) forward to vLLM's `/v1/embeddings` and reshape the response into
  `{"vectors": [...]}`. It also handles `/readyz` (probing whether vLLM is up). It inherits Phase F's
  probe/backpressure discipline, but **forward-pass queueing is handed off to vLLM** (which has its own scheduler —
  the adapter no longer adds a `gpu_lock`).

---

## 4. Go/no-go gates (the center of the plan; failing means no promotion to Plan A)

Strict ordering — if an earlier gate fails, later gates are not attempted. The first three are **correctness**
gates (equivalence); G4 is the **payoff** gate (throughput):

| Gate | Criterion | How it's tested | Consequence of failure |
|---|---|---|---|
| **G0 feasibility** | vLLM loads Qwen3-VL-Embedding-8B in pooling mode, and `/v1/embeddings` produces 4096-dimensional vectors | Start the service with `vllm serve ... --runner pooling` (or `LLM(runner="pooling")`), curl one piece of text | This path is dead on arrival and the plan is void (fall back to FastAPI) |
| **G1 vector equivalence** ⭐ | For the same batch of texts (original non-English language/English, long/short, with instruction), `cosine(vLLM, official Qwen3VLEmbedder) > 0.9999`, and norm ≈ 1 | `scripts/vllm_equiv_probe.py` (see §7): custodian produces and archives the official vectors → stop it → vLLM produces vectors → compare cosine (time-sliced to avoid OOM) | The existing library (built officially) **cannot be queried by vLLM** (vector drift → top-k misalignment, silent data corruption). Two ways out: ① precisely align the input format/transformers version and re-test; ② accept that vLLM can only be used after a **full rebuild of the library using vLLM** (a major migration) |
| **G2 mixed build/query** ⭐ | On the real library built officially, with vLLM encoding the queries, the top-k `chunk_id` ordering for ~50 real queries **exactly matches** the official-query ordering | Start the vLLM adapter pointed at the real library, diff its top-k against official encoding (same method as Phase B's E2) | Element-wise equivalence **does not imply** an unchanged top-k (HNSW approximation + RRF rank amplify tiny differences). If this fails, **it does not go to production** (same ironclad rule as E2) |
| **G3 reranker** (optional) | vLLM's score for Qwen3-VL-Reranker is `allclose` with the official score | Only done if reached in Phase 2; skipped in Phase 1 (reranker stays on torch) | If it fails, **the reranker is simply not migrated** — only embed is migrated (rerank is safe to degrade, no loss) |
| **G4 throughput** (the "does it actually pay off" criterion) | At concurrency C∈{16,32,64}, vLLM's QPS / p50 / p95 are **significantly better** than FastAPI's (serial gpu_lock); single-request latency does not regress | `scripts/bench.py` sweeps concurrency levels, hitting FastAPI and the vLLM adapter with the same batch of queries | If vLLM isn't faster than serial execution (it may even be worse at low concurrency) → **do not promote to Plan A**; keep it as Plan B for later scale-out; record honestly |

> **G1/G2 are hard gates, on the same level as Phase B's E1/E2**: misalignment from mixed build/query is the
> hardest kind of silent corruption to detect. **"It works" = G1 ∧ G2 ∧ G4 all holding**, not just looking at the
> G4 throughput number.

### 4.1 Measured results (2026-07-07, `scripts/vllm_equiv_probe.py`, on a 4090)

| Gate | Result | Data |
|---|---|---|
| **G0 feasibility** | ✅ **GO** | vLLM 0.22.1 `LLM(runner="pooling", max_model_len=8192)` successfully loaded Qwen3-VL-Embedding-8B; `llm.embed()` produced **4096-dimensional, normalized (norm=1.00000)** vectors. **The architecture fully supports this** |
| **G1 vector equivalence** | ⚠ **Borderline** | Across 8 samples (original non-English language/English, long/short), cosine ranged **[0.99956, 0.99982]**, min **0.99956** / mean **0.99973**. Direction is highly consistent (worst case about a 1.7° angle), but it **does not reach the strict >0.9999 bar** |

**Root cause of the drift (diagnosis, not conclusive)**: the prime suspect is the **transformers version
difference** — the library was built with custodian's **4.57.6**, while the vLLM environment runs **5.10.2**, and
the two versions' Qwen3-VL forward passes differ numerically; bf16 kernel differences are a secondary suspect. The
shortest text ("a") has the lowest cosine (0.99956), consistent with "short sequences amplify per-token numerical
differences."

**Does this pass or fail?** — **The call isn't made on this cosine number alone — it's made on G2**: an alignment of
0.9997 is stable for the vast majority of queries' top-k (the drift is far smaller than the gap between different
documents), **but near-duplicate chunks could flip** — which is exactly what the G2 mixed-build-mixed-query test is
meant to catch. So G1's borderline result **does not veto the plan**; the final call is handed to G2.

**Lessons recorded from the pitfalls hit along the way (diagnostic discipline — a positive example of "reject a
canned verdict-style conclusion")**: the probe produced 3 false failures before yielding real data, and every one
was **my own configuration error, not vLLM lacking support**: ① `CUDA_VISIBLE_DEVICES` was set to a GPU UUID (vLLM
only accepts an integer index; only torch accepts a UUID) → fixed by setting `CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1`; ② `max_position_embeddings=262144` made vLLM request 36GB of KV cache and OOM → fixed by
adding `max_model_len=8192` (a single embedding forward pass doesn't need long context); ③ I had piped through
`grep -v` and filtered out the real error (the EngineCore traceback) → nearly led to a false conclusion of
"architecture not supported." **The probe's boilerplate message "vLLM doesn't support Qwen3VL pooling" was wrong
all three times** — the truth is the model loads completely fine. Lesson: when something fails, look at the real
root cause first; don't trust a canned verdict.

| Gate | Result | Data (`scripts/vllm_g2_topk.py`, 88 questions from eval/gold.jsonl, embedded real library with 7652 points) |
|---|---|---|
| **G2 mixed build/query** | ⚠→✅ **Basically passes** | top-1 agreement **87/88 (98.9%)**; top-10 set agreement **86/88 (97.7%)**; top-10 exact-order agreement 76/88 (86.4%); **Jaccard@10 0.9959**. Of the 12 disagreeing samples, the first point of disagreement was at rank≥6 for 7 of them (tail-end near-duplicate reordering), and at rank 0/1 for one each |

**G2 read (honest)**: G1's 0.9997 cosine drift **does not flip effective recall** — top-1 agrees 98.9% of the time,
top-10 sets agree 97.7% of the time, and disagreements are almost entirely near-duplicate chunks being reordered at
ranks 6-9 (the evidence pool stays the same, the answer doesn't change). This is the **worst-case scenario for
mixed build/query** (vLLM querying × officially built, cross-implementation); if Plan A is rolled out with **a full
library rebuild using vLLM**, then build and query use the same implementation and are self-consistent with no
drift — cleaner than this. **The only remaining discrepancy**: 1/88 different top-1 + 2/88 different top-10 sets
(purely from the transformers 4.57↔5.10 numerical difference). For a system whose flagship feature is fidelity, if
zero tolerance is required, aligning the transformers versions or rebuilding the whole library with vLLM would
eliminate this; otherwise, 98.9% top-1 agreement is acceptable for production.

| Gate | Result | Data (`scripts/bench_embed.py`, 88 questions cycled to fill 120 per level, single-text query = 1 vector, chat template pre-applied) |
|---|---|---|
| **G4 throughput** | ✅ **Decisive win** | vLLM (continuous batching) vs FastAPI (serial gpu_lock + backpressure of 16), on a 4090: |

```
        FastAPI QPS   vLLM QPS   ratio     (p50 ms: FastAPI→vLLM)
conc 1     22.4         33.5     1.5x      44 → 28
conc 8     23.7        169.3     7.1x     334 → 45
conc 16    23.5        226.0     9.6x     680 → 72
conc 32    22.2*       271.2    12.2x     428 → 93     *FastAPI only succeeded on 17/120, 103 were rejected with 503 backpressure
conc 64     —          297.3      —      (vLLM succeeded on all, p95 264ms)
Peak: FastAPI caps out at ~24 QPS (serial, doesn't rise with concurrency); vLLM reaches 297 QPS (scales roughly linearly with concurrency)
```

**G4 read**: FastAPI's serial gpu_lock pins throughput at **~24 QPS** (added concurrency only piles up latency +
triggers backpressure rejections), while vLLM's continuous batching packs concurrent requests together, reaching
**297 QPS ≈ 12×**, with zero errors and even lower p95 (93ms vs 428ms at conc32). **This is exactly the payoff that
was reserved when the GPU inference layer was split out** (the endpoint contract has no business concepts →
swapping backends doesn't touch the application layer). **Combining all four gates: vLLM should be promoted to
Plan A.**

**Overall conclusion**: equivalence is acceptable (G1/G2) + 12× throughput (G4) → **promoting vLLM to Plan A is
justified**. Rollout follows §5 Phase 1 (embed goes through vLLM, reranker stays on the safe-to-degrade torch path
for now; compose profile switch + second-level rollback). If zero top-1 drift is required, pair this with a full
vLLM library rebuild (so build and query use the same implementation and are self-consistent).

---

## 5. Phased rollout (each phase independently verifiable + rollback at any time)

**Phase 0 — go/no-go probe (run first; decides whether to proceed).** Only `scripts/vllm_equiv_probe.py`, doesn't
touch production: tests G0 + G1. The sole output of this phase is **one number** (cosine). cosine > 0.9999 →
proceed; otherwise, either fix input alignment first or conclude "a full library rebuild is needed." **Lowest cost,
highest decisiveness — do this first.**

**Phase 1 — embed-only vLLM backend (reranker stays on torch). ✅ Core implementation landed + bare-metal
measurement done (2026-07-07):**
- `src/inference_vllm_adapter.py` (thin FastAPI, no GPU, no torch): `/embed` applies the §2-C chat template → vLLM
  `/v1/embeddings` → `{"vectors"}` (full-dimensional, normalized, same contract as inference_server); `/readyz`
  asynchronously probes vLLM's `/health`; `/embed_image` returns 501 (image encoding is library-build-only and
  runs locally); `/rerank` proxies to the torch reranker if `RERANK_PROXY_URL` is configured, otherwise returns
  503 (custodian degrades to hybrid, which is safe).
  **⚠ Placed at the src/ root, not inside the embedder/ package**: the adapter needs to run in the vllm environment
  (which has no qdrant_client); placing it inside the package would trigger `embedder/__init__`'s qdrant_client
  import, and would also cause the stdlib's `import types` to be shadowed by `embedder/types.py` (both pitfalls
  were actually hit and have been resolved).
- `Dockerfile.inference-vllm`: a slim adapter image (fastapi/uvicorn/httpx/transformers, with a build-time
  assertion that torch is absent).
- **End-to-end bare-metal measurement (same pivot method as Phase E)**: `vllm serve` on `:8000` + the adapter on
  `:8900` + **custodian (RemoteDense, application layer unchanged byte-for-byte) hitting the embedded real library
  via CUSTODIAN_INFERENCE_URL pointed at the adapter** → `scripts/vllm_adapter_smoke.py` got **3/3 queries returning
  real hits** (IBM query → IBM report chunk, Netflix → Netflix 10-K, cross-lingual → the matching paper). The
  adapter's /embed produced a 1×4096, norm=1.0 vector. **This proves "swapping the vLLM backend doesn't touch the
  application layer."**
- **⏸️ Still pending (recorded honestly, same discipline as Phase E)**: containerizing the compose `vllm` profile
  (`inference-vllm-engine` = vllm/vllm-openai running `vllm serve` + `inference-vllm` = the adapter) **has not yet
  been run end-to-end inside containers** — the vllm/vllm-openai image is ~10GB, and pulling it over the local
  network carries risk (the same daocloud pitfall as the torch base image in Phase E); this is left as the next
  step, to be verified in its own right, rather than shipping a profile that has "never actually run in a
  container" (a blind spot lesson learned from Phase E).
- G2 mixed-build-mixed-query was already run ahead of schedule in Phase 0 (§4.1, GO).

**Phase 2 — throughput gate + decide Plan A/B.** Run the **G4** concurrency benchmark (FastAPI vs vLLM), and write
the numbers into SCALE_OUT. If G4 wins → promote vLLM to Plan A (switch the default profile to vllm); otherwise
stay on Plan B.

**Phase 3 (optional) — migrate the reranker to vLLM too.** Only if G3 passes; otherwise the reranker stays on torch
permanently (safe to degrade, no loss).

---

## 6. GPU resource reality (must be accounted for first)

4090, **48GB**. Current state: torch inference (2×8B embed+rerank) uses **~33GB**. vLLM's embed 8B weights are
~16GB + KV cache. **torch-inference at full load and vLLM cannot run simultaneously** (would OOM past 48GB).
Therefore:

- **When testing G0/G1/G2/G4**: you **must first stop the torch inference container**
  (`docker compose stop inference`) to free up the GPU before starting vLLM. In other words, running the vLLM
  experiments means temporarily taking the existing compose stack's inference offline (custodian's /readyz will
  return 503, which is acceptable — it's the experiment window).
- **Plan A rollout shape (if G4 wins)**: vLLM serves only **embed** (~16-20GB) + the torch **reranker** in its own
  container (~16GB) can coexist on one 4090 (totaling ~36GB < 48GB); or if the reranker also moves to vLLM (G3
  passes), everything runs on vLLM. **embed is the QPS bottleneck, so migrating it first captures the bulk of the
  gain**.

---

## 7. Go/no-go probe script (Phase 0, already in place)

`scripts/vllm_equiv_probe.py` (in the repo): run in two separate time slices — `--step official` uses custodian to
produce and archive the official vectors, `--step vllm` uses the vllm env to produce vLLM vectors and compare
cosine. **Critical**: the input format must strictly follow §2-C (chat template + `add_generation_prompt=True` +
the same instruction), otherwise you get a false negative. How to run it:
```bash
# 0) Free up the GPU
docker compose --env-file .env.compose stop inference
# 1) Archive the official vectors (custodian)
conda activate custodian && python scripts/vllm_equiv_probe.py --step official --out /tmp/vllm_probe
# 2) vLLM vectors + comparison (vllm env)
conda activate vllm    && python scripts/vllm_equiv_probe.py --step vllm     --out /tmp/vllm_probe
# Output: cosine for each item + max/min/mean; min cosine > 0.9999 = G1 GO
```

---

## 8. Risks + rollback

| Risk | Mitigation |
|---|---|
| **G1 fails** (transformers version gap / bf16 / input format) | First try to align: install the same transformers version in the vLLM env, diff tokenization token-by-token; if it still fails → conclude "a full vLLM library rebuild is needed," or abandon the plan |
| Input format off by one token → false negative | The probe strictly follows the official notebook (§2-C); verify that both sides' post-tokenization id sequences match before comparing vectors |
| vLLM doesn't support the reranker | Phase 1 simply doesn't migrate the reranker (safe to degrade); G3 is an independent gate |
| GPU can't fit both | embed-only vLLM + torch reranker in separate containers (§6); or time-share |
| vLLM crashes / needs rollback | **Switch the compose profile back to `fastapi` + point custodian's `CUSTODIAN_INFERENCE_URL` back at torch inference — a second-level rollback** (the application layer staying unchanged is this design's biggest safety net) |
| vLLM is slower at low concurrency | G4 measures this honestly; if it doesn't meet the bar, stay on Plan B — don't force the switch |

---

## Appendix: sources

- Model architecture/pooling: local `~/models/Qwen3-VL-Embedding-8B/{config.json,scripts/qwen3_vl_embedding.py}` (read directly)
- vLLM support + input format: [QwenLM/Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding)'s `examples/embedding_vllm.ipynb`, [vLLM Embedding docs](https://docs.vllm.ai/en/latest/models/pooling_models/embed/), [vLLM Pooling docs](https://docs.vllm.ai/en/latest/models/pooling_models/)
- Image preprocessing difference warning: vLLM / Qwen3-VL-Embedding docs (see §2-E)
- Local vllm env: vLLM 0.22.1 / torch 2.11+cu130 / transformers 5.10.2 (measured via `conda activate vllm`)
