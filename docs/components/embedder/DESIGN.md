# embedder design

> **Note (this fork):** this document is kept as the historical design record from when the system's non-English tokenization
> target was Chinese (`jieba`), including the measured numbers below (e.g. the chars/token calibration in §"est_tokens → real
> tokenizer"). This fork later replaced Chinese-language support with Telugu; the current sparse tokenizer is a Telugu-Unicode-range
> regex (no dictionary segmenter needed, since Telugu is written with spaces between words) in `src/embedder/sparse.py`, and current
> behavior may differ from what's narrated below. This file has not been rewritten to match, since the measurements here are
> specific to the old jieba/Chinese implementation and would be fabricated if simply relabeled "Telugu." See the "Sparse/Telugu
> tokenization" row in the top-level README's Technology stack table for what's actually implemented today.

> The embed + retrieval component of the RAG pipeline, taking chunker's `Chunk[]` as input. **Local 4090** deployment: dense = Qwen3-VL-Embedding-8B (multimodal), sparse = **BM25 (jieba tokenization + Qdrant IDF)**, vector store = Qdrant (dense+sparse hybrid + payload filter). Location: `src/embedder/` (a package alongside `src/chunker/` in the same repo).

## 1. Position in the pipeline

```
ingest ─► parse ─► chunker ─► [ embedder ] ─► (Qdrant)
                   Chunk[]        │                at query time:
                                  ▼                query ─► embed ─► query_points (hybrid + ACL filter)
                          dense + sparse vectors        ─► chunker.assemble_big (small-to-big, ACL-gated) ─► LLM
```

The component **only understands chunker's `Chunk`** (text/image_path/image_only/acl/doc_meta/source_indices/doc_type/section_anchor/...). Every interface chunker leaves for it is fully honored here.

## 2. Tech stack (settled + APIs verified)

| Layer | Choice | Key interfaces |
|---|---|---|
| Dense | **Qwen3-VL-Embedding-8B** | `model.encode([{"text":..}/{"image":path}])` (sentence-transformers); text and images share the same 4096-dim space; MRL truncates down from 64–4096; queries get an added `prompt=` instruction; ~16–18GB GPU locally |
| Sparse | **BM25** (jieba tokenization + Qdrant `Modifier.IDF`) | Client-side jieba tokenization → tokens; doc sparse = term frequency, query sparse = 1; Qdrant computes the BM25 score server-side using IDF. **Zero GPU, CPU is enough** |
| Vector store | **Qdrant** | named dense+sparse; `query_points(prefetch=[dense,sparse], query=FusionQuery(RRF), query_filter=Filter)` for server-side hybrid |

**Why sparse = BM25 rather than BGE-M3**: three rounds of research converged on this — ① BM25 is the dominant production choice for sparse in the industry (the standard in frameworks/Qdrant); ② the precise-term matching your Chinese financial reports/legal documents need (numbers, statutory clause numbers, model numbers) is exactly BM25's strength (empirically, on financial documents BM25 beats even the strongest commercial dense models); ③ zero GPU, and cleanly decoupled from the Qwen3-VL dense model; ④ BGE-M3's sparse mode is a minority choice and weak on Chinese (MIRACL-zh 36.3), and its mainstream value (dense) is already replaced by Qwen3-VL. **Kept for the record: the measured comparison of BM25 vs BGE-M3-sparse on real precise-term queries** (see §7, "to be verified").

## 3. Qdrant collection

```python
client.create_collection(
    "rag_chunks",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},   # Qwen3-VL MRL=1024 as a starting point
    sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},    # BM25: Qdrant scores via IDF
)
# payload (schemaless dict): chunk_id/doc_id/kind/text/doc_meta/doc_type/section_id/section_anchor/
#   source_indices/page_start/page_end/image_path/flags/lang
#   + ACL: acl_unset(bool) / acl_visibility(str) / acl_tenant(str) / acl_allow(list[str])
# payload index (to speed up ACL filtering): acl_tenant, acl_visibility, acl_allow, acl_unset
```

`point.vector = {"dense": [...], "sparse": SparseVector(indices=token_ids, values=tf)}`; **the sparse key is omitted for image_only chunks** (a pure image is only recalled via its dense image vector).

## 4. ACL hard filtering (the security core, honoring the chunker contract)

At embed time, `chunk.acl` dict fields get split out into 4 ACL fields in the payload. Retrieval filter (`user.tenant` / `user.principals = groups + [uid]`):

```python
Filter(must=[
    FieldCondition(key="acl_unset", match=MatchValue(value=False)),       # fail-closed: exclude unauthorized documents
    FieldCondition(key="acl_tenant", match=MatchValue(value=user.tenant)),# tenant isolation
    Filter(should=[                                                       # nested: (allow ANY OR public)
        FieldCondition(key="acl_allow", match=MatchAny(any=user.principals)),
        FieldCondition(key="acl_visibility", match=MatchValue(value="public")),
    ]),
])
```

- **The nested should is security-critical**: wrapping `(allow OR public)` in a nested `Filter` inside `must` unambiguously expresses "tenant AND (allowed OR public)," without relying on the semantics of a flat must+should combination. This corresponds to chunker INTEGRATION §6's "the parentheses cannot be flattened."
- **Fail-closed**: `acl_unset==true` is excluded by default. **deny is not automatic** (per the contract, use `must_not` explicitly if needed).
- Two-layer ACL: the retrieval filter plus material selection via `chunker.assemble_big(admit=acl_admits(user))`.

## 5. Data flow

**At index time** (`embed.py`):
```
for chunk in chunks:
    if "image_only" in chunk.flags:           # pure image: dense only (image vector)
        dense = qwen3vl.encode_image(abs(image_path)); sparse = None
    elif chunk.kind in (image, chart):        # an image with a caption: dense image + BM25 on the caption
        dense = qwen3vl.encode_image(abs(image_path)); sparse = bm25_sparse(chunk.text)
    else:                                      # text/table: dense text + BM25
        dense = qwen3vl.encode_text(chunk.text); sparse = bm25_sparse(chunk.text)
    vec = {"dense": dense} | ({"sparse": sparse} if sparse else {})
    client.upsert("rag_chunks", [PointStruct(id, vector=vec, payload=acl_split(chunk)+meta(chunk))])

# bm25_sparse(text): tokens = jieba.cut(text) (Chinese); stopwords removed; token->uint32 stable hash;
#                    returns SparseVector(indices=hashes, values=term_counts)  ← Qdrant's IDF modifier does the scoring
```

**At query time** (`retrieve.py`):
```python
qd = qwen3vl.encode_text(query, prompt="Retrieve relevant documents for the query.")
qs = bm25_sparse(query)                          # query-side values use 1.0
acl = acl_filter(user)
hits = client.query_points("rag_chunks",
    # The ACL filter must be pushed down into every prefetch: embedded QdrantLocal drops the top-level
    # query_filter's should clauses under fusion (measured, see §7 "to be verified" #4) — only prefetch-level
    # filters actually make should take effect. The top-level filter is kept as a second layer of defense.
    prefetch=[Prefetch(query=qd, using="dense", filter=acl, limit=50),
              Prefetch(query=qs, using="sparse", filter=acl, limit=50)],
    query=FusionQuery(fusion=Fusion.RRF),
    query_filter=acl, limit=k, with_payload=True).points
for h in dedup_by_section(hits):                  # dedup: section_id=None is not collapsed (dedup by chunk_id instead) — review finding #6
    mn, tg, mx = BUDGETS.get(h.doc_type, DEFAULT_BUDGET)   # per-doc_type budget (whole sections for slides/policy) — review finding #8
    # the acl_index default path (only pulls original text with the same ACL as the hit) rather than admit:
    # big.acl=hit_acl is then naturally accurate and checkable at the exit point — review finding #2
    big = chunker.assemble_big(shim(h.payload), secs, els, target=tg, min_tokens=mn, max_tokens=mx,
                               banners=banners, acl_index=acl_index)
    if not acl_admits(big.acl, user):            # second check at the exit point (invariant 5): don't deliver this context if unauthorized
        big = None
```

## 6. Modules

| Module | Responsibility | Status |
|---|---|---|
| `config.py` | Model paths/dimensions/Qdrant connection/collection/stopwords/sidecar directory | ✅ |
| `dense.py` | Qwen3-VL wrapper (reuses the official `Qwen3VLEmbedder`, encode_text/image, instruction, MRL truncation, locked to the 4090 by name) — the only thing that uses the GPU | ✅ verified (text and images share a space) |
| `sparse.py` | **BM25**: jieba tokenization + regex to preserve exact strings + token→uint32 stable hash → `SparseVector` (doc=tf / query=1) | ✅ unit tested |
| `store.py` | Qdrant collection (sparse modifier=IDF) / payload index / upsert / **hybrid + ACL hard filtering** (filter pushed down into prefetch) | ✅ ACL unit tested |
| `acl.py` | ACL client logic (centralized for auditability): `acl_split` (splits into indexed fields) + `acl_admits` (retrieval-side predicate), semantically identical to the store's server-side logic | ✅ |
| `embed.py` | Routes each chunk (image_only→image vector/everything else→text+BM25) + ACL splitting + payload + per-doc sidecar (version + elements/sections/acl_index) | ✅ end-to-end |
| `retrieve.py` | Hybrid recall + **optional rerank** + dedup_by_section + hooks into `chunker.assemble_big` (ACL-aware small-to-big) | ✅ end-to-end |
| `rerank.py` | **Qwen3-VL-Reranker-8B** cross-encoder re-ranking (reuses the official `Qwen3VLReranker`, re-ranks hybrid top-N, locked to the 4090 by name) — optional, the second thing that uses the GPU | ✅ evaluated, MRR 0.566→0.867 |
| `types.py` | `User` (tenant+principals) / `Hit` retrieval-result contract | ✅ |

## 7. Decisions and trade-offs

- **sparse=BM25**: the industry-standard choice + strong on precise terms + zero GPU (only dense uses the GPU, which greatly simplifies the environment).
- **dense MRL=1024 as a starting point**: 4096 costs 4x, and 1024 usually costs almost nothing in recall; adjust further based on retrieval quality.
- **Chinese tokenization**: jieba is a variable here — could also switch to Qdrant 1.15+'s server-side CJK tokenizer (`Document` + BM25). Starting with client-side jieba (more controllable), with a measured comparison.
- **token→uint32**: a stable hash (some small chance of collision, acceptable); doc and query must use the same hash function.
- **est_tokens → real tokenizer**: ✅ measured against 2927 real chunks (Qwen3-VL tokenizer) — for prose, chars/token ≈ 3.85 (≈ the current value of 4.0, error < 4%), with the deviation concentrated entirely in number/table-dense documents (financial reports 5.08, government documents 5.35, Chinese research reports 1.51); a single value can't serve both clusters, and changing the mean would actually hurt prose — **keeping the current value, not re-calibrating** (see the comment on core.est_tokens).
- **The 4th tier, oversplitting**: retrieve uses `source_indices` to stitch back together xlsx records that were split across columns/row groups.

**To be verified (measured, all at once once the environment is ready)**:
1. ✅ **Already verified (BM25 confirmed)**: 14 documents/1067 chunks, 25 precise-term queries (programmatically mined for rare strings) + 31 semantic queries (agent-generated, avoiding the original wording),
   single-route MRR/Recall@10 (`scratchpad/eval/`):
   - **Precise-term**: bm25 **0.738**/0.96 > bgem3 0.584/0.88 ≫ dense 0.149/0.24 — BM25 **clearly wins** over BGE-M3 on its home turf (numbers/model numbers/amounts matched exactly);
   - **Semantic**: dense **0.794**/0.94 ≫ bgem3 0.347 > bm25 0.210 — semantic matching is dense's job, both sparse routes are weak here;
   - **Overall**: dense 0.506 > bgem3 0.453 ≈ bm25 0.446 (the two sparse routes are roughly tied).
   **Conclusion: choosing BM25 is correct** — the precise-term job that sparse is responsible for, BM25 wins; semantics is handled by dense (BGE-M3's semantic advantage is redundant inside the hybrid),
   and BM25 is zero model/zero GPU/zero maintenance versus BGE-M3's 2.3GB+GPU. Note: this is a synthetic set at small scale, so the trend is credible but absolute values are for reference only.
2. ✅ **Already verified**: Qwen3-VL's cross-modal text↔image recall (`scratchpad/verify_dense.py`, 2 real paper figures + accurate/irrelevant descriptions with cross-similarity).
   An accurate text description → similarity to the corresponding figure 0.74/0.49, → similarity to the non-corresponding figure only 0.37/0.17, and an irrelevant description (a golden retriever on a beach) gives 0.07–0.11 against both figures.
   **A text query being able to cross-modally recall an image_only chunk holds up**, and the model aligns on the figure's actual semantic content (even abstract descriptions can hit a flowchart).
3. ✅ **MRL verified**: 1024 versus full 4096 dimensions is almost lossless — diagonal similarity 0.7388/0.4930 → 0.7363/0.4483, separation stays healthy
   (A=0.34/B=0.30), irrelevant descriptions stay ≤0.11 throughout. **dense_dim=1024 as a starting point is confirmed** (saves 4× storage/retrieval cost).
   ✅ **RRF weighting verified**: sweeping the dense+bm25 weighted RRF, the peak is at w_dense≈0.4, overall MRR **0.541** > pure dense 0.510 > pure sparse 0.449
   — **hybrid genuinely beats any single route** (dense handles semantics, sparse handles precise terms, and they complement each other). Equal weighting (0.50) MRR 0.499 is already close to the peak.
   **Decision: keep the store on Qdrant's standard RRF (equal weighting, simple server-side fusion), don't switch to client-side weighting** — a 0.04 gain isn't worth the client-side fusion complexity,
   and the optimal 0.4 strongly depends on the query mix (exact:semantic, ≈1:1 in this set), which will drift with the real-world distribution; principle: **don't set the sparse weight too low**;
4. ✅ **Already verified**: Qdrant's `Modifier.IDF` **is supported** in embedded local mode (the sparse route returns results); BUT testing found
   that **embedded QdrantLocal silently drops the top-level `query_filter`'s `should` clauses in fusion (RRF) mode** (both top-level and nested
   should clauses fail — only flat must-equality conditions actually take effect) — this would **degrade the ACL's "(allow ANY) OR public" down to only filtering on tenant,
   leaking unauthorized documents (fail-open)**. The correct fix: **push the filter down into every `Prefetch(filter=acl)`**, so fusion only merges already-filtered results
   (filtering still happens at the recall layer = fail-closed remains intact, and limit isn't polluted by unauthorized results). This is now implemented in `store.py`,
   with `tests/test_store.py` guarding it via an assertion that "unauthorized content cannot be recalled." Diagnostic script: `scratchpad/diag_acl.py`.

## 8. Environment prerequisites (must be ready before implementation/testing)

- **GPU environment**: **only Qwen3-VL-8B dense requires CUDA torch** (the current default python is CPU torch); BM25 sparse is pure CPU (jieba), no GPU needed.
- **Dependencies**: `sentence-transformers transformers>=4.57 torch(cuda) qwen-vl-utils qdrant-client jieba` (drop FlagEmbedding).
- **Qdrant**: embedded (`QdrantClient(path=...)`, zero docker) as a starting point. **Constraint: only one client is allowed on the same path at a time**
  (within a single process, embed+retrieve must share the Store/Dense — `Embedder`/`Retriever` accept `store=`/`dense=` to reuse them; otherwise
  you get `AlreadyLocked` + the 8B model getting loaded twice). Once scale grows, switch to server mode to remove this constraint, and payload index only takes effect there.
- **Model**: the official Qwen3-VL 8B (~16GB, downloaded via modelscope to `~/models`) + the bundled `scripts/qwen3_vl_embedding.py` (reused by dense.py); BM25 needs no model.

## 9. End-to-end verification (MVP milestone)

`scratchpad/e2e.py`: a real paper (acl.long.386, MinerU output) walks the full path parse→`from_mineru`→`Chunker.chunk` (ACL stamped)→
`Embedder.index_document`→`Retriever.search_with_context`. Measured results:

- **Pipeline**: 247 elements → 58 chunks → 58 indexed (0 pure images, 6 image/chart chunks with captions go through the text route).
- **ACL fail-closed, end to end**: an authorized identity (t1/g_research) gets precise recall (top hit scores 0.83, exactly matching the query); **cross-tenant (t2) = 0 hits, same-tenant but no matching group (g_other) = 0 hits** — an unauthorized caller gets zero chunks at the Qdrant filter layer.
- **small-to-big ACL awareness**: `big.acl` is correctly stamped, climbed/tokens are reasonable.
- **Text-to-image recall**: a query describing a figure → top1 hits the image chunk (this document goes through the caption route; the pure-image-vector route is independently verified in `verify_dense.py`: accurate description ↔ corresponding figure 0.74/0.49, irrelevant 0.07–0.11).

## 10. Sign-off adversarial review (before moving to the next stage, 4 dimensions of finders × independent verification of each finding)

The adversarial workflow (4 finders covering ACL security/data flow/retrieval contract/robustness → each finding independently verified by a separate reviewer) turned up 13 confirmed findings, deduplicated down to 10 real issues, all fixed + locked in with regression tests:

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | An empty tenant self-matching an empty-tenant user → tenant isolation fails open | high | `acl_split` treats an empty tenant as unset; `acl_admits` rejects an empty tenant on both sides (`test_acl`) |
| 2 | Under the admit path, `BigBlock.acl` under-reports the stricter ACL actually present in big.text (invariant 5) | medium | Switched to the `acl_index` default path (only pulls same-ACL content) + a second check at the exit point via `acl_admits(big.acl,user)` (`test_retrieve`) |
| 3 | `point_id=uuid5(chunk_id)` — reusing a doc_id silently overwrites | low | Documented as a contract (doc_id must be globally unique) |
| 4 | Changing `dense_dim` lets `ensure_collection` return early → dimension drift causes a crash or bad retrieval | high | An existing branch now asserts size==dense_dim, failing fast (`verify_seal4`) |
| 5 | BM25 doc term frequency double-counts alphanumeric tokens (jieba+regex) → asymmetric distortion | medium | `tokenize` now only supplements exact strings jieba didn't fully extract (`test_sparse`) |
| 6 | `dedup_by_section` was collapsing every `section_id=None` into a single entry → lost recall | high | `None` now degrades to dedup by chunk_id (`test_retrieve`) |
| 7 | `assemble_big`'s lang check only recognized `ch`; the `zh` alias fell through to `en` → Chinese token counts underestimated by 2.35× | high | Aligned with `est_tokens`'s `startswith(("ch","zh"))` (chunker regression) |
| 8 | The embedder used a free function that dropped the per-doc_type budget → slides/policy got truncated | medium | `_ChunkShim` now carries doc_type, and picks the budget from `BUDGETS` accordingly (e2e: 1286→800) |
| 9 | Sidecar writes were non-atomic → a crash could leave a half-written JSON file → assemble would crash on that doc permanently | high | Now writes to .tmp + fsync + `os.replace` (atomic) |
| 10 | Missing/corrupt sidecar had no fallback → one bad hit could take down the whole query | high | `_load_sidecar` now raises explicitly + `search_with_context` degrades gracefully via try/except (`test_retrieve`) |

Embedder unit tests: all 17 green (test_acl/test_sparse/test_store/test_retrieve) + chunker regression + e2e (including e2e_xlsx) + verify_seal4. Ready for sign-off.

**Polish after sign-off** (lazy-tree review, list C): sidecar schema version binding — on the write side, `_write_sidecar` stamps `SIDECAR_VERSION` (`config.py`); on the read side, `_load_sidecar` validates it before deserializing, and raises a loud `ValueError` on a mismatch (including old sidecars with no `version` field at all). **Deliberately kept outside the `(FileNotFoundError, JSONDecodeError)` degrade-and-continue catch in `search_with_context`**: a missing file/bad JSON is a transient, single-document issue (degrade gracefully), while a version mismatch is a systemic schema drift (every sidecar is stale), which should prompt a full rebuild rather than being silently degraded one document at a time. Whenever the sidecar structure changes (Element/Section fields, acl_index encoding), `SIDECAR_VERSION` must be bumped. Regression: `test_retrieve` covers the write-side stamping plus two read-side mismatch cases (an explicit old version number / a missing field).
