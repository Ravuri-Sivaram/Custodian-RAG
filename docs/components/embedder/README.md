# embedder

The **embed + hybrid retrieval** component of the RAG pipeline (takes `chunker`'s `Chunk[]` as input).
`parse → chunk → **embed → retrieve**`.

## Architecture

- **dense**: `Qwen/Qwen3-VL-Embedding-8B` (multimodal, text and images share a single 4096-dim space). Reuses the model's own bundled
  `scripts/qwen3_vl_embedding.py`'s `Qwen3VLEmbedder` (last-token pooling), rather than a custom implementation. MRL truncated to 1024
  (measured almost lossless). The only thing that uses the GPU, locked to the 4090 by name.
- **sparse**: **BM25** (regex-based Telugu-script tokenization + regex to preserve exact strings like `gpt-4`/`42000000` + FNV stable hash → uint32,
  scoring handled by Qdrant's `Modifier.IDF`). Zero models, pure CPU. *(This component previously used `jieba` for Chinese word segmentation;
  this fork replaced Chinese-language support with Telugu, and since Telugu is written with spaces between words, a plain Unicode-range regex
  replaces the dictionary-based segmenter — see `src/embedder/sparse.py`.)*
- **vector store**: **Qdrant** (embedded, as a starting point). Named dense+sparse, `query_points` fuses via RRF.
- **ACL hard filtering** (the security boundary): fail-closed, tenant isolation + "allow ANY OR public." **The filter is pushed down into every
  prefetch** (embedded Qdrant drops the top-level `query_filter`'s should clauses under fusion — see `docs/DESIGN.md` §7#4).
- **rerank** (optional): a `Qwen3-VL-Reranker-8B` cross-encoder re-ranks the hybrid recall's top-N (measured MRR 0.566→0.867); enabled with `search(rerank=True)`, costs +16GB VRAM, off by default.
- **small-to-big**: at retrieval time, hooks into `chunker.assemble_big`, ACL-aware (only pulls original text the user is authorized to see).

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Configuration (models/dimensions/Qdrant/sidecar/stopwords) |
| `dense.py` | Qwen3-VL wrapper (GPU) |
| `sparse.py` | BM25 (CPU) |
| `store.py` | Qdrant collection / upsert / hybrid + ACL hard filtering |
| `acl.py` | ACL client logic (`acl_split` splits into indexed fields / `acl_admits` retrieval predicate), semantically identical to the store |
| `embed.py` | `Embedder`: routes each chunk to embed + payload + per-doc sidecar |
| `retrieve.py` | `Retriever`: hybrid recall + optional rerank + dedup + small-to-big |
| `rerank.py` | `Reranker`: Qwen3-VL-Reranker-8B cross-encoder re-ranking (optional, GPU) |
| `types.py` | `User` / `Hit` |

## Usage

```python
from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir
from embedder import EmbedConfig, Embedder, Retriever, User

cfg = EmbedConfig()                                   # defaults: ~/models, ~/qdrant_data, ~/rag_sidecar
elements = from_mineru_dir("parsed/<doc>")
result = Chunker().chunk(elements, doc_id="d1", doc_type="academic_paper", lang="en", acl=acl)

emb = Embedder(cfg)
emb.index_document("d1", elements, result, image_root="parsed/<doc>")

# Within the same process, reuse store+dense (embedded Qdrant's single client + avoid reloading the 8B model)
ret = Retriever(cfg, store=emb.store, dense=emb.dense)
for r in ret.search_with_context("...query...", User(tenant="t1", principals=["g_research"])):
    print(r["hit"].text, r["context"].text)
```

## Environment

WSL Ubuntu conda env `custodian` (torch 2.8.0+cu128, GPU=4090). See `requirements.txt` for dependencies.
The dense model is downloaded via modelscope to `~/models/Qwen3-VL-Embedding-8B` (~16GB).

## Status

Has passed **R1–R5 adversarial review** (R1: ACL, 0 findings / R2: retrieval fixed 6 window-block-state issues / R4: assets — content_raw now counted in the budget, etc.); `eval/acl_regression.py`
has 44+ assertions all passing (including "after forbidding exit-point `acl_admits`, RRF fusion still shows 0 leakage," proving the prefetch push-down itself blocks unauthorized access). Unit tests: **34 passed**:
`test_sparse.py` (exact strings), `test_store.py` (ACL — unauthorized content can't be recalled), `test_acl.py` (ACL predicates), `test_retrieve.py` (window-block state/exit-point ACL/sidecar versioning/dedup fallback).
See [OVERVIEW §7](../../OVERVIEW.md) for the system-level end-to-end evaluation. Optional enhancements (non-blocking): a BM25 vs BGE-M3-sparse weight sweep, hooking est_tokens up to the Qwen3-VL tokenizer.
