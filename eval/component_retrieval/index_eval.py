"""Evaluation step2 (GPU): indexes 1067 chunks into Qdrant, three vectors in the same library: dense (Qwen3-VL
1024) + bm25 + bge-m3.
Important (learned the hard way): all logic must be inside main()+if __name__=='__main__' — otherwise the child
processes spawned by BGE-M3's multiprocessing will re-import this module as __main__ and re-run the entire script
(infinite re-entry + RuntimeError); also locks bge-m3 to a single GPU via devices='cuda:0' (a single GPU doesn't
spin up a multiprocessing pool)."""
import json
import os
import uuid

EVAL = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from qdrant_client import QdrantClient, models

    from embedder.config import EmbedConfig
    from embedder.dense import Dense
    from embedder.sparse import doc_sparse

    chunks = [json.loads(l) for l in open(os.path.join(EVAL, "chunks.jsonl"), encoding="utf-8")]
    texts = [c["text"] for c in chunks]
    print(f"{len(chunks)} chunks", flush=True)

    # 1) dense (Qwen3-VL 1024 MRL), batched
    dense = Dense(EmbedConfig(dense_dim=1024))
    dvecs = []
    B = 16
    for i in range(0, len(texts), B):
        dvecs.extend(dense.encode_text(texts[i:i + B]))
        if (i // B) % 15 == 0:
            print(f"  dense {min(i + B, len(texts))}/{len(texts)}", flush=True)
    print("dense done", flush=True)

    # 2) bm25 (our doc_sparse)
    bm25 = [doc_sparse(t) for t in texts]

    # 3) bge-m3 sparse (lexical_weights), locked to a single GPU -> no multiprocessing
    from FlagEmbedding import BGEM3FlagModel
    m3 = BGEM3FlagModel(os.path.expanduser("~/models/bge-m3"), use_fp16=True, devices="cuda:0")
    out = m3.encode(texts, batch_size=16, max_length=2048,
                    return_dense=False, return_sparse=True, return_colbert_vecs=False)

    def to_sparse(d):
        items = [(int(t), float(w)) for t, w in (d or {}).items() if float(w) > 0]
        return models.SparseVector(indices=[i for i, _ in items], values=[w for _, w in items]) if items else None

    bgem3 = [to_sparse(d) for d in out["lexical_weights"]]
    print("bge-m3 done", flush=True)

    # 4) qdrant, three vectors in the same library
    client = QdrantClient(path=os.path.join(EVAL, "qdrant"))
    COLL = "eval"
    if client.collection_exists(COLL):
        client.delete_collection(COLL)
    client.create_collection(
        COLL,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
                               "bgem3": models.SparseVectorParams()})
    points = []
    for i, c in enumerate(chunks):
        vec = {"dense": dvecs[i].tolist()}
        if bm25[i] is not None:
            vec["bm25"] = bm25[i]
        if bgem3[i] is not None:
            vec["bgem3"] = bgem3[i]
        points.append(models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["chunk_id"])),
            vector=vec, payload={"chunk_id": c["chunk_id"], "lang": c["lang"], "doc_type": c["doc_type"]}))
    client.upsert(COLL, points=points)
    print(f"indexed {len(points)} -> {EVAL}/qdrant", flush=True)


if __name__ == "__main__":
    main()
