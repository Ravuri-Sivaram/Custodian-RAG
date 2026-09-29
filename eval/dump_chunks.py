"""Scans the evaluation library and produces "unit files" (eval/_units/*.json) + a manifest, for
[Claude sub-agents to construct gold from]. Each unit file = one sub-agent's input, a 1:1 mapping that avoids
index-slicing ambiguity. Three kinds:

  single_NN.json   single-hop: a batch (~10) of independent chunks; the agent produces (question, golden answer) for
                   each one, golden_chunk_ids=[that chunk]
  doc_NN.json      single-doc multi-hop: ~8 chunks from one document; the agent constructs questions that need ≥2
                   chunks combined to answer
  type_N.json      cross-document multi-hop: 2 docs of the same type, ~3 chunks each; the agent constructs questions
                   that need cross-document comparison/synthesis

The chunk source text only lands in these files (sub-agents read them via Read), it never enters the orchestrator's
context. CPU only, no GPU needed.
Run (requires index_eval_corpus.py first):
  CUSTODIAN_EVAL_SRC=~/rag_eval_big CUSTODIAN_EVAL_COLLECTION=evalbig python eval/dump_chunks.py
"""
from __future__ import annotations

import json
import os
import tempfile

from _common import EVAL_COLLECTION, EVAL_SRC, copy_demo, scroll_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = os.path.join(HERE, "_units")

SKIP_KINDS = {"table", "image", "figure", "banner", "caption", "formula", "equation"}
MIN_CHARS = 150
MAX_SRC = 1200
SINGLE_PER_DOC = 4          # single-hop: how many to sample per document
SINGLE_BATCH = 10           # single-hop: how many go into each agent file
DOC_CHUNKS = 8              # single-doc multi-hop: how many chunks to give the agent per document
CROSS_DOCS_PER_TYPE = 2     # cross-document: how many docs to select per type
CROSS_CHUNKS = 3            # cross-document: how many chunks to give per document


def pick_evenly(items: list, k: int) -> list:
    n = len(items)
    if n <= k:
        return items
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def main():
    # Same lock-avoidance approach as gen_gold/run_eval: copytree into a temp copy before opening, so it doesn't
    # contend for the live lock or pollute the original library
    work = os.path.join(tempfile.gettempdir(), "rag_eval_dump")
    qpath, _ = copy_demo(work)
    payloads = scroll_chunks(qpath, EVAL_COLLECTION)
    print(f"scanned {len(payloads)} points (library={EVAL_SRC} coll={EVAL_COLLECTION})", flush=True)

    by_doc: dict[str, list[dict]] = {}
    for p in payloads:
        kind = (p.get("kind") or "text").lower()
        text = (p.get("text") or "").strip()
        if kind in SKIP_KINDS or len(text) < MIN_CHARS:
            continue
        by_doc.setdefault(p.get("doc_id", ""), []).append(p)
    for d in by_doc.values():
        d.sort(key=lambda p: (p.get("page_start", 0), p.get("chunk_id", "")))

    def meta(p):
        return {"chunk_id": p.get("chunk_id"), "doc_id": p.get("doc_id"),
                "doc_type": p.get("doc_type"), "title": (p.get("doc_meta") or {}).get("title") or p.get("doc_id"),
                "text": (p.get("text") or "").strip()[:MAX_SRC], "page": p.get("page_start", 0)}

    if os.path.exists(UNITS):
        import shutil
        shutil.rmtree(UNITS)
    os.makedirs(UNITS)
    manifest = {"singles": [], "docs": [], "types": []}

    # --- single-hop: sample per document -> flatten -> batch ---
    singles = []
    for doc_id, chunks in sorted(by_doc.items()):
        for p in pick_evenly(chunks, SINGLE_PER_DOC):
            singles.append(meta(p))
    for i in range(0, len(singles), SINGLE_BATCH):
        path = os.path.join(UNITS, f"single_{i // SINGLE_BATCH:02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "single", "items": singles[i:i + SINGLE_BATCH]}, f, ensure_ascii=False, indent=1)
        manifest["singles"].append(path)

    # --- single-doc multi-hop: one file per document ---
    for doc_id, chunks in sorted(by_doc.items()):
        if len(chunks) < 3:
            continue
        sel = pick_evenly(chunks, DOC_CHUNKS)
        m = meta(sel[0])
        path = os.path.join(UNITS, f"doc_{len(manifest['docs']):02d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "multi_intra", "doc_id": doc_id, "doc_type": m["doc_type"], "title": m["title"],
                       "chunks": [{"id": meta(p)["chunk_id"], "text": meta(p)["text"]} for p in sel]},
                      f, ensure_ascii=False, indent=1)
        manifest["docs"].append(path)

    # --- cross-document multi-hop: 2 documents of the same type ---
    by_type: dict[str, list[str]] = {}
    for doc_id, chunks in by_doc.items():
        dt = meta(chunks[0])["doc_type"]
        by_type.setdefault(dt, []).append(doc_id)
    for dt, doc_ids in sorted(by_type.items()):
        doc_ids = sorted(doc_ids)[:CROSS_DOCS_PER_TYPE]
        if len(doc_ids) < 2:
            continue
        docs = []
        for doc_id in doc_ids:
            sel = pick_evenly(by_doc[doc_id], CROSS_CHUNKS)
            m = meta(sel[0])
            docs.append({"doc_id": doc_id, "title": m["title"],
                         "chunks": [{"id": meta(p)["chunk_id"], "text": meta(p)["text"]} for p in sel]})
        path = os.path.join(UNITS, f"type_{len(manifest['types'])}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "multi_cross", "doc_type": dt, "docs": docs}, f, ensure_ascii=False, indent=1)
        manifest["types"].append(path)

    with open(os.path.join(UNITS, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"single-hop {len(singles)} candidates -> {len(manifest['singles'])} batch files; "
          f"single-doc multi-hop {len(manifest['docs'])} docs; cross-document {len(manifest['types'])} groups", flush=True)
    print(f"manifest -> {os.path.join(UNITS, 'manifest.json')}", flush=True)


if __name__ == "__main__":
    main()
