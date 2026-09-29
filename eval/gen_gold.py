"""7.0 Fully automatic gold construction: samples text chunks from the demo index (4 docs), has DeepSeek generate
a (question, golden answer) pair for [each chunk] -> gold.jsonl. **Pure CPU + API, no GPU needed** (only scrolls
payloads, doesn't encode).

Why "questions are generated from a single chunk": it keeps golden_chunk_id unambiguous — the question should be
answerable by that passage, so citation/retrieval recall can be judged programmatically, no manual labeling needed.
The cost is that questions skew toward single-hop factual ones (can't test multi-hop), which is exactly the blind
spot that run_eval --mode agentic's dual-layer attribution is meant to fill.

Usage: conda activate custodian && python eval/gen_gold.py [--per-doc 6]
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

from _common import JUDGE_MODEL, JsonLLM, copy_demo, load_env, scroll_chunks

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold.jsonl")

SKIP_KINDS = {"table", "image", "figure", "banner", "caption", "formula", "equation"}
MIN_CHARS = 150          # passages shorter than this can't yield a meaningful factual question
MAX_SRC_CHARS = 1600     # cap on the source text fed to the generator (saves tokens, enough to pose one question)

SYS = (
    "You are a RAG evaluation data generator. Given a source passage from a document, produce a "
    "[specific, factual question that this passage answers clearly and completely] and a "
    "[short, accurate answer based only on this passage (1-3 sentences)]. Requirements: the "
    "question should read naturally, like something a real user would ask -- it must not refer "
    "back to the source with phrases like 'according to this passage/this text', and it must not "
    "be so broad that the passage can't fully answer it. The question and answer must be in the "
    "same language as the source passage (a Telugu passage gets a Telugu question, an English "
    "passage gets an English question). Output a single JSON object only."
)


def pick_evenly(items: list, k: int) -> list:
    """Pick k items evenly spaced from an already-sorted list (deterministic, no randomness), covering the start, middle, and end of the document."""
    n = len(items)
    if n <= k:
        return items
    idx = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=6, help="how many chunks to sample per document (default 6)")
    args = ap.parse_args()

    load_env()
    work = os.path.join(tempfile.gettempdir(), "rag_eval_gold")
    qpath, _ = copy_demo(work)
    payloads = scroll_chunks(qpath)
    print(f"scanned {len(payloads)} points", flush=True)

    by_doc: dict[str, list[dict]] = {}
    for p in payloads:
        kind = (p.get("kind") or "text").lower()
        text = (p.get("text") or "").strip()
        if kind in SKIP_KINDS or len(text) < MIN_CHARS:
            continue
        by_doc.setdefault(p.get("doc_id", ""), []).append(p)

    llm = JsonLLM(model=JUDGE_MODEL)
    rows = []
    for doc_id, chunks in sorted(by_doc.items()):
        chunks.sort(key=lambda p: (p.get("page_start", 0), p.get("chunk_id", "")))
        sample = pick_evenly(chunks, args.per_doc)
        title = ((sample[0].get("doc_meta") or {}).get("title")) or doc_id
        print(f"\n[{doc_id}] eligible chunks {len(chunks)} -> sampled {len(sample)}", flush=True)
        for p in sample:
            src = (p.get("text") or "").strip()[:MAX_SRC_CHARS]
            try:
                qa = llm.ask(SYS, f"Document title: {title}\nSource passage:\n{src}\n\n"
                                  'Output JSON: {"question": "...", "answer": "..."}')
            except Exception as e:
                print(f"  skipped {p.get('chunk_id')}: {e}", flush=True)
                continue
            q, a = (qa.get("question") or "").strip(), (qa.get("answer") or "").strip()
            if len(q) < 5 or len(a) < 2:
                print(f"  skipped {p.get('chunk_id')}: generated Q/A too short", flush=True)
                continue
            rows.append({"query": q, "golden_answer": a,
                         "golden_chunk_id": p.get("chunk_id"), "golden_doc_id": doc_id,
                         "doc_title": title, "source_text": src})
            print(f"  + {q[:60]}", flush=True)

    with open(GOLD, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} rows -> {GOLD}", flush=True)


if __name__ == "__main__":
    main()
