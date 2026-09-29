"""gold table-question supplement: targeted sampling of kind=table chunks from the eval index, has DeepSeek generate
numeric questions from content_raw (the table body).

**Why a separate script**: gen_gold.py explicitly SKIPped tables back then — table chunk text was caption-only at
the time, so no question could be formed. Only after the table-retrieval text enhancement (chunker 2bd97a5) did
content_raw become genuinely usable as question material; and gold's all-prose bias was empirically found to create
a measurement blind spot where "table-facing changes only show up as cost, never benefit" (see custodian
TESTING §3).

**Programmatic QC (guards against hallucinated questions)**: the key numeric value in golden_answer must appear
verbatim in the table body (substring match after comma normalization); if not, the question is discarded outright
— a number the question-writer made up doesn't belong on the exam.

Usage: conda activate custodian && python eval/gen_gold_tables.py [--per-doc 2 --cap 16]
Produces: gold_tables.jsonl (kept separately) + appended to gold.jsonl (the original 72 questions are backed up
first as gold_backup_prose72.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile

from _common import EVAL_COLLECTION, JUDGE_MODEL, JsonLLM, copy_demo, load_env, scroll_chunks
from gen_gold import pick_evenly

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold.jsonl")
GOLD_TABLES = os.path.join(HERE, "gold_tables.jsonl")
BACKUP = os.path.join(HERE, "gold_backup_prose72.jsonl")

MIN_RAW = 120            # a table this small can't yield a meaningful numeric question
MAX_RAW = 3000           # cap on the table body fed to the question generator

SYS = (
    "You are a RAG evaluation data generator. Given a table (HTML) and the document/section it "
    "appears in, produce a [specific question that can only be answered using a value from the "
    "table] and a [short, accurate answer]. Requirements: 1) the answer must contain a specific "
    "value from the table (with its unit), copied verbatim from the table; 2) the question must "
    "specify scope (which year / which segment / which metric) to avoid ambiguity; 3) the question "
    "should read naturally, like something a real user would ask -- it must not use phrases like "
    "'according to this table'; 4) the question and answer must be in the same language as the "
    "document (a Telugu research report gets a Telugu question, an English financial report gets "
    "an English question). Output a single JSON object only."
)

_NUM_RE = re.compile(r"\d[\d,\.]*\d|\d")


def _norm_nums(s: str) -> str:
    return (s or "").replace(",", "").replace(" ", "")


def answer_grounded(answer: str, raw: str) -> bool:
    """At least one >=2-digit number in the answer (after removing commas) must appear in the table body — a programmatic gate against hallucinated questions."""
    raw_n = _norm_nums(raw)
    nums = [_norm_nums(m) for m in _NUM_RE.findall(answer or "")]
    nums = [n for n in nums if len(n) >= 2]
    return any(n in raw_n for n in nums) if nums else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-doc", type=int, default=2)
    ap.add_argument("--cap", type=int, default=16)
    args = ap.parse_args()

    load_env()
    work = os.path.join(tempfile.gettempdir(), "rag_eval_gold_tables")
    qpath, _ = copy_demo(work)
    payloads = scroll_chunks(qpath, collection=EVAL_COLLECTION)

    by_doc: dict[str, list[dict]] = {}
    for p in payloads:
        if (p.get("kind") or "").lower() != "table":
            continue
        raw = (p.get("content_raw") or "").strip()
        if len(raw) < MIN_RAW:
            continue
        by_doc.setdefault(p.get("doc_id", ""), []).append(p)
    print(f"documents with eligible tables: {len(by_doc)} / table chunks: {sum(len(v) for v in by_doc.values())}", flush=True)

    llm = JsonLLM(model=JUDGE_MODEL)
    rows = []
    for doc_id, chunks in sorted(by_doc.items()):
        if len(rows) >= args.cap:
            break
        chunks.sort(key=lambda p: (p.get("page_start", 0), p.get("chunk_id", "")))
        title = ((chunks[0].get("doc_meta") or {}).get("title")) or doc_id
        for p in pick_evenly(chunks, args.per_doc):
            if len(rows) >= args.cap:
                break
            raw = (p.get("content_raw") or "").strip()[:MAX_RAW]
            sec = p.get("section_path") or ""
            try:
                qa = llm.ask(SYS, f"Document title: {title}\nSection: {sec}\nTable HTML:\n{raw}\n\n"
                                  'Output JSON: {"question": "...", "answer": "..."}')
            except Exception as e:
                print(f"  skipped {p.get('chunk_id')}: {e}", flush=True)
                continue
            q, a = (qa.get("question") or "").strip(), (qa.get("answer") or "").strip()
            if len(q) < 5 or len(a) < 2:
                print(f"  skipped {p.get('chunk_id')}: Q/A too short", flush=True)
                continue
            if not answer_grounded(a, raw):
                print(f"  ✗ QC rejected (answer's number not in the table) {p.get('chunk_id')}: {a[:50]}", flush=True)
                continue
            rows.append({"query": q, "golden_answer": a,
                         "golden_chunk_ids": [p.get("chunk_id")], "golden_doc_ids": [doc_id],
                         "hop": "single", "doc_type": p.get("doc_type") or "", "asset": "table"})
            print(f"  + [{doc_id[:36]}] {q[:56]}", flush=True)

    with open(GOLD_TABLES, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not os.path.exists(BACKUP):
        shutil.copy(GOLD, BACKUP)                          # keep the original 72 questions on record, backed up only once
    with open(GOLD, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} table questions -> {GOLD_TABLES} and appended to {GOLD} (original backed up to {BACKUP})", flush=True)


if __name__ == "__main__":
    main()
