"""Evaluation benchmark step1 (CPU): select corpus -> chunk -> dump chunks.jsonl + programmatic exact-term
queries queries_exact.jsonl. Exact-term queries: mine rare exact strings from chunks (statute/clause numbers,
amounts, model numbers), use a string with df==1 as the query and the chunk containing it as golden — this tests
sparse's exact-match hits (BM25's strength). Semantic queries are generated separately by an agent."""
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir

TELUGU = re.compile(r"[\u0C00-\u0C7F]")
PICK_TYPES = {"academic_paper": 3, "law": 3, "financial_report_en": 3,
              "financial_research_te": 3, "government": 2}

# Candidate exact strings (statute/clause numbers, amounts, model numbers, long IDs) — all strings that are
# only correct if matched character-for-character
PATTERNS = [
    re.compile(r"\bSection\s+\d+(?:\.\d+)+\b"),
    re.compile(r"\bArticle\s+\d+\b"),
    re.compile(r"第\s*\d+\s*条"),
    re.compile(r"\b[A-Z][A-Za-z]{1,}-\d+(?:\.\d+)?\b"),       # things like GPT-4, BERT-3, v1.2
    re.compile(r"\$\s?\d[\d,]{3,}(?:\.\d+)?"),                # amounts like $1,234
    re.compile(r"\b\d{6,}\b"),                                 # long IDs / large numbers
]


def is_telugu(t):
    return len(TELUGU.findall(t)) / max(len(t), 1) > 0.2


# 1) select corpus + chunk
docs = sorted(glob.glob(os.path.join(REPO, "parsed", "*")))
picked, counts = [], {}
for d in docs:
    if not os.path.isdir(d):
        continue
    dtype = os.path.basename(d).split("__")[0]
    if dtype in PICK_TYPES and counts.get(dtype, 0) < PICK_TYPES[dtype]:
        picked.append(d); counts[dtype] = counts.get(dtype, 0) + 1

chunks = []
for d in picked:
    name = os.path.basename(d)
    try:
        els = from_mineru_dir(d)
        res = Chunker().chunk(els, doc_id=name, doc_type=name.split("__")[0],
                              lang="te" if "_te" in name else "en")
    except Exception as e:
        print("skip", name, repr(e)[:60]); continue
    for c in res.chunks:
        t = (c.text or "").strip()
        if len(t) < 30:
            continue
        chunks.append({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_type": c.doc_type or name.split("__")[0],
                       "lang": "te" if is_telugu(t) else "en", "section_id": c.section_id, "text": t})

with open(os.path.join(OUT, "chunks.jsonl"), "w", encoding="utf-8") as f:
    for ch in chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")
print(f"corpus: {len(picked)} documents -> {len(chunks)} chunks (Telugu {sum(c['lang']=='te' for c in chunks)})")

# 2) exact-term queries: mine rare exact strings with df==1
df = {}                                   # string -> set(chunk_id)
for ch in chunks:
    seen = set()
    for pat in PATTERNS:
        for m in pat.findall(ch["text"]):
            s = m if isinstance(m, str) else m[0]
            s = s.strip()
            if len(s) >= 3:
                seen.add(s)
    for s in seen:
        df.setdefault(s, set()).add(ch["chunk_id"])

exact = [{"query": s, "golden": list(ids)[0], "kind": "exact"}
         for s, ids in df.items() if len(ids) == 1]
# take at most 1 exact query per golden chunk (avoids one chunk flooding the list), and cap the total
by_golden, dedup = {}, []
for q in exact:
    if by_golden.get(q["golden"], 0) < 1:
        dedup.append(q); by_golden[q["golden"]] = 1
exact = dedup[:80]

with open(os.path.join(OUT, "queries_exact.jsonl"), "w", encoding="utf-8") as f:
    for q in exact:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"exact-term queries: {len(exact)} (df==1 rare strings)")
print("examples:", [q["query"] for q in exact[:12]])
