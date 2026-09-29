"""Selects candidate chunks for semantic queries from chunks.jsonl (substantial prose passages, cross-document,
mixed Telugu/English), dumps candidates.jsonl for an agent to generate questions from."""
import json
import os
import re

OUT = os.path.dirname(os.path.abspath(__file__))
TELUGU = re.compile(r"[\u0C00-\u0C7F]")

chunks = [json.loads(l) for l in open(os.path.join(OUT, "chunks.jsonl"), encoding="utf-8")]

# select substantial prose passages: 150-900 chars, not pure numeric/table fragments (enough letter or Telugu
# character ratio), capped per document, both languages included
def is_prose(t):
    alpha = sum(1 for c in t if c.isalpha() or TELUGU.match(c))
    return alpha / max(len(t), 1) > 0.6

cand, per_doc = [], {}
for ch in chunks:
    t = ch["text"]
    if not (150 <= len(t) <= 900) or not is_prose(t):
        continue
    if per_doc.get(ch["doc_id"], 0) >= 4:
        continue
    cand.append(ch); per_doc[ch["doc_id"]] = per_doc.get(ch["doc_id"], 0) + 1

# ensure Telugu has a share
te = [c for c in cand if c["lang"] == "te"]
en = [c for c in cand if c["lang"] == "en"]
sel = en[:40] + te[:15]
with open(os.path.join(OUT, "candidates.jsonl"), "w", encoding="utf-8") as f:
    for c in sel:
        f.write(json.dumps({"chunk_id": c["chunk_id"], "lang": c["lang"], "text": c["text"]}, ensure_ascii=False) + "\n")
print(f"semantic candidates: {len(sel)} (en {len([c for c in sel if c['lang']=='en'])}, te {len([c for c in sel if c['lang']=='te'])})")
