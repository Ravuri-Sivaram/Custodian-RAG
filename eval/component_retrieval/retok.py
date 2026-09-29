"""#3 tokenizer re-calibration: uses the real Qwen3-VL tokenizer to measure the true char/token ratio for
Telugu/English chunks, comparing against est_tokens's current heuristic (Telugu 1.0 placeholder / English 4.0), to
decide whether BUDGETS needs re-calibrating. Pure CPU (only uses the tokenizer). This is exactly the measurement
the Telugu divisor in chunking.py's est_tokens needs before it can be trusted -- run this against real Telugu
documents and update the divisor from whatever this reports."""
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir

MODEL = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
TELUGU = re.compile(r"[\u0C00-\u0C7F]")


def is_telugu(t):
    return len(TELUGU.findall(t)) / max(len(t), 1) > 0.2


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

docs = sorted(glob.glob(os.path.join(REPO, "parsed", "*")))
pick = [d for d in docs if os.path.isdir(d) and any(
    k in os.path.basename(d) for k in
    ["academic_paper", "financial_research_te", "law__", "financial_report_en", "government", "policy"])]

agg = {"te": [0, 0, 0], "en": [0, 0, 0]}   # [n_chunk, total_char, total_token]
per_type = {}
for d in pick:
    name = os.path.basename(d)
    dtype = name.split("__")[0]
    try:
        els = from_mineru_dir(d)
        res = Chunker().chunk(els, doc_id=name, lang="te" if "_te" in name else "en")
    except Exception as e:
        print("skip", name, repr(e)[:60]); continue
    for c in res.chunks:
        t = (c.text or "").strip()
        if len(t) < 20:
            continue
        n = len(tok.encode(t, add_special_tokens=False))
        lang = "te" if is_telugu(t) else "en"
        agg[lang][0] += 1; agg[lang][1] += len(t); agg[lang][2] += n
        pt = per_type.setdefault(dtype, {"te": [0, 0, 0], "en": [0, 0, 0]})
        pt[lang][0] += 1; pt[lang][1] += len(t); pt[lang][2] += n

print("\n=== char/token ratio (= est_tokens's divisor) ===")
for lang, cur in (("te", 1.0), ("en", 4.0)):
    n, ch, tk = agg[lang]
    if tk:
        r = ch / tk
        print(f"{lang}: n={n:5d} chunks  measured char/token={r:.3f}  current heuristic={cur}  "
              f"deviation={'underestimate' if r > cur else 'overestimate'} {abs(r-cur)/cur*100:.0f}%")

print("\n=== by document type (char/token) ===")
for dtype, d in sorted(per_type.items()):
    parts = []
    for lang in ("te", "en"):
        n, ch, tk = d[lang]
        if tk and n >= 3:
            parts.append(f"{lang} {ch/tk:.2f}(n={n})")
    if parts:
        print(f"  {dtype:24} {'  '.join(parts)}")
