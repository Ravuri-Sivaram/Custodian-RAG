"""Expand corpus: selects a set of table-dense documents from parsed/ that can form thematic clusters, and ingests
them into a larger evaluation library ~/rag_eval_big. All tenant=demo/public (ACL isolation is tested separately by
acl_regression.py); doc_id = directory name, doc_type/lang set per type.

Selection: academic_paper (en, NLP cluster, results tables) / financial_research_te (te, research reports,
table-dense) / financial_report_en (en, earnings reports). Two languages + two strong table-content categories ->
stress-tests both the "table numeric grounding" bottleneck and multi-hop/cross-document evaluation at the same time.

NOTE: this matches directories under parsed/ by NAME PREFIX (financial_research_te__...). If your real parsed/
corpus still has directories prefixed with this project's previous non-English doc_type label, either rename
those directories or change the PLAN dict below to match -- this script has no way to know what's actually
sitting on your disk, and renaming a doc_type label here doesn't rename real files for you.

Run: conda activate custodian && python eval/index_eval_corpus.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)   # the engine package is already installed with pip install -e ., no need for sys.path injection

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir
from embedder import EmbedConfig, Embedder

PARSED = os.path.join(REPO, "parsed")
BIG = os.path.expanduser("~/rag_eval_big")
ACL = {"tenant": "demo", "allow": [], "visibility": "public", "unset": False}

# type -> (lang, how many docs to take)
PLAN = {
    "academic_paper": ("en", 5),
    "financial_research_te": ("te", 6),
    "financial_report_en": ("en", 4),
}


def select() -> list[tuple]:
    """Returns [(parsed_dir, doc_id, doc_type, lang, title)]. Within each type, sorts by name and takes the first N."""
    out = []
    for dt, (lang, n) in PLAN.items():
        dirs = sorted(d for d in os.listdir(PARSED)
                      if d.startswith(dt + "__") and os.path.isdir(os.path.join(PARSED, d)))
        for d in dirs[:n]:
            out.append((d, d, dt, lang, d))   # doc_id=title=directory name (stable, unique)
    return out


def main():
    cfg = EmbedConfig(qdrant_path=os.path.join(BIG, "qdrant"),
                      sidecar_dir=os.path.join(BIG, "sidecar"), dense_dim=1024, collection="evalbig")
    emb = Embedder(cfg)
    docs = select()
    print(f"selected {len(docs)} documents:", flush=True)
    total = 0
    for d, doc_id, dt, lang, title in docs:
        ddir = os.path.join(PARSED, d)
        try:
            els = from_mineru_dir(ddir)
            res = Chunker().chunk(els, doc_id=doc_id, doc_type=dt, lang=lang,
                                  doc_meta={"title": title}, acl=ACL)
            stat = emb.index_document(doc_id, els, res, image_root=ddir)
            total += len(res.chunks)
            print(f"  {dt:22s} {len(res.chunks):4d} chunk  {doc_id[:46]}", flush=True)
        except Exception as e:
            print(f"  skipped {doc_id}: {type(e).__name__}: {e}", flush=True)
    print(f"\nDONE -> {BIG}  total {total} chunks / {len(docs)} documents", flush=True)


if __name__ == "__main__":
    main()
