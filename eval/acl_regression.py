"""7.C ACL isolation regression (end-to-end, exercises the real retrieval path). The demo library is all-public so it
can't exercise isolation, so this builds a 2-tenant synthetic library instead, ingests it with the real
Chunker+Embedder (real vectors), then runs the Retriever under different identities and asserts that
[cross-tenant / no-permission / unset identities get 0 recall on restricted content].

This is the fail-closed regression floor: it takes each document's [unique sentinel text] as the query and searches
as an unauthorized identity — even an exact match must still return 0 recall (proving ACL hard-filtering happens
before relevance ranking, rather than relying on the luck of "couldn't find it anyway"). It also covers the Batch2/3
document-level direct-read surfaces (get_document/expand fail-closed).

Policy (aligned with acl_admits: cross-tenant is always denied; public is visible only within the same tenant;
visible only if allow ∩ principals is non-empty, or the doc is public):
  docA  t1 / restricted / allow=[g_research]   -> visible only to g_research (same tenant)
  docB  t1 / public                            -> visible to any identity in t1
  docC  t2 / public                            -> visible only to t2 (not visible cross-tenant to t1)
  docD  t1 / restricted / allow=[g_finance]    -> visible only to g_finance
Usage: conda activate custodian && python eval/acl_regression.py   (exit code 0 = all passed, 1 = leakage found)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)   # the engine package is already installed with pip install -e ., no need for sys.path injection

from chunker import Chunker
from chunker.types import Element
from embedder import EmbedConfig, Embedder, Retriever, User

# Per document: title + unique sentinel body text (used as the query for an exact hit, to verify that an
# unauthorized identity can't recall it even with an exact match)
DOCS = {
    "docA": ("t1", "restricted", ["g_research"],
             "Project Alpha Confidential",
             "Project Alpha reported secret quarterly revenue of forty two million dollars in the restricted research ledger."),
    "docB": ("t1", "public", [],
             "Bravo Public Roadmap",
             "Bravo public roadmap announces the next-year product launch open to everyone in tenant one."),
    "docC": ("t2", "public", [],
             "Charlie Tenant-Two Memo",
             "Charlie memo belongs to tenant two and discusses an unrelated logistics reorganization plan."),
    "docD": ("t1", "restricted", ["g_finance"],
             "Delta Finance Only",
             "Delta finance-only confidential gross margin data shows a sensitive eighty seven percent figure."),
}

# identity -> the set of docs that should be visible to it (everything else must be 0 recall)
USERS = {
    "u_research(t1,g_research)": (User(tenant="t1", principals=["g_research"]), {"docA", "docB"}),
    "u_finance(t1,g_finance)":   (User(tenant="t1", principals=["g_finance"]), {"docB", "docD"}),
    "u_t2(t2,-)":                (User(tenant="t2", principals=[]), {"docC"}),
    "u_noperm(t1,g_none)":       (User(tenant="t1", principals=["g_none"]), {"docB"}),
    "u_unset('',-)":             (User(tenant="", principals=[]), set()),
}


def build_doc(title: str, body: str) -> list[Element]:
    return [Element(idx=0, kind="text", text=title, text_level=1, page=1),
            Element(idx=1, kind="text", text=body, page=1)]


def main():
    work = os.path.join(tempfile.gettempdir(), "rag_acl_regression")
    if os.path.exists(work):
        shutil.rmtree(work, ignore_errors=True)
    cfg = EmbedConfig(qdrant_path=os.path.join(work, "qdrant"),
                      sidecar_dir=os.path.join(work, "sidecar"), dense_dim=1024, collection="aclreg")
    emb = Embedder(cfg)
    for doc_id, (tenant, vis, allow, title, body) in DOCS.items():
        acl = {"tenant": tenant, "allow": allow, "visibility": vis, "unset": False}
        els = build_doc(title, body)
        res = Chunker().chunk(els, doc_id=doc_id, doc_type="memo", lang="en",
                              doc_meta={"title": title}, acl=acl)
        emb.index_document(doc_id, els, res, image_root=None)
    retr = Retriever(cfg, store=emb.store, dense=emb.dense)

    fails = []

    def check(desc, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}", flush=True)
        if not ok:
            fails.append(desc)

    # 1) Recall isolation matrix: for every identity x every doc's sentinel query -> the set of hit docs must be a subset of the authorized set
    print("=== 1) search recall isolation (exact sentinel queries) ===", flush=True)
    for uname, (user, allowed) in USERS.items():
        for doc_id, (_t, _v, _a, _title, body) in DOCS.items():
            hits = retr.search(body, user, top_k=10)
            got = {h.doc_id for h in hits}
            leaked = got - allowed
            check(f"{uname} searching《{doc_id}》text -> hits {sorted(got) or '∅'}; authorized {sorted(allowed) or '∅'}"
                  + (f"  ⚠LEAK {sorted(leaked)}" if leaked else ""),
                  not leaked and (doc_id in got if doc_id in allowed else True))

    # 2) Authorization positive check: an authorized identity must be able to find what it's allowed to see (guards against a "deny everything" false pass)
    print("\n=== 2) authorization positive check (guards against deny-everything false pass) ===", flush=True)
    for uname, (user, allowed) in USERS.items():
        for doc_id in allowed:
            body = DOCS[doc_id][4]
            got = {h.doc_id for h in retr.search(body, user, top_k=10)}
            check(f"{uname} should be able to recall authorized《{doc_id}》", doc_id in got)

    # 3) doc-level direct-read fail-closed: get_document on a restricted doc by an unauthorized identity -> PermissionError
    print("\n=== 3) get_document direct-read fail-closed ===", flush=True)
    for uname, (user, allowed) in USERS.items():
        for doc_id in DOCS:
            if doc_id in allowed:
                continue
            try:
                retr.get_document(doc_id, user)
                check(f"{uname} get_document《{doc_id}》(unauthorized) should be denied", False)
            except PermissionError:
                check(f"{uname} get_document《{doc_id}》(unauthorized) -> PermissionError", True)

    # 4) expand cross-ACL fail-closed: use docA's real chunk_id, expand as an unauthorized identity -> None
    print("\n=== 4) expand cross-ACL fail-closed ===", flush=True)
    owner = USERS["u_research(t1,g_research)"][0]
    a_hits = retr.search(DOCS["docA"][4], owner, top_k=5)
    a_cid = next((h.chunk_id for h in a_hits if h.doc_id == "docA"), None)
    check("could fetch docA's chunk_id (precondition)", a_cid is not None)
    if a_cid:
        for uname in ("u_finance(t1,g_finance)", "u_t2(t2,-)", "u_noperm(t1,g_none)", "u_unset('',-)"):
            user = USERS[uname][0]
            check(f"{uname} expand(docA.chunk) (unauthorized) -> None", retr.expand(a_cid, user) is None)

    # 5) Isolation for RRF fusion prefetch pushdown (checks the fusion-level filter, not just the exit gate): disable the
    #    exit-side acl_admits recheck (force it to always return True), leaving only acl_filter's prefetch-level
    #    pushdown. If cross-tenant/unauthorized access still yields 0 recall, that proves the fusion pushdown *itself*
    #    blocks it (rather than being masked by the exit gate), locking in the fix for the bug where the embedded
    #    QdrantLocal fusion dropped the top-level `should`. The English sentinel text has sparse tokens -> exercises
    #    the RRF fusion of both the dense+sparse prefetch routes.
    print("\n=== 5) RRF fusion prefetch pushdown isolation (must still be 0 leakage with the exit gate disabled) ===", flush=True)
    import embedder.store as _store
    _orig_admits = _store.acl_admits
    _store.acl_admits = lambda acl, user: True          # disable the exit-side recheck
    try:
        for uname, (user, allowed) in USERS.items():
            for doc_id, (_t, _v, _a, _title, body) in DOCS.items():
                got = {h.doc_id for h in retr.search(body, user, top_k=10)}   # default hybrid=RRF fusion
                leaked = got - allowed
                check(f"[exit gate disabled] {uname} searching《{doc_id}》-> hits {sorted(got) or '∅'}"
                      + (f"  ⚠fusion pushdown did not block it, leaked {sorted(leaked)}" if leaked else ""), not leaked)
    finally:
        _store.acl_admits = _orig_admits                # restore, doesn't affect anything afterward

    print(f"\n{'='*48}\n{'ALL PASSED ✓' if not fails else f'{len(fails)} leak(s)/failure(s) ✗'}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
