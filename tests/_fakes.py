"""Test doubles: fake retriever / user / app constructors (pure CPU, doesn't touch
Qdrant/GPU/network). Shape aligned with tests/engine/test_tools.py's _MockRet (same
duck-typing contract)."""
from __future__ import annotations

from types import SimpleNamespace

from custodian import config as pconfig
from custodian.service import create_app


def make_hit(cid="c1", doc="d1", title="Title A", section="Chapter 1 > 1.1", score=0.83,
             text="raw hit chunk text", kind="text", **payload_extra):
    payload = {"doc_meta": {"title": title}, "section_path": section,
               "page_start": 4, "page_end": 4, "kind": kind,
               "acl": {"tenant": "t1", "allow": [], "visibility": "public", "unset": False}}
    payload.update(payload_extra)
    return SimpleNamespace(chunk_id=cid, doc_id=doc, text=text, score=score, kind=kind, payload=payload)


def make_res(hit, ctx_text=None, status="full_section", anchor=None, n_tokens=42):
    ctx = None
    if ctx_text is not None:
        ctx = SimpleNamespace(text=ctx_text, anchor=anchor, resolved_section="sec#x",
                              n_tokens=n_tokens, climbed=0)
    return {"hit": hit, "context": ctx, "context_status": status}


class FakeRetriever:
    """Duck-typing aligned with the surface of embedder.Retriever that toolcore/Generator consume.
    results_factory produces fresh results on each call (toolcore._demote mutates the dict in
    place, so reusing the same object would leak state between calls)."""

    def __init__(self, results_factory=None, docs=None, document=None, outline=None,
                 expand_big=None, grouped=None):
        self._rf = results_factory or (lambda: [])
        self._document, self._outline = document, outline
        self._expand, self._grouped = expand_big, grouped
        self.calls: list = []
        self.store = SimpleNamespace(list_documents=lambda user: list(docs or []))

    def search_with_context(self, query, user, top_k=None, rerank=False, doc_ids=None,
                            doc_type=None, kind=None, assemble=True, strategy="hybrid", rerank_top_n=None):
        self.calls.append({"query": query, "top_k": top_k, "rerank": rerank, "assemble": assemble,
                           "doc_ids": doc_ids, "doc_type": doc_type, "kind": kind, "strategy": strategy,
                           "user_tenant": getattr(user, "tenant", None),        # multi-identity: proves identity flows through to the engine
                           "user_principals": list(getattr(user, "principals", []) or [])})
        return self._rf()

    def get_document(self, doc_id, user, max_tokens=6000):
        if self._document is None:
            raise PermissionError
        return dict(self._document)

    def get_outline(self, doc_id, user):
        if self._outline is None:
            raise PermissionError
        return self._outline

    def expand(self, chunk_id, user, target_tokens=1500):
        return self._expand

    def search_grouped(self, query, user, doc_ids, top_k=3, rerank=False):
        return self._grouped or {}


def make_user(tenant="t1", principals=("g",)):
    return SimpleNamespace(tenant=tenant, principals=list(principals))


def make_cfg(**kw):
    kw.setdefault("tenant", "t1")
    return pconfig.CustodianConfig(**kw)


def make_app(retriever=None, user=None, cfg=None, generator_factory=None, keys=None):
    return create_app(cfg=cfg or make_cfg(), retriever=retriever or FakeRetriever(),
                      user=user or make_user(), generator_factory=generator_factory, keys=keys)
