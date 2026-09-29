"""Generator: the "G" in RAG -- retrieval -> prompt assembly -> LLM -> citation parsing. Both the LLM
and the retriever are dependency-injected (decoupled).

ACL exit gate: Generator only ever consumes what retriever.search_with_context returns (hits have
already passed Qdrant's hard ACL filter, and context has already passed the exit-gate secondary
check) -- it never introduces any new content, so every piece of text fed to the LLM is something the
user is authorized to see. When context is None (the sidecar file is missing or corrupted), it falls
back to the hit chunk's own text (a single chunk is likewise already authorized), so the answer isn't
lost.
"""
from __future__ import annotations

import re

from chunker.chunking import est_tokens
from .llm import LLMClient
from .prompt import CITE_RE, PromptBuilder
from .types import Answer, Citation


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


class Generator:
    def __init__(self, retriever, llm: LLMClient, prompt_builder: PromptBuilder | None = None, acl_check=None,
                 max_context_tokens: int | None = None):
        self.retriever = retriever
        self.llm = llm
        self.pb = prompt_builder or PromptBuilder()
        # Optional (acl:dict, user) -> bool for defense in depth: if not passed, the retriever is
        # trusted to have already done a hard ACL filter. When reusing this with a retriever that
        # doesn't hard-filter and returns raw hits directly, inject this to do a fail-closed secondary
        # ACL check on each hit chunk.
        self.acl_check = acl_check
        # Soft total budget on context size for the closed pipeline (in estimated tokens; whole
        # contexts are trimmed off the tail once the budget is exceeded); None = unlimited (the
        # default, unchanged behavior). This has different semantics from the tool-facing
        # CUSTODIAN_MAX_CONTEXT_TOKENS (in toolcore, which governs what retrieval delivers): this one
        # governs the prompt actually fed to the LLM -- it guards against overflowing the context
        # window (400 errors) or having the backend silently truncate the SYSTEM message when
        # switching to a small-context (8k/32k) backend.
        self.max_context_tokens = max_context_tokens

    def answer(self, query: str, user, top_k=None, rerank: bool = False,
               doc_ids=None, doc_type=None, kind=None, strategy=None, extra_legs=None,
               max_context_tokens: int | None = None) -> Answer:
        # Optional retrieval filters/routing are passed through to the retriever (added after a Custodian
        # review). They're passed only when needed -- unset parameters don't appear in the call at all,
        # so older, narrower-signature retrievers (unit-test mocks / smoke tests) and existing callers
        # (eval) are completely unaffected. Motivation, backed by testing: for questions where "the
        # number is buried in a table," a generic phrasing of the question gets the table chunk pushed
        # out of the top-k by MD&A prose; kind='table' hits it directly.
        kw: dict = {}
        if doc_ids is not None:
            kw["doc_ids"] = doc_ids
        if doc_type is not None:
            kw["doc_type"] = doc_type
        if kind is not None:
            kw["kind"] = kind
        if strategy is not None:
            kw["strategy"] = strategy
        results = list(self.retriever.search_with_context(query, user, top_k=top_k, rerank=rerank, **kw))
        # smart-ask multi-leg retrieval (unioned, not replaced -- a decompose experiment demonstrated
        # that replacing narrows the result set): each leg is one retrieval call with its own
        # filter/rerank parameters, deduped by chunk_id and appended after the main hits (so the main
        # ranking is left undisturbed). Typical usage: add a DEFAULT_TABLE_LEG (signals.py) for numeric
        # questions to recover a table that prose pushed out of the main window.
        if extra_legs:
            seen = {r["hit"].chunk_id for r in results}
            for leg in extra_legs:
                lkw = {k: leg[k] for k in ("doc_ids", "doc_type", "kind", "strategy", "rerank_top_n")
                       if leg.get(k) is not None}
                for r in self.retriever.search_with_context(query, user, top_k=leg.get("top_k"),
                                                            rerank=bool(leg.get("rerank")), **lkw):
                    cid = r["hit"].chunk_id
                    if cid not in seen:
                        seen.add(cid)
                        results.append(r)
        contexts, meta = [], []
        for r in results:
            hit, ctx = r["hit"], r["context"]
            payload = hit.payload or {}
            if self.acl_check is not None and not self.acl_check(payload.get("acl") or {}, user):
                continue                                               # defense in depth: with injection, unauthorized hits don't reach the prompt
            text = (ctx.text if ctx is not None else hit.text) or ""   # prefer the big-block; fall back to the hit chunk if absent (both are already authorized)
            # Table/chart grounding: assemble_big's big-block only takes el.text/caption, which doesn't
            #   include the asset's content_raw -> numbers inside a chart/table (which live in
            #   chunk.content_raw) get excluded, causing the LLM to "retrieve it but still not be able
            #   to answer." When an asset is hit, splice content_raw back in (testing showed a large
            #   share of correctness failures trace back to this). The asset chunk has already passed
            #   ACL itself, and what's being added back is authorized content from that same hit chunk,
            #   so this doesn't cross any authorization boundary.
            if payload.get("kind") in ("chart", "table"):
                craw = (payload.get("content_raw") or "").strip()
                # Short asset data (a single cell or number, e.g. "42") is always spliced back in --
                # a whitespace-stripped substring match would misfire here (e.g. "42" happening to
                # appear in the prose and being treated as a duplicate and suppressed, reproducing the
                # same failure described above). Dedup ("don't re-feed if it's already in text") is
                # only applied to longer content_raw, to avoid re-feeding a long table's HTML twice.
                if craw and (len(_norm(craw)) < 40 or _norm(craw) not in _norm(text)):
                    text = (text + "\n" + craw).strip() if text.strip() else craw
            if not text.strip():
                continue
            dm = payload.get("doc_meta") or {}
            # Scope evidence goes into the prompt (paired with the SYSTEM message's numeric-scope
            # constraint): a financial report's "segment / sub-period" information often only appears
            # in a section heading (not mentioned anywhere in the table body itself) -- if section_path
            # weren't merged into the source line, the model would have no way to know a number is
            # segment-scoped data; testing showed this leads to segment revenue being wrongly
            # generalized into total company revenue (the Custodian N3 case; the root cause is missing
            # scope evidence, not the model failing to follow instructions).
            src = dm.get("title") or hit.doc_id
            sec = payload.get("section_path") or ""
            contexts.append({"text": text, "source": f"{src} § {sec}" if sec else src})
            meta.append({"chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "title": dm.get("title") or hit.doc_id,
                         "section": payload.get("section_path") or "", "page": payload.get("page_start", 0),
                         "text": text})   # falls back to doc_id when title is missing (matches the source fed to the LLM, so provenance never lands on an empty string)
        # Soft total context budget: once exceeded, whole contexts are trimmed off the tail in list
        # order (main hits come first, extra_legs after, giving natural priority), and meta is trimmed
        # in lockstep -- citation numbers are only generated in build(), so they stay aligned naturally
        # after trimming. The first context is always kept: if a single context alone already exceeds
        # the budget, dropping it would degrade into a zero-recall refusal, and that information loss
        # is worse than exceeding the budget slightly (a single context already has its own upper bound
        # from the chunker's BUDGETS). est_tokens approximates using an English-oriented divisor (which
        # underestimates Telugu text), but that's fine for a soft budget.
        budget = max_context_tokens if max_context_tokens is not None else self.max_context_tokens
        if budget and budget > 0 and contexts:
            total, n_keep = 0.0, 0
            for c in contexts:
                total += est_tokens(c["text"], "")
                if total > budget and n_keep >= 1:
                    break
                n_keep += 1
            contexts, meta = contexts[:n_keep], meta[:n_keep]
        messages = self.pb.build(query, contexts)
        if not contexts:   # Zero recall -> deterministically return "not enough information" rather than leaving the refusal fallback up to the LLM's compliance
            return Answer(text="I don't have enough information in the provided context to answer.",
                          citations=[], n_contexts=0, finish_reason=None, raw_messages=messages)
        raw = self.llm.complete(messages)
        # finish_reason is snapshotted into the Answer immediately after complete() returns: the llm
        # instance attribute gets overwritten by later calls (if a smart-ask retry is discarded, the
        # instance is left holding the value from that discarded round), so reading it after the fact
        # would necessarily be misaligned.
        return Answer(text=raw, citations=self._parse_citations(raw, meta), n_contexts=len(contexts),
                      finish_reason=getattr(self.llm, "last_finish_reason", None), raw_messages=messages)

    @staticmethod
    def _parse_citations(answer_text: str, meta: list[dict]) -> list[Citation]:
        """Extracts [n] markers from the answer and maps them back to context sources (keeping only
        markers that are actually cited and have a valid number, deduped and sorted ascending)."""
        # Shares CITE_RE with prompt._neutralize (lenient about whitespace/case): any variant the
        # parser accepts must also be caught by the neutralizer (this symmetry is intentional), so
        # switching to a non-DeepSeek backend that drifts to something like "[cite: 1]" doesn't
        # silently drop citations. Only [cite:n] is recognized; bare [n] appearing in body text is
        # never picked up.
        used = sorted({int(m) for m in CITE_RE.findall(answer_text)})
        out = []
        for n in used:
            if 1 <= n <= len(meta):                  # an out-of-range [n] (a hallucinated number from the LLM) is discarded rather than mapped to the wrong source
                m = meta[n - 1]
                out.append(Citation(marker=n, chunk_id=m["chunk_id"], doc_id=m["doc_id"], title=m["title"],
                                    section=m["section"], page=m["page"], text=m["text"]))
        return out
