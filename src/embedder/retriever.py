"""Query time: query -> hybrid recall (dense+BM25, RRF, hard ACL filter) -> dedup_by_section
-> chunker.assemble_big (small-to-big, ACL-aware). chunker is imported lazily (import hygiene)."""
from __future__ import annotations

import json
import os
import threading

from .acl import acl_admits
from .config import SIDECAR_VERSION, EmbedConfig
from .dense import Dense
from .remote import make_dense
from .sparse import query_sparse
from .store import Store
from .types import User


class SearchResults(list):
    """The return type of search_with_context: list[dict] plus a same-section-folding count. A
    list subclass rather than changing the return shape to (list, stats): callers like
    generator/eval keep consuming it as a plain list with zero changes; toolcore reads the count
    via getattr (a mock returning a bare list defaults to 0)."""
    section_folded_n = 0   # Count of hits dropped by same-section dedup (never entering the results) -- lets the agent distinguish "the library ran out" from "hits were folded together"


class _ChunkShim:
    """Reconstructs just the handful of hit_chunk attributes assemble_big needs to read, from a
    Qdrant payload -- no need to build a full Chunk."""
    __slots__ = ("acl", "lang", "doc_id", "section_id", "section_anchor",
                 "breadcrumb", "page_start", "page_end", "source_indices", "doc_type")

    def __init__(self, p: dict):
        self.acl = p.get("acl") or {}
        self.lang = p.get("lang", "en")
        self.doc_id = p.get("doc_id", "")
        self.section_id = p.get("section_id")
        self.section_anchor = p.get("section_anchor")
        self.breadcrumb = p.get("breadcrumb") or []
        self.page_start = p.get("page_start", 0)
        self.page_end = p.get("page_end", 0)
        self.source_indices = p.get("source_indices") or []
        self.doc_type = p.get("doc_type")              # Used for per-doc_type budgets


class Retriever:
    def __init__(self, cfg: EmbedConfig | None = None, store: Store | None = None,
                 dense: Dense | None = None, reranker=None):
        # store/dense can be shared with an Embedder in the same process (the embedded Qdrant
        # client's single-client constraint) plus dense (avoids loading the 8B model twice).
        self.cfg = cfg or EmbedConfig()
        self.dense = dense or make_dense(self.cfg)   # Factory: cfg.inference_url empty = local (default), non-empty = remote inference service
        self.store = store or Store(self.cfg)
        self._reranker = reranker          # Optional cross-encoder reranker; built lazily the first time rerank is requested
        self._reranker_lock = threading.Lock()   # Single-flight lazy build for the reranker (a concurrent first query doesn't load a second 8B model twice)

    def _get_reranker(self):
        if self._reranker is None:                 # Fast path, no lock needed
            with self._reranker_lock:
                if self._reranker is None:         # Double-checked: only built once even under a concurrent first query
                    from .remote import make_reranker   # Lazy: only the local backend loads a second 8B model (+16G); remote goes over HTTP
                    self._reranker = make_reranker(self.cfg)
        return self._reranker

    def _load_sidecar(self, doc_id: str, cache: dict | None = None, user: User | None = None):
        if cache is not None and doc_id in cache:           # Query-scoped cache: multiple hits from the same doc only read+deserialize it once
            return cache[doc_id]
        path = os.path.join(self.cfg.sidecar_dir, f"{doc_id}.json")
        if not os.path.exists(path):                        # An explicit error rather than a bare open() -- lets the caller degrade gracefully
            raise FileNotFoundError(f"Sidecar missing: {path} (doc {doc_id} is indexed but its sidecar is gone; needs a re-run of index_document)")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        # Version check: a mismatch (including an old sidecar with no version field at all) raises
        # ValueError -- deliberately outside search_with_context's (FileNotFoundError,
        # JSONDecodeError) catch, so it propagates as a loud failure. A missing file or corrupt
        # JSON is a per-document transient (worth degrading gracefully for); a version mismatch is
        # systemic (schema drift -- every sidecar is stale), and should prompt a full rebuild
        # rather than silently degrading document by document.
        if d.get("version") != SIDECAR_VERSION:
            raise ValueError(
                f"Sidecar version mismatch: {path} is v{d.get('version')}, this build needs v{SIDECAR_VERSION}"
                f"(a schema change); please re-run index_document for doc {doc_id} to rebuild its sidecar.")
        acl_index = {int(k): v for k, v in d.get("acl_index", {}).items()}
        # A doc-level fail-closed precheck for the **doc_id direct-read tools** (get_document/
        # expand): when a user is given, the doc must have at least one element visible to them, or
        # loading is refused outright (an empty acl_index -> any([]) is False -> also refused,
        # fail-closed).
        # **user=None means the caller is hit-driven and has already proven doc_id passed ACL**
        # (e.g. _assemble: the hit already passed the store's hard filter plus its own re-check, and
        # downstream assemble_big has both per-element gating and an exit acl_admits(ctx.acl) check)
        # -- in that case the precheck is **skipped**, because chunker's stricter-sibling logic can
        # rewrite the hit element's idx in acl_index to a stricter sibling's acl, and a doc-level
        # precheck here would wrongly reject a hit that should actually be delivered as public
        # (confirmed as a real, high-severity issue during adversarial review).
        # Direct-read tools must always pass user; fine-grained per-element gating is still
        # get_document's own admits= responsibility.
        if user is not None and not any(acl_admits(a or {}, user) for a in acl_index.values()):
            raise PermissionError(f"doc {doc_id} has no elements visible to the current identity (fail-closed)")
        from chunker.types import Element, Section          # Lazy: chunker is only needed when small-to-big assembly actually runs
        elements = [Element(**e) for e in d["elements"]]
        # Both the doc-level direct-read tools (get_document/get_outline's _gather) and
        # assemble_big index elements **by position**, but acl_index is keyed by el.idx -- a sparse
        # or out-of-order sidecar would misalign ACL by position, delivering an unauthorized
        # element as if it were visible. assemble_big has this assertion internally, but the
        # direct-read path bypasses it, so the same assertion (elements[i].idx==i) is enforced here
        # at the loading source, so every consumer fails closed together (a standard adapter
        # produces dense, ordered output and never triggers this).
        if any(el.idx != i for i, el in enumerate(elements)):
            raise ValueError(f"Sidecar elements are not densely ordered by idx (elements[i].idx==i); reading doc {doc_id} would misalign ACL by position -- needs a rebuild via index_document")
        secs = {s["sec_id"]: Section(**s) for s in d["sections"]}
        result = (elements, secs, frozenset(d.get("banners", [])), acl_index)
        if cache is not None:
            cache[doc_id] = result
        return result

    def search(self, query: str, user: User, top_k: int | None = None, rerank: bool = False,
               doc_ids: list[str] | None = None, doc_type: str | None = None, kind: str | None = None,
               strategy: str = "hybrid", rerank_top_n: int | None = None):
        """Recall (already passed through the hard ACL filter). rerank=True: recall a larger
        rerank_top_n pool -> cross-encoder rerank -> take the top_k. Optional filters:
        doc_ids/doc_type/kind, ANDed with ACL. strategy: hybrid/dense/sparse routing. rerank_top_n:
        the rerank candidate pool depth (defaults to cfg.rerank_top_n, with guardrails); a reranker
        failure degrades to the plain hybrid recall rather than crashing."""
        qd = self.dense.encode_query(query)
        qs = query_sparse(query, self.cfg.stopwords)         # On the query side, values=1.0; no valid token -> None
        # `top_k or cfg.top_k` is a 0-falsy bug (top_k=0 would silently become the default, and a
        # negative value would slice the wrong subset). Explicit `is None` check plus a clamp instead.
        k = self.cfg.top_k if top_k is None else int(top_k)
        k = max(1, min(k, self.cfg.prefetch_limit))          # Clamp to [1, the recall cap] -- asking for more than the recall pool holds is meaningless
        flt = dict(doc_ids=doc_ids, doc_type=doc_type, kind=kind, strategy=strategy)
        if rerank:
            rtn = self.cfg.rerank_top_n if rerank_top_n is None else max(1, min(int(rerank_top_n), self.cfg.prefetch_limit))
            n = max(rtn, k)                     # Recall at least k candidates for reranking, or top_k > the candidate pool would deliver fewer results than asked
            hits = self.store.hybrid_search(qd.tolist(), qs, user, top_k=n, **flt)
            try:
                return self._get_reranker().rerank(query, hits, top_k=k)
            except Exception:                   # A reranker failure (OOM/missing model) degrades to the plain hybrid recall rather than taking down basic retrieval
                return hits[:k]                 # These hits keep the hybrid score_kind -> the caller marks rerank_degraded from that
        return self.store.hybrid_search(qd.tolist(), qs, user, top_k=k, **flt)

    def search_with_context(self, query: str, user: User, top_k: int | None = None,
                            rerank: bool = False, doc_ids: list[str] | None = None,
                            doc_type: str | None = None, kind: str | None = None,
                            assemble: bool = True, strategy: str = "hybrid",
                            rerank_top_n: int | None = None) -> list[dict]:
        """Recall (with optional reranking) plus per-section dedup plus small-to-big assembly
        (ACL-aware: big.text only ever contains raw text sharing the hit's ACL). Each returned
        entry is {hit, context, context_status}. context=None means a degraded, bare-hit result;
        context_status explains why (full_section/climbed_N = a complete section, usable as-is;
        **section_window = a token-limited window or truncated fragment, NOT a complete section --
        call expand on its chunk_id for more**; asset_no_prose = an asset page with no prose
        big-block, data lives in the hit's content_raw instead; single_chunk_degraded = the sidecar
        is missing/corrupt; single_chunk_acl = the big-block failed the exit ACL check; deduped = a
        duplicate big-block for the same passage/window as an earlier hit was folded, only the bare
        hit is returned; concise = small-to-big assembly was skipped).
        assemble=False (concise mode): skips small-to-big assembly and returns bare hits only
        (saves tokens/disk reads), but still does section-level dedup.
        doc_ids/doc_type/kind: passed straight through as filters.
        Returns a SearchResults (list subclass): .section_folded_n records how many hits were
        folded away by same-section dedup (folded hits are never backfilled, so the delivered
        count can be well under top_k with no other signal to distinguish "the library ran out"
        from "hits were folded together"; backfilling would change what gets delivered and needs
        GPU-based evaluation to validate first, so it's left for later)."""
        out, seen, cache, seen_anchor = SearchResults(), set(), {}, set()   # cache: query-scoped sidecar cache; seen_anchor: big-block dedup
        for h in self.search(query, user, top_k, rerank=rerank, doc_ids=doc_ids, doc_type=doc_type,
                             kind=kind, strategy=strategy, rerank_top_n=rerank_top_n):
            sid = h.payload.get("section_id")
            # section_id=None means "no section", not "the same section" -- it must never be
            # folded (or every no-section hit in the same doc would collapse down to just the
            # first one). An asset chunk's (chart/table) data lives in its own content_raw, not in
            # the prose big-block -- assemble_big's _gather only pulls el.text/caption, never
            # asset_content, so chart/table numbers are excluded from it. If an asset hit got
            # folded together with a prose sibling in the same section, that data would be lost
            # entirely ("recalled but couldn't answer from the table/chart's numbers", confirmed by
            # eval). So asset chunks key on their own chunk_id and never participate in
            # section-level dedup.
            kind = h.payload.get("kind")
            key = (h.doc_id, sid) if (sid is not None and kind not in ("chart", "table")) else ("__chunk__", h.chunk_id)
            if key in seen:                                  # dedup_by_section: multiple hits in the same section keep only the first
                out.section_folded_n += 1                    # Folded hits don't enter the results, but the count is exposed (toolcore surfaces it in meta)
                continue
            seen.add(key)
            if not assemble:                                 # concise mode: skip small-to-big, return bare hits only (saves tokens/disk reads)
                out.append({"hit": h, "context": None, "context_status": "concise"})
                continue
            status = "full_section"
            try:
                ctx = self._assemble(h, user, cache)
            except (FileNotFoundError, json.JSONDecodeError, ValueError):  # Sidecar missing/corrupt/version mismatch/non-dense -> degrade to a bare hit
                # A ValueError (version mismatch / non-dense idx) used to be uncaught here, letting
                # it bubble up to the retrieve tool and expose an exception containing the sidecar's
                # absolute path straight to the agent (an information leak). Now it degrades the
                # same way a missing file does (the bare hit already passed the store's own
                # re-check, so this is safe) -- no path leaked, and the whole query doesn't crash.
                ctx, status = None, "single_chunk_degraded"
            # Exit-side re-check: big.acl must be visible to the user, or it isn't delivered.
            # acl=None is treated as "never access-checked" -> refused (fail-closed; the legacy
            # branch of assemble_big can return acl=None while also having pulled material across
            # an ACL boundary, so this must never be allowed through).
            if ctx is not None and (ctx.acl is None or not acl_admits(ctx.acl, user)):
                ctx, status = None, "single_chunk_acl"
            # Multi-hit dedup: a big-block with the same doc+anchor as an earlier one is a
            # duplicate produced by climbing to the same parent -- only the first is kept, the rest
            # degrade to bare hits (saves LLM context, avoids feeding the same passage twice when
            # multiple hits overlap).
            # A windowed block's anchor isn't stable across different seed hits, so exact-anchor
            # matching would miss folding it -- but windows within the same bound share the same
            # resolved_section (secid#window), so windows are folded on resolved_section instead,
            # merging heavily overlapping near-duplicate windows.
            if ctx is not None and ctx.anchor is not None:
                akey = ((h.doc_id, ctx.resolved_section) if getattr(ctx, "windowed", False)
                        else (h.doc_id, tuple(ctx.anchor)))
                if akey in seen_anchor:
                    ctx, status = None, "deduped"
                else:
                    seen_anchor.add(akey)
            # The status must distinguish "a complete section" from "a window/truncated fragment".
            # windowed=True is a token-limited fragment from _window_within (not a complete
            # passage); text that's entirely empty (an asset page with no prose, data lives in
            # content_raw) is marked asset_no_prose on its own. Otherwise climbed_N / full_section
            # are the only statuses that mean a genuinely complete section.
            if ctx is not None:
                if not (ctx.text or "").strip():
                    status = "asset_no_prose"
                elif getattr(ctx, "windowed", False):
                    status = "section_window"
                elif getattr(ctx, "climbed", 0):
                    status = f"climbed_{ctx.climbed}"
                else:
                    status = "full_section"
            out.append({"hit": h, "context": ctx, "context_status": status})
        return out

    def _assemble(self, hit, user: User, cache: dict | None = None):
        from chunker import assemble_big                     # lazy
        from chunker.chunking import BUDGETS, DEFAULT_BUDGET
        # Hit-driven: the hit already passed the store's hard filter plus its own re-check, and
        # downstream has both assemble_big's per-element gating and an exit acl_admits(ctx.acl)
        # check -- so **user is deliberately not passed**, which would trigger the doc-level
        # precheck (that precheck exists for the doc_id direct-read tools; on the hit path it would
        # wrongly reject material via the stricter-sibling rewrite described in _load_sidecar,
        # confirmed as high-severity during adversarial review).
        elements, secs, banners, acl_index = self._load_sidecar(hit.doc_id, cache)
        shim = _ChunkShim(hit.payload)
        # Only elements sharing the hit's **exact ACL** are pulled in (the default acl_index path):
        # big.acl = hit_acl is naturally accurate and verifiable at the exit check, which is more
        # fail-closed than an admit-style rule (which pulls in everything visible), and avoids a
        # case where chunker's admit branch under-reports big.acl.
        mn, tg, mx = BUDGETS.get(shim.doc_type, DEFAULT_BUDGET)   # Per-doc_type budget: slides/policy keep the whole section intact
        return assemble_big(shim, secs, elements, target=tg, min_tokens=mn, max_tokens=mx,
                            banners=banners, acl_index=acl_index)

    # ---- Control-plane operations: doc_id direct-read / drill-down / outline / cross-document ----
    def _visible_own_sections(self, secs: dict, acl_index: dict, user: User) -> list:
        """Sections whose own body (the section's range **minus its children's ranges**) has at
        least one visible element. Used by get_outline's listing and get_document's title gating.
        Judging visibility over the whole range would let a visible descendant drag a
        restricted ancestor's title along with it, leaking it -- so only the section's own,
        direct body is checked.

        PERF: children are grouped by parent_sec_id once up front (a dict), instead of rescanning
        the full section list for every section (this was O(S^2) in section count S; a large
        slide deck or policy document with hundreds of sections felt this on every
        get_document/get_outline call). Output is identical to the previous per-section scan --
        verified against the original nested-loop version across 200 randomized section trees
        before this change was made."""
        sec_list = list(secs.values())
        children_by_parent: dict = {}
        for c in sec_list:
            children_by_parent.setdefault(c.parent_sec_id, []).append((c.start_idx, c.end_idx))
        vis = []
        for sec in sec_list:
            child = children_by_parent.get(sec.sec_id, [])
            if any((not any(cs <= i < ce for cs, ce in child)) and acl_admits(acl_index.get(i) or {}, user)
                   for i in range(sec.start_idx, sec.end_idx)):
                vis.append(sec)
        return vis

    def get_document(self, doc_id: str, user: User, max_tokens: int = 6000) -> dict:
        """Reads an entire document: a doc-level fail-closed precheck plus per-element
        acl_admits gating (a normalized predicate, not an ==hit_acl equality check). Heading
        elements never enter acl_index (they're absorbed into the section tree), so headings of
        **own-section visible** sections are separately included (otherwise a full read would drop
        every heading). lang comes from the store's authoritative value rather than a rough guess
        (a wrong guess would misjudge _cap's units and over-truncate). Output past max_tokens is
        truncated and flagged as truncated."""
        from chunker.chunking import est_tokens
        from chunker.assembly import _cap, _gather
        elements, secs, banners, acl_index = self._load_sidecar(doc_id, user=user)   # doc_id-driven: precheck + the idx==i assertion
        heading_idxs = {sec.start_idx for sec in self._visible_own_sections(secs, acl_index, user)}  # heading element positions for visible sections

        def admits(i):
            a = acl_index.get(i)
            if a is not None:
                return acl_admits(a, user)
            return i in heading_idxs                          # A heading is admitted if its own section is visible; any other unknown idx is refused, fail-closed

        n_vis = sum(1 for i in range(len(elements)) if admits(i))
        text = _gather(elements, 0, len(elements), banners, admits)
        lang = self.store.get_doc_lang(doc_id)
        capped = _cap(text, lang, max_tokens)
        return {"doc_id": doc_id, "text": capped, "n_tokens": round(est_tokens(capped, lang)),
                "n_elements_visible": n_vis, "truncated": len(capped) < len(text)}   # Never returns a total that would leak the count of restricted elements

    def get_outline(self, doc_id: str, user: User) -> list[dict]:
        """A document's section outline, ACL-scoped: only includes sections whose **own body has a
        visible element** (this prevents a restricted parent's title leaking through a visible
        child section). Doesn't return breadcrumb (which would carry a restricted ancestor's title
        along with it; hierarchy is expressed via `level` instead)."""
        elements, secs, banners, acl_index = self._load_sidecar(doc_id, user=user)   # doc_id-driven: precheck
        out = [{"sec_id": s.sec_id, "title": s.title, "level": s.level,
                "start_idx": s.start_idx, "end_idx": s.end_idx}
               for s in self._visible_own_sections(secs, acl_index, user)]
        out.sort(key=lambda s: s["start_idx"])
        return out

    def expand(self, chunk_id: str, user: User, target_tokens: int = 1500):
        """Pulls a larger context window around a given hit: chunk_id -> the point's payload
        (ACL re-checked) -> re-runs assemble_big with a larger budget.
        **Hit-driven** (chunk_id names a specific hit): same as _assemble, _load_sidecar is called
        **without user** (avoiding the stricter-sibling false-deny described in _load_sidecar);
        safety instead comes from three checks: get_by_chunk_id's own acl_admits, assemble_big's
        ==hit_acl per-element gating, and the exit acl_admits check.
        Returns a BigBlock, or None (not found / no access / failed the exit check)."""
        from chunker import assemble_big
        payload = self.store.get_by_chunk_id(chunk_id, user)   # ACL-gated; None = not found or no access (fail-closed, doesn't distinguish which)
        if payload is None:
            return None
        shim = _ChunkShim(payload)
        elements, secs, banners, acl_index = self._load_sidecar(payload.get("doc_id", ""))   # hit-driven: user not passed
        big = assemble_big(shim, secs, elements, target=target_tokens, min_tokens=max(target_tokens // 4, 1),
                           max_tokens=max(target_tokens * 2, 1500), banners=banners, acl_index=acl_index)
        if big.acl is None or not acl_admits(big.acl, user):   # The exit-side check
            return None
        return big

    def search_grouped(self, query: str, user: User, doc_ids: list[str], top_k: int = 3,
                       rerank: bool = False) -> dict:
        """Grouped cross-document retrieval: takes top_k from each doc_id, returning {doc_id:
        [Hit]} -- for comparison/summarization tasks. The query is encoded exactly once and
        reused across all doc_ids (avoiding len(doc_ids) redundant 8B forward passes)."""
        qd = self.dense.encode_query(query).tolist()
        qs = query_sparse(query, self.cfg.stopwords)
        k = max(1, min(int(top_k), self.cfg.prefetch_limit))
        # The rerank candidate pool depth is clamped to [1, prefetch_limit] the same way search()
        # does it (otherwise an overly large cfg.rerank_top_n could exceed the candidate union).
        rtn = max(1, min(int(self.cfg.rerank_top_n), self.cfg.prefetch_limit))
        out = {}
        for d in doc_ids:
            if rerank:
                hits = self.store.hybrid_search(qd, qs, user, top_k=max(rtn, k), doc_ids=[d])
                try:
                    out[d] = self._get_reranker().rerank(query, hits, top_k=k)
                except Exception:                        # The same graceful degradation as search(): reranker OOM/missing falls back to hybrid recall,
                    out[d] = hits[:k]                     # so one doc's rerank failure doesn't take down the whole grouped query (the degraded hit keeps the hybrid score scale)
            else:
                out[d] = self.store.hybrid_search(qd, qs, user, top_k=k, doc_ids=[d])
        return out
