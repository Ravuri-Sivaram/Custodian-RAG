"""Retrieval-side small-to-big assembly. Pure: takes a hit Chunk + sections + elements.

Reads the hit's section_anchor and grows the 'big' block to a token target — climbing to
parent sections, and (when a jump would exceed max) growing a window WITHIN the ancestor
that pulls in adjacent SIBLING-section content. Genuinely tiny docs stay small (correct).

SECURITY (small-to-big must not cross an ACL boundary): the block is gathered from RAW elements
by idx range, so a public hit could otherwise sweep a tightened sibling's text into big.text. Pass
`acl_index` ({idx: acl}, from ChunkResult.acl_index()) and the assembler gathers ONLY elements whose
acl matches the hit's (fail-closed: unknown idx is excluded). Pass `admit(acl)->bool` for a custom
caller-visibility rule instead. With neither, assembly is legacy (no ACL awareness, single-ACL-doc
assumption) and BigBlock.acl is None to flag that the text was NOT access-verified."""
from __future__ import annotations

from .chunking import NOISE_KINDS, _banner_texts, _norm, est_tokens
from .types import BigBlock


def _elem_text(el):
    if el.text:
        return el.text
    if el.list_items:
        return " ".join(el.list_items)
    return el.caption or ""


def _gather(elements, start, end, banners=frozenset(), admits=None):
    parts = []
    for i in range(max(0, start), min(end, len(elements))):
        el = elements[i]
        if el.kind in NOISE_KINDS or _norm(el.text) in banners:   # round-3: keep banners out of big-blocks too
            continue
        if admits is not None and not admits(i):                  # ACL: skip elements the caller can't see
            continue
        t = _elem_text(el).strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


def _counts_toward(elements, i, banners, admits):
    el = elements[i]
    if el.kind in NOISE_KINDS or _norm(el.text) in banners:
        return False
    return admits is None or admits(i)


def _cap(txt, lang, max_tokens):
    """Truncate above max_tokens, to keep a big-block from blowing out the LLM's context (flagged in
    the lazy-tree design review). First accumulate along line (element) boundaries; if a single
    oversized paragraph can't be trimmed that way, fall back to a hard character-level cut (keeping
    a word boundary), guaranteeing est_tokens <= max."""
    if not txt or est_tokens(txt, lang) <= max_tokens:
        return txt
    kept, tok = [], 0.0
    for ln in txt.split("\n"):
        t = est_tokens(ln, lang)
        if kept and tok + t > max_tokens:
            break
        kept.append(ln)
        tok += t
    out = "\n".join(kept)
    if est_tokens(out, lang) > max_tokens:                 # a single oversized paragraph can't be trimmed by line -> hard character cut
        # Word-boundary-safe truncation applies regardless of language here: this used to
        # special-case scripts with no inter-word spaces (where a raw character cut was the only
        # option), but Telugu IS written with spaces between words, same as English -- so the
        # char-budget just needs to match est_tokens' divisor, and the actual cut always backs
        # off to the last space.
        is_te = (lang or "").lower().startswith("te")
        cap_char = int(max_tokens * (1.0 if is_te else 4.0))
        out = out[:cap_char].rsplit(" ", 1)[0] or out[:cap_char]
    return out


def _window_within(elements, hit_idxs, bound, lang, target, note, banners=frozenset(), admits=None, max_tokens=1500):
    s, e = (min(hit_idxs), max(hit_idxs)) if hit_idxs else (bound["start"], bound["start"])
    s, e = max(s, bound["start"]), min(e, bound["end"] - 1)        # clamp the seed range into bound (review note: hit_idxs can be out of range)
    total = est_tokens(_gather(elements, s, e + 1, banners, admits), lang)
    while total < target:
        grew = False
        if e + 1 < bound["end"]:
            e += 1; grew = True            # grow window; only count tokens the gather will keep
            if _counts_toward(elements, e, banners, admits):
                total += est_tokens("\n" + _elem_text(elements[e]), lang)   # include _gather's newline separator, otherwise this systematically underestimates -> overshoot
        if total < target and s - 1 >= bound["start"]:
            s -= 1; grew = True
            if _counts_toward(elements, s, banners, admits):
                total += est_tokens("\n" + _elem_text(elements[s]), lang)
        if not grew:
            break
    txt = _cap(_gather(elements, s, e + 1, banners, admits), lang, max_tokens)   # cap at max_tokens
    return BigBlock(text=txt, resolved_section=bound["sec_id"] + "#window",
                    breadcrumb=bound["crumb"], n_tokens=round(est_tokens(txt, lang)),
                    climbed=0, anchor=[s, e + 1], note=note, acl=bound.get("acl"), windowed=True)


def _make_admit(hit_chunk, acl_index, admit):
    """Return (admits_fn_or_None, big_block_acl). admits(idx)->bool gates which elements may enter
    big.text; big_block_acl is what to stamp on BigBlock (None == legacy/unverified)."""
    hit_acl = getattr(hit_chunk, "acl", None) or {}
    if admit is not None:                                          # caller-supplied visibility rule
        if acl_index is None:                                     # admit judges EACH element's acl -> it
            raise ValueError(                                    # REQUIRES acl_index; without it we'd judge
                "assemble_big: admit= requires acl_index= (per-element acl). Pass acl_index, or "
                "drop admit for the default same-ACL equivalence.")  # else: judge by hit_acl yet stamp
        def admits(i):                                           # big.acl as verified -> silent cross-ACL leak
            return admit(acl_index.get(i))
        return admits, hit_acl
    if acl_index is not None:                                      # same-ACL-as-hit equivalence class
        def admits(i):                                            # unknown idx -> denied (fail-closed)
            return acl_index.get(i) == hit_acl
        return admits, hit_acl
    return None, None                                             # legacy: no acl info -> no gating


def assemble_big(hit_chunk, sections_by_id, elements, target=800, min_tokens=200, max_tokens=1500,
                 banners=None, acl_index=None, admit=None):
    """sections_by_id: {sec_id: Section}. elements: list[Element] (idx-aligned). Returns BigBlock.
    banners: precomputed running-banner set (from ChunkResult.banners); recomputed if None.
    acl_index: {idx: acl} (from ChunkResult.acl_index()) -> gather only same-ACL-as-hit elements.
    admit: acl->bool, a custom caller-visibility rule (overrides the equivalence default).
    Requires elements to be densely and contiguously ordered by idx (elements[i].idx==i): both
    text-gathering and ACL gating index by position, so sparse/out-of-order elements would pull the
    wrong span (see the assertion below)."""
    # fail-closed: _gather/_window index elements by POSITION, while start/source_indices/acl_index are
    # all el.idx values -> we enforce elements[i].idx==i here to rule out a future sparse/subset
    # `elements` list causing gather to grab the wrong span and leak ACL by position (flagged in the
    # lazy-tree design review, item 1).
    if any(el.idx != i for i, el in enumerate(elements)):
        raise ValueError("assemble_big: elements must be densely and contiguously ordered by idx "
                         "(elements[i].idx==i); sparse/out-of-order input would make query-time "
                         "assembly grab the wrong span and leak ACL by position.")
    lang = "te" if (hit_chunk.lang or "").lower().startswith("te") else "en"  # align with est_tokens' Telugu handling (see the note there)
    doc_id = hit_chunk.doc_id
    if banners is None:                        # reuse chunk-time set when caller passes it (perf)
        banners = _banner_texts(elements)
    admits, block_acl = _make_admit(hit_chunk, acl_index, admit)
    doc_end = (max((e.idx for e in elements), default=-1) + 1)
    doc_root = {"sec_id": f"{doc_id}::root", "start": 0, "end": doc_end,
                "crumb": hit_chunk.breadcrumb, "acl": block_acl}
    hit_idxs = hit_chunk.source_indices

    sec = sections_by_id.get(hit_chunk.section_id) if hit_chunk.section_id else None
    if sec is None:
        # No section: open a window over the neighborhood of the hit's page(s) (whether one page or
        # several), to avoid pulling in the entire document for a headingless multi-page doc (flagged
        # in the lazy-tree design review).
        pages = set(range(hit_chunk.page_start, hit_chunk.page_end + 1))
        pidx = [e.idx for e in elements if e.page in pages]
        if pidx:
            bound = {"sec_id": f"{doc_id}::p{hit_chunk.page_start}", "start": min(pidx),
                     "end": max(pidx) + 1, "crumb": hit_chunk.breadcrumb, "acl": block_acl}
            return _window_within(elements, hit_idxs, bound, lang, target,
                                  "no section -> page window", banners, admits, max_tokens)
        return _window_within(elements, hit_idxs, doc_root, lang, target,
                              "no section -> doc window", banners, admits, max_tokens)

    def toks(s):
        return est_tokens(_gather(elements, s.start_idx, s.end_idx, banners, admits), lang)

    if toks(sec) > max_tokens:
        bound = {"sec_id": sec.sec_id, "start": sec.start_idx, "end": sec.end_idx,
                 "crumb": sec.breadcrumb, "acl": block_acl}
        return _window_within(elements, hit_idxs, bound, lang, target, "section>max -> window", banners, admits, max_tokens)

    cur, climbed, seen = sec, 0, {sec.sec_id}
    while toks(cur) < target:
        parent = sections_by_id.get(cur.parent_sec_id)
        if parent is None or parent.sec_id in seen:        # a cycle/self-reference (corrupted sidecar data) -> stop and fall back, to prevent an infinite loop (review note #2)
            break
        seen.add(parent.sec_id)
        if toks(parent) > max_tokens:
            bound = {"sec_id": parent.sec_id, "start": parent.start_idx, "end": parent.end_idx,
                     "crumb": parent.breadcrumb, "acl": block_acl}
            return _window_within(elements, hit_idxs, bound, lang, target,
                                  "climb overshoot -> sibling window within parent", banners, admits, max_tokens)
        cur, climbed = parent, climbed + 1

    txt = _cap(_gather(elements, cur.start_idx, cur.end_idx, banners, admits), lang, max_tokens)   # cap at max_tokens
    if est_tokens(txt, lang) < min_tokens:
        return _window_within(elements, hit_idxs, doc_root, lang, target,
                              "top-too-small -> doc window", banners, admits, max_tokens)
    return BigBlock(text=txt, resolved_section=cur.sec_id, breadcrumb=cur.breadcrumb,
                    n_tokens=round(est_tokens(txt, lang)), climbed=climbed,
                    anchor=[cur.start_idx, cur.end_idx], acl=block_acl)
