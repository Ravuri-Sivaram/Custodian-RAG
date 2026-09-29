"""Pure chunking core (v2 Heading-Skeleton). Consumes Element[] -> ChunkResult.
No file IO. Ported from the harness chunk_document.py (tested on 77 docs / 5337 headings)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .types import Chunk, ChunkResult, Element, Section

NOISE_KINDS = {"header", "footer", "page_number", "aside_text"}   # round-3: aside_text = page-edge
# watermark / rotated spine (arXiv stamp, brokerage-report spine text), never body — confirmed noise on 11/11 corpus cases
TOC_RE = re.compile(r"(\.{3,}|…)\s*\d+\s*$")
TOC_TAIL_RE = re.compile(r"^\d+(?:\.\d+)*\s+.+\s+\d{1,3}\s*$")
NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")
YEAR_RE = re.compile(r"^(19|20)\d\d$")
BULLET_RE = re.compile(r"^\s*[-–—•·●○▪◦]\s")   # review F3: bullet glyph = body, never a heading
LAW_SEC_RE = re.compile(r"^\s*(SECTION|SEC\.)\s+\d+", re.I)
MAX_HEADING_LEN = 120

# (min, target, max) token budget per doc_type. Numbers cluster — one default works; the
# real differentiation is the layout MODE (slides/policy keep whole sections).
DEFAULT_BUDGET = (200, 800, 1500)
BUDGETS = {
    "financial_research_te": (250, 550, 900), "academic_paper": (300, 750, 1100),
    "law": (300, 700, 1200), "financial_report_en": (250, 650, 1000),
    "government": (250, 650, 1000), "research_report": (200, 550, 800),
    "admin_industry": (250, 650, 1000), "tech_report": (300, 750, 1100),
    "form": (150, 400, 800), "brochure": (150, 400, 700), "guidebook": (150, 400, 700),
    "slides_tutorial": (0, 9999, 9999), "policy": (0, 9999, 9999), "news": (200, 550, 800),
}
PAGE_GROUPED = {"slides_tutorial"}        # one slide(page) = one chunk

# FAIL-CLOSED default access policy: a chunk whose document was NOT (yet) wired to a permission
# source is RESTRICTED, never public. Retrieval MUST hard pre-filter on chunk.acl; a chunk with
# `unset=True` / empty `allow` is denied to everyone but an explicit admin/open-mode caller.
RESTRICTED_ACL = {"visibility": "restricted", "allow": [], "unset": True}


def est_tokens(text, lang):
    # HEURISTIC char/divisor. The English value (4.0) is empirically validated (2026-06, against
    # the real Qwen3-VL tokenizer, 2927 real chunks): for English prose, academic/law char/token ~=
    # 3.85 (error < 4%); the deviation comes almost entirely from number/table-dense documents
    # (financial reports 5.08 / government 5.35). This is an inherent limitation of a char-based
    # heuristic — a single value can't serve both the "prose" and the "number-dense" clusters at
    # once, and switching to the mean (English 4.34) would actually hurt the prose-dominant
    # majority. Both error directions are covered: overestimating tokens -> chunk runs small ->
    # small-to-big retrieval compensates (no data lost); underestimating -> chunk runs large but
    # stays well under the Qwen3-VL 32k context limit. So we keep the current value rather than
    # recalibrate it.
    #
    # The Telugu value (1.0) is a CONSERVATIVE PLACEHOLDER, NOT empirically measured the way the
    # English/former-CJK values above were. Telugu is an abugida script (base consonant + vowel
    # sign + optional virama per visual character), and tokenizers trained mostly on Latin/CJK
    # corpora tend to fragment underrepresented Indic scripts into individual codepoints or even
    # UTF-8 byte-pairs rather than whole syllables — often a WORSE (lower) chars-per-token ratio
    # than even Chinese got here previously. 1.0 is deliberately conservative (assumes heavy token
    # use per character) to avoid the same truncation risk the CJK alias was added to prevent, but
    # it must be measured against real Telugu documents through the actual Qwen3-VL/DeepSeek
    # tokenizer before this is trusted in production — see the English paragraph above for exactly
    # this kind of validation.
    return len(text or "") / (1.0 if (lang or "").lower().startswith("te") else 4.0)


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _safe_rel(p):
    """SECURITY (seal review): an image_path must be a SAFE RELATIVE ref under the MinerU output root —
    reject absolute / UNC / drive paths, URL schemes (://), and '..' traversal, so a poisoned parse can't
    make the embed stage read an arbitrary file or SSRF off it. Returns the cleaned rel path, or None."""
    import posixpath
    if not isinstance(p, str) or not p.strip():
        return None
    q = p.strip().replace("\\", "/")
    if "://" in q or q.startswith("/") or re.match(r"^[A-Za-z]:", q):
        return None                                  # url scheme / absolute / UNC / windows drive letter
    norm = posixpath.normpath(q)
    if norm == ".." or norm.startswith("../") or norm.startswith("/"):
        return None                                  # traversal escaping the output root
    return norm


def _asset_desc(s, cap=800):
    """An image's VLM-extracted content (OCR text / mermaid / chart data) or alt-text, cleaned for
    use as retrieval text. For mermaid (flowchart/diagram), keep only the node + edge LABELS — the
    real retrieval signal — and drop scaffolding (`graph TD`, `-->`, `style X fill:#f9f`), which is
    zero-value noise in the embedding (review F2). Cap at a word boundary, not mid-token."""
    if not s:
        return ""
    s = re.sub(r"```[a-z]*|```", " ", s)
    if re.search(r"\bgraph\s+(?:TD|TB|LR|RL|BT)\b|\bflowchart\b|--?>", s):
        labels = re.findall(r'[\[\(\{|]\s*"?([^"\[\]\(\)\{\}|]+?)"?\s*[\]\)\}|]', s)
        s = " ".join(x.strip() for x in labels if x.strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s[:cap].rsplit(" ", 1)[0] if len(s) > cap else s


_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _table_signal(body, cap=400):
    """Retrieval signal for a table: the header row(s) (first 2 rows, since financial reports commonly
    use a two-tier header) plus the first non-empty cell of every remaining row (the row label),
    deduplicated and concatenated, capped at a word boundary. **Why**: a table chunk's `text` used to
    be just the caption+footnote sentence — but the column headers / row labels (Revenues / Net
    income / year, etc.) are actually the semantic signal for "what question does this table answer",
    and they were locked away in `content_raw` (which doesn't participate in embedding/sparse
    retrieval). This was empirically shown to cause number-heavy question/table chunks to get pushed
    out of the top-k by prose (diagnosed in Custodian TESTING §3). We only extract labels, not data
    cells: the numbers themselves carry no retrieval semantics and would just add noise."""
    if not body:
        return ""
    rows = [[_TAG_RE.sub(" ", c).strip() for c in _CELL_RE.findall(r)] for r in _TR_RE.findall(body)]
    parts = []
    for r in rows[:2]:                                   # header rows (may be two-tier)
        parts.extend(r)
    for r in rows[2:]:                                   # row label = first non-empty cell of the row
        first = next((c for c in r if c), "")
        if first:
            parts.append(first)
    seen, out = set(), []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    sig = " | ".join(out)
    return sig[:cap].rsplit(" ", 1)[0] if len(sig) > cap else sig


def _banner_texts(elements):
    """Running banners/watermarks parser mislabeled as content: identical text appearing on
    >= half the pages (review round-2). Page-fraction (not raw count) avoids killing legit
    recurring headings ('SALARIES AND EXPENSES' x4, 'Engine Sensors' x6 are NOT banners).

    Round-3: a banner must NOT end with a colon — colon labels like 'Prompt:' / 'GPT-4V:'
    recur on 60% of pages in dialogue/multimodal papers but are CONTENT (speaker turns),
    not page furniture. (bbox position-stability would be a stronger signal — deferred.)"""
    npages = max((e.page for e in elements), default=0) + 1
    if npages < 4:
        return set()
    pages_of, freq = defaultdict(set), Counter()
    for e in elements:
        if e.kind in NOISE_KINDS:
            continue
        t = _norm(e.text)
        if t and not t.endswith((":", "：")):
            pages_of[t].add(e.page); freq[t] += 1
    return {t for t in freq if freq[t] >= 3 and len(pages_of[t]) >= npages * 0.5}


def _structural_nonheading(txt):
    """Text-shape rejects shared by the leveling pass and the reset-aware pre-pass."""
    return bool(TOC_RE.search(txt) or TOC_TAIL_RE.match(txt)
                or len(txt) > MAX_HEADING_LEN or BULLET_RE.match(txt))


def _leading_number(txt):
    """Parse a leading section/list number -> ('dotted', depth) | ('bare', value) | None.
    Year-leading (2026 ...) is treated as not-a-number (review F1/F4)."""
    m = NUM_RE.match(txt)
    if not m:
        return None
    tok = m.group(1)
    if YEAR_RE.match(tok.split(".")[0]):
        return None
    if "." in tok:
        return ("dotted", tok.count(".") + 1)
    if tok.isdigit():
        return ("bare", int(tok))
    return None


def heading_level(el: Element, doc_type, promote_bare=True):
    """text_level primary (free from parser). DOTTED numbering (2.1) always refines depth;
    a BARE integer promotes to L1 only when the doc's numbering is a monotonic outline
    (`promote_bare`) — otherwise it is a list ordinal and we defer to text_level (review F1)."""
    txt = (el.text or "").strip()
    if _structural_nonheading(txt):
        return None
    if doc_type == "law" and LAW_SEC_RE.match(txt):
        return 1
    if not el.text_level:
        return None
    num = _leading_number(txt)
    if num:
        kind, val = num
        if kind == "dotted":
            return val
        if kind == "bare" and promote_bare:
            return 1
    return int(el.text_level)


def build_sections(headings, doc_id, doc_end):
    """headings: [{idx, level, title, crumb}] in reading order -> section tree (idx ranges)."""
    sections = []
    for i, h in enumerate(headings):
        end = doc_end
        for j in range(i + 1, len(headings)):
            if headings[j]["level"] <= h["level"]:
                end = headings[j]["idx"]; break
        parent = next((headings[k]["idx"] for k in range(i - 1, -1, -1)
                       if headings[k]["level"] < h["level"]), None)
        sections.append(Section(
            sec_id=f"{doc_id}::s{h['idx']}", doc_id=doc_id, level=h["level"], title=h["title"],
            breadcrumb=h["crumb"], start_idx=h["idx"], end_idx=end,
            parent_sec_id=f"{doc_id}::s{parent}" if parent is not None else None))
    return sections


def _sentence_split(text, hi, lang):
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    out, cur = [], ""
    for p in parts:
        if est_tokens(cur + p, lang) > hi and cur:
            out.append(cur.strip()); cur = ""
        cur += p + " "
    if cur.strip():
        out.append(cur.strip())
    return out or [text]


def assemble_text(blocks, lang, budget):
    """Greedy accumulate consecutive same-section text blocks to a token budget. THE knob."""
    lo, target, hi = budget
    groups, cur, cur_tok = [], [], 0.0

    def flush():
        nonlocal cur, cur_tok
        if cur:
            groups.append(cur); cur = []; cur_tok = 0.0

    for b in blocks:
        bt = est_tokens(b["text"], lang)
        if bt > hi:
            flush()
            for piece in _sentence_split(b["text"], hi, lang):
                groups.append([{**b, "text": piece}])
            continue
        if cur and not b.get("merge") and cur_tok + bt > target:
            flush()
        cur.append(b); cur_tok += bt
    flush()

    merged = []
    for g in groups:
        gt = sum(est_tokens(x["text"], lang) for x in g)
        if merged and gt < lo and sum(est_tokens(x["text"], lang) for x in merged[-1]) + gt <= hi:
            merged[-1].extend(g)
        else:
            merged.append(g)

    return [{"text": "\n".join(x["text"] for x in g), "pages": sorted({x["page"] for x in g}),
             "idxs": [x["idx"] for x in g], "stitched": any(x.get("merge") for x in g)}
            for g in merged]


class Chunker:
    """The chunker component. `chunk(elements, ...)` -> ChunkResult. Stateless config holder."""

    def __init__(self, target=None, min_tokens=None, max_tokens=None, budgets=None, page_grouped=None):
        self._override = None
        if target is not None:
            self._override = (min_tokens or 200, target, max_tokens or 1500)
        self.budgets = {**BUDGETS, **(budgets or {})}
        self.page_grouped = page_grouped if page_grouped is not None else PAGE_GROUPED

    def _budget(self, doc_type):
        return self._override or self.budgets.get(doc_type, DEFAULT_BUDGET)

    def chunk(self, elements: list[Element], *, doc_id: str, doc_type: str | None = None,
              lang: str = "en", doc_meta: dict | None = None, acl: dict | None = None) -> ChunkResult:
        budget = self._budget(doc_type)
        banners = _banner_texts(elements)          # round-2: drop per-page running banners
        kept = [e for e in elements
                if e.kind not in NOISE_KINDS and _norm(e.text) not in banners]

        # reset-aware switch (review F1): bare-integer numbering only promotes to L1 when
        # the doc's bare numbers form a monotonic outline. A restart (1..9,1.. = list usage)
        # means the numbers are list ordinals, not chapters -> defer to text_level.
        bare_seq = []
        for el in kept:
            txt = (el.text or "").strip()
            if not el.text_level or _structural_nonheading(txt):
                continue
            num = _leading_number(txt)
            if num and num[0] == "bare":
                bare_seq.append(num[1])
        promote_bare = all(bare_seq[i] > bare_seq[i - 1] for i in range(1, len(bare_seq)))

        # heading stack -> breadcrumb + deepest-section per element; collect headings
        stack, headings, enriched = [], [], []
        for el in kept:
            lvl = heading_level(el, doc_type, promote_bare)
            if lvl is not None:
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                title = (el.text or "").strip()
                stack.append((lvl, title, el.idx))
                headings.append({"idx": el.idx, "level": lvl, "title": title,
                                 "crumb": [t for _, t, _ in stack]})
                continue
            el._crumb = [t for _, t, _ in stack]
            el._sec_head = stack[-1][2] if stack else None
            enriched.append(el)

        # emit chunks: assets atomic; text grouped by (section [, page])
        leaves, text_run, run_key = [], [], None

        def section_key(el):
            # group by the DEEPEST SECTION (its heading idx — unique per heading instance), NOT the
            # breadcrumb TEXT (seal review): two same-named sibling sections (repeated subheadings in
            # forms/appendices/multi-party filings) share an identical crumb, which would merge their
            # distinct bodies into one chunk and mis-anchor it to the first section — irreversibly wrong
            # data in the vector store. _sec_head (core:236) is unique per heading instance.
            base = getattr(el, "_sec_head", None)
            return (base, el.page) if doc_type in self.page_grouped else base

        def flush_text():
            nonlocal text_run, run_key
            if not text_run:
                return
            crumb = list(text_run[0].get("crumb") or [])     # breadcrumb from the run's elements, not the key
            for g in assemble_text(text_run, lang, budget):
                leaves.append(self._text_chunk(doc_id, lang, doc_type, g, crumb))
            text_run = []; run_key = None

        for el in enriched:
            if el.kind in ("table", "chart", "image"):
                flush_text()
                ch = self._asset_chunk(doc_id, lang, doc_type, el)
                if ch is not None:               # skip captionless + contentless placeholders (review F3)
                    leaves.append(ch)
            else:   # text/list + previously-dropped kinds: equation/code/page_footnote/ref_text
                txt = el.text or " ".join(el.list_items or [])
                if not txt.strip():
                    continue
                key = section_key(el)
                if run_key is not None and key != run_key:
                    flush_text()
                run_key = key
                text_run.append({"text": txt, "page": el.page, "idx": el.idx,
                                 "merge": el.merge_prev, "crumb": getattr(el, "_crumb", [])})
        flush_text()

        # section tree + stamp each leaf with its enclosing section anchor
        doc_end = (max((e.idx for e in elements), default=-1) + 1)
        sections = build_sections(headings, doc_id, doc_end)
        idx2head = {e.idx: getattr(e, "_sec_head", None) for e in enriched}
        sec_by_head = {s.start_idx: s for s in sections}
        # stamp doc-level metadata + access policy onto every chunk (the chunker STAMPS; ingest
        # EXTRACTS). acl defaults FAIL-CLOSED so an un-permissioned doc is never accidentally public.
        import copy
        dm = doc_meta or {}
        ac = acl if acl else RESTRICTED_ACL
        for n, ch in enumerate(leaves):
            ch.chunk_id = f"{doc_id}#{n:04d}"
            first = ch.source_indices[0] if ch.source_indices else None
            sec = sec_by_head.get(idx2head.get(first))
            ch.section_id = sec.sec_id if sec else None
            ch.section_anchor = [sec.start_idx, sec.end_idx] if sec else None
            ch.doc_meta = copy.deepcopy(dm)      # deep copy -> per-chunk (per-section) override is safe
            ch.acl = copy.deepcopy(ac)
            ch.doc_type = doc_type               # seal review: assemble_big reads this for query-time budget
        return ChunkResult(chunks=leaves, sections=sections, banners=frozenset(banners))

    def assemble_big(self, hit_chunk, result: ChunkResult, elements, target=None,
                     min_tokens=None, max_tokens=None, admit=None):
        from .assembly import assemble_big as _ab
        b = self._budget(getattr(hit_chunk, "doc_type", None))
        return _ab(hit_chunk, result.sections_by_id(), elements,
                   target=target or b[1], min_tokens=min_tokens or b[0], max_tokens=max_tokens or b[2],
                   banners=result.banners,               # reuse the banner set computed at chunk time
                   acl_index=result.acl_index(), admit=admit)   # SECURITY: gather only same-ACL-as-hit text

    # -- chunk constructors --
    def _text_chunk(self, doc_id, lang, doc_type, g, crumb):
        return Chunk(
            chunk_id="", doc_id=doc_id, kind="text", text=g["text"], content_raw=None,
            breadcrumb=crumb, section_path=" > ".join(crumb), section_id=None, section_anchor=None,
            page_start=g["pages"][0], page_end=g["pages"][-1], source_indices=g["idxs"],
            n_tokens=round(est_tokens(g["text"], lang)), trust="high",
            flags=[f for f in ["multi_page" if len(g["pages"]) > 1 else None,
                               "merge_prev_stitched" if g.get("stitched") else None] if f],
            lang=lang)

    def _asset_chunk(self, doc_id, lang, doc_type, el: Element):
        cap, foot = (el.caption or "").strip(), (el.footnote or "").strip()
        body = el.table_body if el.kind == "table" else el.asset_content
        img = _safe_rel(el.image_path) if el.kind in ("image", "chart") else None   # sanitize (seal #2) + ''->None
        parts = [cap, foot]
        if el.kind in ("image", "chart"):        # fold the image's best description (VLM-extracted
            parts.append(_asset_desc(el.asset_content))   # content/OCR/mermaid, or alt-text) into the
        if el.kind == "table" and (cap or foot or el.table_body):   # Table: breadcrumb (scoping info like
            parts.append(" > ".join(getattr(el, "_crumb", [])))     # segment/sub-period often lives only in
            parts.append(_table_signal(el.table_body))              # the section heading) + header/row labels
            # (the table's semantic signal, which used to be locked in content_raw and unretrievable;
            # see Custodian TESTING §3).
            # WARNING: the gate (cap|foot|body) matters — a placeholder table with no caption and no
            # table body must keep the F3 drop-it semantics. If the breadcrumb alone were allowed to
            # make `retrieval` non-empty, it would "revive" a ghost chunk (text = breadcrumb only, no
            # real content), and that would shift every later chunk id in the same document (breaking
            # alignment with any existing gold labels/citations built on the old index). Empirically,
            # without this gate the corpus went from 7652 to 7675 chunks (+23 ghost chunks); with the
            # gate it's back to 7652.
        retrieval = " ".join(x for x in parts if x).strip()   # retrieval text so the image is findable
        if not retrieval and not body and not img:    # (1) a bare picture (img only) is STILL retrievable via
            return None                               # VL image embedding; only a truly-empty placeholder is dropped
        flags = []
        if not cap:
            flags.append("captionless")
        if el.kind in ("chart", "image") and body:
            flags.append("vlm_content")
        if img and not retrieval and not body:   # (1) pure picture: text is a placeholder -> downstream embeds
            flags.append("image_only")           # the IMAGE itself (skip the sparse/text path for this chunk)
        return Chunk(
            chunk_id="", doc_id=doc_id, kind=el.kind,
            text=retrieval or f"[{el.kind} without caption]", content_raw=body,
            breadcrumb=getattr(el, "_crumb", []), section_path=" > ".join(getattr(el, "_crumb", [])),
            section_id=None, section_anchor=None, page_start=el.page, page_end=el.page,
            source_indices=[el.idx], n_tokens=round(est_tokens(retrieval, lang)),
            trust="low" if "vlm_content" in flags else "high", flags=flags, lang=lang,
            image_path=img)                       # normalized; None for text/table (only images get image embeddings)
