# Heading-Skeleton RAG — Design v2 (revised after adversarial review and real-data evidence)

> The v1 idea was "keep only a minimal numbered skeleton at ingest, then lazily rebuild the hierarchy on a hit." **Adversarial review (4 lenses, each independently verified, run against 77 documents / 5337 headings of real data) overturned v1's two load-bearing assumptions**, and v2 is rewritten from that evidence.
> In one line: **the default flips from "lazy" to "eager-but-cheap"; leveling flips from "bet on numbering" to a multi-signal fusion led by text_level; pre-TOC stripping moves into ingest; "lazy" is demoted to a rare special case.**

---

## 0. Review findings (where v1 was wrong, and why v2 changes what it does)

Evidence (`analysis/per_doc_stats.csv` + `parsed/*/content_list.json`):

1. **The core bet, "numbered → zero LLM," only covers ~13%**: across the whole corpus, only 10.6% of headings can be parsed into a number; by type, academic is 0.73 while financial/law/policy/slides/brochure are ≈0. **Unnumbered is the majority, not the tail** → "zero LLM by default" cannot be treated as the selling point.
2. **font_rank is not a reliable fallback**: bbox height doesn't separate levels (a 10-K's L1/L2 are the same height); legal/financial documents' `SEC./TITLE/DIVISION/Item` collapse num_depth down to depth=1.
3. **"Lazy" was premature optimization**: building the whole tree eagerly across all 77 documents took only 66.6ms total (<1ms/document), and uses the same algorithm as query-time Step A → laziness saves almost nothing, while instead moving zero-cost work onto the hot query path and adding cache-invalidation complexity.
4. **A real strength**: on **academic papers**, which have dense numbering, numbered-based leveling is reliable and beats MinerU's flat leveling — v2 keeps it as a strong signal for that document type.
5. **Things I over-worried about** (corrected by real data): reading-order violations where idx<E, heading misdetection at 1.3%, oversized sections at 1% — none of these turned out to matter; the real common case is **undersized sections (median 42 tokens)**.

---

## 1. Overall (v2): eager multi-signal skeleton, lazy reserved for special cases

```
ingest (once per document, deterministic, sub-millisecond, zero LLM by default):
  parse (MinerU md+layout) → candidate detection → pre-TOC stripping → multi-signal leveling → monotonic-stack tree building
  → fold breadcrumb + section_anchor into every leaf chunk (no inlined text)
query (read on hit, no rebuilding):
  flat retrieval hit → read the breadcrumb + section_anchor already attached to the chunk
  → fetch a surrounding region sized to the target token count (if too small, climb up or merge siblings) → assemble
lazy (special case only): when a document has an extremely large number of headings (e.g. news with 1335) AND queries are very sparse AND ingest has a hard budget constraint, skip pre-building and instead build-on-hit with caching
```

---

## 2. Ingest: eager multi-signal hierarchy skeleton (deterministic)

### 2.1 Candidate detection + mandatory pre-TOC stripping
- Candidates = elements MinerU marked with `text_level` ∪ elements matched by a leading-numbering regex.
- **Pre-TOC stripping (missing in v1, now mandatory)**: use `(\.{3,}|…|\s)\d{1,3}\s*$` (dot leaders/ellipsis + trailing page number) plus a `toc_like_headings` signal to **exclude** table-of-contents/list-of-figures entries from the candidate index; also apply **segment isolation** for **numbering resets** (the same token sequence recurring multiple times, e.g. 1.1 per chapter) to prevent cross-segment contamination of the monotonic stack.

### 2.2 Multi-signal leveling (fusion, not a bet on numbering) — in priority order
1. **text_level (primary signal, free)**: already provided by MinerU, used directly as the baseline level. **v1's mistake was discarding it**; v2 treats it as primary.
2. **Numbered segments (strongest when present, used as a correction)**: parse a numbering token by `.`-separated segments → `2`=depth 1, `2.1`=depth 2. **Only overrides text_level in numbering-dense documents (academic)**; also extended with a genre-prefix-to-depth mapping: `SEC./Article/TITLE [roman]/PART/Item N/Article N (Chinese-style)/first, second... (Chinese-style)` → an explicit depth table (deterministic, still zero LLM).
   - **Guard**: a pure numeric token that is a year (19xx/20xx) or a quantity >40 is **not treated as numbering** (guards against false triggers like "2020 results", "52 Places" — measured at ~4%).
3. **Font/layout (weak fallback, tiebreak among same-level items only)**: **bbox height is never treated as a globally comparable level**; when font size can't separate levels, **explicitly mark `level_unreliable` and degrade to a linear breadcrumb (only the nearest preceding heading, without forcing a parent-child structure)**, rather than silently guessing wrong.
4. **Conflict resolution**: when numbering and text_level disagree, trust numbering in numbering-dense documents, otherwise trust text_level.

### 2.3 Tree building + folding into chunks
- Monotonic-stack tree building (microsecond-scale). Every leaf chunk is annotated with: `breadcrumb` (the chain of ancestor headings), `section_anchor` (the index range of its containing section), and `level_reliable: bool`.
- **Zero LLM.** The LLM is not on the ingest main path.

### 2.4 Where the LLM honestly belongs (not "a rare tail case")
- For unnumbered, multi-level documents (the bulk of the corpus), if text_level also can't be trusted, **either accept text_level as-is (cheap, breadcrumb might only be one level deep, best-effort)**, **or, offline/at ingest time (not on the query hot path), selectively run an LLM disambiguation pass per doc_type once and cache the result**.
- **To be explicit**: this is an **optional enhancement routed by doc_type**, not "mostly zero LLM." The cost model treats documents with `numbered_frac<0.2` as a separate case.

---

## 3. Query: read directly, no rebuilding

1. A flat retrieval hit → read the `breadcrumb` already attached to the chunk at ingest time (a few dozen tokens, free context).
2. **Fetch a surrounding region sized by actual token count, not by fragile font-based levels**: use the section_anchor's index range plus an actually-computed assemble_text token count.
   - Too small (the main case — median is only 42 tokens) → **climb up the breadcrumb to the grandparent, or merge adjacent sibling sections, until reaching the target** (driven by the actually-computed size of the anchor range, not by font level).
   - Too large (rare, 1%) → take a window around the hit's neighborhood.
3. Multiple hits in the same section are deduplicated (similar to auto-merge); an isolated single hit can be given just the breadcrumb plus the hit itself.
4. For documents with `level_unreliable`: the breadcrumb degrades to "the nearest preceding heading" as linear context, **explicitly labeled best-effort**.

---

## 4. Storage

- Vector store: leaf chunks (carrying breadcrumb + section_anchor + level_reliable).
- Docstore: the per-document element stream (for fetching surrounding text and climbing up).
- **No query-time cache layer needed** (eager pre-building already covers it); only the lazy special case needs a cache.

---

## 5. Optional: cross-reference expansion (a differentiator, off by default)
At ingest time, cheaply regex-extract references like "see Appendix G / Table 5 / §2.3" and store them as edges; at query time, if the hit region contains a reference relevant to the query, additionally pull in the referenced target section. This fills the gap of "neighbors/parent alone can't reach Appendix G."

---

## 6. When "lazy" actually wins (the narrowed set of preconditions)
Only go lazy when **all** of these hold, otherwise use eager-cheap:
1. The corpus has **high-frequency local updates** (a single document is rewritten/re-versioned at second-level frequency) such that eager's total budget also becomes expensive; **and**
2. Hits are **extremely sparse over a long tail** (most sections are never hit, so eager pre-building is wasted work); **and**
3. A single document has an **extremely large number of headings** (e.g. news with 1335, form-p17 with 193; only 3/77 documents in this corpus).
None of the three hold for this corpus → **eager by default**.

---

## 7. Failure modes (updated by evidence: ✅ real problem / ❌ overthought)

- ✅ **Unnumbered is the majority** (74% of documents have frac<0.2) → text_level as primary, LLM routed by type (§2.2/2.4).
- ✅ **font_rank can't be leveled** → never treated as a global level, degrades to linear breadcrumb (§2.2.3).
- ✅ **Legal/financial numbering collapses** → genre-prefix depth table (§2.2.2).
- ✅ **Mixed leveling scales** → text_level is uniformly the baseline, numbering only corrects it, avoiding directly comparing num_depth vs. font_rank (§2.2).
- ✅ **TOC contamination + numbering resets** → ingest-time pre-TOC stripping + segment isolation (§2.1).
- ✅ **Sections are commonly too small** → climbing up/merging siblings as the main path, driven by real token counts (§3.2).
- ✅ **Dotted-leader false triggers on years/quantities** → guarded (§2.2.2).
- ❌ Reading-order idx<E violations: measured at 0 violations, **not a real problem**, no longer defended against.
- ❌ Heading misdetection: measured at 1.3%, no complex fallback added (a single regex cleanup pass suffices for scanned documents).
- ❌ Oversized sections needing windowing: measured at 1%, downgraded to a minor concern.

---

## 8. In one sentence (v2 positioning)
**Not "lazily rebuild the hierarchy," but "eagerly build a cheap multi-signal skeleton (text_level as primary + numbering correction + TOC stripping), fold breadcrumb/section_anchor straight into the chunk; at query time, read directly and fetch a surrounding region sized by actual tokens (climbing up if too small)."** Numbered-segment leveling is a genuine strength for academic documents; "lazy" is reserved only for the rare special case of huge heading counts + extremely sparse queries + high update frequency. This design has essentially converged on the ingest-time structuring approach of mature systems (Knowhere/parent-child) — adversarial review pushed a clever but narrow idea into an honest, data-driven design.

## 9. Parameters
`target=800, min=200, max=1500` tokens; numbering year-guard 19xx/20xx; LLM routed by doc_type (for types with numbered_frac<0.2), offline and optional, never on the query hot path.
