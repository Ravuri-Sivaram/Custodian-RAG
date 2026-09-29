"""BM25 sparse vectors (the sparse path of hybrid retrieval). Tokens get hashed stably into
uint32 indices, with term frequency (tf) as the values; scoring itself is left to Qdrant's
`Modifier.IDF` (which computes BM25 using the collection's document frequency statistics). No
model involved, pure CPU. On the doc side values=tf, on the query side values=1.

Tokenization has two tiers, chosen at import time based on what's installed:

1. **Preferred: `indic-nlp-library`** (AI4Bharat), if installed. Its `IndicNormalizer` applies
   Unicode canonical normalization *plus* Telugu-specific glyph normalization (e.g. collapsing
   different valid Unicode encodings of the same visible vowel-sign/conjunct into one canonical
   form -- without this, two visually-identical words can hash to different tokens and silently
   miss each other in doc/query matching), and `indic_tokenize` handles Telugu word boundaries
   more carefully than a raw script-range regex (e.g. punctuation glued onto a word). This is a
   real, actively-maintained Indic-NLP library, not a from-scratch regex -- the better technology
   choice for actual Telugu text.
   ⚠ **Not installable or testable in the sandbox this code was written in** (no network access to
   PyPI at the time), so this path is implemented but unverified. Before relying on it in
   production: `pip install indic-nlp-library` (or `pip install -e '.[telugu]'`, see
   pyproject.toml) and confirm `_HAS_INDIC_NLP` below actually comes out `True` and the tokenizer
   handles your real documents as expected.
2. **Fallback: script-boundary regex** (always available, zero extra dependency). English/numeric
   text is split on the same "preserve alphanumeric runs whole" rule BM25 needs anyway (see _ALNUM
   below), and Telugu text is split by extracting each contiguous run of Telugu-script characters
   as one token. This works because Telugu is written with spaces between words, so ordinary
   whitespace and punctuation already do most of the word-boundary-finding job -- there's no need
   for a dictionary-based segmenter for scripts written without inter-word spaces (this pipeline
   previously used `jieba` for that; it had no Telugu equivalent, so it was dropped when this
   project's language support switched to Telugu). Both tiers apply `unicodedata.normalize("NFC",
   ...)` first regardless -- that's a zero-dependency partial mitigation for the same
   glyph-encoding problem the indic-nlp normalizer solves more completely.

Caveat (both tiers): neither is a morphological analyzer. Telugu is agglutinative (case/number/
tense markers attach as suffixes onto a root, e.g. "పుస్తకం" book vs "పుస్తకాలు" books), so unlike
a real Telugu stemmer, inflected forms of the same word are treated as different tokens entirely --
this weakens recall for inflected forms. `indic-nlp-library` ships an unsupervised morphological
analyzer too, but it's not wired in here; see README for the tradeoff this was chosen under.

Exact-term matching is the whole point of choosing BM25: the same _ALNUM regex below still fully
preserves alphanumeric strings like 'GPT-4', 'v1.2', or reference numbers exactly, so model names,
regulation/article numbers, and numeric IDs are matched character-for-character on both the doc
and query side."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from qdrant_client import models

try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize import indic_tokenize

    _INDIC_NORMALIZER = IndicNormalizerFactory().get_normalizer("te")
    _HAS_INDIC_NLP = True
except Exception:
    # Expected in any environment without indic-nlp-library installed (see module docstring) --
    # falls through to the regex tokenizer below. Not narrowed to ImportError because the library
    # can also fail at its own internal resource-loading step, which we still want to degrade from.
    _INDIC_NORMALIZER = None
    _HAS_INDIC_NLP = False

_MASK = (1 << 32) - 1
_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)
# Exact strings: contiguous alphanumeric runs, allowing internal - _ . / connectors (gpt-4, v1.2, 42000000, no.42)
_ALNUM = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*")
# Telugu Unicode block (U+0C00-U+0C7F): a contiguous run of Telugu-script characters is one token.
# Disjoint from _ALNUM's character class (Latin letters/digits), so the two regexes can never
# double-extract the same substring -- unlike the old jieba-based tokenizer, there's no need to
# track "already seen" tokens to avoid double-counting term frequency. Used by the fallback tier
# only; the indic-nlp tier does its own word-boundary segmentation instead.
_TELUGU_WORD = re.compile(r"[ఀ-౿]+")


def _tok_id(tok: str) -> int:
    """Stable hash (FNV-1a 32-bit). NOT Python's hash() -- that one is randomized across
    processes, which would make doc and query hashes not match."""
    h = 0x811C9DC5
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 0x01000193) & _MASK
    return h


def _tokenize_indic_nlp(text: str, stopwords: frozenset[str]) -> list[str]:
    """Preferred tier: indic-nlp-library's normalizer + tokenizer for the Telugu-script portion,
    same _ALNUM exact-string preservation as the fallback tier for everything else."""
    normalized = _INDIC_NORMALIZER.normalize(text)
    toks: list[str] = []
    for t in indic_tokenize.trivial_tokenize(normalized, lang="te"):
        t = t.strip()
        if t and _TELUGU_WORD.search(t) and not _PUNCT.match(t) and t not in stopwords:
            toks.append(t)
    for m in _ALNUM.findall(text):
        if len(m) >= 2 and m not in stopwords:
            toks.append(m)
    return toks


def _tokenize_regex(text: str, stopwords: frozenset[str]) -> list[str]:
    """Fallback tier: script-boundary regex, no external dependency (see module docstring)."""
    toks: list[str] = []
    for t in _TELUGU_WORD.findall(text):
        if t and not _PUNCT.match(t) and t not in stopwords:
            toks.append(t)
    for m in _ALNUM.findall(text):
        if len(m) >= 2 and m not in stopwords:
            toks.append(m)
    return toks


def tokenize(text: str, stopwords: frozenset[str] = frozenset()) -> list[str]:
    """Telugu-script runs plus fully preserved alphanumeric exact strings. Strips pure
    punctuation/whitespace/stopwords. Uses indic-nlp-library when installed (see module
    docstring), otherwise the zero-dependency regex tokenizer -- same call signature either way."""
    text = unicodedata.normalize("NFC", (text or "").lower())
    if _HAS_INDIC_NLP:
        return _tokenize_indic_nlp(text, stopwords)
    return _tokenize_regex(text, stopwords)


def doc_sparse(text: str, stopwords: frozenset[str] = frozenset()) -> models.SparseVector | None:
    """Document side: term frequency (tf) as values (Qdrant's Modifier.IDF multiplies in IDF later). No valid tokens -> None (falls back to dense only)."""
    counts = Counter(_tok_id(t) for t in tokenize(text, stopwords))
    if not counts:
        return None
    return models.SparseVector(indices=list(counts.keys()), values=[float(c) for c in counts.values()])


def query_sparse(text: str, stopwords: frozenset[str] = frozenset()) -> models.SparseVector | None:
    """Query side: any match counts as 1.0 (the BM25 score is determined by document tf x IDF, so the query weight is just 1)."""
    ids = {_tok_id(t) for t in tokenize(text, stopwords)}
    if not ids:
        return None
    return models.SparseVector(indices=list(ids), values=[1.0] * len(ids))
