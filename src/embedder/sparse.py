"""BM25 sparse vectors (the sparse path of hybrid retrieval). Tokens get hashed stably into
uint32 indices, with term frequency (tf) as the values; scoring itself is left to Qdrant's
`Modifier.IDF` (which computes BM25 using the collection's document frequency statistics). No
model involved, pure CPU. On the doc side values=tf, on the query side values=1.

Tokenization is regex-based, not a real segmenter: English/numeric text is split on the same
"preserve alphanumeric runs whole" rule BM25 needs anyway (see _ALNUM below), and Telugu text is
split by extracting each contiguous run of Telugu-script characters as one token. This works
because Telugu (unlike Chinese) is written with spaces between words, so ordinary whitespace and
punctuation already do the word-boundary-finding job -- there's no need for a dictionary-based
segmenter the way Chinese requires one (this pipeline previously used jieba for that; jieba is
Chinese-specific and has no Telugu equivalent, so it was dropped along with Chinese support).

Caveat: this is a script-boundary tokenizer, not a morphological one. Telugu is agglutinative
(case/number/tense markers attach as suffixes onto a root, e.g. "పుస్తకం" book vs "పుస్తకాలు"
books), so unlike a real Telugu stemmer/analyzer, this tokenizer treats those as different tokens
entirely -- it will not realize they share a root. This weakens recall for inflected forms of the
same word. A real Indic NLP stemmer would fix this but is a new, unverified-in-this-environment
dependency; see README for the tradeoff this was chosen under.

Exact-term matching is the whole point of choosing BM25: the same _ALNUM regex below still fully
preserves alphanumeric strings like 'GPT-4', 'v1.2', or reference numbers exactly, so model names,
regulation/article numbers, and numeric IDs are matched character-for-character on both the doc
and query side."""
from __future__ import annotations

import re
from collections import Counter

from qdrant_client import models

_MASK = (1 << 32) - 1
_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)
# Exact strings: contiguous alphanumeric runs, allowing internal - _ . / connectors (gpt-4, v1.2, 42000000, no.42)
_ALNUM = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*")
# Telugu Unicode block (U+0C00-U+0C7F): a contiguous run of Telugu-script characters is one token.
# Disjoint from _ALNUM's character class (Latin letters/digits), so the two regexes can never
# double-extract the same substring -- unlike the old jieba-based tokenizer, there's no need to
# track "already seen" tokens to avoid double-counting term frequency.
_TELUGU_WORD = re.compile(r"[ఀ-౿]+")


def _tok_id(tok: str) -> int:
    """Stable hash (FNV-1a 32-bit). NOT Python's hash() -- that one is randomized across
    processes, which would make doc and query hashes not match."""
    h = 0x811C9DC5
    for ch in tok.encode("utf-8"):
        h = ((h ^ ch) * 0x01000193) & _MASK
    return h


def tokenize(text: str, stopwords: frozenset[str] = frozenset()) -> list[str]:
    """Telugu-script runs plus fully preserved alphanumeric exact strings. Strips pure
    punctuation/whitespace/stopwords."""
    text = (text or "").lower()
    toks: list[str] = []
    for t in _TELUGU_WORD.findall(text):
        if t and not _PUNCT.match(t) and t not in stopwords:
            toks.append(t)
    for m in _ALNUM.findall(text):
        if len(m) >= 2 and m not in stopwords:
            toks.append(m)
    return toks


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
