"""BM25 sparse unit tests (pure CPU). Focus: verifying exact strings are preserved -- the whole
reason for choosing BM25 in the first place."""

import unicodedata

from embedder.sparse import _tok_id, doc_sparse, query_sparse, tokenize


def test_exact_string_preserved():
    # Exact strings (model numbers/versions/IDs/numbers) must come through as whole tokens, or
    # BM25's exact-match advantage is destroyed
    toks = tokenize("Using GPT-4 and v1.2, per Article 42, revenue 42000000 dollars")
    assert "gpt-4" in toks, toks
    assert "v1.2" in toks, toks
    assert "42000000" in toks, toks


def test_doc_query_consistent_hash():
    # doc and query with the same token -> the same hash index (otherwise the sparse leg of
    # hybrid retrieval can't line up)
    d = doc_sparse("Company revenue 42000000 dollars")
    q = query_sparse("42000000")
    assert d is not None and q is not None
    assert set(q.indices) & set(d.indices), "the exact string 42000000 doesn't line up between doc/query"


def test_empty_returns_none():
    assert doc_sparse("") is None
    assert doc_sparse(". , ! ") is None          # pure punctuation -> no valid tokens


def test_telugu_word_extracted_as_one_token():
    # A run of Telugu-script characters must come out as a single token (word-boundary
    # extraction, not per-character), and doc/query hashing must line up for it same as for
    # alphanumeric text.
    toks = tokenize("ఆదాయం విశ్లేషణ నివేదిక")   # "revenue analysis report" (best-effort; see sparse.py docstring)
    assert len(toks) == 3, toks
    d = doc_sparse("ఆదాయం విశ్లేషణ నివేదిక")
    q = query_sparse("ఆదాయం")
    assert d is not None and q is not None
    assert set(q.indices) & set(d.indices), "a Telugu token doesn't line up between doc/query"


def test_nfc_normalization_collapses_equivalent_telugu_encodings():
    # The same visible Telugu text can be encoded as different (but canonically-equivalent) Unicode
    # codepoint sequences. Concretely: the vowel sign AI (U+0C48, "ై") has a canonical decomposition
    # to vowel sign E + the AI length mark (U+0C46 U+0C56) -- text containing it can arrive either
    # precomposed or decomposed depending on the source tool, and without normalization these two
    # byte-for-byte-different encodings of the identical visible word would hash to different tokens
    # and silently fail to match between doc and query. tokenize() applies
    # unicodedata.normalize("NFC", ...) up front specifically to prevent this.
    raw = "నైనం"                              # contains the precomposed vowel sign AI (U+0C48)
    decomposed = unicodedata.normalize("NFD", raw)   # same visible word, decomposed encoding
    assert raw != decomposed, "test fixture isn't actually exercising a decomposed variant"
    assert tokenize(raw) == tokenize(decomposed)


def test_no_double_count_tf():
    # An alphanumeric token is never re-extracted by the Telugu-script regex (the two character
    # classes are disjoint) -> doc tf isn't double-counted.
    assert tokenize("model model").count("model") == 2   # genuinely 2, not 4
    d = doc_sparse("gpt-4")
    gid = _tok_id("gpt-4")
    assert d.values[list(d.indices).index(gid)] == 1.0, (d.indices, d.values)


if __name__ == "__main__":
    test_exact_string_preserved()
    test_doc_query_consistent_hash()
    test_empty_returns_none()
    test_telugu_word_extracted_as_one_token()
    test_no_double_count_tf()
    print("sparse tests OK")
    print("sample tokens:", tokenize("Article42 GPT-4 revenue42000000 netprofit"))
