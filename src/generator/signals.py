"""Query signals: lightweight rule-based judgments (zero LLM calls) that let the product layer of the
closed pipeline do cheap routing.

This module is shared by Custodian smart-ask and eval --smart-tables (a single source of truth, to
prevent the two word lists from drifting apart). The judgments are deliberately lenient: the cost of a
false positive is just unioning in a few extra table chunks (cheap), while the cost of a false negative
is an incomplete answer to a numeric question (expensive).
"""
from __future__ import annotations

import re

# NOTE on the Telugu segment below: best-effort, NOT written or reviewed by a fluent Telugu
# speaker. Telugu numeral/quantity words and common financial terms are included on the same
# lenient-matching principle as the rest of this regex (false positives just union in a few extra
# table chunks, cheap; false negatives risk an incomplete numeric answer, expensive).
# Get this reviewed by a native speaker before relying on it in production.
_NUM_HINT = re.compile(
    r"[0-9౦-౯]"                                                      # Arabic digits + Telugu digits (౦-౯)
    r"|ఎంత|శాతం|నిష్పత్తి|వృద్ధి|పెరుగుదల|తగ్గుదల"                        # "how much", "percent", "ratio", "growth", "increase", "decrease"
    r"|ఆదాయం|లాభం|ఖర్చు|రుసుము|మొత్తం|ధర|మార్కెట్\s*విలువ|అమ్మకాలు"         # revenue/profit/cost/fee/amount/price/market value/sales
    r"|how\s+(?:much|many)|percent|revenue|income|profit|price|cost|amount|total",
    re.I)

# Recommended supplemental table-retrieval "leg" for numeric questions (unioned with the main
# retrieval, not a replacement): kind=table removes competition from prose chunks, and the smaller
# rerank pool corrects a cross-language bias (observed in testing: for a non-English year-by-year net
# profit question, the five-year table went from outside the top 20 for the whole corpus to rank 4).
# WARNING: this must be used in a failure-driven way (only retried with the extra leg after the first
# round refuses to answer) -- it must not be applied up front. Tested on 88 questions: applying the
# leg up front raised table-question accuracy from 0.625 to 0.875, but at the same time broke 5
# previously-correct prose questions (nearby numbers introduced bias / over-caution), dropping prose
# accuracy from 0.861 to 0.792. Under the failure-driven approach, questions that are already answered
# correctly never trigger the extra leg, so there is zero collateral damage. See custodian TESTING section 3
# for the write-up.
# rerank_top_n=50 (full depth by default): a hard-won lesson from testing -- at 30, the five-year table
# (ranked 31-50 by the coarse ranker in mixed retrieval) never made it into the rerank pool, making the
# retry leg useless. The rerank pool depth must be >= "the worst coarse-rank position the correct chunk
# can land at"; cross-language table questions were observed in testing to need 50.
DEFAULT_TABLE_LEG = {"kind": "table", "top_k": 5, "rerank": True, "rerank_top_n": 50}

# Refusal / partial-refusal patterns (Telugu and English). Shared by custodian smart-ask and
# eval --smart-tables (a single source of truth).
# The Telugu phrases are best-effort, NOT written or reviewed by a fluent Telugu speaker -- same
# caveat as _NUM_HINT above. Get this reviewed before relying on it in production.
_REFUSAL = re.compile(
    r"తగినంత సమాచారం లేదు|సమాచారం సరిపోదు|సమాధానం చెప్పలేను|సమాధానం ఇవ్వలేను|నిర్ధారించలేను"
    r"|లెక్కించలేను|అందించబడలేదు|సంబంధిత సమాచారం లేదు|కనుగొనబడలేదు|దొరకలేదు"
    r"|don'?t have enough|not contain|no (?:relevant )?information|unable to answer|cannot answer"
    r"|not explicitly stated|not (?:directly )?provided",
    re.I)


def looks_numeric(query: str) -> bool:
    """Whether the question is likely to need a numeric answer (digits / quantity words /
    financial amount words / "how much", etc.)."""
    return bool(_NUM_HINT.search(query or ""))


def is_refusal(answer_text: str) -> bool:
    """Whether the answer is a refusal or partial refusal (including partial-missing-data statements
    like "no data available for year X")."""
    return bool(_REFUSAL.search(answer_text or ""))
