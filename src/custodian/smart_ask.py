"""The smart-ask product layer (the logic that makes /v1/ask "smarter"; see DESIGN.md D9):

Layer 1, hints: when the answer is a refusal or a partial refusal, generate actionable
suggestions from this request's parameters (teaching the user/agent which knobs to turn).
Layer 2, failure-driven table retry: for numeric questions, **only when the first round is a
refusal or partial refusal**, re-query once with an extra retrieval leg using kind=table (unioned
with, not replacing, the main retrieval). Note: we deliberately do NOT run this leg up front --
in testing on 88 questions, running it up front raised table-question accuracy from 0.625 to
0.875 but also broke 5 prose questions that were previously answered correctly (nearby numeric
values biasing the answer, or excessive caution). Because it's failure-driven, questions that were
already answered correctly never trigger it, so there's zero blast radius there.

Design constraints: no looping/iterative reasoning (a hard cap of 1 retry round; in testing,
agent-style orchestration was a net negative on this workload, and that's the MCP exit point's
responsibility anyway); every automatic action leaves a trace in the response's `auto` field;
CUSTODIAN_SMART_ASK=off switches everything back to plain mode with one flag.
The parameters for numeric detection, refusal detection, and the retry leg all come from the
engine's generator.signals (the same source eval --smart-tables uses, to prevent drift).
"""
from __future__ import annotations

import re

from generator import is_refusal as _ir

_TELUGU = re.compile(r"[ఀ-౿]")


def is_refusal(answer_text: str) -> bool:
    return _ir(answer_text)


def build_hints(query: str, *, auto: list[str], req_kind, req_rerank: bool, numeric: bool) -> list[str]:
    """Next-step suggestions for a refusal. Doesn't repeat an action already taken automatically
    or already used by the caller; at most 3, ordered by cost-effectiveness."""
    retried = any(a.startswith("table_leg") for a in auto)
    hints: list[str] = []
    if numeric and req_kind is None and not retried:
        hints.append('For numeric questions, try adding "kind": "table" to search only within tables (CLI: --kind table).')
    if _TELUGU.search(query or ""):
        hints.append("This library is mostly English documents: for numeric/terminology questions, using keywords in the document's language (e.g. net income) usually improves hit rate significantly.")
    if not req_rerank and not retried:
        hints.append('Try adding "rerank": true for a reranking pass that corrects ordering (a few seconds slower, especially effective for tables/cross-language cases).')
    hints.append("You can also use /v1/retrieve (mode=concise, with a larger top_k) to check where the correct content ranks, to figure out which knob to adjust.")
    return hints[:3]
