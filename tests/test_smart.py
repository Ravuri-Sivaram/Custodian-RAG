"""smart-ask (D9, failure-driven version) unit tests: first round is pure / only a refusal
triggers the table-leg retry / auto trail / explicit user params take priority / the switch /
hints. The up-front-leg version was rejected by real testing on 88 questions (it hurt prose
questions), do not revert to it."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from _fakes import FakeRetriever, make_app, make_cfg, make_hit, make_res
from custodian import smart_ask as smart
from generator import Generator, MockLLM


def _gen_factory(r, c):
    return Generator(r, MockLLM())


def _app(retriever, **cfg_kw):
    return make_app(retriever=retriever, cfg=make_cfg(**cfg_kw), generator_factory=_gen_factory)


def test_numeric_good_answer_no_retry():
    # First round already answers (not a refusal) -> no retry triggered, zero extra calls (lesson from the up-front-leg approach)
    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="Net profit $122,641 thousand")])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "What was net profit in 2015?"}).json()
    assert r["status"] == "ok" and r["auto"] == [] and r["hints"] == []
    assert len(ret.calls) == 1


def test_numeric_refusal_triggers_retry_discarded_when_still_refusal():
    # Empty retrieval -> first round refuses -> triggers retry; retry still refuses -> **discarded** (keep the first round's honest refusal), traced as discarded
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "What was Netflix's net profit in 2011?"}).json()
    assert r["auto"] == ["table_leg_retry_discarded"]
    assert len(ret.calls) == 3                       # 1 for the first round + retry round (1 main + 1 leg)
    leg = ret.calls[2]
    assert leg["kind"] == "table" and leg["top_k"] == 5 and leg["rerank"] is True


def test_retry_adopted_when_fully_answered():
    # First round: empty retrieval, refuses; retry round: retrieval returns results -> fully answered (not a refusal) -> retry answer is adopted
    state = {"n": 0}

    def factory():
        state["n"] += 1
        return [] if state["n"] == 1 else [make_res(make_hit(), ctx_text="Net income 226,126 thousand")]

    ret = FakeRetriever(results_factory=factory)
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "What was Netflix's net profit in 2011?"}).json()
    assert r["auto"] == ["table_leg_retry"] and r["status"] == "ok"
    assert "[cite:1]" in r["answer"] and r["hints"] == []          # fully answered, no hints


def test_non_numeric_refusal_no_retry():
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "What is the core method idea of this paper"}).json()
    assert r["auto"] == [] and len(ret.calls) == 1


def test_explicit_kind_respected_no_retry():
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "What was net profit in 2015?", "kind": "text"}).json()
    assert r["auto"] == [] and len(ret.calls) == 1


def test_smart_off_pure_mode():
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret, smart_ask=False)) as c:
        r = c.post("/v1/ask", json={"query": "What was net profit in 2015?"}).json()
    assert r["auto"] == [] and r["hints"] == [] and len(ret.calls) == 1


def test_refusal_hints_present_and_no_duplicate_suggestion():
    # Still refuses after retry -> hints appear; table leg already auto-attempted -> no longer
    # suggests kind=table; a Telugu-script query -> language-mismatch hint (best-effort Telugu
    # phrase, not reviewed by a fluent speaker -- see signals.py; roughly "What was Netflix's 2011
    # net profit?")
    ret = FakeRetriever(results_factory=lambda: [])
    with TestClient(_app(ret)) as c:
        r = c.post("/v1/ask", json={"query": "నెట్‌ఫ్లిక్స్ 2011 నికర లాభం ఎంత?"}).json()
    assert r["auto"] == ["table_leg_retry_discarded"] and r["hints"]
    assert not any("kind" in h and "table" in h for h in r["hints"])
    assert any("English documents" in h or "document's language" in h for h in r["hints"])


def test_discarded_retry_keeps_first_round_finish_reason():
    # Fix: when the retry is discarded, the first round's answer is returned, so finish_reason
    # must also be the first round's (taken from the Answer snapshot) -- previously this read
    # gen.llm.last_finish_reason, which held the discarded second round's value, misaligning the
    # truncation diagnostic signal with the answer.
    class _TwoRoundRefusalLLM:
        """Both rounds refuse but with different finish_reason values: round one is length
        (truncation causes the refusal), round two is stop."""
        def __init__(self):
            self.n = 0
            self.last_finish_reason = None
        def complete(self, messages):
            self.n += 1
            self.last_finish_reason = "length" if self.n == 1 else "stop"
            return "I don't have enough information to answer."

    ret = FakeRetriever(results_factory=lambda: [make_res(make_hit(), ctx_text="prose text, no digits")])
    app = make_app(retriever=ret, generator_factory=lambda r, c: Generator(r, _TwoRoundRefusalLLM()))
    with TestClient(app) as c:
        r = c.post("/v1/ask", json={"query": "What was net profit in 2015?"}).json()
    assert r["auto"] == ["table_leg_retry_discarded"]        # confirms it took the discard branch
    assert r["finish_reason"] == "length"                    # the first round's value (the old implementation returned the second round's "stop")


def test_is_refusal_patterns():
    assert smart.is_refusal("For 2011, I don't have enough information.")
    assert smart.is_refusal("I don't have enough information to answer.")
    assert smart.is_refusal("The context does not provide this data, unable to answer.")
    assert smart.is_refusal("No net profit data found for Netflix for 2011 and 2012.")
    assert not smart.is_refusal("Net profit for 2015 was $122,641 thousand [cite:1].")
