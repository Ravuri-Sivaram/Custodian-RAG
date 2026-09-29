"""Generator unit tests (pure CPU, MockLLM + fake retriever, duck-typed with no dependency on
embedder). Covers: citation parsing, ACL exit-point degradation (context=None falls back to
hit.text), the grounding fallback (no context), and out-of-range citations being dropped."""


from generator.synthesis import Generator
from generator.llm import MockLLM


class _Hit:
    def __init__(self, cid, doc, text, payload):
        self.chunk_id, self.doc_id, self.text, self.payload = cid, doc, text, payload


class _Ctx:
    def __init__(self, text):
        self.text = text


class _Ret:
    def __init__(self, results):
        self._r = results

    def search_with_context(self, query, user, top_k=None, rerank=False):
        return self._r


def test_answer_with_citations():
    results = [
        {"hit": _Hit("c1", "d1", "hit1", {"doc_meta": {"title": "Doc One"}, "section_path": "S1", "page_start": 3}),
         "context": _Ctx("big context one")},
        {"hit": _Hit("c2", "d2", "hit2", {"doc_meta": {"title": "Doc Two"}}), "context": _Ctx("big context two")},
    ]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.n_contexts == 2
    assert [c.marker for c in ans.citations] == [1, 2]          # MockLLM cites [1][2]
    assert ans.citations[0].title == "Doc One" and ans.citations[0].page == 3
    assert "big context one" in ans.raw_messages[1].content     # context made it into the prompt


def test_acl_exit_degrade_to_hit():
    # context=None (sidecar missing / didn't clear the exit check) -> falls back to hit.text
    # (a single chunk that's already authorized), the answer isn't lost
    results = [{"hit": _Hit("c1", "d1", "hit text only", {}), "context": None}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.n_contexts == 1 and "hit text only" in ans.raw_messages[1].content


def test_no_context_grounding():
    # no retrieval results -> no context -> MockLLM takes the grounding fallback (insufficient information), 0 citations
    ans = Generator(_Ret([]), MockLLM()).answer("q", user=None)
    assert ans.n_contexts == 0 and ans.citations == [] and "enough information" in ans.text.lower()


def test_citation_out_of_range_dropped():
    # the answer hallucinates an out-of-range [cite:99] -> dropped, not mapped to the wrong source
    class _BadLLM:
        def complete(self, messages):
            return "answer [cite:1] and [cite:99]"
    results = [{"hit": _Hit("c1", "d1", "t", {}), "context": _Ctx("ctx")}]
    ans = Generator(_Ret(results), _BadLLM()).answer("q", user=None)
    assert [c.marker for c in ans.citations] == [1]            # [cite:99] is out of range, dropped


def test_context_bracket_not_polluting():
    # a bare [1]/[99] the LLM copied verbatim from the body must not be treated as a citation; only
    # [cite:n] counts (review #1's core point: provenance must not be polluted by body text)
    class _EchoLLM:
        def complete(self, messages):
            return "Doc says [1] and [99] are footnotes. Real cite [cite:2]."
    results = [{"hit": _Hit("c1", "d1", "t1", {}), "context": _Ctx("ctx1")},
               {"hit": _Hit("c2", "d2", "t2", {}), "context": _Ctx("ctx2")}]
    ans = Generator(_Ret(results), _EchoLLM()).answer("q", user=None)
    assert [c.marker for c in ans.citations] == [2]           # body's [1][99] don't pollute, only [cite:2] counts


def test_asset_content_raw_appended():
    # ③ table/chart grounding: the big-block (ctx) doesn't include asset data -> content_raw is added back into the prompt; non-asset blocks aren't
    results = [
        {"hit": _Hit("c1", "d1", "chart caption", {"kind": "chart", "content_raw": "DATA revenue=42M"}),
         "context": _Ctx("surrounding prose without the number")},
        {"hit": _Hit("c2", "d2", "t2", {"kind": "text", "content_raw": "SHOULD_NOT_APPEAR"}),
         "context": _Ctx("plain text ctx")},
    ]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    prompt = ans.raw_messages[1].content
    assert "DATA revenue=42M" in prompt              # the chart's content_raw is added back (the big-block is missing it)
    assert "SHOULD_NOT_APPEAR" not in prompt         # non-asset blocks' content_raw is not added


def test_asset_long_content_raw_not_duplicated():
    # a long content_raw already in ctx (the big-block) -> not appended again (dedup only applies
    # to long assets, to avoid feeding a long table's HTML twice)
    blob = "Quarterly revenue table: Q1 100 Q2 200 Q3 300 Q4 400 total 1000 units"   # >40 chars
    results = [{"hit": _Hit("c1", "d1", "cap", {"kind": "table", "content_raw": blob}),
               "context": _Ctx("Preamble. " + blob + " Footnote.")}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.raw_messages[1].content.count("Q2 200") == 1     # the long content_raw is already in ctx, not duplicated


def test_asset_short_content_raw_always_appended():
    # R2#2: short asset data (like "42") is always added back even when it happens to be a
    # substring of the prose -- substring-based dedup would wrongly suppress it, reproducing bug ③
    results = [{"hit": _Hit("c1", "d1", "cap", {"kind": "table", "content_raw": "42"}),
               "context": _Ctx("revenue grew by 42 percent overall")}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.raw_messages[1].content.count("42") >= 2         # short data is always added back (otherwise wrongly suppressed)


def test_empty_context_deterministic_grounding():
    # R3 E: zero recall -> deterministically returns "insufficient information", never calls the
    # LLM at all (the grounding fallback doesn't rely on the model behaving)
    class _BoomLLM:
        def complete(self, messages):
            raise AssertionError("zero recall must not call the LLM")
    ans = Generator(_Ret([]), _BoomLLM()).answer("q", user=None)
    assert ans.n_contexts == 0 and ans.citations == [] and "enough information" in ans.text.lower()


def test_citation_title_falls_back_to_doc_id():
    # R3 D: no doc_meta.title -> Citation.title falls back to doc_id (never an empty string, consistent with the source fed to the LLM)
    results = [{"hit": _Hit("c1", "mydoc", "t", {}), "context": _Ctx("ctx")}]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.citations and ans.citations[0].title == "mydoc"


def test_acl_check_filters():
    # when acl_check is injected, an unauthorized hit is blocked from reaching the prompt (defense in depth, review #3)
    results = [{"hit": _Hit("c1", "d1", "ok text", {"acl": {"v": "pub"}}), "context": _Ctx("ctx1")},
               {"hit": _Hit("c2", "d2", "SECRET", {"acl": {"v": "secret"}}), "context": _Ctx("ctx2")}]
    g = Generator(_Ret(results), MockLLM(), acl_check=lambda acl, u: acl.get("v") == "pub")
    ans = g.answer("q", user=None)
    assert ans.n_contexts == 1 and "SECRET" not in ans.raw_messages[1].content   # the secret block is blocked from the prompt


def test_filters_forwarded_only_when_set():
    # N3: filter params are forwarded on demand -- not set, not forwarded (compatible with an old
    # narrow-signature retriever); when set, forwarded to the retriever as-is
    class _CapRet:
        def __init__(self):
            self.kw = None

        def search_with_context(self, query, user, top_k=None, rerank=False, **kw):
            self.kw = kw
            return []

    cap = _CapRet()
    gen = Generator(cap, MockLLM())
    gen.answer("q", user=None)                                   # no filters set
    assert cap.kw == {}
    gen.answer("q", user=None, kind="table", doc_ids=["d1"], strategy="sparse")
    assert cap.kw == {"kind": "table", "doc_ids": ["d1"], "strategy": "sparse"}
    # an old narrow signature (no **kw) still works when no filters are passed
    class _Narrow:
        def search_with_context(self, query, user, top_k=None, rerank=False):
            return []
    Generator(_Narrow(), MockLLM()).answer("q", user=None)       # doesn't raise TypeError


if __name__ == "__main__":
    test_answer_with_citations()
    test_acl_exit_degrade_to_hit()
    test_no_context_grounding()
    test_citation_out_of_range_dropped()
    test_context_bracket_not_polluting()
    test_asset_content_raw_appended()
    test_asset_long_content_raw_not_duplicated()
    test_asset_short_content_raw_always_appended()
    test_empty_context_deterministic_grounding()
    test_citation_title_falls_back_to_doc_id()
    test_acl_check_filters()
    test_filters_forwarded_only_when_set()
    print("generate tests OK")


def test_system_numeric_scope_constraint():
    # Numeric-scope constraint (a narrow target, guards against "a segment number being extrapolated
    # to the total" -- an actual wrong-answer case found in Custodian N3): goes into SYSTEM's prompt
    from generator.prompt import SYSTEM, PromptBuilder
    assert "scoped" in SYSTEM and "NEVER present it as the total" in SYSTEM
    msgs = PromptBuilder().build("q", [{"text": "Domestic streaming revenues 4,180,339", "source": "s"}])
    assert "NEVER present it as the total" in msgs[0].content


def test_extra_legs_union_dedup():
    # smart-ask multi-leg retrieval: leg hits are unioned in after the main hits, deduped by chunk_id; passing no legs has zero effect
    class _CapRet:
        def __init__(self):
            self.calls = []

        def search_with_context(self, query, user, top_k=None, rerank=False, **kw):
            self.calls.append({"top_k": top_k, "rerank": rerank, **kw})
            if kw.get("kind") == "table":
                return [{"hit": _Hit("t1", "d1", "table chunk", {"kind": "table"}), "context": _Ctx("five-year table")},
                        {"hit": _Hit("c1", "d1", "duplicate", {}), "context": _Ctx("dup")}]   # c1 duplicates the main leg
            return [{"hit": _Hit("c1", "d1", "prose chunk", {}), "context": _Ctx("primary-path context")}]

    cap = _CapRet()
    from generator.signals import DEFAULT_TABLE_LEG
    ans = Generator(cap, MockLLM()).answer("2015 net profit?", user=None, extra_legs=[DEFAULT_TABLE_LEG])
    assert len(cap.calls) == 2 and cap.calls[1]["kind"] == "table" and cap.calls[1]["rerank"] is True
    assert cap.calls[1]["top_k"] == 5 and cap.calls[1]["rerank_top_n"] == 50
    assert ans.n_contexts == 2                            # main leg's c1 + the leg's new t1; duplicate c1 deduped
    assert "primary-path context" in ans.raw_messages[1].content and "five-year table" in ans.raw_messages[1].content


def test_looks_numeric():
    from generator.signals import looks_numeric
    assert looks_numeric("What was Netflix's net profit for each year from 2011 to 2015?")
    assert looks_numeric("How has the company's revenue share changed")
    assert looks_numeric("How much were total revenues?")
    assert not looks_numeric("What is the core methodological idea of this paper")
    assert not looks_numeric("What is the main contribution of this paper?")


# ---------- Fix: finish_reason is snapshotted into Answer (no longer read from the llm instance's attribute after the fact) ----------
class _FRLLM:
    """A fake LLM with last_finish_reason: it changes value on each call (simulating the instance
    attribute being overwritten by a later call)."""

    def __init__(self, reasons=("length", "stop"), text="answer [cite:1]"):
        self._reasons, self._text, self.n = list(reasons), text, 0
        self.last_finish_reason = None

    def complete(self, messages):
        self.last_finish_reason = self._reasons[min(self.n, len(self._reasons) - 1)]
        self.n += 1
        return self._text


def test_finish_reason_snapshot_into_answer():
    results = [{"hit": _Hit("c1", "d1", "t", {}), "context": _Ctx("ctx")}]
    llm = _FRLLM(reasons=("length", "stop"))
    gen = Generator(_Ret(results), llm)
    a1 = gen.answer("q", user=None)
    a2 = gen.answer("q", user=None)
    assert a1.finish_reason == "length" and a2.finish_reason == "stop"
    assert a1.finish_reason == "length"          # the second call doesn't rewrite the first Answer (a snapshot, not an instance reference)


def test_finish_reason_none_on_zero_recall_not_residual():
    # a zero-recall early exit never calls the LLM: finish_reason must be None, not the leftover
    # value from a previous request on the same llm instance
    llm = _FRLLM()
    llm.last_finish_reason = "length"            # simulates leftover state from a previous request
    ans = Generator(_Ret([]), llm).answer("q", user=None)
    assert ans.n_contexts == 0 and ans.finish_reason is None


# ---------- Fix: citation parsing and neutralization share the same loose CITE_RE (tolerates whitespace/case) ----------
def test_citation_spacing_variants_parsed():
    class _LooseLLM:
        def complete(self, messages):
            return "answer [cite: 1] and [Cite:2]"     # a formatting drift common on non-DeepSeek backends
    results = [{"hit": _Hit("c1", "d1", "t1", {}), "context": _Ctx("x1")},
               {"hit": _Hit("c2", "d2", "t2", {}), "context": _Ctx("x2")}]
    ans = Generator(_Ret(results), _LooseLLM()).answer("q", user=None)
    assert [c.marker for c in ans.citations] == [1, 2]   # previously, under the strict regex, citations=[] (silently dropped all of them)


# ---------- Fix: a soft token budget on the total context (over budget truncates the tail, meta stays in sync) ----------
def test_context_token_budget_truncates_tail():
    # est_tokens("x"*400, "")=100 per item; budget 250 -> keeps the first 2 items (the main hits
    # come first, so it's the tail that gets truncated)
    results = [{"hit": _Hit(f"c{i}", "d1", f"t{i}", {}), "context": _Ctx("x" * 400)} for i in range(5)]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None, max_context_tokens=250)
    assert ans.n_contexts == 2
    assert [c.chunk_id for c in ans.citations] == ["c0", "c1"]   # meta is truncated in sync, numbering doesn't misalign
    assert "x" * 400 in ans.raw_messages[1].content


def test_context_budget_keeps_first_even_if_over():
    # a single item that alone exceeds the budget is not cleared: degrading to zero context (and a
    # refusal) loses more information than going over budget (a single item already has its own
    # ceiling from the chunker's BUDGETS)
    results = [{"hit": _Hit("c1", "d1", "t", {}), "context": _Ctx("y" * 4000)}]
    gen = Generator(_Ret(results), MockLLM(), max_context_tokens=100)
    ans = gen.answer("q", user=None)
    assert ans.n_contexts == 1


def test_context_budget_default_off():
    # default None: behavior is completely unchanged (all context goes into the prompt)
    results = [{"hit": _Hit(f"c{i}", "d1", f"t{i}", {}), "context": _Ctx("x" * 400)} for i in range(5)]
    ans = Generator(_Ret(results), MockLLM()).answer("q", user=None)
    assert ans.n_contexts == 5


# ---------- Fix: OpenAICompatibleLLM raises on empty content + an abnormal finish_reason (instead of treating it as a legitimate empty answer) ----------
def _openai_llm_with_stub(content, finish_reason, reasoning=None):
    import pytest
    pytest.importorskip("openai")
    from types import SimpleNamespace

    from generator.llm import OpenAICompatibleLLM
    llm = OpenAICompatibleLLM(model="m", api_key="test-key")
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)])
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: resp)))
    return llm


def _msgs():
    from generator.types import Message
    return [Message(role="user", content="q")]


def test_empty_content_content_filter_raises():
    import pytest
    llm = _openai_llm_with_stub("", "content_filter")
    with pytest.raises(RuntimeError, match="content_filter"):
        llm.complete(_msgs())


def test_empty_content_length_with_reasoning_raises():
    # thinking burned through max_tokens on reasoning_content, leaving content empty -> raises (does not return status=ok with an empty answer)
    import pytest
    llm = _openai_llm_with_stub("", "length", reasoning="a very long chain of thought...")
    with pytest.raises(RuntimeError, match="length"):
        llm.complete(_msgs())


def test_empty_content_stop_passes_through():
    # conservative: finish_reason=stop (the model genuinely answered empty) is not wrongly flagged
    assert _openai_llm_with_stub("", "stop").complete(_msgs()) == ""


def test_nonempty_content_length_returns():
    # truncated but has content: returned as usual (the truncation signal travels via finish_reason, for the caller to handle)
    llm = _openai_llm_with_stub("partial answer", "length")
    assert llm.complete(_msgs()) == "partial answer"
    assert llm.last_finish_reason == "length"


# ---------- Fix: send_thinking's auto-detection widened to also check base_url or the model name (gateway/proxy scenarios) ----------
def test_send_thinking_gateway_by_model_name():
    import pytest
    pytest.importorskip("openai")
    from generator.llm import OpenAICompatibleLLM
    gw = "https://llm-gw.corp/v1"                # a gateway URL with no "deepseek" substring
    assert OpenAICompatibleLLM(model="deepseek-v4-flash", base_url=gw, api_key="k").send_thinking is True
    assert OpenAICompatibleLLM(model="qwen3-32b", base_url=gw, api_key="k").send_thinking is False
    assert OpenAICompatibleLLM(model="qwen3-32b", base_url="https://api.deepseek.com",
                               api_key="k").send_thinking is True
    # an explicit override always takes priority over auto-detection
    assert OpenAICompatibleLLM(model="deepseek-v4-flash", base_url=gw, api_key="k",
                               send_thinking=False).send_thinking is False
