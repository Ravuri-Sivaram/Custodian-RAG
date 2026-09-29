"""LLM client: a pluggable protocol + MockLLM (for scaffolding verification) + OpenAICompatibleLLM
(a real backend).

A real implementation only needs `complete(messages) -> str`. The interface uses the OpenAI-compatible
messages shape, which is compatible with local backends (vLLM) as well as the various hosted APIs.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from .types import Message


class LLMClient(Protocol):
    """Any object that implements complete(messages) -> str can be plugged into Generator."""

    def complete(self, messages: list[Message]) -> str: ...


class MockLLM:
    """A fake LLM: doesn't call a real model, just echoes back an answer with citations built from the
    context numbers in the prompt, used to verify the generator scaffolding.
    Behavior: cites the first min(2, number of distinct markers) contexts; returns "not enough
    information" when there's no context (to test the grounding fallback)."""

    def complete(self, messages: list[Message]) -> str:
        user = next((m.content for m in messages if m.role == "user"), "")
        # Only counts the [cite:n] markers added by PromptBuilder (bare [n] in body text doesn't
        # count) -> the decision is based on the real context count, uncontaminated by body text.
        markers = sorted({int(x) for x in re.findall(r"\[cite:(\d+)\]", user)})
        if not markers:
            return "I don't have enough information in the provided context to answer."
        refs = "".join(f"[cite:{n}]" for n in markers[:2])
        return f"Based on the retrieved context {refs}, here is a grounded mock answer citing the provided sources."


def messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Converts a list[Message] into the [{'role','content'}] shape each LLM API expects. Reused by
    real backends (OpenAI/vLLM/Claude) when implementing complete(), so each backend doesn't have to
    write its own error-prone version of this conversion."""
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAICompatibleLLM:
    """An LLMClient for any OpenAI-compatible chat/completions endpoint -- works with DeepSeek, a GLM
    proxy, or a local vLLM alike. Model/endpoint/thinking toggle are all configuration, never hardcoded:
    switching backends just means changing base_url+model, with zero lines of complete() touched.
    Requires `pip install openai`.

    Thinking mode (DeepSeek V4): `thinking=True` -> extra_body {"thinking":{"type":"enabled"}}; when
    off, `disabled` is sent explicitly -- V4 Flash's thinking may default to on, and not explicitly
    turning it off causes the backend to require reasoning_content to be sent back, resulting in a 400.
    The chain of thought comes back via `reasoning_content`, kept separate from `content` so it doesn't
    contaminate the answer's [cite:n] format; each call stores it to self.last_reasoning (kept separate,
    not fed back or shown by default).

    Three knobs for evaluation: model (flash/pro), thinking (on/off), reasoning_effort (high/max) --
    swept together in eval, with the results used to pick the configuration. Grounded RAG defaults to
    temperature=0 (for faithfulness and reproducibility). The key is read from api_key= or an
    environment variable (default DEEPSEEK_API_KEY); keep it in .env and don't commit it."""

    def __init__(self, *, model: str, base_url: str = "https://api.deepseek.com",
                 api_key: str | None = None, api_key_env: str = "DEEPSEEK_API_KEY",
                 thinking: bool = False, reasoning_effort: str | None = None,
                 max_tokens: int = 2000, temperature: float = 0.0, timeout: float = 120,
                 send_thinking: bool | None = None):
        from openai import OpenAI       # lazy import: importing generator doesn't trigger importing openai (only needed when actually constructing this LLM backend; openai is a core dependency)
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"Missing API key: pass api_key= or set the {api_key_env} environment variable (recommended: put it in .env, and don't commit it).")
        self._client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        # `thinking` is a DeepSeek-specific extra_body field; native OpenAI, most vLLM setups, and other
        # compatible proxies will 400 on an unrecognized body field. So it's only injected for
        # DeepSeek-family backends (overridable explicitly via send_thinking=), preserving the
        # "works with any OpenAI-compatible backend" contract.
        # The automatic detection looks at both base_url and the model name: in a corporate gateway/
        # proxy setup, the URL may not contain the "deepseek" substring even though the model name
        # usually still does -- checking the URL alone would silently fail to send `disabled`, and if
        # V4 Flash's thinking defaults to on, that means anywhere from doubled latency/cost to an
        # outright 400 (see the class docstring).
        self.send_thinking = send_thinking if send_thinking is not None else (
            "deepseek" in base_url.lower() or "deepseek" in model.lower())
        self.last_reasoning: str | None = None      # the most recent chain of thought (reasoning_content), stored separately, never merged into the answer
        self.last_finish_reason: str | None = None  # the most recent finish reason; == 'length' means the answer was truncated by max_tokens (a trailing [cite:n] may have been cut off)

    def complete(self, messages: list[Message]) -> str:
        extra: dict = {}
        if self.send_thinking:
            extra["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
            if self.thinking and self.reasoning_effort:
                extra["reasoning_effort"] = self.reasoning_effort
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages_to_dicts(messages),
            max_tokens=self.max_tokens, temperature=self.temperature, extra_body=extra or None)
        # A content-filter hit or an upstream error can come back wrapped as choices=[]/None -> raise a
        # clear error here rather than letting it IndexError or silently returning an empty answer
        # (which needs to be distinguished from "the model legitimately answered with nothing").
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        if choice is None:
            raise RuntimeError("LLM returned empty choices (possibly a content-filter hit or an upstream error); unable to answer")
        self.last_finish_reason = getattr(choice, "finish_reason", None)   # surface the truncation signal for callers/eval to filter on
        self.last_reasoning = getattr(choice.message, "reasoning_content", None)   # chain of thought kept separate, never mixed into content
        content = choice.message.content or ""
        # Additional check: if content is empty and finish_reason is not a normal one (e.g. the content
        # filter zeroed it out, or thinking mode burned through max_tokens entirely inside
        # reasoning_content), raise via the same path as the empty-choices case above. Otherwise an
        # empty string is returned as a "legitimate empty answer" -- is_refusal("") is False, so
        # hints/retries/observability's error counters would all be bypassed (a false-green result of
        # status=ok plus an empty answer). stop/None are conservatively allowed through unchanged (the
        # model may have legitimately answered with nothing, or the backend doesn't return a
        # finish_reason at all -- neither case should be flagged as an error).
        if not content.strip() and self.last_finish_reason not in ("stop", None):
            raise RuntimeError(f"LLM returned empty content (finish_reason={self.last_finish_reason}; "
                               f"possibly a content-filter hit or thinking mode exhausting max_tokens); unable to answer")
        return content
