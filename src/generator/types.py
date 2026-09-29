"""Data contracts for generator (pure stdlib)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str            # system | user | assistant (OpenAI-compatible)
    content: str


@dataclass
class Citation:
    marker: int          # the [n] marker in the answer
    chunk_id: str
    doc_id: str
    title: str           # doc_meta.title (shown with the citation)
    section: str         # section_path / breadcrumb
    page: int
    text: str            # the context text being cited (for provenance)


@dataclass
class Answer:
    text: str                                 # the LLM-generated answer (with [n] citation markers)
    citations: list[Citation]                 # sources actually cited in the answer (deduped, sorted by marker)
    n_contexts: int                           # number of contexts fed into the prompt
    # Snapshot of the finish_reason from the LLM call that produced this answer ('length' means it was
    # truncated by max_tokens). This travels with the Answer rather than being read off the llm instance
    # attribute afterward -- the instance-level last_finish_reason gets overwritten by later calls (if a
    # smart-ask retry is discarded, the instance is left holding the second round's value, which would then
    # be mismatched against the first-round answer that was actually returned). None when the LLM was never
    # called because retrieval returned zero hits.
    finish_reason: str | None = None
    raw_messages: list[Message] = field(default_factory=list)   # debug: the messages sent to the LLM
