"""generator: the "G" (generation) layer of the RAG pipeline (retrieval -> LLM answer synthesis,
with citations, an ACL exit gate, and grounding).

LLM-agnostic: the `LLMClient` protocol is pluggable (local Qwen / OpenAI-compatible API / Claude, etc.),
with `MockLLM` provided for scaffolding verification. Dependency injection: `Generator(retriever, llm)`
-- the retriever is supplied by the embedder, so the core code has zero runtime dependencies (pure stdlib).
"""
from .synthesis import Generator
from .llm import LLMClient, MockLLM, OpenAICompatibleLLM, messages_to_dicts
from .prompt import PromptBuilder
from .signals import DEFAULT_TABLE_LEG, is_refusal, looks_numeric
from .types import Answer, Citation, Message

__all__ = ["Generator", "LLMClient", "MockLLM", "OpenAICompatibleLLM", "messages_to_dicts",
           "PromptBuilder", "Answer", "Citation", "Message",
           "looks_numeric", "is_refusal", "DEFAULT_TABLE_LEG"]
