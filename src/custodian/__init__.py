"""Custodian: a self-contained, multi-format agentic RAG system (built for a small team's internal
knowledge base).

The Lighthouse of Alexandria stood beside the Library of Alexandria, guiding ships home --
Custodian guides you through your own private library.

A single-repo, end-to-end pipeline: parse -> chunk (chunker) -> hybrid retrieval + rerank + ACL
(embedder) -> generate + cite (generator), with two entry points sharing the same retrieval
semantics (custodian.toolcore):
  - `custodian serve`: an HTTP daemon (holding the embedded Qdrant and GPU models exclusively),
                    offering closed-pipeline Q&A at /v1/ask plus 6 retrieval endpoints;
  - `custodian mcp`  : a thin, GPU-independent MCP adapter (stdio -> HTTP) that lets agents like
                    Claude Code do agentic RAG (`--direct` is a no-daemon fallback that connects
                    stdio straight to the engine).
"""
__version__ = "0.3.0"
