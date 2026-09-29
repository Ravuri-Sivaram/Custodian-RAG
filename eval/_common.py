"""Shared utilities for end-to-end / agentic RAG evaluation: env loading, demo library copying (to avoid the live
MCP single-client lock), DeepSeek text/JSON clients. Reused by gen_gold / run_eval / acl_regression.

**Why copy the demo library**: embedded Qdrant holds an exclusive single-client lock. The MCP server you're
connected to in Claude Code is currently holding the ~/rag_demo/qdrant lock, so an evaluation script that opens the
same directory directly gets "already accessed by another instance". The fix is to always copytree into a temp
working directory first and open that instead — this neither contends for the lock nor pollutes the original
library.
"""
from __future__ import annotations

import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)   # = the custodian repo root; load_env reads REPO/.env (i.e. custodian/.env). The engine package is already installed with pip install -e .

DEMO_SRC = os.path.expanduser("~/rag_demo")
DEMO_COLLECTION = "demo"
DENSE_DIM = 1024


def load_env(path: str | None = None) -> None:
    """Minimal .env loader (avoids depending on python-dotenv); an already-set environment variable takes precedence (setdefault)."""
    path = path or os.path.join(REPO, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# Fix from review: load .env first, so the module-level constants below can actually pick up CUSTODIAN_EVAL_* from
# it (the original call happened after the constants -> .env was silently ineffective)
load_env()

# Evaluation library source (can be overridden via env to point at a larger library, e.g. the ~/rag_eval_big /
# collection=evalbig that index_eval_corpus.py builds)
EVAL_SRC = os.environ.get("CUSTODIAN_EVAL_SRC") or os.environ.get("RAG_EVAL_SRC", DEMO_SRC)
EVAL_COLLECTION = os.environ.get("CUSTODIAN_EVAL_COLLECTION") or os.environ.get("RAG_EVAL_COLLECTION", DEMO_COLLECTION)

# Evaluation model tiers (all via DeepSeek; see README's "honest warnings" for the same-vendor self-evaluation
# circular bias). RAG_EVAL_* is kept as one deprecated alias generation.
# GEN_MODEL defaults to falling back to the production CUSTODIAN_LLM_MODEL — this ensures the actual deployed config
# is what's being tested (can still be explicitly overridden via CUSTODIAN_EVAL_GEN_MODEL).
GEN_MODEL = (os.environ.get("CUSTODIAN_EVAL_GEN_MODEL") or os.environ.get("CUSTODIAN_LLM_MODEL")
             or os.environ.get("RAG_EVAL_GEN_MODEL", "deepseek-v4-flash"))
JUDGE_MODEL = os.environ.get("CUSTODIAN_EVAL_JUDGE_MODEL") or os.environ.get("RAG_EVAL_JUDGE_MODEL", "deepseek-v4-flash")


def copy_demo(dest: str, src: str | None = None) -> tuple[str, str]:
    """Copies the evaluation library source (default EVAL_SRC, overridable via env CUSTODIAN_EVAL_SRC) to dest
    (qdrant + sidecar), returns (qdrant_path, sidecar_dir). Doesn't contend for the live lock, doesn't pollute the
    original library."""
    src = src or EVAL_SRC
    if not os.path.isdir(src):
        raise FileNotFoundError(f"evaluation library does not exist: {src}; run index_demo.py or index_eval_corpus.py first")
    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)
    return os.path.join(dest, "qdrant"), os.path.join(dest, "sidecar")


def make_retriever(work: str):
    """Builds a Retriever on a copy of the evaluation library (GPU lazy-loads Qwen3-VL 8B on the first query). The library and collection are determined by EVAL_SRC/EVAL_COLLECTION."""
    from embedder import EmbedConfig, Retriever
    qpath, sdir = copy_demo(work)
    cfg = EmbedConfig(qdrant_path=qpath, sidecar_dir=sdir, dense_dim=DENSE_DIM, collection=EVAL_COLLECTION)
    return Retriever(cfg), cfg


def demo_user():
    """The demo library is all tenant=demo/public — this identity can see all 4 documents."""
    from embedder import User
    return User(tenant="demo", principals=[])


def scroll_chunks(qdrant_path: str, collection: str = EVAL_COLLECTION) -> list[dict]:
    """Scans all point payloads in a Qdrant library (CPU only, no GPU needed). Used by gen_gold for sampling."""
    from qdrant_client import QdrantClient
    client = QdrantClient(path=qdrant_path)
    out, offset = [], None
    while True:
        pts, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
        out.extend(p.payload for p in pts)
        if offset is None:
            break
    return out


# ---------------- DeepSeek clients ----------------

class TextLLM:
    """The generation LLM used by the system under test (real service config: thinking off / temp 0). Directly reuses generator.OpenAICompatibleLLM."""

    def __init__(self, model: str = GEN_MODEL):
        from generator import OpenAICompatibleLLM
        self.impl = OpenAICompatibleLLM(model=model, thinking=False)

    def complete(self, messages) -> str:
        return self.impl.complete(messages)


class JsonLLM:
    """Used for gold generation + judging: DeepSeek JSON mode (response_format json_object) + one retry. Returns a dict.

    thinking is explicitly disabled (V4 Flash may default to thinking on, and without disabling it, it can require
    reasoning_content to be sent back and 400 otherwise); json_object hard-constrains the output to be parseable.
    """

    def __init__(self, model: str = JUDGE_MODEL, temperature: float = 0.0, max_tokens: int = 1200):
        from openai import OpenAI
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("missing DEEPSEEK_API_KEY (put it in .env, don't commit it)")
        self.client = OpenAI(base_url="https://api.deepseek.com", api_key=key, timeout=120)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask(self, system: str, user: str) -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, temperature=self.temperature,
                max_tokens=self.max_tokens, response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}})
            content = resp.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if attempt == 0:
                    msgs.append({"role": "assistant", "content": content})
                    msgs.append({"role": "user", "content": "The previous message was not valid JSON. Output only a single valid JSON object, with no extra text."})
                    continue
                raise ValueError(f"judge/generation returned invalid JSON (twice): {content[:200]}")
        raise AssertionError("unreachable")
