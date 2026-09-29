"""The Custodian CLI: serve (the HTTP daemon) / mcp (the thin adapter) / index (build the index) /
parse (parse a corpus) / ask (Q&A) / health.

serve/index need the WSL custodian environment (GPU + engine dependencies); parse needs
MINERU_TOKEN_*; mcp/ask/health only need httpx (plus the mcp package).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import __version__, config


def _client(url: str | None):
    import httpx
    config.load_env()                      # .env must be loaded even when --url is given (CUSTODIAN_API_KEY lives in it)
    base = url or config.adapter_base_url()
    headers = {}
    if os.environ.get("CUSTODIAN_API_KEY"):
        headers["X-API-Key"] = os.environ["CUSTODIAN_API_KEY"]
    return httpx.Client(base_url=base, timeout=httpx.Timeout(600.0, connect=5.0), headers=headers)


def cmd_serve(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import uvicorn
    from .service import create_app
    cfg = config.from_env()
    host, port = args.host or cfg.host, args.port or cfg.port
    print(f"Custodian v{__version__}  http://{host}:{port}  index={cfg.qdrant_path} "
          f"collection={cfg.collection} tenant={cfg.tenant or '(not set, fail-closed)'}", flush=True)
    # Graceful shutdown: on SIGTERM, uvicorn stops accepting new requests and drains in-flight
    # ones, forcing an exit after at most 25s. 25 < the compose stop_grace_period of 30s -> this
    # exits cleanly before Docker sends SIGKILL (an in-flight /v1/ask is never cut off mid-request).
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info",
                timeout_graceful_shutdown=25)


def cmd_mcp(args) -> None:
    if getattr(args, "direct", False):
        # A no-daemon fallback: connect stdio directly to the engine (holding the exclusive
        # embedded Qdrant lock and a resident GPU model itself), bypassing the daemon entirely.
        from .mcp_stdio import main as stdio_main
        stdio_main()
    else:
        from .mcp_adapter import main as adapter_main
        adapter_main()


def cmd_index(args) -> None:
    from .indexer import run_index
    cfg = config.from_env()
    run_index(cfg, corpus=args.corpus, dest=args.dest, collection=args.collection,
              tenant=args.tenant, visibility=args.visibility, allow=args.allow,
              only=args.only, limit=args.limit)


def cmd_parse(args) -> None:
    from .parser import run_parse
    cfg = config.from_env()
    manifest = args.manifest or os.path.join(config.REPO_ROOT, "sample_manifest.csv")
    dest = os.path.expanduser(args.dest or cfg.corpus_dir or os.path.join(config.REPO_ROOT, "parsed"))
    corpus_root = args.corpus_root or config.REPO_ROOT
    run_parse(manifest, dest, corpus_root)


def cmd_ask(args) -> None:
    with _client(args.url) as c:
        try:
            r = c.post("/v1/ask", json={"query": args.query, "top_k": args.top_k,
                                        "rerank": args.rerank, "include_contexts": args.contexts,
                                        "doc_ids": args.doc_id or None, "doc_type": args.doc_type,
                                        "kind": args.kind, "strategy": args.strategy})
        except Exception as e:
            raise SystemExit(f"Could not reach the Custodian daemon ({c.base_url}): {type(e).__name__}. Run `custodian serve` first.")
        try:
            data = r.json()
        except ValueError:                 # A 5xx or non-JSON response shouldn't dump a raw traceback
            raise SystemExit(f"The daemon returned something unexpected (HTTP {r.status_code}, not JSON); see the server logs.")
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if data.get("status") != "ok":
        raise SystemExit(f"[{data.get('status')}] {data.get('hint', '')}")
    print(data["answer"])
    if data.get("citations"):
        print("\nSources:")
        for c_ in data["citations"]:
            sec = f" §{c_['section']}" if c_.get("section") else ""
            print(f"  [{c_['marker']}] {c_['title']} (p.{c_['page']}{sec})  {c_['chunk_id']}")
    if data.get("hints"):
        print("\nHints:")
        for h in data["hints"]:
            print(f"  - {h}")
    if data.get("finish_reason") == "length":
        print("\nWarning: the answer was truncated by max_tokens (finish_reason=length).", file=sys.stderr)


def cmd_keys_new(args) -> None:
    from .identity import append_key
    config.load_env()
    path = args.file or os.environ.get("CUSTODIAN_KEYS_FILE") or os.path.expanduser("~/custodian.keys.json")
    principals = [p.strip() for p in args.principals.split(",") if p.strip()]
    key = append_key(path, name=args.name, tenant=args.tenant, principals=principals, admin=args.admin)
    print(f"Written to {path} (recommended: chmod 600; do not commit this file to git)")
    print(f"Identity {args.name} (tenant={args.tenant}, principals={principals}, admin={args.admin})")
    print(f"\nAPI key (shown only this once -- hand it off to the user securely):\n  {key}")
    print("\nTo take effect: make sure the server's CUSTODIAN_KEYS_FILE points at this file, then `sudo systemctl restart custodian`")


def cmd_health(args) -> None:
    with _client(args.url) as c:
        try:
            r = c.get("/healthz")
        except Exception as e:
            raise SystemExit(f"Could not reach the Custodian daemon ({c.base_url}): {type(e).__name__}. Run `custodian serve` first.")
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="custodian", description="Custodian: a multi-format agentic RAG service")
    p.add_argument("--version", action="version", version=f"custodian {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("serve", help="Start the HTTP daemon (holds the index exclusively, plus the GPU model)")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.set_defaults(fn=cmd_serve)

    sp = sub.add_parser("mcp", help="Start the thin MCP adapter (stdio -> HTTP, for Claude Code)")
    sp.add_argument("--direct", action="store_true",
                    help="A no-daemon fallback: connect stdio directly to the engine (holding the exclusive Qdrant lock and a resident GPU model itself, bypassing the daemon)")
    sp.set_defaults(fn=cmd_mcp)

    sp = sub.add_parser("index", help="Build the index from a MinerU parse output directory (needs GPU; stop serve first)")
    sp.add_argument("--corpus", default=None, help="The corpus directory (MinerU parse output); defaults to CUSTODIAN_CORPUS_DIR")
    sp.add_argument("--dest", default=None, help="The index output directory (defaults to CUSTODIAN_INDEX_DIR=~/rag_real)")
    sp.add_argument("--collection", default=None)
    sp.add_argument("--tenant", default=None, help="The ACL tenant (defaults to CUSTODIAN_TENANT or 'demo')")
    sp.add_argument("--visibility", default="public", choices=["public", "restricted"])
    sp.add_argument("--allow", default="", help="Comma-separated principals (used with restricted visibility)")
    sp.add_argument("--only", default=None, help="Only build documents whose directory name starts with this prefix")
    sp.add_argument("--limit", type=int, default=None, help="Only build the first N documents (for a smoke test)")
    sp.set_defaults(fn=cmd_index)

    sp = sub.add_parser("parse", help="Batch-parse corpus PDFs with MinerU -> parsed/<doc_id>/ (for index; needs MINERU_TOKEN_*)")
    sp.add_argument("--manifest", default=None, help="A parse manifest CSV (defaults to the repo root's sample_manifest.csv; generated by scripts/select_sample.py)")
    sp.add_argument("--dest", default=None, help="The output directory (defaults to CUSTODIAN_CORPUS_DIR, then the repo root's parsed/)")
    sp.add_argument("--corpus-root", default=None, dest="corpus_root", help="The root the manifest's paths are relative to (defaults to the repo root)")
    sp.set_defaults(fn=cmd_parse)

    sp = sub.add_parser("ask", help="A one-shot question (via the daemon's /v1/ask -- closed pipeline with citations)")
    sp.add_argument("query")
    sp.add_argument("--top-k", type=int, default=None, dest="top_k")
    sp.add_argument("--rerank", action="store_true")
    sp.add_argument("--kind", default=None, choices=["text", "table", "image", "chart"],
                    help="Only retrieve from this chunk kind (use 'table' for numeric/tabular questions)")
    sp.add_argument("--doc-type", default=None, dest="doc_type", help="Restrict to a document type (e.g. financial_report_en)")
    sp.add_argument("--doc-id", action="append", default=None, dest="doc_id", help="Restrict to specific documents (can be given multiple times)")
    sp.add_argument("--strategy", default=None, choices=["hybrid", "dense", "sparse"],
                    help="Retrieval routing (default: hybrid)")
    sp.add_argument("--contexts", action="store_true", help="Include the cited passage's raw text in citations")
    sp.add_argument("--json", action="store_true", help="Print the raw JSON response")
    sp.add_argument("--url", default=None, help="The daemon's address (defaults to CUSTODIAN_URL)")
    sp.set_defaults(fn=cmd_ask)

    sp = sub.add_parser("health", help="Check the daemon's health")
    sp.add_argument("--url", default=None)
    sp.set_defaults(fn=cmd_health)

    sp = sub.add_parser("keys", help="Multi-identity key management")
    ksub = sp.add_subparsers(dest="keys_cmd", required=True)
    kn = ksub.add_parser("new", help="Generate a new identity and write it to the keys file (the key is only ever printed this once)")
    kn.add_argument("name", help="An identity name (goes into request logs; must not contain sensitive information)")
    kn.add_argument("--tenant", required=True, help="The ACL tenant (the sample library is built with tenant=demo)")
    kn.add_argument("--principals", default="", help="Comma-separated principals")
    kn.add_argument("--admin", action="store_true", help="Can read /v1/stats")
    kn.add_argument("--file", default=None, help="The keys file (defaults to CUSTODIAN_KEYS_FILE, or ~/custodian.keys.json)")
    kn.set_defaults(fn=cmd_keys_new)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
