"""`custodian index`: batch-builds a Custodian index (qdrant + sidecar) from a directory of MinerU
parse outputs.

Productized from the engine repo's mcp_server/index_real.py (kept there as a historical script):
corpus/target/collection/ACL are now all parameterized instead of being hardcoded to parsed/ and
~/rag_real. Corpus requirements: under the corpus directory, one MinerU output directory per
document (content_list.json + layout.json, ...); the directory name is the doc_id, and a
`<doc_type>__<name>` prefix convention is optional.

Requires a GPU (Qwen3-VL-Embedding-8B). Warning: the embedded Qdrant client is single-client only,
so you cannot write to a target index while the daemon has it open -- stop `custodian serve` first
before indexing, or index into a new directory and then switch CUSTODIAN_INDEX_DIR.
"""
from __future__ import annotations

import os

from chunker import Chunker
from chunker.adapters.mineru import from_mineru_dir
from embedder import EmbedConfig, Embedder

from . import config


def detect_lang(elements) -> str:
    """Detect language from the proportion of Telugu-script characters in the content (more
    reliable than guessing from doc_type; carried over from the engine's index_real.py, which
    originally used a similar script-ratio check for the non-English language this project
    supported before it was replaced with Telugu)."""
    sample = "".join((e.text or "") for e in elements[:40])[:2000]
    if not sample:
        return "en"
    telugu = sum(1 for c in sample if "ఀ" <= c <= "౿")
    return "te" if telugu > len(sample) * 0.12 else "en"


def run_index(cfg: config.CustodianConfig, corpus: str | None = None, dest: str | None = None,
              collection: str | None = None, tenant: str | None = None, visibility: str = "public",
              allow: str = "", only: str | None = None, limit: int | None = None) -> int:
    allow_list = [a.strip() for a in allow.split(",") if a.strip()]
    # Fixed after review (tracked as C1): restricted visibility with an empty allow list means the
    # document is unreachable by any identity (fail-closed), but indexing completes with zero
    # warnings and it only shows up at query time as "empty results", which is extremely hard to
    # debug. So we reject this explicitly at the indexing entry point and require principals.
    if visibility == "restricted" and not allow_list:
        raise SystemExit("--visibility restricted requires at least one --allow principal: "
                         "a restricted document with an empty allow list is invisible to every "
                         "identity (fail-closed); indexing would \"succeed\" while the document "
                         "is silently unretrievable by anyone.")
    corpus = corpus or cfg.corpus_dir
    if not corpus:
        raise SystemExit("No corpus directory specified: use --corpus or set CUSTODIAN_CORPUS_DIR "
                         "(the directory of MinerU parse outputs; this data stays outside the "
                         "repo and is not migrated into it).")
    # The write path used for indexing must match the read path used by serve (a key finding from
    # the phase-F review, tracked as S3): serve uses cfg.qdrant_path/cfg.sidecar_dir (already
    # derived by from_env from CUSTODIAN_QDRANT_PATH/CUSTODIAN_SIDECAR_DIR or index_dir). If indexer
    # derived its own sidecar path from dest, a deployment that explicitly sets
    # CUSTODIAN_SIDECAR_DIR would end up with **mismatched write/read paths** -- indexing would write
    # to dest/sidecar while serve reads cfg.sidecar_dir, so a query hit would resolve to nothing
    # and silently degrade (single_chunk_degraded). So: when --dest isn't given explicitly, use
    # cfg's paths directly; only derive from dest when --dest is given explicitly (the user
    # deliberately chose to index elsewhere and owns the consistency of that choice).
    explicit_dest = dest is not None
    dest = os.path.expanduser(dest or cfg.index_dir)
    qdrant_path = os.path.join(dest, "qdrant") if explicit_dest else cfg.qdrant_path
    sidecar_dir = os.path.join(dest, "sidecar") if explicit_dest else cfg.sidecar_dir
    collection = collection or cfg.collection
    acl = {"tenant": (tenant or cfg.tenant or "demo"), "allow": allow_list,
           "visibility": visibility, "unset": False}
    if not os.path.isdir(corpus):
        raise SystemExit(f"Corpus directory does not exist: {corpus}")

    # The model path/gpu_name are also passed through from cfg (per the phase-F review):
    # otherwise, when CUSTODIAN_DENSE_MODEL_PATH is customized, indexing would use the default path
    # and end up out of sync with the model serve loads -- a vector-space mismatch with no
    # warning.
    ecfg = EmbedConfig(qdrant_path=qdrant_path, qdrant_url=cfg.qdrant_url,
                       sidecar_dir=sidecar_dir,
                       dense_dim=cfg.dense_dim, collection=collection,
                       dense_model_path=cfg.dense_model_path, rerank_model_path=cfg.rerank_model_path,
                       gpu_name_must_contain=cfg.gpu_name)   # server mode (qdrant_url set) writes to the server; qdrant_path is ignored by that branch of Store
    try:
        emb = Embedder(ecfg)
    except Exception as e:
        if "already accessed" in str(e):
            raise SystemExit(f"Index directory is in use (embedded Qdrant is single-client only): {dest}\n"
                             f"Stop custodian serve first before indexing, or use --dest to point at a new directory.") from e
        raise

    dirs = sorted(d for d in os.listdir(corpus) if os.path.isdir(os.path.join(corpus, d)))
    if only:
        dirs = [d for d in dirs if d.startswith(only)]
    if limit:
        dirs = dirs[:limit]
    print(f"{len(dirs)} documents under {corpus}, indexing -> {dest} (collection={collection}, "
          f"acl tenant={acl['tenant']}/{acl['visibility']})", flush=True)
    ok = total = 0
    failed: list[str] = []
    for d in dirs:
        ddir = os.path.join(corpus, d)
        doc_type = d.split("__")[0] if "__" in d else "unknown"
        try:
            els = from_mineru_dir(ddir)
            if not els:
                print(f"  skipping {d}: empty", flush=True)
                continue
            lang = detect_lang(els)
            res = Chunker().chunk(els, doc_id=d, doc_type=doc_type, lang=lang,
                                  doc_meta={"title": d}, acl=acl)
            emb.index_document(d, els, res, image_root=ddir)
            ok += 1
            total += len(res.chunks)
            print(f"  [{ok:2d}] {doc_type:20s} {lang} {len(res.chunks):4d} chunk  {d[:44]}", flush=True)
        except Exception as e:
            # Fixed after review (tracked as embedder fix 1): no longer silently "skips" a
            # failure -- failed docs are recorded in a list, summarized in the DONE line, and
            # cause a non-zero exit. If index_document fails after having already deleted the old
            # entry, it emits its own "delisted" FATAL warning; the list recorded here gives you
            # what to re-run once the batch finishes.
            failed.append(d)
            print(f"  failed {d}: {type(e).__name__}: {e}", flush=True)
    print(f"\nDONE -> {dest}  {ok} documents / {total} chunks; {len(failed)} failed; "
          f"collection={collection} dense_dim={cfg.dense_dim}", flush=True)
    if failed:
        raise SystemExit(f"Indexing had failed documents (cannot be treated as a successful build): {', '.join(failed)}\n"
                         f"Re-run with --only <doc_id prefix>; if the log contains a \"delisted\" FATAL warning, "
                         f"that document's old index entry has already been removed and it must be re-run.")
    return ok
