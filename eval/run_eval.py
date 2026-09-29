"""7.A end-to-end + 7.B dual-layer attribution. Runs the real system (real Retriever + real DeepSeek generation) on
a copy of the evaluation library, and computes, for each gold item:

  Programmatic metrics (don't need a judge, computed directly by this script):
    retrieval_recall  the fraction of golden chunks (can be multiple) that were retrieved / retrieval_full whether
                       all of them were / MRR (best rank)
    citation_recall   the fraction of golden chunks the answer's [cite:n] actually cites
  Subjective metrics (--judge decides who judges):
    faithfulness / correctness — `--judge deepseek` uses DeepSeek self-judging (same vendor, has circular bias);
    `--judge none` skips judging and only produces rows (containing answer + the context fed in + golden_answer),
    handed off to [Claude sub-agent judging] for debiasing.

Multi-hop: gold uses golden_chunk_ids (a list); single-hop = length 1. retrieval/citation recall are computed as a
"hit fraction". Modes --mode single|agentic|both: single = closed-pipeline single hop; agentic = DeepSeek-driven
query rewriting for multi-hop (= the agent under test, not the judge).

Usage (debias full pipeline):
  CUSTODIAN_EVAL_SRC=~/rag_eval_big CUSTODIAN_EVAL_COLLECTION=evalbig \
    python eval/run_eval.py --mode both --judge none --gold eval/gold.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

from _common import JsonLLM, TextLLM, demo_user, load_env, make_retriever

HERE = os.path.dirname(os.path.abspath(__file__))

FAITH_SYS = (
    "You are a strict RAG faithfulness judge. Determine whether every factual claim in the "
    "[System Answer] is supported by the [Provided Context]. "
    "If even one claim has no basis in the context (introduced from outside knowledge or fabricated) "
    "=> faithful=false. "
    "If the system answer is a reasonable refusal such as 'insufficient information', treat it as "
    "faithful=true. Output JSON only."
)
CORRECT_SYS = (
    "You are a strict question-answering correctness judge. Determine whether the [System Answer] "
    "factually answers the [Question] correctly, using the [Reference Answer] as the standard. "
    "Different wording, more detail, or a more concise answer are all fine; correct=true as long as "
    "the core facts match the reference answer with no contradiction. "
    "If the system answers 'insufficient information' but the reference answer has substantive "
    "content, correct=false. Output JSON only."
)
SUFFICIENCY_SYS = (
    "You are driving a retrieval agent. Given the [Question] and the [Retrieved Context So Far], "
    "determine whether the context is sufficient to fully answer the question. "
    "If not, provide a [Rewritten Retrieval Query] more likely to hit the missing information "
    "(different wording / added keywords / a split sub-question), in the same language as the "
    "question. Output JSON only."
)
DECOMPOSE_SYS = (
    "You are driving a retrieval agent. Split the [Question] into 1-4 [independently retrievable "
    "sub-questions], each targeting one entity/aspect/document. "
    "A cross-document comparison question must be split into one sub-question per object "
    "(e.g. 'compare A and B's X' -> ['A's X', 'B's X']); "
    "a single-fact question needs no splitting -- just return the original question as the sole "
    "sub-question. Sub-questions must be in the same language as the original question. Output "
    "JSON only."
)


def _ctx_block(contexts):
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(contexts, 1)) or "(no context)"


def _dedup(seq):
    """Order-preserving dedup: agentic/decompose's union_ids is appended round by round and contains duplicates; only after deduping is best_rank/MRR comparable to single's ordered hits."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def judge_faithful(jllm, ctx_text, answer):
    return jllm.ask(FAITH_SYS, f"Context:\n{ctx_text}\n\nSystem Answer:\n{answer}\n\n"
                               'Output JSON: {"faithful": true/false, "reason": "..."}')


def judge_correct(jllm, query, golden, answer):
    return jllm.ask(CORRECT_SYS, f"Question: {query}\nReference Answer: {golden}\nSystem Answer: {answer}\n\n"
                                 'Output JSON: {"correct": true/false, "reason": "..."}')


def best_rank(gcids, retrieved):
    ranks = [i for i, c in enumerate(retrieved, 1) if c in set(gcids)]
    return min(ranks) if ranks else None


def run_single(retriever, gen, query, user, k, rerank, smart_tables=False):
    """--smart-tables: **failure-driven** supplementary table retrieval (same strategy and parameters as custodian
    smart-ask, sharing generator.signals) — the first round is clean; when a numeric question is refused or
    partially refused, it re-asks once with an extra kind=table leg (hard cap of 1 retry).
    ⚠ An earlier version that added the leg up front was rejected by the 88-question benchmark (tables +0.25 but
    hurt 5 prose questions, prose 0.861->0.792) — do not revert to that.
    The retrieval metric on retry = primary retrieval ∪ leg retrieval (deduped, order-preserving) — this union is
    exactly what the system actually delivers to the LLM."""
    hits = retriever.search(query, user, top_k=k, rerank=rerank)
    ids = [h.chunk_id for h in hits]
    ans = gen.answer(query, user, top_k=k, rerank=rerank)
    retried = retry_kept = False
    if smart_tables:
        from generator import DEFAULT_TABLE_LEG, is_refusal, looks_numeric
        if looks_numeric(query) and is_refusal(ans.text):
            retried = True
            leg = dict(DEFAULT_TABLE_LEG)
            lkw = {kk: leg[kk] for kk in ("doc_ids", "doc_type", "kind", "strategy", "rerank_top_n")
                   if leg.get(kk) is not None}
            ans2 = gen.answer(query, user, top_k=k, rerank=rerank, extra_legs=[leg])
            # Same strategy as custodian service.py (changing one must update the other in sync): only adopt the
            # retry result if it's a **fully answered** response — a partial answer can carry an incorrect claim of
            # missing information (measured faithfulness 1.0->0.93), so conservatively keep the first round's
            # honest refusal instead.
            if not is_refusal(ans2.text):
                retry_kept = True
                ans = ans2
                ids += [h.chunk_id for h in retriever.search(query, user, top_k=leg.get("top_k"),
                                                             rerank=bool(leg.get("rerank")), **lkw)]
                ids = _dedup(ids)
    ctx_text = next((m.content for m in ans.raw_messages if m.role == "user"), "")
    return ans.text, [c.chunk_id for c in ans.citations], ids, ctx_text, retried, retry_kept


def run_agentic(retriever, tllm, jllm, query, user, k, rounds):
    from generator import PromptBuilder
    collected, seen, union_ids, queries = [], set(), [], [query]
    cur_q, n_rounds = query, 0
    for rnd in range(rounds):
        n_rounds = rnd + 1
        for r in retriever.search_with_context(cur_q, user, top_k=k):
            hit, ctx = r["hit"], r["context"]
            text = (ctx.text if ctx is not None else hit.text) or ""
            union_ids.append(hit.chunk_id)
            if hit.chunk_id in seen or not text.strip():
                continue
            seen.add(hit.chunk_id)
            dm = (hit.payload or {}).get("doc_meta") or {}
            collected.append({"chunk_id": hit.chunk_id, "text": text, "source": dm.get("title") or hit.doc_id})
        if rnd == rounds - 1:
            break
        dec = jllm.ask(SUFFICIENCY_SYS, f"Question: {query}\nRetrieved Context So Far:\n{_ctx_block(collected)}\n\n"
                                        'Output JSON: {"sufficient": true/false, "refined_query": "..."}')
        if dec.get("sufficient") or not (dec.get("refined_query") or "").strip():
            break
        cur_q = dec["refined_query"].strip()
        queries.append(cur_q)
    messages = PromptBuilder().build(query, [{"text": c["text"], "source": c["source"]} for c in collected])
    answer = tllm.complete(messages)
    cited = sorted({int(n) for n in re.findall(r"\[cite:(\d+)\]", answer)})
    cited_ids = [collected[n - 1]["chunk_id"] for n in cited if 1 <= n <= len(collected)]
    ctx_text = next((m.content for m in messages if m.role == "user"), "")
    return answer, cited_ids, _dedup(union_ids), ctx_text, n_rounds, queries


def run_decompose(retriever, tllm, jllm, query, user, k, max_subs=4, cap=14):
    """Splits the query into sub-questions -> retrieves each separately -> **union** (no replacement, no hop
    dropped) -> synthesizes. Distinct from run_agentic's "rewrite and replace the query" approach (which can
    narrow down and lose another hop on multi-hop questions). Cross-document comparisons benefit especially
    (each object gets its own retrieval)."""
    from generator import PromptBuilder
    dec = jllm.ask(DECOMPOSE_SYS, f"Question: {query}\nOutput JSON: {{\"sub_queries\": [\"...\"]}}")
    subs = [s for s in (dec.get("sub_queries") or []) if (s or "").strip()][:max_subs] or [query]
    collected, seen, union_ids = [], set(), []
    for sq in subs:
        for r in retriever.search_with_context(sq, user, top_k=k):
            hit, ctx = r["hit"], r["context"]
            text = (ctx.text if ctx is not None else hit.text) or ""
            union_ids.append(hit.chunk_id)
            if hit.chunk_id in seen or not text.strip():
                continue
            seen.add(hit.chunk_id)
            dm = (hit.payload or {}).get("doc_meta") or {}
            collected.append({"chunk_id": hit.chunk_id, "text": text, "source": dm.get("title") or hit.doc_id})
    collected = collected[:cap]                              # cap it, to keep a large union across many sub-questions from blowing out the context
    messages = PromptBuilder().build(query, [{"text": c["text"], "source": c["source"]} for c in collected])
    answer = tllm.complete(messages)
    cited = sorted({int(n) for n in re.findall(r"\[cite:(\d+)\]", answer)})
    cited_ids = [collected[n - 1]["chunk_id"] for n in cited if 1 <= n <= len(collected)]
    ctx_text = next((m.content for m in messages if m.role == "user"), "")
    return answer, cited_ids, _dedup(union_ids), ctx_text, len(subs), subs


def evaluate(mode, retriever, gen, tllm, jllm, gold, user, k, rerank, rounds, judge, smart_tables=False):
    rows = []
    for i, g in enumerate(gold, 1):
        gcids = g.get("golden_chunk_ids") or ([g["golden_chunk_id"]] if g.get("golden_chunk_id") else [])
        q, hop = g["query"], g.get("hop", "single")
        if mode == "single":
            ans, cited, retrieved, ctx_text, retried, retry_kept = run_single(
                retriever, gen, q, user, k, rerank, smart_tables=smart_tables)
            n_rounds, queries = 1, [q]
        elif mode == "decompose":
            ans, cited, retrieved, ctx_text, n_rounds, queries = run_decompose(retriever, tllm, jllm, q, user, k)
            retried = retry_kept = False
        else:
            ans, cited, retrieved, ctx_text, n_rounds, queries = run_agentic(retriever, tllm, jllm, q, user, k, rounds)
            retried = retry_kept = False
        rset, cset, gset = set(retrieved), set(cited), set(gcids)
        hit_frac = len(gset & rset) / len(gset) if gset else 0.0
        cit_frac = len(gset & cset) / len(gset) if gset else 0.0
        faith = correct = None
        fr = cr = ""
        if judge == "deepseek":
            jf, jc = judge_faithful(jllm, ctx_text, ans), judge_correct(jllm, q, g["golden_answer"], ans)
            faith, correct, fr, cr = bool(jf.get("faithful")), bool(jc.get("correct")), jf.get("reason", ""), jc.get("reason", "")
        rows.append({"query": q, "hop": hop, "golden_chunk_ids": gcids, "golden_doc_ids": g.get("golden_doc_ids", []),
                     "golden_answer": g["golden_answer"], "answer": ans, "ctx_text": ctx_text,
                     "retrieval_hit_frac": hit_frac, "retrieval_full": gset <= rset and bool(gset),
                     "rank": best_rank(gcids, retrieved), "citation_recall": cit_frac, "n_citations": len(cited),
                     "n_rounds": n_rounds, "queries": queries, "faithful": faith, "correct": correct,
                     "faith_reason": fr, "correct_reason": cr,
                     "retried": retried, "retry_kept": retry_kept})
        mk = lambda v: ("✓" if v else "·")
        jm = "" if judge == "none" else f" {('✓' if faith else '✗')}faith {('✓' if correct else '✗')}correct"
        print(f"  [{i:2d}/{len(gold)}] {hop[:5]:5s} retr{hit_frac:.2f} cite{cit_frac:.2f}{jm} r{n_rounds}  {q[:40]}", flush=True)
    return rows


def agg(rows, judge):
    n = len(rows)
    out = {"n": n,
           "retrieval_recall": sum(r["retrieval_hit_frac"] for r in rows) / n,
           "retrieval_full": sum(1 for r in rows if r["retrieval_full"]) / n,
           "mrr": sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows) / n,
           "citation_recall": sum(r["citation_recall"] for r in rows) / n,
           "avg_rounds": sum(r["n_rounds"] for r in rows) / n}
    if judge != "none":
        out["faithfulness"] = sum(1 for r in rows if r["faithful"]) / n
        out["correctness"] = sum(1 for r in rows if r["correct"]) / n
    return out


def show(tag, a):
    print(f"\n=== {tag} (n={a['n']}) ===")
    print(f"  retrieval_recall {a['retrieval_recall']:.3f}  full_recall {a['retrieval_full']:.3f}  MRR {a['mrr']:.3f}")
    print(f"  citation_recall {a['citation_recall']:.3f}   avg_rounds {a['avg_rounds']:.2f}")
    if "faithfulness" in a:
        print(f"  faithfulness {a['faithfulness']:.3f}   correctness {a['correctness']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "agentic", "decompose", "both"], default="both")
    ap.add_argument("--judge", choices=["deepseek", "none"], default="deepseek",
                    help="none=only compute programmatic metrics + emit rows for Claude sub-agent judging (debiased); deepseek=DeepSeek self-judging (has circular bias)")
    ap.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--smart-tables", action="store_true", dest="smart_tables",
                    help="automatically add a supplementary table-retrieval leg for numeric questions (shares generator.signals with custodian smart-ask)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load_env()
    if not os.path.exists(args.gold):
        raise SystemExit(f"missing gold: {args.gold}")
    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]
    n0 = len(gold)
    gold = [g for g in gold if g.get("golden_chunk_ids") or g.get("golden_chunk_id")]   # empty golden would pollute the retrieval/citation recall denominator
    if len(gold) < n0:
        print(f"⚠ skipped {n0 - len(gold)} gold entries with empty golden_chunk_ids", flush=True)
    if args.limit:
        gold = gold[:args.limit]
    print(f"gold {len(gold)} entries; mode={args.mode} judge={args.judge} top_k={args.top_k} rerank={args.rerank} rounds={args.rounds}", flush=True)

    from generator import Generator
    retriever, _ = make_retriever(os.path.join(tempfile.gettempdir(), "rag_eval_run"))
    tllm, jllm = TextLLM(), JsonLLM()
    gen = Generator(retriever, tllm)
    user = demo_user()

    aggs = {}
    for m in (["single", "agentic"] if args.mode == "both" else [args.mode]):
        print(f"\n----- running {m} -----", flush=True)
        rows = evaluate(m, retriever, gen, tllm, jllm, gold, user, args.top_k, args.rerank, args.rounds,
                        args.judge, smart_tables=args.smart_tables)
        aggs[m] = agg(rows, args.judge)
        show(m, aggs[m])
        with open(os.path.join(HERE, f"results_{m}.json"), "w", encoding="utf-8") as f:
            json.dump({"agg": aggs[m], "rows": rows}, f, ensure_ascii=False, indent=2)

    if args.mode == "both" and args.judge != "none":
        s, a = aggs["single"], aggs["agentic"]
        print("\n=== dual-layer attribution (7.B): agentic − single ===")
        print(f"  correctness {s['correctness']:.3f} -> {a['correctness']:.3f} (Δ{a['correctness']-s['correctness']:+.3f})")
        print(f"  retrieval_recall {s['retrieval_recall']:.3f} -> {a['retrieval_recall']:.3f} (Δ{a['retrieval_recall']-s['retrieval_recall']:+.3f})")
    print(f"\nresults -> eval/results_*.json" + ("(judge=none: faithfulness/correctness pending Claude sub-agent judging)" if args.judge == "none" else ""), flush=True)


if __name__ == "__main__":
    main()
