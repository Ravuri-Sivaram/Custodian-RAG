"""Merges [programmatic metrics (results_*.json)] + [Claude's two independent judging passes (verdicts.json)] into
a final report.

verdicts.json: {"pass1":[{id,faithful,correct}], "pass2":[...]}. id=f"{mode}#{i:03d}" (row order in results).
Automatically includes whichever modes exist (single/agentic/decompose). Dual-layer attribution takes a **paired**
difference over [the common set of questions judged in both modes].

Fixes applied here:
- Fingerprint verification: compares _judge/fingerprint.json against the current results' query fingerprints;
  a mismatch aborts with an error rather than emitting numbers (guards against silently mismatched rows from
  "re-ran results without re-judging").
- Loudly reports missing verdicts: prints n_judged per mode/per hop; questions with no verdict are marked "n/a"
  rather than silently mixed in as nan.
- Dual-layer attribution uses the common judged question set, so the denominator matches before subtracting.
- Missing fields are excluded as None, rather than bool(None)=False silently counting as a fail.

Run: python eval/aggregate.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = ["single", "agentic", "decompose"]


def qhash(q):
    return hashlib.sha1((q or "").encode("utf-8")).hexdigest()[:12]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else math.nan


def fmt(x):
    return "  n/a " if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.3f}"


def load_verdicts():
    v = json.load(open(os.path.join(HERE, "verdicts.json"), encoding="utf-8"))
    def idx(p):
        return {x["id"]: x for x in v.get(p, [])}
    return idx("pass1"), idx("pass2")


def andflag(p1, p2, rid, key):
    """Both passes judged and both true -> True; both judged but not both true -> False; either unjudged/field missing -> None (excluded)."""
    a, b = p1.get(rid), p2.get(rid)
    if a is None or b is None or a.get(key) is None or b.get(key) is None:
        return None
    return bool(a[key]) and bool(b[key])


def main():
    p1, p2 = load_verdicts()
    fp_path = os.path.join(HERE, "_judge", "fingerprint.json")
    fp = json.load(open(fp_path, encoding="utf-8")) if os.path.exists(fp_path) else None
    if fp is None:
        print("⚠ no _judge/fingerprint.json, skipping results<->verdicts consistency check (recommend re-running dump_judge_units)")

    modes = {}
    for m in MODES:
        path = os.path.join(HERE, f"results_{m}.json")
        if not os.path.exists(path):
            continue
        rows = json.load(open(path, encoding="utf-8"))["rows"]
        # Fingerprint verification: only aligns if the results' query matches what was there at judging time
        if fp is not None:
            for i, r in enumerate(rows):
                rid = f"{m}#{i:03d}"
                if rid in fp and fp[rid] != qhash(r["query"]):
                    raise SystemExit(f"✗ alignment failed: {rid}'s query fingerprint doesn't match what it was at judging time — results was re-run but not re-judged, refusing to emit numbers. Please re-run dump_judge_units and re-judge.")
        for i, r in enumerate(rows):
            rid = f"{m}#{i:03d}"
            r["_id"] = rid
            r["_correct"] = andflag(p1, p2, rid, "correct")
            r["_faith"] = andflag(p1, p2, rid, "faithful")
        modes[m] = rows

    def agg(rows):
        n = len(rows)
        judged = [r for r in rows if r["_correct"] is not None]
        return {"n": n, "n_judged": len(judged),
                "retrieval_recall": mean(r["retrieval_hit_frac"] for r in rows),
                "retrieval_full": mean(1.0 if r["retrieval_full"] else 0.0 for r in rows),
                "mrr": mean((1.0 / r["rank"]) if r["rank"] else 0.0 for r in rows),
                "citation_recall": mean(r["citation_recall"] for r in rows),
                "avg_rounds": mean(r["n_rounds"] for r in rows),
                "correctness": mean(r["_correct"] for r in judged) if judged else None,
                "faithfulness": mean(r["_faith"] for r in judged) if judged else None}

    print("=" * 74)
    for m, rows in modes.items():
        a = agg(rows)
        miss = a["n"] - a["n_judged"]
        print(f"\n### {m}  (n={a['n']}, judged={a['n_judged']}" + (f", ⚠missing verdicts {miss}" if miss else "") + ")")
        print(f"  retrieval_recall {fmt(a['retrieval_recall'])}  full_recall {fmt(a['retrieval_full'])}  MRR {fmt(a['mrr'])}  citation_recall {fmt(a['citation_recall'])}  avg_rounds {fmt(a['avg_rounds'])}")
        print(f"  faithfulness {fmt(a['faithfulness'])}   correctness {fmt(a['correctness'])}   (judged {a['n_judged']}/{a['n']})")

    # Split by hop (each mode has its own n_judged; unjudged marked n/a)
    print("\n" + "=" * 74 + "\nsplit by hop (correctness AND / retrieval_recall / n_judged):")
    hops = ["single", "multi_intra", "multi_cross"]
    head = "  ".join(f"{m:>26}" for m in modes)
    print(f"  {'hop':12} | {head}")
    for hop in hops:
        cells = []
        for m, rows in modes.items():
            rs = [r for r in rows if r.get("hop") == hop]
            judged = [r for r in rs if r["_correct"] is not None]
            corr = mean(r["_correct"] for r in judged) if judged else None
            rec = mean(r["retrieval_hit_frac"] for r in rs)
            cells.append(f"{fmt(corr)}/{fmt(rec)}/j{len(judged)}of{len(rs)}".rjust(26))
        print(f"  {hop:12} | " + "  ".join(cells))

    # Dual-layer attribution: paired, subtracting only over the [common index] judged in both single and other
    if "single" in modes:
        base = {int(r["_id"].split("#")[1]): r["_correct"] for r in modes["single"] if r["_correct"] is not None}
        print("\n" + "=" * 74 + "\ndual-layer attribution (paired, common judged question set):")
        for m in ("agentic", "decompose"):
            if m not in modes:
                continue
            other = {int(r["_id"].split("#")[1]): r["_correct"] for r in modes[m] if r["_correct"] is not None}
            common = sorted(set(base) & set(other))
            if not common:
                print(f"  single vs {m}: no common judged questions, skipping")
                continue
            sc = mean(base[i] for i in common)
            oc = mean(other[i] for i in common)
            print(f"  single vs {m:9}: correctness {sc:.3f} -> {oc:.3f}  (Δ{oc - sc:+.3f}, common n={len(common)})")


if __name__ == "__main__":
    main()
