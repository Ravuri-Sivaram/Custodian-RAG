# eval baseline anchor (metrics anchor)

> Records only metric numbers + provenance, not per-question gold/answers (those contain research-report excerpts and are gitignored). Reproduction steps: [README.md](README.md) / [../docs/RUNBOOK.md](../docs/RUNBOOK.md).

## Tier1 (DeepSeek self-judging, reproducible within the repo)

- **Config**: `--mode single --judge deepseek --smart-tables`, gold set of 88 questions (72 prose + 16 table), eval library `~/rag_eval_big` (collection `evalbig`, 15 docs / 1409 chunks).
- **Reproduced on**: after folding the engine into the single custodian repo (`master`, merge commit `b6fe69d`), 2026-07-05, WSL custodian / RTX 4090.

| Metric | This run (single-repo master) | Pre-migration authoritative baseline | Δ |
|---|---|---|---|
| Retrieval recall@k | **0.830** | 0.818 | +0.012 |
| Full recall | **0.761** | — (0.792 is on the 72-question basis) | — |
| MRR | **0.629** | 0.627 | +0.002 |
| Citation recall | **0.778** | 0.767 | +0.011 |
| Faithfulness | **0.977** | 0.977 | 0 (exact) |
| Correctness | **0.830** | 0.818 | +0.012 |
| Avg. rounds | 1.00 | 1.00 | 0 |

**Conclusion**: all five metrics fall within the **±2-question noise band** of a single-pass DeepSeek judge (0.012×88 ≈ 1 question), and faithfulness matches exactly.
**Migration preserved end-to-end fidelity** — same code (byte-identical engine) + same gold + same eval library, reproduced within the single repo with numbers matching the pre-migration values.

## Tier2 (dual-Claude cross-vendor judging, authoritative debiased)

Tier1 is same-vendor DeepSeek self-judging (circular bias, useful for trends). **Authoritative debiasing** uses cross-vendor Claude judging: 2 independent Claude passes, seeing the **full context**, dual-pass AND (counted true only if both passes agree). This run used Claude Code multi-agent orchestration (single-repo master, 2026-07-05, 88 questions, single), fingerprint verification passed:

| Metric (88 questions, single) | Tier2 dual-Claude (authoritative) | Tier1 DeepSeek self-judging |
|---|---|---|
| **Faithfulness** | **0.989** | 0.977 |
| **Correctness** | **0.818** | 0.830 |
| Retrieval recall / MRR / Citation recall | 0.830 / 0.629 / 0.778 (programmatic, judge-independent) | same |

**Per-hop correctness (AND)**: single-hop **0.889** (n=54) / multi_intra **0.828** (n=29) / multi_cross **0.000** (n=5).
- Faithfulness 0.989 ≈ watertight — grounding contract holds (reproduces the R5 "≈1.0" finding: the system would rather decline than fabricate); the two passes disagreed on only 1 faithfulness question and 1 correctness question (high judge consistency).
- Cross-vendor Claude judging is **slightly stricter** than same-vendor DeepSeek self-judging (correctness 0.818 vs 0.830, a difference of ~1 question), in the expected direction (debiasing).
- The remaining hard problem is still **multi_cross (cross-document synthesis) at 0.000** (n=5, small sample): stuck on "getting chunks from both documents but still failing to produce a comparison" — a synthesis problem, not a retrieval one.

**Why this is "not reproducible within the repo"**: this judging pipeline (2 independent Claude passes + full context) relies on Claude Code multi-agent orchestration; `verdicts.json`/`_judge/` are not checked into the repo (gitignored, regenerable) — reproducing this exact set of authoritative numbers requires an environment with Claude orchestration. There is a separate set of numbers on the historical 72-question basis (before the table questions were added) (faithfulness ≈1.0 / correctness 0.847), which is **not directly comparable** to the 88-question set.

## Baseline discontinuity note

The 88-question set (starting 2026-07-03 when 16 table questions were added) and the historical 72-question aggregate numbers are **not directly comparable**. gold/results/verdicts/baseline_*.json are all gitignored (contain private data, regenerable).
