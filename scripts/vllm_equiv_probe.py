#!/usr/bin/env python3
"""vLLM <-> official Qwen3VLEmbedder text-vector equivalence go/no-go probe
(docs/VLLM_PLAN.md Phase 0 / G0+G1).

**The single number that decides whether the vLLM plan lives or dies**: cosine(vLLM's query
vector, the official script's vector). >0.9999 = G1 GO (vLLM can query the existing store);
otherwise the existing store cannot be queried by vLLM (vector drift -> top-k misalignment), and
either the two encoders must be aligned first or the whole store must be rebuilt.

**Why it's split into two time-sliced steps**: the official encoder (custodian, transformers 4.57)
and vLLM (vllm env, transformers 5.10) live in different conda environments, and loading the 8B
model twice would OOM. So `--step official` produces and saves the vectors first -> `--step vllm`
then produces vLLM's vectors and compares.

**Preconditions for equivalence (copied verbatim from the official recipe, otherwise you get a
false negative)**: both sides use the `[{system:instruction},{user:text}]` conversation format +
`apply_chat_template(add_generation_prompt=True, tokenize=False)` (verified by reading the
official `qwen3_vl_embedding.py::_preprocess_inputs` and confirming it matches vLLM's
`examples/embedding_vllm.ipynb`). If the instruction doesn't end in punctuation, append '.'
(the official format_model_input rule).

Run:
  # 0) free the GPU (stop the torch inference container)
  docker compose --env-file .env.compose stop inference
  # 1) official vectors (custodian)
  conda activate custodian && python scripts/vllm_equiv_probe.py --step official
  # 2) vLLM vectors + comparison (vllm env)
  conda activate vllm    && python scripts/vllm_equiv_probe.py --step vllm
"""
from __future__ import annotations

import argparse
import os
import sys
import unicodedata

import numpy as np

# To match the store: dense_dim=1024 is the production truncation dim; but equivalence is first
# compared at **full 4096 dims** (truncation is a deterministic post-processing step, so if the
# full-dim vectors are equivalent, the truncated ones must be too).
EMB_MODEL = os.path.expanduser("~/models/Qwen3-VL-Embedding-8B")
DEFAULT_INSTRUCTION = "Represent the user's input."          # the official wrapper's default_instruction
QUERY_INSTRUCTION = "Retrieve relevant documents for the query."   # custodian EmbedConfig.query_instruction

# Coverage: long/short, special characters, different instructions. The query path is plain text
# only, so only text is tested here.
SAMPLES = [
    ("What were Netflix's revenue and net profit in 2015?", QUERY_INSTRUCTION),
    ("What was IBM's total revenue growth year over year?", QUERY_INSTRUCTION),
    ("Mixed script test 123 !@#$% special characters and punctuation.", QUERY_INSTRUCTION),
    ("a", QUERY_INSTRUCTION),
    ("The company's cash flow statement for the reporting period shows that net cash flow from "
     "operating activities rose year over year, mainly due to faster receivables turnover and "
     "improved inventory management, while an increase in cash outflow from investing activities "
     "reflects expanding capital expenditure.", QUERY_INSTRUCTION),
    ("quarterly earnings per share diluted", DEFAULT_INSTRUCTION),
    ("Board resolution on profit distribution", DEFAULT_INSTRUCTION),
    ("supply chain risk and mitigation strategy", DEFAULT_INSTRUCTION),
]


def _system_instruction(instr: str) -> str:
    """Mirrors the official format_model_input: if the instruction doesn't end in punctuation (Unicode P*), append '.'."""
    instr = (instr or DEFAULT_INSTRUCTION).strip()
    if instr and not unicodedata.category(instr[-1]).startswith("P"):
        instr = instr + "."
    return instr


def _conversation(text: str, instr: str) -> list:
    """The conversation structure shared by both sides (text-only)."""
    return [
        {"role": "system", "content": [{"type": "text", "text": _system_instruction(instr)}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _out_path(out_dir: str) -> str:
    return os.path.join(out_dir, "official_vecs.npz")


def step_official(out_dir: str) -> None:
    """custodian: produces full-dim vectors with the official Qwen3VLEmbedder and saves them. This is the **authoritative** reference (the existing store was built with it)."""
    import torch
    scripts = os.path.join(EMB_MODEL, "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from qwen3_vl_embedding import Qwen3VLEmbedder

    print(f"[official] Loading {EMB_MODEL} (bf16)…", flush=True)
    model = Qwen3VLEmbedder(model_name_or_path=EMB_MODEL, torch_dtype=torch.bfloat16)
    vecs = []
    for text, instr in SAMPLES:
        emb = model.process([{"text": text, "instruction": instr}], normalize=True)  # (1, 4096) already normalized
        vecs.append(np.asarray(emb.float().cpu().numpy()[0], dtype=np.float32))
    arr = np.stack(vecs)
    os.makedirs(out_dir, exist_ok=True)
    # Only saves the numeric arrays (texts/instrs already live in the SAMPLES constant, no need to
    # persist them) -> the read side uses allow_pickle=False, so there's no deserialization risk.
    np.savez(_out_path(out_dir), vecs=arr)
    print(f"[official] Saved {arr.shape} -> {_out_path(out_dir)}; norm range "
          f"[{np.linalg.norm(arr,axis=1).min():.5f}, {np.linalg.norm(arr,axis=1).max():.5f}]", flush=True)


def step_vllm(out_dir: str) -> None:
    """vllm env: produces vectors via vLLM pooling and compares cosine against the official saved vectors -> G0/G1 verdict."""
    p = _out_path(out_dir)
    if not os.path.exists(p):
        raise SystemExit(f"Official saved vectors not found at {p}; run --step official in custodian first.")
    ref = np.load(p)                # allow_pickle=False (default): only reads numeric arrays, no deserialization risk
    ref_vecs = ref["vecs"]

    # G0: can vLLM even load this VL embedding model in pooling mode
    try:
        from vllm import LLM
    except Exception as e:
        raise SystemExit(f"[G0 FAIL] import vllm failed: {e} (confirm conda activate vllm)")
    print(f"[vllm] Loading {EMB_MODEL} via LLM(runner='pooling')…(first load may be slow)", flush=True)
    try:
        llm = LLM(model=EMB_MODEL, runner="pooling", dtype="bfloat16", trust_remote_code=True,
                  enforce_eager=True,          # rule out cudagraph as a variable, focus purely on equivalence
                  max_model_len=8192,          # CRITICAL: the model's max_position_embeddings=262144 would make vLLM
                  #                               want a 36GB KV cache, but embedding is a single forward pass with no
                  #                               need for long context; the official wrapper also uses MAX_LENGTH=8192.
                  #                               Omitting this will OOM.
                  gpu_memory_utilization=0.90)
    except Exception as e:
        # Do not conclude "architecture unsupported" here -- print the real error verbatim and let the
        # caller diagnose it (the previous two "failures" were actually device UUID / KV cache config
        # issues, not an architecture problem).
        raise SystemExit(f"[vLLM init failed] {type(e).__name__}: {e}\n"
                         f"-> See the EngineCore traceback above for the real root cause (could be VRAM/"
                         f"max_model_len/version, not necessarily unsupported architecture).")

    tok = llm.get_tokenizer()
    prompts = []
    for (text, instr) in SAMPLES:
        s = tok.apply_chat_template(_conversation(text, instr), tokenize=False, add_generation_prompt=True)
        prompts.append({"prompt": s})
    outputs = llm.embed(prompts)
    vllm_vecs = np.stack([np.asarray(o.outputs.embedding, dtype=np.float32) for o in outputs])

    # Dimension-alignment check (full dim should be 4096)
    if vllm_vecs.shape[1] != ref_vecs.shape[1]:
        print(f"[warn] Dimension mismatch: vLLM={vllm_vecs.shape[1]} official={ref_vecs.shape[1]}"
              f"(if vLLM's output isn't normalized/truncated the same way, cosine can still compare direction)", flush=True)
    d = min(vllm_vecs.shape[1], ref_vecs.shape[1])

    # Per-sample cosine (both sides should already be normalized; explicitly renormalize before the dot product just in case)
    def _cos(a, b):
        a = a[:d] / (np.linalg.norm(a[:d]) + 1e-12)
        b = b[:d] / (np.linalg.norm(b[:d]) + 1e-12)
        return float(np.dot(a, b))

    print("\n=== G1 vector equivalence (cosine per sample) ===")
    cosines = []
    for i, (text, _) in enumerate(SAMPLES):
        c = _cos(vllm_vecs[i], ref_vecs[i])
        cosines.append(c)
        vnorm = float(np.linalg.norm(vllm_vecs[i]))
        print(f"  [{i}] cos={c:.7f}  vllm_norm={vnorm:.5f}  «{text[:32]}»")
    cosines = np.array(cosines)
    mn, mean = cosines.min(), cosines.mean()
    print(f"\n--- min cos={mn:.7f}  mean={mean:.7f} ---")
    if mn > 0.9999:
        print("PASS: G1 GO: min cosine > 0.9999 -- vLLM's query vectors are mathematically equivalent to the "
              "existing store (built with the official encoder); it can query the existing store.")
    elif mn > 0.999:
        print("MARGINAL: G1 borderline: 0.999 < min < 0.9999 -- direction is highly consistent but with some "
              "micro-drift (suspect transformers version/bf16). Must run G2 cross-build/cross-query to check "
              "whether top-k actually flips; if needed, align transformers versions in the vllm env and re-test.")
    else:
        print("FAIL: G1 NO-GO: min cosine <= 0.999 -- significant vector drift. The existing store cannot be "
              "queried directly by vLLM. First check whether the input format/tokenization matches token-for-token "
              "(a common cause of false negatives); otherwise the whole store must be rebuilt with vLLM (see VLLM_PLAN §8).")


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM <-> official text-vector equivalence probe (G0/G1)")
    ap.add_argument("--step", required=True, choices=["official", "vllm"])
    ap.add_argument("--out", default="/tmp/vllm_probe", help="directory for saved vectors (shared by both steps)")
    args = ap.parse_args()
    if args.step == "official":
        step_official(args.out)
    else:
        step_vllm(args.out)


if __name__ == "__main__":
    main()
