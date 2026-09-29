"""GPU equivalence test (P1-2: the basis for local<->remote retrieval-vector mathematical
equivalence). Requires a real GPU + the Qwen3-VL model; CI has no GPU so this is auto-skipped.
Run manually with: conda activate custodian && pytest tests/engine/test_equivalence_gpu.py -v

**This only covers consistency of the truncation implementation (torch vs numpy), not full
local<->remote equivalence** (clarified in the Stage B review, M1):
- `_mrl` (torch) and `_mrl_np` (numpy) truncate the same fp32 full-dim vector, verifying the two
  implementations agree element-wise;
- the **bf16->fp32 step** (the real build-index entry point: model.process emits bf16 -> _mrl.float())
  is guarded in test_remote.py (pure CPU);
- the **end-to-end** path (spinning up the service, bf16->JSON->fp32 transport, mixed build/query E2)
  is covered by the scripts/equiv_gpu.py timesharing script in docs/SCALE_OUT.md Section 5-B
  (measured: encode cosine=1.0000000 / maxdiff 2.98e-08 / rerank maxdiff 0.00)."""
import os

import numpy as np
import pytest

from embedder.config import EmbedConfig


def _gpu_ready() -> bool:
    try:
        import torch
        return (torch.cuda.is_available()
                and os.path.isdir(os.path.expanduser("~/models/Qwen3-VL-Embedding-8B/scripts")))
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_ready(), reason="requires a real GPU + the Qwen3-VL model (custodian environment); skipped in CI")


def test_mrl_torch_numpy_consistent_on_fp32_output():
    """On real model output (already .float()-ed to fp32 via encode_text), Dense._mrl (torch) and
    RemoteDense._mrl_np (numpy) agree element-wise.
    Honest boundary warning (Stage B review, M1): this test starts from **the same fp32** vector and
    only verifies "fp32 truncation: torch ~= numpy" -- it does **not** cover the bf16->fp32 step.
    - the guard for the real build-index entry point (bf16 tensors into _mrl) ->
      test_remote.py::test_mrl_fp32_normalize_on_bf16_input (pure CPU);
    - measured end-to-end local<->remote (including bf16->JSON->fp32 transport) ->
      scripts/equiv_gpu.py timesharing (cosine=1.0/maxdiff 2.98e-8, SCALE_OUT Section 5-B)."""
    import torch

    from embedder.dense import Dense
    from embedder.remote import RemoteDense

    D = 1024
    texts = ["What was Netflix's revenue in 2015", "The company reported strong quarterly growth.",
             "Mixed script test 123 !@#$%", "a"]
    # Get the real model's full dim (dense_dim=huge -> _mrl doesn't truncate, returns the full bf16->fp32
    # dimension). Loads the model only once, to avoid OOM.
    full = Dense(EmbedConfig(dense_dim=10 ** 9)).encode_text(texts)          # (n, 4096) fp32
    assert full.shape[0] == len(texts) and full.shape[1] > D

    v_torch = Dense(EmbedConfig(dense_dim=D))._mrl(torch.from_numpy(full))   # local truncation (torch); this Dense doesn't load a model
    rd = RemoteDense.__new__(RemoteDense)                                    # skip __init__ (don't build an httpx client)
    rd.cfg = EmbedConfig(dense_dim=D)
    v_numpy = rd._mrl_np(full)                                              # remote truncation (numpy)

    assert v_torch.shape == v_numpy.shape == (len(texts), D)
    assert np.allclose(v_torch, v_numpy, atol=1e-6), \
        f"the torch/numpy truncation paths are not equivalent, maxdiff={np.abs(v_torch - v_numpy).max():.2e}"
    # fp32 normalize -> norm should both = 1 (if _mrl falls back to bf16 normalize, local norm~=1.002, and this assertion fails)
    assert np.allclose(np.linalg.norm(v_torch, axis=-1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(v_numpy, axis=-1), 1.0, atol=1e-5)

# Note: the P1-1 lower-bound assertion guard test (pure CPU) has been moved to
# tests/engine/test_remote.py::test_mrl_np_lower_bound_assertion -- keeping it here would let the
# module-level GPU skipif swallow it (Stage F review: removing fail-loud in CPU/CI wouldn't turn the
# test red = a false green).
