"""Diagnostic-snapshot stash tests for AdamPolarProductLoRA.

Verifies that with `optimizer.snapshot_pair_tensors = True`, the batched step
writes A_pre, B_pre, u_A_pre, u_B_pre clones into pair_state with the right
shapes and values, and that save_checkpoint round-trips them.

CPU-only; reuses the toy fixtures from test_checkpoint.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.checkpoint import (
    ckpt_dir_for_step,
    load_checkpoint,
    save_checkpoint,
)
from tests.test_checkpoint import (
    _PeftLikeWrapper,
    _ToyModel,
    _make_optimizer,
    _step_once,
)

_OPT = "adam-polar-product-lora-coupled-spectral-chord-tight"


def _capture_pre_step_AB(optimizer):
    """Return list of (A.detach().clone(), B.detach().clone()) per pair, in
    optimizer pair-index order. These are the values FED INTO the step."""
    return [(A.detach().clone().float(), B.detach().clone().float())
            for (A, B) in optimizer.pairs]


def test_snapshot_off_by_default(tmp_path):
    torch.manual_seed(0)
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, _OPT)
    assert getattr(opt, "snapshot_pair_tensors") is False

    x = torch.randn(2, 8); y = torch.randn(2, 8)
    _step_once(model.inner, opt, x, y)
    for ps in opt.pair_state.values():
        for key in ("A_pre", "B_pre", "u_A_pre", "u_B_pre"):
            assert key not in ps, f"unexpected snapshot key {key} when flag off"


def test_snapshot_stash_shapes_and_values(tmp_path):
    """With the flag on, pair_state[i] gains the 4 stash keys with the right
    shapes; A_pre / B_pre match the pre-step values bitwise."""
    torch.manual_seed(0)
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, _OPT)
    pre_AB = _capture_pre_step_AB(opt)

    opt.snapshot_pair_tensors = True
    x = torch.randn(2, 8); y = torch.randn(2, 8)
    _step_once(model.inner, opt, x, y)

    assert len(opt.pair_state) == len(pre_AB)
    for i, (A_pre_ref, B_pre_ref) in enumerate(pre_AB):
        ps = opt.pair_state[i]
        for key in ("A_pre", "B_pre", "u_A_pre", "u_B_pre"):
            assert key in ps, f"missing snapshot key {key} at pair {i}"
            assert isinstance(ps[key], torch.Tensor)
            assert ps[key].dtype == torch.float32
        assert ps["A_pre"].shape == A_pre_ref.shape
        assert ps["B_pre"].shape == B_pre_ref.shape
        assert ps["u_A_pre"].shape == A_pre_ref.shape
        assert ps["u_B_pre"].shape == B_pre_ref.shape
        # A_pre, B_pre must match the values fed into the step (bitwise).
        assert torch.equal(ps["A_pre"], A_pre_ref), \
            f"A_pre at pair {i} does not match pre-step A"
        assert torch.equal(ps["B_pre"], B_pre_ref), \
            f"B_pre at pair {i} does not match pre-step B"


def test_snapshot_u_pre_matches_adam_rms_formula():
    """u_A_pre / u_B_pre should equal m_hat / (sqrt(v_hat) + eps), computed
    from the post-step m_A, v_A and the current step counter. This pins the
    stash to its definition before σ_max normalization."""
    torch.manual_seed(0)
    model = _PeftLikeWrapper(_ToyModel())
    opt = _make_optimizer(model.inner, _OPT)
    opt.snapshot_pair_tensors = True
    x = torch.randn(2, 8); y = torch.randn(2, 8)
    _step_once(model.inner, opt, x, y)

    b1, b2, eps = opt.beta1, opt.beta2, opt.eps
    # Group buffers hold the post-step m/v stacked across pairs.
    for gs in opt.group_state:
        indices = gs["indices"]
        t = opt.pair_state[indices[0]]["step"]
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        u_A_grp = (gs["m_A"] / bc1) / ((gs["v_A"] / bc2).sqrt() + eps)
        u_B_grp = (gs["m_B"] / bc1) / ((gs["v_B"] / bc2).sqrt() + eps)
        for j, gi in enumerate(indices):
            assert torch.allclose(opt.pair_state[gi]["u_A_pre"], u_A_grp[j],
                                  atol=1e-6, rtol=1e-6)
            assert torch.allclose(opt.pair_state[gi]["u_B_pre"], u_B_grp[j],
                                  atol=1e-6, rtol=1e-6)


def test_snapshot_save_load_round_trip(tmp_path):
    """save_checkpoint persists the stash; load_checkpoint restores it
    bitwise into a fresh optimizer's pair_state."""
    torch.manual_seed(0)
    src_model = _PeftLikeWrapper(_ToyModel())
    src_opt = _make_optimizer(src_model.inner, _OPT)
    src_opt.snapshot_pair_tensors = True

    x = torch.randn(2, 8); y = torch.randn(2, 8)
    _step_once(src_model.inner, src_opt, x, y)

    ckpt = ckpt_dir_for_step(tmp_path, step=1)
    save_checkpoint(
        ckpt,
        bare_model=src_model, optimizer=src_opt, scheduler=None,
        step=1, total_tokens=42, resume_segment=0,
        cfg_snapshot={"command": "test"},
    )

    # Fresh dst (different RNG seed → different inits before load).
    torch.manual_seed(99)
    dst_model = _PeftLikeWrapper(_ToyModel())
    dst_opt = _make_optimizer(dst_model.inner, _OPT)
    info = load_checkpoint(ckpt, bare_model=dst_model, optimizer=dst_opt,
                           scheduler=None)
    assert info is not None
    for i, src_ps in src_opt.pair_state.items():
        dst_ps = dst_opt.pair_state[i]
        for key in ("A_pre", "B_pre", "u_A_pre", "u_B_pre"):
            assert key in dst_ps, f"snapshot key {key} not restored at pair {i}"
            assert torch.equal(dst_ps[key], src_ps[key]), \
                f"snapshot key {key} drifted after load at pair {i}"
