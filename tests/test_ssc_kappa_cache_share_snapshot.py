"""Snapshot-based behavioral verification for SSC κ-adaptive Picard-share
cache (Optimization A).

The unit test (`test_ssc_kappa_cache_share.py`) verifies the cache slots
are wired correctly. THIS test verifies the realized `dA`, `dB`
trajectory difference between share=True and share=False is within the
predicted snapshot-drift envelope, on real per-pair optimizer state.

Procedure: load one snapshot step from a recorded chord-tight production
run, replay the chord-tight-clean pipeline at picard=2, κ=0.6 twice
(share=False vs True), and compare realized per-pair dA/dB.

Pass thresholds (loose, since at picard=2 the n=1 cross-coupling
multiplies a small c-diff into a slightly larger dA diff):
- median per-pair max(‖ΔdA‖_F/‖dA‖_F, ‖ΔdB‖_F/‖dB‖_F) < 3%
- p99 < 8%
- max < 15%

If snapshot files not present, test skips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from lora_playground.optim import AdamPolarProductLoRA
from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, load_snapshot,
)


def _have_snapshot(run_key):
    root = SNAP_ROOTS.get(run_key)
    return (root is not None and root.exists()
            and STEPS_BY_ROOT.get(run_key))


def _build_model_from_snapshot(snap, max_pairs=None):
    """Build a fake LoRA model whose adapters carry A, B from the snapshot
    pair_state. Returns (model, pair_aux) where pair_aux is the list of
    (u_A, u_B) per adapter — used as a synthetic grad to drive the step.

    Only includes pairs where the snapshot has 'A', 'B', 'u_A', 'u_B'
    (i.e., the §9 diagnostic stash).
    """
    pair_state = snap["pair_state"]
    items = []
    for pi, p in pair_state.items():
        if not all(k in p for k in ("A", "B", "u_A", "u_B")):
            continue
        items.append((pi, p))
    if max_pairs is not None:
        items = items[:max_pairs]
    if not items:
        return None, None, None

    class _Pair(nn.Module):
        def __init__(self, A, B):
            super().__init__()
            r, d_in = A.shape
            d_out = B.shape[0]
            self.lora_A = nn.ModuleDict({"default": nn.Linear(d_in, r, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(r, d_out, bias=False)})
            with torch.no_grad():
                self.lora_A["default"].weight.copy_(A.float())
                self.lora_B["default"].weight.copy_(B.float())

    class _Model(nn.Module):
        def __init__(self, pairs):
            super().__init__()
            self.adapters = nn.ModuleList([_Pair(A, B) for A, B in pairs])

    pairs_AB = [(p["A"].float().cpu(), p["B"].float().cpu()) for _, p in items]
    u_pairs = [(p["u_A"].float().cpu(), p["u_B"].float().cpu()) for _, p in items]
    model = _Model(pairs_AB)
    return model, u_pairs, [pi for pi, _ in items]


def _seed_grads_from_u(model, u_pairs):
    """Drive the optimizer step with u_A, u_B as the gradients. The
    optimizer will compute Adam(grad) → u_adam internally; we don't
    perfectly reproduce the snapshot's u (which is post-Adam-RMS), but
    the shape/scale is realistic and identical across share=True/False —
    only their c-cache strategy differs, which is what we're measuring.
    """
    for adapter, (uA, uB) in zip(model.adapters, u_pairs):
        adapter.lora_A["default"].weight.grad = uA.clone()
        adapter.lora_B["default"].weight.grad = uB.clone()


def _make_opt(model, share_picard):
    return AdamPolarProductLoRA(
        model,
        lr=3e-2,
        betas=(0.9, 0.999),
        delta=1e-6,
        eps=1e-8,
        ns_steps=5,
        lora_plus_multiplier=1.0,
        log_basic_diagnostics=False,
        picard_iters=2,
        precond_refresh_every=1,
        precond_method="higham",
        magnitude_rule="spectral_chord_tight_clean",
        ns_form="rect",
        polar_method="ssc",
        ssc_kappa=0.6,
        ssc_nsteps=10,
        ssc_kappa_refresh_every=1,
        ssc_kappa_warmup_steps=0,
        ssc_kappa_solver="eigvalsh",
        ssc_kappa_cache_share_picard=share_picard,
    )


def _snapshot_deltas(A_before, A_after, B_before, B_after):
    """Return per-pair (ΔA, ΔB) as fp32 CPU tensors."""
    return (
        (A_after.detach().cpu().float() - A_before.detach().cpu().float()),
        (B_after.detach().cpu().float() - B_before.detach().cpu().float()),
    )


def _run_one(run_key, step, max_pairs=16):
    snap = load_snapshot(step, root=SNAP_ROOTS[run_key])
    model_no, u_pairs, pair_ids = _build_model_from_snapshot(snap, max_pairs=max_pairs)
    if model_no is None:
        pytest.skip(f"{run_key} step {step} has no pairs with §9 diagnostic stash")
    # Two parallel models with identical initial state.
    model_sh = type(model_no)(
        [(adapter.lora_A["default"].weight.detach().clone(),
          adapter.lora_B["default"].weight.detach().clone())
         for adapter in model_no.adapters]
    )

    # Capture initial A, B for each.
    A0_no = [a.lora_A["default"].weight.detach().clone() for a in model_no.adapters]
    B0_no = [a.lora_B["default"].weight.detach().clone() for a in model_no.adapters]
    A0_sh = [a.lora_A["default"].weight.detach().clone() for a in model_sh.adapters]
    B0_sh = [a.lora_B["default"].weight.detach().clone() for a in model_sh.adapters]

    opt_no = _make_opt(model_no, share_picard=False)
    opt_sh = _make_opt(model_sh, share_picard=True)

    _seed_grads_from_u(model_no, u_pairs)
    _seed_grads_from_u(model_sh, u_pairs)

    opt_no.step()
    opt_sh.step()

    diffs = []
    for i, adapter in enumerate(model_no.adapters):
        dA_no, dB_no = _snapshot_deltas(
            A0_no[i], adapter.lora_A["default"].weight,
            B0_no[i], adapter.lora_B["default"].weight,
        )
        dA_sh, dB_sh = _snapshot_deltas(
            A0_sh[i], model_sh.adapters[i].lora_A["default"].weight,
            B0_sh[i], model_sh.adapters[i].lora_B["default"].weight,
        )
        nA = torch.linalg.norm(dA_no)
        nB = torch.linalg.norm(dB_no)
        if nA < 1e-20 or nB < 1e-20:
            continue
        relA = float(torch.linalg.norm(dA_sh - dA_no) / nA)
        relB = float(torch.linalg.norm(dB_sh - dB_no) / nB)
        diffs.append(max(relA, relB))
    return diffs


def _assert_envelope(diffs, run_label, step):
    assert diffs, f"{run_label} step {step}: no comparable pairs"
    arr = np.array(diffs)
    p50 = float(np.median(arr))
    p99 = float(np.percentile(arr, 99))
    pmax = float(arr.max())
    print(f"\n[{run_label} step={step}] N={len(arr)}  "
          f"p50={p50:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    assert p50 < 0.03, f"{run_label}: median diff {p50:.4f} >= 3%"
    assert p99 < 0.08, f"{run_label}: p99 diff {p99:.4f} >= 8%"
    assert pmax < 0.15, f"{run_label}: max diff {pmax:.4f} >= 15%"


@pytest.mark.parametrize("run_key,label", [
    (RUN_A, "RUN_A-r64-lr3e-2"),
    (RUN_B, "RUN_B-r256-lr1e-1"),
])
def test_share_picard_drift_within_envelope(run_key, label):
    if not _have_snapshot(run_key):
        pytest.skip(f"snapshot files for {run_key} not present")
    steps = STEPS_BY_ROOT[run_key]
    # Pick a mid-training step (avoid step 0 init artifacts and the last
    # step which may be partial).
    step = steps[len(steps) // 2] if len(steps) > 1 else steps[0]

    diffs = _run_one(run_key, step, max_pairs=16)
    _assert_envelope(diffs, label, step)
