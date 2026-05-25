"""Snapshot-based behavioral verification for Optimization B
(cross-shape-group eigvalsh batching for κ-adaptive SSC).

The unit test (`test_ssc_cross_group_eigvalsh.py`) verifies on a tiny
synthetic 3-shape-group model. THIS test replays on real per-pair
optimizer state (OLMo-2-1B all-linear LoRA — which has 3 shape groups:
qkvo / gate-up / down) for RUN_A (r=64 step≥1) and RUN_B (r=256 step≥1).

Pass thresholds (tight — this is a pure batching refactor; eigvalsh on
stacked Grams is bit-identical to per-group eigvalsh modulo cuSOLVER
batch-axis bookkeeping):
- median per-pair max(‖ΔdA‖_F/‖dA‖_F, ‖ΔdB‖_F/‖dB‖_F) < 0.1%
- p99 < 0.3%
- max < 1.0%

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
    """Same pattern as test_ssc_kappa_cache_share_snapshot.py — wraps the
    snapshot pair_state into a Fake LoRA model whose shape-group
    distribution mirrors the snapshot (qkvo/gate-up/down for OLMo-2-1B).
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
    for adapter, (uA, uB) in zip(model.adapters, u_pairs):
        adapter.lora_A["default"].weight.grad = uA.clone()
        adapter.lora_B["default"].weight.grad = uB.clone()


def _make_opt(model, xgroup):
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
        ssc_kappa_cache_share_picard=True,
        ssc_kappa_cross_group_eigvalsh=xgroup,
    )


def _snapshot_deltas(A_before, A_after, B_before, B_after):
    return (
        (A_after.detach().cpu().float() - A_before.detach().cpu().float()),
        (B_after.detach().cpu().float() - B_before.detach().cpu().float()),
    )


def _run_one(run_key, step, max_pairs=None):
    snap = load_snapshot(step, root=SNAP_ROOTS[run_key])
    model_pg, u_pairs, _ = _build_model_from_snapshot(snap, max_pairs=max_pairs)
    if model_pg is None:
        pytest.skip(f"{run_key} step {step} has no pairs with §9 diagnostic stash")
    model_xg = type(model_pg)(
        [(adapter.lora_A["default"].weight.detach().clone(),
          adapter.lora_B["default"].weight.detach().clone())
         for adapter in model_pg.adapters]
    )

    A0_pg = [a.lora_A["default"].weight.detach().clone() for a in model_pg.adapters]
    B0_pg = [a.lora_B["default"].weight.detach().clone() for a in model_pg.adapters]
    A0_xg = [a.lora_A["default"].weight.detach().clone() for a in model_xg.adapters]
    B0_xg = [a.lora_B["default"].weight.detach().clone() for a in model_xg.adapters]

    opt_pg = _make_opt(model_pg, xgroup=False)
    opt_xg = _make_opt(model_xg, xgroup=True)

    if len(opt_pg.group_state) <= 1:
        pytest.skip(
            f"{run_key} step {step}: snapshot has {len(opt_pg.group_state)} "
            "shape group(s) — cross-group preflight needs ≥2"
        )

    _seed_grads_from_u(model_pg, u_pairs)
    _seed_grads_from_u(model_xg, u_pairs)

    opt_pg.step()
    opt_xg.step()

    diffs = []
    for i, adapter in enumerate(model_pg.adapters):
        dA_pg, dB_pg = _snapshot_deltas(
            A0_pg[i], adapter.lora_A["default"].weight,
            B0_pg[i], adapter.lora_B["default"].weight,
        )
        dA_xg, dB_xg = _snapshot_deltas(
            A0_xg[i], model_xg.adapters[i].lora_A["default"].weight,
            B0_xg[i], model_xg.adapters[i].lora_B["default"].weight,
        )
        nA = torch.linalg.norm(dA_pg)
        nB = torch.linalg.norm(dB_pg)
        if nA < 1e-20 or nB < 1e-20:
            continue
        relA = float(torch.linalg.norm(dA_xg - dA_pg) / nA)
        relB = float(torch.linalg.norm(dB_xg - dB_pg) / nB)
        diffs.append(max(relA, relB))
    return diffs


@pytest.mark.parametrize("run_key,label", [
    (RUN_A, "RUN_A-r64-lr3e-2"),
    (RUN_B, "RUN_B-r256-lr1e-1"),
])
def test_cross_group_drift_within_envelope(run_key, label):
    """Replay all available snapshot steps × all pairs and assert the
    realized dA/dB diff between cross_group=True and False is within
    fp32 noise. Pure batching refactor — the eigvalsh-on-stacked-Grams
    is the same kernel as per-group eigvalsh on each Gram individually.
    """
    if not _have_snapshot(run_key):
        pytest.skip(f"snapshot files for {run_key} not present")
    steps = [s for s in STEPS_BY_ROOT[run_key] if s > 0]
    if not steps:
        pytest.skip(f"{run_key} has no non-zero snapshot steps")

    all_diffs = []
    per_step_rows = []
    for step in steps:
        diffs = _run_one(run_key, step, max_pairs=None)
        if not diffs:
            continue
        arr = np.array(diffs)
        per_step_rows.append((step, len(arr), float(np.median(arr)),
                              float(np.percentile(arr, 99)), float(arr.max())))
        all_diffs.extend(diffs)
    print(f"\n[{label}] coverage = {len(steps)} steps × all-pairs:")
    for step, n, p50, p99, pm in per_step_rows:
        print(f"  step={step:>4}  N={n:>3}  p50={p50:.2e}  p99={p99:.2e}  max={pm:.2e}")

    assert all_diffs, f"{label}: no comparable pairs across any step"
    arr = np.array(all_diffs)
    p50 = float(np.median(arr))
    p99 = float(np.percentile(arr, 99))
    pmax = float(arr.max())
    print(f"  POOLED N={len(arr)}  p50={p50:.2e}  p99={p99:.2e}  max={pmax:.2e}")
    # Pure batching refactor — fp32 noise floor.
    assert p50 < 1e-3, f"{label}: pooled median {p50:.2e} >= 0.1%"
    assert p99 < 3e-3, f"{label}: pooled p99 {p99:.2e} >= 0.3%"
    assert pmax < 1e-2, f"{label}: pooled max {pmax:.2e} >= 1%"
