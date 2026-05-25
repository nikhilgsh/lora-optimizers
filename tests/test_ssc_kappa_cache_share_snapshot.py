"""Snapshot-based behavioral verification for SSC κ-adaptive Picard-share
cache (Optimization A) — REALISTIC variant.

Replays the chord-tight-clean polar pipeline DIRECTLY with snapshot
(A, B, u_A, u_B) — bypasses Adam moment re-derivation so the X the
polar pipeline sees matches what production sees on the snapshot step.

This is the same X-construction the c-drift bench uses
(`scripts/bench/ssc_kappa_drift_snapshot.py`).

Coverage:
- All 4 snapshot keys (RUN_A, RUN_C r=64; RUN_B r=256 high-lr; RUN_D r=256 low-lr).
- All available snapshot steps per key (skip step 0 — degenerate B=0 init).
- All 112 pairs per snapshot (qkvo + gate/up + down shape groups).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from lora_playground.optim import (
    _ssc_adaptive_kappa_batched, _ssc_misr_batched,
)
from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, RUN_C, RUN_D, load_snapshot,
)
from lora_playground.utils import spd_frac_power_inv


def _have_snapshot(run_key):
    root = SNAP_ROOTS.get(run_key)
    return (root is not None and root.exists()
            and STEPS_BY_ROOT.get(run_key))


def _spd_half_inv_loop(S, eps=1e-6):
    out = torch.empty_like(S)
    for i in range(S.shape[0]):
        out[i] = spd_frac_power_inv(S[i], gamma=0.5, eps=eps)
    return out


def _group_by_shape(pair_state):
    groups = defaultdict(list)
    for pi, p in pair_state.items():
        if not all(k in p for k in ("A", "B", "u_A", "u_B")):
            continue
        A, B = p["A"].float(), p["B"].float()
        u_A, u_B = p["u_A"].float(), p["u_B"].float()
        key = (A.shape[0], A.shape[1], B.shape[0])
        groups[key].append((pi, A, B, u_A, u_B))
    return groups


def _stack(group):
    A = torch.stack([g[1] for g in group])
    B = torch.stack([g[2] for g in group])
    u_A = torch.stack([g[3] for g in group])
    u_B = torch.stack([g[4] for g in group])
    return A, B, u_A, u_B


def _picard_step_per_group(A, B, u_A, u_B, kappa, eta, picard_iters, ssc_nsteps,
                          share_picard, eps=1e-6):
    """Replay one chord-tight-clean polar pipeline call (per shape group) at
    picard=2 with the given cache_share policy.

    Returns dA, dB (same shape as A, B) realized by the pipeline. Mirrors
    `_chord_tight_clean_polar_pipeline` body but is decoupled from the
    optimizer-class wiring so the test isolates the c-cache-policy effect.
    """
    N, r, _ = A.shape
    # Whitening (Higham not used here; eigh-based for test determinism).
    SA = A @ A.transpose(-2, -1)
    SB = B.transpose(-2, -1) @ B
    SA_half_inv = _spd_half_inv_loop(SA, eps)
    SB_half_inv = _spd_half_inv_loop(SB, eps)

    # σ_max(A), σ_max(B) → ρ
    sigma_A = torch.stack([torch.linalg.matrix_norm(a, ord=2) for a in A])
    sigma_B = torch.stack([torch.linalg.matrix_norm(b, ord=2) for b in B])
    rho = eta / (sigma_A + sigma_B + 1e-30)

    # Whitened input + §2.5 pre-rescale
    X_A = SB_half_inv @ u_A
    X_B = u_B @ SA_half_inv
    sigma_XA = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X_A])
    sigma_XB = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X_B])
    inv_XA = (1.0 / (sigma_XA + 1e-30)).view(N, 1, 1)
    inv_XB = (1.0 / (sigma_XB + 1e-30)).view(N, 1, 1)
    X_A = X_A * inv_XA
    X_B = X_B * inv_XB
    u_A_r = u_A * inv_XA
    u_B_r = u_B * inv_XB

    dA = torch.zeros_like(A)
    dB = torch.zeros_like(B)
    cache_A = {}
    cache_B = {}

    for n in range(picard_iters):
        if n == 0:
            X_A_eff, X_B_eff = X_A, X_B
        else:
            u_A_eff = u_A_r + (1.0 / eta) * (B.transpose(-2, -1) @ dB @ A)
            u_B_eff = u_B_r + (1.0 / eta) * (B @ dA @ A.transpose(-2, -1))
            X_A_eff = SB_half_inv @ u_A_eff
            X_B_eff = u_B_eff @ SA_half_inv

        # c-cache logic — mirrors _chord_tight_clean_polar_pipeline._polar.
        if share_picard:
            # One slot per side, shared across n.
            if 'c_A' not in cache_A or n == 0:
                _, c_A = _ssc_adaptive_kappa_batched(X_A_eff, kappa=kappa, nsteps=ssc_nsteps)
                cache_A['c_A'] = c_A
            c_A = cache_A['c_A']
            P_A = _ssc_misr_batched(X_A_eff, c=c_A, nsteps=ssc_nsteps)

            if 'c_B' not in cache_B or n == 0:
                _, c_B = _ssc_adaptive_kappa_batched(X_B_eff, kappa=kappa, nsteps=ssc_nsteps)
                cache_B['c_B'] = c_B
            c_B = cache_B['c_B']
            P_B = _ssc_misr_batched(X_B_eff, c=c_B, nsteps=ssc_nsteps)
        else:
            P_A, c_A = _ssc_adaptive_kappa_batched(X_A_eff, kappa=kappa, nsteps=ssc_nsteps)
            P_B, c_B = _ssc_adaptive_kappa_batched(X_B_eff, kappa=kappa, nsteps=ssc_nsteps)

        geo_A = SB_half_inv @ P_A
        geo_B = P_B @ SA_half_inv
        op_geoA = torch.stack([torch.linalg.matrix_norm(g, ord=2) for g in geo_A])
        op_geoB = torch.stack([torch.linalg.matrix_norm(g, ord=2) for g in geo_B])
        dA = -(rho / (op_geoA + 1e-30)).view(N, 1, 1) * geo_A
        dB = -(rho / (op_geoB + 1e-30)).view(N, 1, 1) * geo_B

    return dA, dB


def _run_step(run_key, step, kappa=0.6, eta=3e-2, picard_iters=2, ssc_nsteps=10):
    snap = load_snapshot(step, root=SNAP_ROOTS[run_key])
    groups = _group_by_shape(snap['pair_state'])
    if not groups:
        return []
    diffs = []
    for shape_key, group in groups.items():
        A, B, u_A, u_B = _stack(group)
        dA_no, dB_no = _picard_step_per_group(
            A, B, u_A, u_B, kappa, eta, picard_iters, ssc_nsteps, share_picard=False)
        dA_sh, dB_sh = _picard_step_per_group(
            A, B, u_A, u_B, kappa, eta, picard_iters, ssc_nsteps, share_picard=True)
        for i in range(A.shape[0]):
            nA = torch.linalg.norm(dA_no[i])
            nB = torch.linalg.norm(dB_no[i])
            if nA < 1e-20 or nB < 1e-20:
                continue
            relA = float(torch.linalg.norm(dA_sh[i] - dA_no[i]) / nA)
            relB = float(torch.linalg.norm(dB_sh[i] - dB_no[i]) / nB)
            diffs.append(max(relA, relB))
    return diffs


@pytest.mark.parametrize("run_key,label,eta", [
    (RUN_A, "RUN_A-r64-lr3e-2", 3e-2),
    (RUN_C, "RUN_C-r64-lr1e-1", 1e-1),
    (RUN_B, "RUN_B-r256-lr1e-1", 1e-1),
    (RUN_D, "RUN_D-r256-lr1e-3", 1e-3),
])
def test_share_picard_drift_realistic(run_key, label, eta):
    """All 4 snapshot keys × all post-init steps × all 112 pairs.

    Realism: invokes the polar pipeline with snapshot's actual (A, B, u_A, u_B)
    — bypasses Adam re-derivation so X matches production.

    RUN_D (η=1e-3, two orders below production) is the saturated-c regime per
    prior snapshot drift analysis — bounds are looser.
    """
    if not _have_snapshot(run_key):
        pytest.skip(f"snapshot files for {run_key} not present")
    steps = [s for s in STEPS_BY_ROOT[run_key] if s > 0]

    is_low_eta = run_key == RUN_D
    p50_thresh = 0.15 if is_low_eta else 0.03
    p99_thresh = 0.50 if is_low_eta else 0.08
    max_thresh = 1.50 if is_low_eta else 0.15

    all_diffs = []
    rows = []
    for step in steps:
        diffs = _run_step(run_key, step, kappa=0.6, eta=eta)
        if not diffs:
            continue
        arr = np.array(diffs)
        rows.append((step, len(arr), float(np.median(arr)),
                     float(np.percentile(arr, 99)), float(arr.max())))
        all_diffs.extend(diffs)

    print(f"\n[{label}] {len(rows)} steps:")
    for step, n, p50, p99, pm in rows:
        print(f"  step={step:>4}  N={n:>3}  p50={p50:.4f}  p99={p99:.4f}  max={pm:.4f}")
    assert all_diffs, f"{label}: no comparable pairs"
    arr = np.array(all_diffs)
    p50 = float(np.median(arr))
    p99 = float(np.percentile(arr, 99))
    pmax = float(arr.max())
    print(f"  POOLED N={len(arr)}  p50={p50:.4f}  p99={p99:.4f}  max={pmax:.4f}")
    assert p50 < p50_thresh, f"{label}: pooled median {p50:.4f} >= {p50_thresh}"
    assert p99 < p99_thresh, f"{label}: pooled p99 {p99:.4f} >= {p99_thresh}"
    assert pmax < max_thresh, f"{label}: pooled max {pmax:.4f} >= {max_thresh}"
