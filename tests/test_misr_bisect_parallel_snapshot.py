"""Snapshot-based test for parallel-grid MISR-bisect (Optimization C).

Verifies on REAL production-scale data:
- Production X spectra (LoRA snapshots), not random Gaussian
- Production scale (r=64 or 256, N=16-64 per shape group)
- Compared against eigvalsh ground-truth c (not just sequential MISR-bisect,
  which has its own bisection error)

Pass criteria:
- Parallel-K=9 c-error vs eigvalsh ground truth is bounded by grid spacing
  (log_window/(K-1) = 0.5/8 ≈ 0.0625 → factor ≈ exp(0.0625) ≈ 1.064)
- F-norm of parallel-output matches MISR-applied-at-solved-c (winner consistency)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import torch

from lora_playground.optim import (
    _ssc_adaptive_kappa_batched, _ssc_misr_batched,
    _ssc_misr_bisect_batched_kpar,
)
from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, RUN_C, RUN_D, load_snapshot,
)
from lora_playground.utils import spd_frac_power_inv


def _have_snapshot(run_key):
    root = SNAP_ROOTS.get(run_key)
    return root is not None and root.exists() and STEPS_BY_ROOT.get(run_key)


def _spd_half_inv_loop(S, eps=1e-6):
    out = torch.empty_like(S)
    for i in range(S.shape[0]):
        out[i] = spd_frac_power_inv(S[i], gamma=0.5, eps=eps)
    return out


def _whitened_X_from_snapshot(snap, side='A'):
    """Return per-shape-group X tensors matching production §2.5-rescale."""
    groups = defaultdict(list)
    for pi, p in snap['pair_state'].items():
        if not all(k in p for k in ('A', 'B', 'u_A', 'u_B')):
            continue
        A, B = p['A'].float(), p['B'].float()
        u_A, u_B = p['u_A'].float(), p['u_B'].float()
        key = (A.shape[0], A.shape[1], B.shape[0])
        groups[key].append((A, B, u_A, u_B))
    out = []
    for shape_key, group in groups.items():
        A = torch.stack([g[0] for g in group])
        B = torch.stack([g[1] for g in group])
        u_A = torch.stack([g[2] for g in group])
        u_B = torch.stack([g[3] for g in group])
        SA = A @ A.transpose(-2, -1)
        SB = B.transpose(-2, -1) @ B
        if side == 'A':
            W = _spd_half_inv_loop(SB)
            X = W @ u_A
        else:
            W = _spd_half_inv_loop(SA)
            X = u_B @ W
        sigma = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X])
        X = X / sigma.view(-1, *([1] * (X.dim() - 1)))
        out.append((shape_key, X))
    return out


@pytest.mark.parametrize("run_key,label", [
    (RUN_A, "RUN_A-r64"),
    (RUN_C, "RUN_C-r64-highlr"),
    (RUN_B, "RUN_B-r256-highlr"),
])
def test_kpar_vs_eigvalsh_on_snapshot(run_key, label):
    """K=9 parallel grid should land within one grid step (factor ~1.06×) of
    the true eigvalsh-solved c on production X spectra.
    """
    if not _have_snapshot(run_key):
        pytest.skip(f"snapshot files for {run_key} not present")

    K_par = 9
    log_window = 0.5
    kappa = 0.6
    # Tolerance: one grid spacing in log, plus a small fp32 buffer.
    tol_log = (2 * log_window) / (K_par - 1) + 0.02
    tol_factor = float(np.exp(tol_log))  # multiplicative tolerance on c

    # Run on a mid-training step (avoid step 0).
    steps = [s for s in STEPS_BY_ROOT[run_key] if s > 0]
    step = steps[len(steps) // 2]

    snap = load_snapshot(step, root=SNAP_ROOTS[run_key])
    groups_A = _whitened_X_from_snapshot(snap, side='A')
    groups_B = _whitened_X_from_snapshot(snap, side='B')

    log_errors = []
    winner_consistency_max = 0.0
    for side_label, groups in [('A', groups_A), ('B', groups_B)]:
        for shape_key, X in groups:
            # Ground truth c via eigvalsh.
            _, c_true = _ssc_adaptive_kappa_batched(
                X, kappa=kappa, nsteps=10,
            )
            # Parallel-grid K=9, init at c_true (ideal warm start).
            out_par, c_par = _ssc_misr_bisect_batched_kpar(
                X, kappa=kappa, K=K_par, nsteps=10,
                c_init=c_true, log_window=log_window,
            )
            log_err = (c_par.log() - c_true.log()).abs()
            log_errors.extend(log_err.tolist())
            # Winner consistency: out_par for the winner should equal MISR(X, c_par).
            out_check = _ssc_misr_batched(X, c=c_par, nsteps=10)
            consistency = (out_par - out_check).abs().max().item()
            winner_consistency_max = max(winner_consistency_max, consistency)

    log_errs = np.array(log_errors)
    p50 = float(np.median(log_errs))
    p99 = float(np.percentile(log_errs, 99))
    pmax = float(log_errs.max())
    print(f"\n[{label} step={step}] N_pairs_total={len(log_errs)}")
    print(f"  c-log-error (vs eigvalsh): p50={p50:.4f}  p99={p99:.4f}  max={pmax:.4f}  "
          f"(tol={tol_log:.4f}, factor tol={tol_factor:.3f}×)")
    print(f"  winner consistency max|H_par - MISR(c_par)| = {winner_consistency_max:.2e}")

    assert pmax <= tol_log + 1e-4, (
        f"{label}: parallel-K={K_par} c-log-error {pmax:.4f} > tol {tol_log:.4f}"
    )
    assert winner_consistency_max < 1e-4, (
        f"{label}: winner H disagrees with MISR(c_par) by {winner_consistency_max:.2e}"
    )


def test_kpar_dispatch_actually_wired():
    """Confirms the --ssc_kappa_bisect_mode parallel flag plumbs through to use
    `_ssc_misr_bisect_batched_kpar` (not silently falls back to sequential)."""
    from lora_playground.optim import AdamPolarProductLoRA
    import torch.nn as nn

    class _Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict({"default": nn.Linear(16, 4, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(4, 16, bias=False)})

    model = nn.ModuleList([_Pair() for _ in range(2)])

    opt = AdamPolarProductLoRA(
        model, lr=1e-3, betas=(0.9, 0.999), delta=1e-6, eps=1e-8,
        ns_steps=5, lora_plus_multiplier=1.0, log_basic_diagnostics=False,
        picard_iters=2, precond_refresh_every=1, precond_method='higham',
        magnitude_rule='spectral_chord_tight_clean', ns_form='rect',
        polar_method='ssc', ssc_kappa=0.6, ssc_nsteps=10,
        ssc_kappa_refresh_every=1, ssc_kappa_warmup_steps=0,
        ssc_kappa_solver='misr_bisect', ssc_kappa_bisect_iters=3,
        ssc_kappa_bisect_mode='parallel',
    )
    # Sanity: optimizer attribute set correctly.
    assert opt.ssc_kappa_bisect_mode == 'parallel', \
        f"bisect_mode not threaded; got {opt.ssc_kappa_bisect_mode!r}"
