"""Measure MISR apply-quality convergence on real snapshots.

For each snapshot (RUN_A/B/C/D), per shape group, per side {A, B}:
1. Build §2.5-rescaled whitened X (matches what the polar pipeline sees).
2. Solve c via the exact eigvalsh path (`_ssc_adaptive_kappa_batched`).
3. Apply MISR at the solved c with nsteps ∈ {10, 20, 30}.
4. Report ‖MISR_n − MISR_30‖_F / ‖MISR_30‖_F per pair.

Answers: at production ssc_nsteps=10, how close is MISR_10 to the true h_c(X)?

CPU-only, no GPU needed.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from lora_playground.optim import (
    _ssc_adaptive_kappa_batched, _ssc_misr_batched,
)
from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, RUN_C, RUN_D, load_snapshot,
)
from lora_playground.utils import spd_frac_power_inv


def _spd_half_inv_loop(S, eps=1e-6):
    out = torch.empty_like(S)
    for i in range(S.shape[0]):
        out[i] = spd_frac_power_inv(S[i], gamma=0.5, eps=eps)
    return out


def _whitened_X_groups(snap):
    """Per-(side, shape_group) tensors of §2.5-rescaled whitened X."""
    groups = defaultdict(list)
    for pi, p in snap['pair_state'].items():
        if not all(k in p for k in ('A', 'B', 'u_A', 'u_B')):
            continue
        A, B = p['A'].float(), p['B'].float()
        u_A, u_B = p['u_A'].float(), p['u_B'].float()
        key = (A.shape[0], A.shape[1], B.shape[0])
        groups[key].append((A, B, u_A, u_B))
    out = {'A': [], 'B': []}
    for shape_key, group in groups.items():
        A = torch.stack([g[0] for g in group])
        B = torch.stack([g[1] for g in group])
        u_A = torch.stack([g[2] for g in group])
        u_B = torch.stack([g[3] for g in group])
        SA = A @ A.transpose(-2, -1)
        SB = B.transpose(-2, -1) @ B
        W_for_A = _spd_half_inv_loop(SB)
        W_for_B = _spd_half_inv_loop(SA)
        X_A = W_for_A @ u_A
        X_B = u_B @ W_for_B
        for side, X in (('A', X_A), ('B', X_B)):
            sigma = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X])
            X = X / sigma.view(-1, *([1] * (X.dim() - 1)))
            out[side].append((shape_key, X))
    return out


def _diagnose_run(run_key, label, kappa=0.6, ref_nsteps=30, test_nsteps=(10, 20)):
    if SNAP_ROOTS.get(run_key) is None or not SNAP_ROOTS[run_key].exists():
        print(f"\n[{label}] SKIP — snapshot files not present")
        return
    steps = [s for s in STEPS_BY_ROOT[run_key] if s > 0]
    print(f"\n[{label}] {len(steps)} snapshot steps; ref nsteps={ref_nsteps}")

    # Per (side, nsteps) → list of per-pair relative F-residuals.
    residuals = {(side, n): [] for side in ('A', 'B') for n in test_nsteps}

    for step in steps:
        snap = load_snapshot(step, root=SNAP_ROOTS[run_key])
        groups = _whitened_X_groups(snap)
        for side in ('A', 'B'):
            for shape_key, X in groups[side]:
                _, c_true = _ssc_adaptive_kappa_batched(X, kappa=kappa, nsteps=ref_nsteps)
                H_ref = _ssc_misr_batched(X, c=c_true, nsteps=ref_nsteps)
                ref_norm = torch.linalg.matrix_norm(H_ref, dim=(-2, -1))  # (N,)
                for n in test_nsteps:
                    H_n = _ssc_misr_batched(X, c=c_true, nsteps=n)
                    diff_norm = torch.linalg.matrix_norm(H_ref - H_n, dim=(-2, -1))
                    rel = (diff_norm / ref_norm.clamp_min(1e-30)).tolist()
                    residuals[(side, n)].extend(rel)

    # Report.
    for n in test_nsteps:
        pooled = []
        for side in ('A', 'B'):
            arr = np.array(residuals[(side, n)])
            p50 = float(np.median(arr))
            p99 = float(np.percentile(arr, 99))
            pmax = float(arr.max())
            print(f"  nsteps={n}  side={side}  N={len(arr):>4}  "
                  f"p50={p50:.2e}  p99={p99:.2e}  max={pmax:.2e}")
            pooled.extend(residuals[(side, n)])
        arr = np.array(pooled)
        p50 = float(np.median(arr))
        p99 = float(np.percentile(arr, 99))
        pmax = float(arr.max())
        print(f"  nsteps={n}  POOLED  N={len(arr):>4}  "
              f"p50={p50:.2e}  p99={p99:.2e}  max={pmax:.2e}")


def main():
    torch.manual_seed(0)
    for run_key, label in [
        (RUN_A, "RUN_A r=64 lr=3e-2"),
        (RUN_C, "RUN_C r=64 lr=1e-1"),
        (RUN_B, "RUN_B r=256 lr=1e-1"),
        (RUN_D, "RUN_D r=256 lr=1e-3"),
    ]:
        _diagnose_run(run_key, label, kappa=0.6,
                      ref_nsteps=60, test_nsteps=(12, 14, 15, 16, 18, 20))


if __name__ == "__main__":
    main()
