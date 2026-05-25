"""Measure Higham inverse-square-root convergence on real snapshots.

Higham iterates compute S^{-1/2} for SPD S (used for whitening in the
chord-tight-clean polar pipeline). Like MISR, it's Newton-Schulz with
local quadratic convergence — but the rate depends on the condition
number of S.

For each snapshot, per shape group, per side {A, B}:
1. Build S = A A^T (for side A whitener) and S = B^T B (for side B).
2. Apply Higham at nsteps ∈ {10, 14, 16, 20} and at reference nsteps=60.
3. Report ‖Higham_n − Higham_60‖_F / ‖Higham_60‖_F per pair.

If nsteps=10 (current default) is materially under-converged, the
whitening step is wrong → downstream polar input is wrong.

CPU-only.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, RUN_C, RUN_D, load_snapshot,
)
from lora_playground.utils import spd_inv_sqrt_higham


def _higham_inv_sqrt(S, nsteps, delta=1e-6):
    return spd_inv_sqrt_higham(S, n_iters=nsteps, eps=delta)


def _diagnose_run(run_key, label, ref_nsteps=60, test_nsteps=(10, 14, 16, 20)):
    if SNAP_ROOTS.get(run_key) is None or not SNAP_ROOTS[run_key].exists():
        print(f"\n[{label}] SKIP — snapshot files not present")
        return
    steps = [s for s in STEPS_BY_ROOT[run_key] if s > 0]
    print(f"\n[{label}] {len(steps)} snapshot steps; ref nsteps={ref_nsteps}")

    residuals = {(side, n): [] for side in ('A', 'B') for n in test_nsteps}
    cond_numbers = {'A': [], 'B': []}

    for step in steps:
        snap = load_snapshot(step, root=SNAP_ROOTS[run_key])
        for pi, p in snap['pair_state'].items():
            if not all(k in p for k in ('A', 'B')):
                continue
            A = p['A'].float()
            B = p['B'].float()
            S_A = A @ A.transpose(-2, -1)            # (r, r) — used to whiten B side
            S_B = B.transpose(-2, -1) @ B            # (r, r) — used to whiten A side
            for side, S in (('A', S_A), ('B', S_B)):
                # Skip degenerate (rank-deficient) S — Higham doesn't converge there.
                eigs = torch.linalg.eigvalsh(S).clamp_min(0)
                cond = (eigs.max() / eigs.min().clamp_min(1e-30)).item()
                cond_numbers[side].append(cond)
                W_ref = _higham_inv_sqrt(S, nsteps=ref_nsteps)
                ref_norm = torch.linalg.matrix_norm(W_ref).item()
                if ref_norm < 1e-30:
                    continue
                for n in test_nsteps:
                    W_n = _higham_inv_sqrt(S, nsteps=n)
                    diff = torch.linalg.matrix_norm(W_ref - W_n).item()
                    residuals[(side, n)].append(diff / ref_norm)

    # Condition number summary.
    for side in ('A', 'B'):
        arr = np.array(cond_numbers[side])
        print(f"  cond(S_{side})  p50={float(np.median(arr)):.2e}  "
              f"p99={float(np.percentile(arr, 99)):.2e}  max={float(arr.max()):.2e}")
    # Residuals.
    for n in test_nsteps:
        pooled = []
        for side in ('A', 'B'):
            arr = np.array(residuals[(side, n)])
            if not len(arr): continue
            p50 = float(np.median(arr))
            p99 = float(np.percentile(arr, 99))
            pmax = float(arr.max())
            print(f"  nsteps={n}  side={side}  N={len(arr):>4}  "
                  f"p50={p50:.2e}  p99={p99:.2e}  max={pmax:.2e}")
            pooled.extend(residuals[(side, n)])
        if pooled:
            arr = np.array(pooled)
            print(f"  nsteps={n}  POOLED  N={len(arr):>4}  "
                  f"p50={float(np.median(arr)):.2e}  "
                  f"p99={float(np.percentile(arr, 99)):.2e}  "
                  f"max={float(arr.max()):.2e}")


def main():
    torch.manual_seed(0)
    for run_key, label in [
        (RUN_A, "RUN_A r=64 lr=3e-2"),
        (RUN_C, "RUN_C r=64 lr=1e-1"),
        (RUN_B, "RUN_B r=256 lr=1e-1"),
        (RUN_D, "RUN_D r=256 lr=1e-3"),
    ]:
        _diagnose_run(run_key, label, ref_nsteps=60, test_nsteps=(10, 14, 16, 20))


if __name__ == "__main__":
    main()
