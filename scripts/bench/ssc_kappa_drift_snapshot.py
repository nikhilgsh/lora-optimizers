"""SSC κ-adaptive c-drift analysis on snapshot data.

Two questions for amortization strategy:

1. **Inner-Picard drift.** At picard=2, how much does the solved c change
   between n=0 and n=1? If small, we can cache c at n=0 and reuse at n=1
   (skip the eigvalsh+bisection at the inner iter).

2. **Cross-step drift.** How much does c at n=0 change between adjacent
   snapshot steps? Tells us how aggressive a between-step refresh schedule
   can be.

Runs batched-by-shape on the allocated GPU. Output: prints percentiles
of |Δc|/c per run + writes per-pair CSV.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, RUN_C, RUN_D, load_snapshot,
)
from lora_playground.optim import (
    _solve_c_from_kappa_batched, _ssc_misr_batched, _ssc_adaptive_kappa_batched,
)
from lora_playground.utils import spd_frac_power_inv


def _spd_half_inv_batched(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """S: (N, r, r) PSD → (N, r, r) inverse-sqrt. Loops over N (eigh is
    launch-bound but N is small — ~10-50 per shape group)."""
    out = torch.empty_like(S)
    for i in range(S.shape[0]):
        out[i] = spd_frac_power_inv(S[i], gamma=0.5, eps=eps)
    return out


def _group_pairs_by_shape(pair_state: dict, max_pairs: int | None):
    """Return {(r, d_in, d_out): list[(pair_idx, A, B, u_A, u_B)]}."""
    groups = defaultdict(list)
    items = list(pair_state.items())
    if max_pairs is not None:
        items = items[:max_pairs]
    for pi, p in items:
        if 'A' not in p or 'u_A' not in p:
            continue
        A, B = p['A'].float(), p['B'].float()
        u_A, u_B = p['u_A'].float(), p['u_B'].float()
        key = (A.shape[0], A.shape[1], B.shape[0])
        groups[key].append((pi, A, B, u_A, u_B))
    return groups


def _stack_group(group, device):
    pi = [g[0] for g in group]
    A = torch.stack([g[1] for g in group]).to(device)   # (N, r, d_in)
    B = torch.stack([g[2] for g in group]).to(device)   # (N, d_out, r)
    u_A = torch.stack([g[3] for g in group]).to(device)  # (N, r, d_in)
    u_B = torch.stack([g[4] for g in group]).to(device)  # (N, d_out, r)
    return pi, A, B, u_A, u_B


def _picard_two_step(A, B, u_A, u_B, kappa, eta, ssc_nsteps, eps=1e-6):
    """Run the chord-tight κ-adaptive SSC pipeline for picard=2 on a
    stacked group. Returns (c_A0, c_A1, c_B0, c_B1) each shape (N,)."""
    N, r, _ = A.shape
    d_out = B.shape[1]
    SA = A @ A.transpose(-2, -1)
    SB = B.transpose(-2, -1) @ B
    SA_half_inv = _spd_half_inv_batched(SA, eps=eps)
    SB_half_inv = _spd_half_inv_batched(SB, eps=eps)

    X_A = SB_half_inv @ u_A                                    # (N, r, d_in)
    X_B = u_B @ SA_half_inv                                    # (N, d_out, r)

    # §2.5 pre-rescale by σ_max(X_*) (use torch.linalg.matrix_norm per-pair).
    sigma_XA = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X_A])
    sigma_XB = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X_B])
    inv_XA = (1.0 / (sigma_XA + 1e-30)).view(N, 1, 1)
    inv_XB = (1.0 / (sigma_XB + 1e-30)).view(N, 1, 1)
    X_A = X_A * inv_XA
    X_B = X_B * inv_XB
    u_A_r = u_A * inv_XA
    u_B_r = u_B * inv_XB

    # n=0
    _, c_A0 = _ssc_adaptive_kappa_batched(X_A, kappa=kappa, nsteps=ssc_nsteps)
    _, c_B0 = _ssc_adaptive_kappa_batched(X_B, kappa=kappa, nsteps=ssc_nsteps)

    # Reproduce n=0 polar output → dA, dB.
    P_A = _ssc_misr_batched(X_A, c=c_A0, nsteps=ssc_nsteps)
    P_B = _ssc_misr_batched(X_B, c=c_B0, nsteps=ssc_nsteps)
    geo_A = SB_half_inv @ P_A
    geo_B = P_B @ SA_half_inv

    sigma_A = torch.stack([torch.linalg.matrix_norm(a, ord=2) for a in A])
    sigma_B = torch.stack([torch.linalg.matrix_norm(b, ord=2) for b in B])
    rho = eta / (sigma_A + sigma_B + 1e-30)                    # (N,)

    op_geoA = torch.stack([torch.linalg.matrix_norm(g, ord=2) for g in geo_A])
    op_geoB = torch.stack([torch.linalg.matrix_norm(g, ord=2) for g in geo_B])
    dA = -(rho / (op_geoA + 1e-30)).view(N, 1, 1) * geo_A
    dB = -(rho / (op_geoB + 1e-30)).view(N, 1, 1) * geo_B

    # n=1 cross-coupling correction.
    u_A_eff = u_A_r + (1.0 / eta) * (B.transpose(-2, -1) @ dB @ A)
    u_B_eff = u_B_r + (1.0 / eta) * (B @ dA @ A.transpose(-2, -1))
    X_A_eff = SB_half_inv @ u_A_eff
    X_B_eff = u_B_eff @ SA_half_inv
    _, c_A1 = _ssc_adaptive_kappa_batched(X_A_eff, kappa=kappa, nsteps=ssc_nsteps)
    _, c_B1 = _ssc_adaptive_kappa_batched(X_B_eff, kappa=kappa, nsteps=ssc_nsteps)

    return c_A0.cpu(), c_A1.cpu(), c_B0.cpu(), c_B1.cpu()


def analyze_run(run_key, kappa, eta, ssc_nsteps, max_pairs, device, csv_writer):
    root = SNAP_ROOTS[run_key]
    if not root.exists():
        print(f"  [skip] {run_key} root missing: {root}")
        return
    steps = STEPS_BY_ROOT[run_key]
    if not steps:
        print(f"  [skip] {run_key} no steps")
        return
    print(f"\n=== {run_key}  η={eta}  κ={kappa}  steps={steps}  device={device} ===")

    inner_records = []   # (step, pair, side, c0, c1)
    series = {}          # (pair, side) -> [(step, c0)]

    for step in steps:
        snap = load_snapshot(step, root=root)
        groups = _group_pairs_by_shape(snap['pair_state'], max_pairs)
        for shape_key, group in groups.items():
            pi, A, B, u_A, u_B = _stack_group(group, device)
            c_A0, c_A1, c_B0, c_B1 = _picard_two_step(
                A, B, u_A, u_B, kappa, eta, ssc_nsteps)
            for j, p_idx in enumerate(pi):
                a0, a1 = float(c_A0[j]), float(c_A1[j])
                b0, b1 = float(c_B0[j]), float(c_B1[j])
                inner_records.append((step, p_idx, 'A', a0, a1))
                inner_records.append((step, p_idx, 'B', b0, b1))
                series.setdefault((p_idx, 'A'), []).append((step, a0))
                series.setdefault((p_idx, 'B'), []).append((step, b0))
        print(f"  step {step:5d} done   pairs={sum(len(g) for g in groups.values())}")

    # Reports.
    rel_inner = np.array([abs(c1 - c0) / max(c0, 1e-30)
                          for (_, _, _, c0, c1) in inner_records])
    print(f"  inner-Picard |Δc|/c   N={len(rel_inner):5d}   "
          f"p50={np.median(rel_inner):.4f}  p90={np.percentile(rel_inner, 90):.4f}  "
          f"p99={np.percentile(rel_inner, 99):.4f}  max={rel_inner.max():.4f}")

    cross_recs = []
    for key, lst in series.items():
        lst.sort()
        for i in range(len(lst) - 1):
            (s0, c0), (s1, c1) = lst[i], lst[i + 1]
            cross_recs.append((key[0], key[1], s1 - s0, abs(c1 - c0) / max(c0, 1e-30)))
    if cross_recs:
        for g in sorted({x[2] for x in cross_recs}):
            arr = np.array([x[3] for x in cross_recs if x[2] == g])
            print(f"  cross-step gap={g:5d}  N={len(arr):4d}   "
                  f"p50={np.median(arr):.4f}  p90={np.percentile(arr, 90):.4f}  "
                  f"p99={np.percentile(arr, 99):.4f}  max={arr.max():.4f}")

    if csv_writer is not None:
        for (step, pi, side, c0, c1) in inner_records:
            csv_writer.writerow({
                'run': str(run_key), 'step': step, 'pair': pi, 'side': side,
                'kind': 'inner', 'gap': 0, 'c0': c0, 'c1': c1,
                'rel_drift': abs(c1 - c0) / max(c0, 1e-30),
            })
        for (pi, side, gap, rel) in cross_recs:
            csv_writer.writerow({
                'run': str(run_key), 'step': -1, 'pair': pi, 'side': side,
                'kind': 'cross', 'gap': gap, 'c0': float('nan'), 'c1': float('nan'),
                'rel_drift': rel,
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappa', type=float, default=0.6)
    ap.add_argument('--ssc_nsteps', type=int, default=10)
    ap.add_argument('--max_pairs', type=int, default=None)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out_csv', type=str,
                    default=str(REPO / 'notebooks/snapshot_analysis/_data/ssc_c_drift_kappa0p6.csv'))
    args = ap.parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, 'w', newline='')
    writer = csv.DictWriter(
        f, fieldnames=['run', 'step', 'pair', 'side', 'kind', 'gap', 'c0', 'c1', 'rel_drift'])
    writer.writeheader()

    runs = [
        (RUN_A, 3e-2),   # r=64,  low lr
        (RUN_C, 1e-1),   # r=64,  high lr
        (RUN_D, 1e-3),   # r=256, low lr
        (RUN_B, 1e-1),   # r=256, high lr
    ]
    for run_key, eta in runs:
        analyze_run(run_key, kappa=args.kappa, eta=eta,
                    ssc_nsteps=args.ssc_nsteps, max_pairs=args.max_pairs,
                    device=args.device, csv_writer=writer)

    f.close()
    print(f"\nCSV: {out_path}")


if __name__ == '__main__':
    main()
