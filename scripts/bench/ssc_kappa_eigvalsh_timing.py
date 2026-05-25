"""SSC κ-adaptive: time the eigvalsh+bisection cost vs the rest of the
polar pipeline, at production shapes loaded from snapshots.

Output: per-call wall (ms) for
  - _ssc_adaptive_kappa_batched (eigvalsh + bisection + MISR)
  - _ssc_misr_batched (apply-only, with a precomputed c)
  - delta = κ-adaptive overhead per call

Plus a projected per-step wall under (picard, inner-cache, refresh-every-N)
configurations, so the refresh budget falls out of the numbers directly.

Run on the allocated Blackwell:
  python scripts/bench/ssc_kappa_eigvalsh_timing.py --kappa 0.6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, load_snapshot,
)
from lora_playground.optim import (
    _ssc_adaptive_kappa_batched, _ssc_misr_batched, _newton_schulz_gram_batched,
)
from lora_playground.utils import spd_frac_power_inv


def _cuda_time(fn, warmup=3, iters=20):
    """Return median wall (ms) over iters after warmup, with cuda.synchronize."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def _load_production_X(run_key, step, device, side='A'):
    """Load a single shape group at `step` from `run_key`, return X tensor
    matching the §2.5-rescaled input to _ssc_adaptive_kappa_batched.
    Picks the most common shape group (typically q/k/v/o projections)."""
    root = SNAP_ROOTS[run_key]
    snap = load_snapshot(step, root=root)
    from collections import Counter
    shapes = Counter()
    for _, p in snap['pair_state'].items():
        if 'A' not in p: continue
        A, B = p['A'], p['B']
        shapes[(A.shape[0], A.shape[1], B.shape[0])] += 1
    target_shape = shapes.most_common(1)[0][0]
    pairs = [p for _, p in snap['pair_state'].items()
             if 'A' in p and (p['A'].shape[0], p['A'].shape[1], p['B'].shape[0]) == target_shape]
    A = torch.stack([p['A'].float() for p in pairs]).to(device)
    B = torch.stack([p['B'].float() for p in pairs]).to(device)
    u_A = torch.stack([p['u_A'].float() for p in pairs]).to(device)
    u_B = torch.stack([p['u_B'].float() for p in pairs]).to(device)

    SB = B.transpose(-2, -1) @ B
    SA = A @ A.transpose(-2, -1)
    SB_half_inv = torch.stack([spd_frac_power_inv(SB[i], 0.5, eps=1e-6)
                               for i in range(SB.shape[0])])
    SA_half_inv = torch.stack([spd_frac_power_inv(SA[i], 0.5, eps=1e-6)
                               for i in range(SA.shape[0])])
    if side == 'A':
        X = SB_half_inv @ u_A                                    # (N, r, d_in)
    else:
        X = u_B @ SA_half_inv                                    # (N, d_out, r)

    # §2.5 pre-rescale by per-pair σ_max.
    sigma = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X])
    X = X / sigma.view(-1, *([1] * (X.dim() - 1)))
    return X, target_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappa', type=float, default=0.6)
    ap.add_argument('--ssc_nsteps', type=int, default=10)
    ap.add_argument('--ns_steps', type=int, default=5)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    print(f"\n# κ={args.kappa}  ssc_nsteps={args.ssc_nsteps}  ns_steps={args.ns_steps}")
    print(f"# device={args.device}  {torch.cuda.get_device_name(0)}")

    # Production scales: r=64 (RUN_A) and r=256 (RUN_B), step 2000.
    cases = [
        ('r=64',  RUN_A, 2000),
        ('r=256', RUN_B, 2000),
    ]
    for label, run_key, step in cases:
        for side in ['A', 'B']:
            X, shape = _load_production_X(run_key, step, args.device, side=side)
            N, m, n = X.shape
            r = X.shape[-2] if X.shape[-2] < X.shape[-1] else X.shape[-1]
            print(f"\n## {label} side={side}  shape={shape}  batched X={tuple(X.shape)}")

            # Solve c once outside the timed loop so MISR can be timed with a real c.
            _, c = _ssc_adaptive_kappa_batched(X, kappa=args.kappa, nsteps=args.ssc_nsteps)

            t_full = _cuda_time(lambda: _ssc_adaptive_kappa_batched(
                X, kappa=args.kappa, nsteps=args.ssc_nsteps))
            t_misr = _cuda_time(lambda: _ssc_misr_batched(
                X, c=c, nsteps=args.ssc_nsteps))
            t_ns = _cuda_time(lambda: _newton_schulz_gram_batched(
                X, nsteps=args.ns_steps))

            eig_overhead = t_full - t_misr
            print(f"  κ-adaptive (eigvalsh + bisect + MISR)   {t_full:7.3f} ms")
            print(f"  MISR-only  (apply with cached c)        {t_misr:7.3f} ms")
            print(f"  NS-gram    (canonical polar baseline)   {t_ns:7.3f} ms")
            print(f"  → eigvalsh+bisection overhead            {eig_overhead:7.3f} ms"
                  f"   ({eig_overhead/max(t_misr,1e-9):.2f}× MISR)")

            # Per-step projections at picard=2, both sides (A and B):
            # The polar pipeline calls _polar twice per Picard iter (A + B).
            # Total eigvalsh calls/step at picard=2:
            #   no cache: 4
            #   inner-cache: 2 (only n=0)
            #   inner-cache + refresh-every-N: 2/N
            # Total MISR calls/step at picard=2 = 4 (always).
            print(f"  per-step projections (picard=2, ONE-SIDE only — double for both):")
            misr_per_step = 2 * t_misr   # one side, 2 Picard iters
            print(f"    no cache             {2*t_full:7.3f} ms  ({2*eig_overhead:.2f} eig overhead)")
            print(f"    inner-cache          {t_full + t_misr:7.3f} ms  ({eig_overhead:.2f} eig overhead)")
            for refresh in (5, 10, 50):
                wall = (1/refresh) * t_full + (2 - 1/refresh) * t_misr
                overhead = (1/refresh) * eig_overhead
                pct_of_full = 100 * overhead / (2 * t_full)
                print(f"    inner-cache + N={refresh:<3d} {wall:7.3f} ms  ({overhead:.3f} eig overhead, "
                      f"{pct_of_full:.1f}% of per-step uncached baseline)")


if __name__ == '__main__':
    main()
