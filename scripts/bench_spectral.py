"""Microbench + accuracy probe for σ_max primitives in `lora_playground.spectral`.

Times each shape-B primitive on synthetic inputs matching production shapes
AND reports rel-error vs exact SVD. Timing without accuracy is misleading —
a 10× speedup for a primitive that overestimates σ_max by 2× would still
be worse for chord-direction (over-estimate → too-conservative step size).

Synthetic inputs span three spectrum regimes to bracket realistic
chord-direction behavior:

  - random:        Gaussian X, Y → roughly flat spectrum, multi-moment loose
  - rank-r-spike:  one dominant σ in BP → multi-moment tight
  - rank-deficient: Y near singular (mimics chord-direction's near-null B
                   early in training) → exposes krylov-chol fragility

  - r ∈ {16, 64, 128, 256}     (LoRA rank)
  - m, n ∈ {2048}              (model dim)
  - N (batch) = 1, 32, 112

Companion: integration bench at the optimizer level (`scripts/bench_chord_direction_step.py`)
is the load-bearing measurement — accuracy and timing both end up reflected
in per-step wall time and convergence rate.

Usage:
    python scripts/bench_spectral.py                    # default shapes
    python scripts/bench_spectral.py --device cuda      # GPU bench
    python scripts/bench_spectral.py --quick            # smaller sweep
    python scripts/bench_spectral.py --accuracy-only    # skip timing
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from lora_playground.spectral import (
    sigma_max_krylov_chol,
    sigma_max_power_iter_nonsym,
    sigma_max_warm_power_iter_unfactored,
    sigma_max_multimoment_upper,
)


def _time_call(fn, n_warmup=3, n_repeat=20, sync_fn=None):
    """Median ms/call over n_repeat trials, after n_warmup warmups."""
    for _ in range(n_warmup):
        fn()
        if sync_fn is not None:
            sync_fn()
    times = []
    for _ in range(n_repeat):
        if sync_fn is not None:
            sync_fn()
        t0 = time.perf_counter()
        fn()
        if sync_fn is not None:
            sync_fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _make_inputs(N, m, r, n, device, dtype=torch.float32, seed=0, regime="random"):
    """Synthetic chord-direction shapes: X (N, m, r), Y (N, r, n).

    `regime` controls the spectrum shape, so we can measure how each
    primitive behaves across the conditions chord-direction sees:
      - 'random': Gaussian factors. Roughly flat singular spectrum.
      - 'rank-r-spike': construct X, Y so BP has one dominant singular
        value (rest are an order of magnitude smaller). Multi-moment
        bound should be tight here.
      - 'rank-deficient': Y is constructed to have rank << r, so G_inner
        is severely rank-deficient. Stresses krylov-chol's damping.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    if regime == "random":
        X = torch.randn(N, m, r, generator=g, device=device, dtype=dtype)
        Y = torch.randn(N, r, n, generator=g, device=device, dtype=dtype)
    elif regime == "rank-r-spike":
        # Build X = U Σ V^T with σ = [10, 1, 1, ..., 1]
        U_X = torch.randn(N, m, r, generator=g, device=device, dtype=dtype)
        U_X, _ = torch.linalg.qr(U_X)
        V_X = torch.randn(N, r, r, generator=g, device=device, dtype=dtype)
        V_X, _ = torch.linalg.qr(V_X)
        sig_X = torch.ones(r, device=device, dtype=dtype)
        sig_X[0] = 10.0
        X = U_X @ torch.diag(sig_X) @ V_X.transpose(-2, -1)
        U_Y = torch.randn(N, r, r, generator=g, device=device, dtype=dtype)
        U_Y, _ = torch.linalg.qr(U_Y)
        V_Y = torch.randn(N, n, r, generator=g, device=device, dtype=dtype)
        V_Y, _ = torch.linalg.qr(V_Y)
        sig_Y = torch.ones(r, device=device, dtype=dtype)
        sig_Y[0] = 10.0
        Y = U_Y @ torch.diag(sig_Y) @ V_Y.transpose(-2, -1)
    elif regime == "rank-deficient":
        # Y has rank 2 within an r-dim subspace
        eff_rank = min(2, r)
        X = torch.randn(N, m, r, generator=g, device=device, dtype=dtype)
        U_Y = torch.randn(N, r, eff_rank, generator=g, device=device, dtype=dtype)
        V_Y = torch.randn(N, eff_rank, n, generator=g, device=device, dtype=dtype)
        Y = U_Y @ V_Y
    else:
        raise ValueError(f"unknown regime: {regime}")
    G_outer = X.transpose(-2, -1) @ X                  # (N, r, r)
    G_inner = Y @ Y.transpose(-2, -1)                  # (N, r, r)
    return X, Y, G_outer, G_inner


def _exact_sigma_max_batch(X, Y):
    """Reference σ_max(X·Y) via SVD on the explicit product. Small-batch only."""
    XY = X @ Y                                          # (N, m, n)
    sig = torch.linalg.svdvals(XY).max(dim=-1).values   # (N,)
    return sig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="smaller shape sweep")
    parser.add_argument("--n-repeat", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    sync_fn = torch.cuda.synchronize if device.type == "cuda" else None
    print(f"Device: {device}, repeats per shape: {args.n_repeat}")
    print()

    if args.quick:
        shapes = [
            (1,   2048, 16,  2048),
            (32,  2048, 64,  2048),
            (112, 2048, 256, 2048),
        ]
    else:
        shapes = [
            (1,   2048, 16,  2048),
            (1,   2048, 64,  2048),
            (1,   2048, 256, 2048),
            (32,  2048, 16,  2048),
            (32,  2048, 64,  2048),
            (32,  2048, 256, 2048),
            (112, 2048, 16,  2048),
            (112, 2048, 64,  2048),
            (112, 2048, 128, 2048),
            (112, 2048, 256, 2048),
        ]

    regimes = ["random", "rank-r-spike", "rank-deficient"]

    for regime in regimes:
        print()
        print(f"=== Regime: {regime} ===")
        print(f"{'N':>5s} {'r':>5s} | "
              f"{'krylov':>9s} {'nonsym':>9s} {'multimom':>9s} "
              f"| {'kry-err':>9s} {'ns-err':>9s} {'mm-err':>9s}")
        print("-" * 100)
        for N, m, r, n in shapes:
            X, Y, G_outer, G_inner = _make_inputs(N, m, r, n, device=device, regime=regime)
            # Accuracy: rel-error vs exact SVD on each primitive (median over batch).
            ref = _exact_sigma_max_batch(X, Y)
            sig_kry = sigma_max_krylov_chol(G_outer, G_inner)
            sig_ns, _ = sigma_max_power_iter_nonsym(G_outer, G_inner)
            sig_mm = sigma_max_multimoment_upper(G_outer, G_inner)
            kry_err = ((sig_kry - ref).abs() / ref).median().item()
            ns_err  = ((sig_ns  - ref).abs() / ref).median().item()
            mm_err  = ((sig_mm  - ref) / ref).median().item()  # signed (upper bound → ≥ 0)
            # Timing
            t_kry = _time_call(
                lambda: sigma_max_krylov_chol(G_outer, G_inner),
                n_repeat=args.n_repeat, sync_fn=sync_fn,
            )
            t_ns = _time_call(
                lambda: sigma_max_power_iter_nonsym(G_outer, G_inner),
                n_repeat=args.n_repeat, sync_fn=sync_fn,
            )
            t_mm = _time_call(
                lambda: sigma_max_multimoment_upper(G_outer, G_inner),
                n_repeat=args.n_repeat, sync_fn=sync_fn,
            )
            print(f"{N:>5d} {r:>5d} | "
                  f"{t_kry:>7.2f}ms {t_ns:>7.2f}ms {t_mm:>7.2f}ms | "
                  f"{kry_err:>9.2e} {ns_err:>9.2e} {mm_err:>+9.2e}")

    print()
    print("Columns:")
    print("  ms-columns:  median wall time over --n-repeat trials (after warmup).")
    print("  err-columns: median across batch of |σ_primitive - σ_SVD| / σ_SVD.")
    print("               mm-err is SIGNED (multi-moment is a strict upper bound;")
    print("               positive = over-estimate).")
    print("Interpret:")
    print("  - kry-err should be < 1e-3 (Krylov power-iter)")
    print("  - unf-err depends on n_iters=8 cold; 1-5% typical at random regime")
    print("  - mm-err quantifies multi-moment's looseness — the load-bearing")
    print("    accuracy question for using it in chord-direction.")


if __name__ == "__main__":
    main()
