"""Microbenchmark: looped per-pair Newton-Schulz vs batched-across-pairs NS.

Hypothesis (from profiling_a6000_2026_05_04.md): _polar_pipeline's
NS_A + NS_B is launch-bound across the 112-pair Python loop, not
per-matrix work. Stacking pairs of identical shape into a 3-D tensor
and using a single batched bmm sequence should compress the launch
storm and dominate the loop.

Test: for each of the three real shape groups in OLMo-2-1B at r=16
(N=64 of (2048,16); N=32 of (8192,16); N=16 of (2048,8192)), time
both implementations on real device tensors. Verify numerical
equivalence (max abs err < 1e-5 fp32) before benchmarking.

This is decoupled from the optimizer — it tests the NS primitive
in isolation. If batched NS is not faster than looped NS in this
microbenchmark, the larger optimizer-integration refactor is not
worth doing.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import _newton_schulz, _newton_schulz_batched


def equivalence_check(N, m, n, nsteps, device, dtype=torch.float32, atol=1e-5):
    """Stacked _newton_schulz_batched should match per-matrix _newton_schulz."""
    torch.manual_seed(0)
    X = torch.randn(N, m, n, device=device, dtype=dtype)
    Y_loop = torch.stack([_newton_schulz(X[i], nsteps=nsteps) for i in range(N)])
    Y_batched = _newton_schulz_batched(X, nsteps=nsteps)
    err = (Y_loop - Y_batched).abs().max().item()
    rel = err / (Y_loop.abs().max().item() + 1e-30)
    print(f"  equivalence (N={N}, {m}×{n}, nsteps={nsteps}): "
          f"max_abs_err={err:.2e}, max_rel_err={rel:.2e}")
    assert err < atol, f"NS batched/loop diverge: {err} > {atol}"


def time_loop(X, nsteps, n_warmup=3, n_reps=20):
    """Per-matrix loop, mirrors current _polar_pipeline."""
    N = X.shape[0]
    for _ in range(n_warmup):
        for i in range(N):
            _ = _newton_schulz(X[i], nsteps=nsteps)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range(N):
            _ = _newton_schulz(X[i], nsteps=nsteps)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def time_batched(X, nsteps, n_warmup=3, n_reps=20):
    for _ in range(n_warmup):
        _ = _newton_schulz_batched(X, nsteps=nsteps)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = _newton_schulz_batched(X, nsteps=nsteps)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--nsteps", type=int, default=5)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_reps", type=int, default=20)
    args = p.parse_args()
    device = torch.device(args.device)

    # Real shape groups for OLMo-2-1B at r=16, target_modules=all-linear:
    # × 64 of A=(16, 2048),  B=(2048, 16)
    # × 32 of A=(16, 2048),  B=(8192, 16)
    # × 16 of A=(16, 8192),  B=(2048, 16)
    # NS sees the polar pipeline's input shapes, which after whitening match
    # the original A/B shapes (S^{-1/2} is r×r, doesn't change outer dims).
    GROUPS = [
        ("A_64x16x2048",   64, 16, 2048),
        ("B_64x2048x16",   64, 2048, 16),
        ("A_32x16x2048",   32, 16, 2048),
        ("B_32x8192x16",   32, 8192, 16),
        ("A_16x16x8192",   16, 16, 8192),
        ("B_16x2048x16",   16, 2048, 16),
    ]

    print(f"# device={device}, nsteps={args.nsteps}, n_reps={args.n_reps}")
    if device.type == "cuda":
        print(f"# gpu={torch.cuda.get_device_name(device)}")

    print("\n# === Equivalence checks ===")
    for name, N, m, n in GROUPS:
        equivalence_check(N, m, n, args.nsteps, device)

    print("\n# === Timing (ms) ===")
    print(f"{'group':<22} {'N':>3} {'m':>5} {'n':>5} "
          f"{'loop_ms':>10} {'batch_ms':>10} {'speedup':>8}")
    print("-" * 80)
    total_loop = 0.0
    total_batch = 0.0
    for name, N, m, n in GROUPS:
        torch.manual_seed(0)
        X = torch.randn(N, m, n, device=device, dtype=torch.float32)
        loop_t = time_loop(X, args.nsteps, args.n_warmup, args.n_reps)
        batch_t = time_batched(X, args.nsteps, args.n_warmup, args.n_reps)
        loop_med = sorted(loop_t)[len(loop_t) // 2]
        batch_med = sorted(batch_t)[len(batch_t) // 2]
        speedup = loop_med / batch_med
        total_loop += loop_med
        total_batch += batch_med
        print(f"{name:<22} {N:>3} {m:>5} {n:>5} "
              f"{loop_med:>10.3f} {batch_med:>10.3f} {speedup:>7.2f}x")
    print("-" * 80)
    print(f"{'TOTAL (sum of medians)':<22} {' ':>3} {' ':>5} {' ':>5} "
          f"{total_loop:>10.3f} {total_batch:>10.3f} {total_loop/total_batch:>7.2f}x")


if __name__ == "__main__":
    main()
