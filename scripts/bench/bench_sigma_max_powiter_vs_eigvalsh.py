"""Benchmark σ_max estimation: warm-start power-iter vs eigvalsh.

Motivation: the chord-tight optimizer's σ_max(A), σ_max(B), σ_max(geo_A),
σ_max(geo_B) computations were switched from warm-start power-iter to eigvalsh
on the r×r Gram (commits 57a932b, 54311ba). Eigvalsh is exact but the kernel-
launch overhead — flagged by the user as a concern, especially since the
project uses higham (Newton-Schulz) over eigh for S^{-1/2} for exactly this
reason — should be benchmarked against warm-start power-iter on production
LoRA shapes before committing eigvalsh for the variant 1 hot path.

Generates synthetic A (r, d_in), B (d_out, r) and geo_A (r, d_in), geo_B
(d_out, r) at the LoRA-realistic shapes used in OLMo-2-1B training:
  r ∈ {16, 64, 128, 256}
  d ∈ {2048, 8192}     (q/k/v/out at 2048, mlp at 8192)
  batch (N pairs)   ∈ {1, 28, 112}  (1 pair, 1 layer's pairs, all 112)

Two methods compared per shape:
  (A) warm-start power-iter at n_iters ∈ {3, 5, 8} via _sigma_max_power_iter
      (or _batched). Cached top vector reused across calls.
  (B) eigvalsh on r×r Gram (smaller side).

Reports wall time per call (µs), accuracy ratio σ_pwr/σ_eigh.

Usage:
    python scripts/bench/bench_sigma_max_powiter_vs_eigvalsh.py [--device cuda] [--dtype fp32]
"""
import argparse
import time
import torch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lora_playground.optim import (
    _sigma_max_power_iter,
    _sigma_max_power_iter_batched,
)


def gram_r_eigvalsh(M):
    """σ_max via eigvalsh on the r×r Gram (smaller side)."""
    if M.shape[-2] <= M.shape[-1]:
        G = M @ M.transpose(-1, -2)
    else:
        G = M.transpose(-1, -2) @ M
    return torch.linalg.eigvalsh(G).clamp_min(0.0).max(dim=-1).values.sqrt()


def time_per_call(fn, n_warmup=5, n_timed=50, device="cuda"):
    """Median per-call wall time in microseconds with CUDA sync."""
    for _ in range(n_warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_timed):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)  # us
    times.sort()
    return times[len(times) // 2]


def run_shape(r, d, N, device, dtype):
    """Bench all methods on shape (N, r, d) (or per-pair if N==1).

    Returns dict of method → (wall_us, accuracy_ratio).
    """
    torch.manual_seed(0)
    if N == 1:
        M = torch.randn(r, d, device=device, dtype=dtype)
    else:
        M = torch.randn(N, r, d, device=device, dtype=dtype)
    # Reference: eigvalsh
    if N == 1:
        ref = gram_r_eigvalsh(M)
        ref_val = float(ref)
    else:
        ref = gram_r_eigvalsh(M)
        ref_val = float(ref.median())

    results = {}

    # eigvalsh wall time
    def eigvalsh_call():
        return gram_r_eigvalsh(M)
    t_eigh = time_per_call(eigvalsh_call, device=device)
    results["eigvalsh"] = (t_eigh, 1.0)

    # power-iter cold-start at n_iters ∈ {3, 5, 8}
    for n_iters in (3, 5, 8):
        if N == 1:
            def pwr_call(n=n_iters):
                s, _ = _sigma_max_power_iter(M, n_iters=n)
                return s
            s, _ = _sigma_max_power_iter(M, n_iters=n_iters)
            acc = float(s) / ref_val
        else:
            def pwr_call(n=n_iters):
                s, _ = _sigma_max_power_iter_batched(M, n_iters=n)
                return s
            s, _ = _sigma_max_power_iter_batched(M, n_iters=n_iters)
            acc = float(s.median()) / ref_val
        t = time_per_call(pwr_call, device=device)
        results[f"powiter_cold_n{n_iters}"] = (t, acc)

    # power-iter warm-start at n_iters=3 (run once cold, then time warm calls)
    if N == 1:
        _, v_warm = _sigma_max_power_iter(M, n_iters=20)  # converged init
        def pwr_warm_call():
            s, _ = _sigma_max_power_iter(M, v_init=v_warm, n_iters=3)
            return s
        s, _ = _sigma_max_power_iter(M, v_init=v_warm, n_iters=3)
        acc = float(s) / ref_val
    else:
        _, v_warm = _sigma_max_power_iter_batched(M, n_iters=20)
        def pwr_warm_call():
            s, _ = _sigma_max_power_iter_batched(M, v_init=v_warm, n_iters=3)
            return s
        s, _ = _sigma_max_power_iter_batched(M, v_init=v_warm, n_iters=3)
        acc = float(s.median()) / ref_val
    t = time_per_call(pwr_warm_call, device=device)
    results["powiter_warm_n3"] = (t, acc)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = parser.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print(f"device={args.device}  dtype={args.dtype}")
    print(f"{'shape':>20s}  {'method':>20s}  {'wall_us':>10s}  {'accuracy':>10s}")
    print("-" * 70)
    shapes = [
        (16, 2048, 1), (16, 2048, 112),
        (64, 2048, 1), (64, 2048, 112),
        (64, 8192, 1), (64, 8192, 28),
        (128, 2048, 1), (128, 2048, 112),
        (256, 2048, 1), (256, 2048, 112),
    ]
    for r, d, N in shapes:
        results = run_shape(r, d, N, args.device, dtype)
        for method, (t, acc) in results.items():
            print(f"{f'r={r} d={d} N={N}':>20s}  {method:>20s}  {t:>10.1f}  {acc:>10.4f}")
        print()


if __name__ == "__main__":
    main()
