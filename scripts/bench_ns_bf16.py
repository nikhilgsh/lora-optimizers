"""Microbench: batched Newton-Schulz in bf16 vs fp32, on real shape groups.

Validates the modded-nanogpt-style "iterate in bf16" pattern for our
LoRA shapes. Two questions:

1. **Convergence:** does bf16 NS at j=5 converge to acceptable orthogonality?
   bf16 mantissa is ~7 bits; orthogonality residual will bottom at ~1e-3
   instead of fp32's ~1e-7. Algorithm 1 only needs the polar direction,
   not its precision — 1e-3 is well within tolerance.

2. **Speedup:** bf16 matmul has 2× throughput on Ampere tensor cores. NS
   is matmul-heavy (5 iters × 3 matmuls × 2 sides). Expect ~2× wall
   speedup on the larger groups; smaller groups may be closer to 1× since
   they're already cheap.

Benches on the same six shape groups as `bench_ns_batched.py`, comparing:
- fp32 batched (current default)
- bf16 batched (new) — pre-norm in fp32, iterate in bf16, cast result to fp32

Reports orthogonality `‖Y^T Y − I‖_F / √r` and direction error vs
SVD polar `‖Y − UV^T‖_F / √min(m,n)` to confirm the algorithm's
qualitative property is preserved.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from lora_playground.optim import _newton_schulz_batched


def svd_polar(X):
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ Vh


def ortho_err(Y):
    tall = Y.shape[-2] >= Y.shape[-1]
    M = Y.transpose(-2, -1) @ Y if tall else Y @ Y.transpose(-2, -1)
    n = M.shape[-1]
    I = torch.eye(n, dtype=M.dtype, device=M.device)
    return (M - I).flatten(-2).norm(dim=-1) / (n ** 0.5)


def time_call(fn, n_warmup, n_reps, device):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(n_reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(device)
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

    GROUPS = [
        ("A_64x16x2048",  64, 16, 2048),
        ("B_64x2048x16",  64, 2048, 16),
        ("A_32x16x2048",  32, 16, 2048),
        ("B_32x8192x16",  32, 8192, 16),
        ("A_16x16x8192",  16, 16, 8192),
        ("B_16x2048x16",  16, 2048, 16),
    ]

    print(f"# device={device}, nsteps={args.nsteps}, n_reps={args.n_reps}")
    if device.type == "cuda":
        print(f"# gpu={torch.cuda.get_device_name(device)}")
    print()
    print(f"{'group':<20} {'N':>3} {'shape':<12} "
          f"{'fp32_ms':>9} {'bf16_ms':>9} {'speedup':>8} "
          f"{'fp32_orth':>11} {'bf16_orth':>11} "
          f"{'fp32_polar':>12} {'bf16_polar':>12}")
    print("-" * 122)
    total_fp32 = 0.0
    total_bf16 = 0.0
    for name, N, m, n in GROUPS:
        torch.manual_seed(0)
        X = torch.randn(N, m, n, device=device, dtype=torch.float32)
        P_true = svd_polar(X)
        scale = (min(m, n)) ** 0.5

        Y_fp32 = _newton_schulz_batched(X, nsteps=args.nsteps)
        Y_bf16 = _newton_schulz_batched(X, nsteps=args.nsteps,
                                         dtype=torch.bfloat16).float()

        oe_fp32 = ortho_err(Y_fp32).max().item()
        oe_bf16 = ortho_err(Y_bf16).max().item()
        pe_fp32 = ((Y_fp32 - P_true).flatten(-2).norm(dim=-1) / scale).max().item()
        pe_bf16 = ((Y_bf16 - P_true).flatten(-2).norm(dim=-1) / scale).max().item()

        t_fp32 = time_call(
            lambda: _newton_schulz_batched(X, nsteps=args.nsteps),
            args.n_warmup, args.n_reps, device)
        t_bf16 = time_call(
            lambda: _newton_schulz_batched(X, nsteps=args.nsteps,
                                            dtype=torch.bfloat16),
            args.n_warmup, args.n_reps, device)
        med_fp32 = sorted(t_fp32)[len(t_fp32) // 2]
        med_bf16 = sorted(t_bf16)[len(t_bf16) // 2]
        total_fp32 += med_fp32
        total_bf16 += med_bf16
        sp = med_fp32 / med_bf16
        print(f"{name:<20} {N:>3} {f'{m}x{n}':<12} "
              f"{med_fp32:>9.3f} {med_bf16:>9.3f} {sp:>7.2f}x "
              f"{oe_fp32:>11.2e} {oe_bf16:>11.2e} "
              f"{pe_fp32:>12.2e} {pe_bf16:>12.2e}")
    print("-" * 122)
    print(f"{'TOTAL':<20} {' ':>3} {' ':<12} "
          f"{total_fp32:>9.3f} {total_bf16:>9.3f} "
          f"{total_fp32/total_bf16:>7.2f}x")


if __name__ == "__main__":
    main()
