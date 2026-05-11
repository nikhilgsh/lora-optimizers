"""Bench accuracy of higham (Newton-Schulz) S^{-1/2} vs eigh ground truth.

The chord-tight optimizer's spectral preconditioners SA^{-1/2}, SB^{-1/2}
are computed via _spd_inv_half with method='higham' by default (selected
over eigh because eigh on r×r at r=256 hits a kernel-launch storm — same
underlying issue as the σ_max bench at large r). This script asks: how
much accuracy do we lose for that speed?

For each (r, higham_iters) combination, generate a realistic SPD matrix
S (= A·A^T for random A at LoRA shape (r, d)) and compute:

  X_eigh  = (S + δI)^{-1/2}  via eigh decomposition           (ground truth)
  X_higham = (S + δI)^{-1/2} via spd_inv_sqrt_higham(n_iters) (production)

Report:
  rel_err_F   = ‖X_higham - X_eigh‖_F / ‖X_eigh‖_F
  residual_F  = ‖X_higham · (S + δI) · X_higham - I‖_F        (independent check)
  rel_err_top = |σ_max(X_higham) - σ_max(X_eigh)| / σ_max(X_eigh)

Also tests condition-number sensitivity: at LoRA init S = A·A^T has
near-flat spectrum (Kaiming uniform A); after training, S could be much
more ill-conditioned. Bench at three condition regimes:
  well-conditioned   (random A, all σ similar)
  moderately ill     (one tail factor at 0.1× the bulk)
  highly ill         (one tail factor at 0.01× the bulk)

Usage:
    python scripts/bench/bench_higham_accuracy.py [--device cuda]
"""
import argparse
import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lora_playground.utils import spd_inv_sqrt_higham
from lora_playground.optim import _spd_inv_half


def make_spd(r, d, condition, device, dtype):
    """Build S = A · A^T with a controlled condition number.

    `condition` in {'well', 'moderate', 'ill'} sets how much the bottom
    singular value is suppressed relative to the bulk.
    """
    torch.manual_seed(0)
    A = torch.randn(r, d, device=device, dtype=dtype)
    if condition != "well":
        U, S, Vt = torch.linalg.svd(A, full_matrices=False)
        if condition == "moderate":
            S[-1] *= 0.1
        elif condition == "ill":
            S[-1] *= 0.01
        A = U @ torch.diag(S) @ Vt
    return A @ A.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--eps", type=float, default=1e-6, help="δ in (S+δI)^{-1/2}")
    args = parser.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]

    print(f"device={args.device}  dtype={args.dtype}  eps={args.eps}")
    print(f"{'r':>4s} {'d':>5s} {'cond':>10s} {'iters':>5s} {'rel_err_F':>12s} {'residual_F':>12s} {'rel_err_top':>12s}")
    print("-" * 80)
    eye = lambda r: torch.eye(r, device=args.device, dtype=dtype)
    for r in (16, 64, 128, 256):
        for d in (2048,):
            for condition in ("well", "moderate", "ill"):
                S = make_spd(r, d, condition, args.device, dtype)
                # Ground truth via eigh
                S_damped = S + args.eps * eye(r)
                X_eigh = _spd_inv_half(S, eps=args.eps, method="eigh")
                # Reference σ_max(X_eigh) for relative top-eig check
                sigma_eigh = torch.linalg.eigvalsh(X_eigh).max()
                for n_iters in (5, 10, 20):
                    X_higham = spd_inv_sqrt_higham(S, n_iters=n_iters, eps=args.eps)
                    rel_err_F = (X_higham - X_eigh).norm() / X_eigh.norm()
                    residual_F = (X_higham @ S_damped @ X_higham - eye(r)).norm()
                    sigma_h = torch.linalg.eigvalsh(X_higham).max()
                    rel_err_top = (sigma_h - sigma_eigh).abs() / sigma_eigh
                    print(f"{r:>4d} {d:>5d} {condition:>10s} {n_iters:>5d} "
                          f"{float(rel_err_F):>12.4e} {float(residual_F):>12.4e} "
                          f"{float(rel_err_top):>12.4e}")
                print()


if __name__ == "__main__":
    main()
