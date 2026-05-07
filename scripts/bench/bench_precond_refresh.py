"""Microbench: precond_refresh primitive — looped eigh vs batched eigh
vs looped higham vs batched higham.

Hypothesis: like NS, precond_refresh is launch-bound across the 112-pair
Python loop. Both SA = A A^T and SB = B^T B are (r, r) for all pairs, so
all 112 stack into a single (112, r, r) tensor — even cleaner than NS.

Tests four implementations on synthetic SPD tensors at r ∈ {16, 64}, the
two canonical operating points:

  loop_eigh   : 112× spd_frac_power_inv (current default in optim.py)
  batched_eigh: torch.linalg.eigh on (112, r, r) + diag-clamp + recompose
  loop_higham : 112× spd_inv_sqrt_higham
  batch_higham: spd_inv_sqrt_higham_batched on (112, r, r)

Reports wall time (median over n_reps) and equivalence error vs the
current production path (loop_eigh).
"""
import argparse
import time

import torch

from lora_playground.utils import (
    spd_frac_power_inv,
    spd_inv_sqrt_higham,
    spd_inv_sqrt_higham_batched,
    spdify,
)


def make_spd_batch(N, r, device, seed=0):
    """Synthetic batch of SPD matrices that look like A A^T / B^T B for
    typical LoRA factors: well-conditioned, modest spectrum."""
    g = torch.Generator(device=device).manual_seed(seed)
    # Generate as outer products of random matrices (mimics A A^T).
    M = torch.randn(N, r, 4 * r, device=device, generator=g, dtype=torch.float32)
    H = M @ M.transpose(-2, -1) / (4 * r)
    return H


def batched_eigh_inv_half(H, eps):
    """Batched (..., n, n) -> (..., n, n) ≈ (H + eps I)^{-1/2}, eigh-based."""
    H = 0.5 * (H + H.transpose(-2, -1))
    eye = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
    H = H + eps * eye
    evals, Q = torch.linalg.eigh(H)
    inv_half = evals.clamp(min=eps).pow(-0.5)
    # Q diag(λ^{-1/2}) Q^T, batched. inv_half: (..., n) -> (..., n, 1)
    # to broadcast along Q^T's row dim (scaling rows of Q^T).
    return Q @ (inv_half.unsqueeze(-1) * Q.transpose(-2, -1))


def time_call(fn, n_warmup, n_reps, device):
    for _ in range(n_warmup):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times = []
    for _ in range(n_reps):
        if device.type == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
        else:
            t0 = time.perf_counter()
        _ = fn()
        if device.type == "cuda":
            e.record()
            torch.cuda.synchronize(device)
            times.append(s.elapsed_time(e))
        else:
            times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--N", type=int, default=112,
                   help="Number of pairs (matches OLMo-2-1B all-linear).")
    p.add_argument("--ranks", nargs="+", type=int, default=[16, 64])
    p.add_argument("--higham_iters", type=int, default=5)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_reps", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-6)
    args = p.parse_args()
    device = torch.device(args.device)

    print(f"# device={device}, N={args.N}, higham_iters={args.higham_iters}, "
          f"n_reps={args.n_reps}")
    if device.type == "cuda":
        print(f"# gpu={torch.cuda.get_device_name(device)}")

    for r in args.ranks:
        H = make_spd_batch(args.N, r, device)

        ref = torch.stack([spd_frac_power_inv(H[i], gamma=0.5, eps=args.eps)
                           for i in range(args.N)])
        be = batched_eigh_inv_half(H, args.eps)
        lh = torch.stack([spd_inv_sqrt_higham(H[i], n_iters=args.higham_iters,
                                              eps=args.eps) for i in range(args.N)])
        bh = spd_inv_sqrt_higham_batched(H, n_iters=args.higham_iters, eps=args.eps)

        def relerr(M):
            return ((M - ref).norm() / (ref.norm() + 1e-30)).item()
        print(f"\n# r={r}: equivalence to looped eigh (relative Frobenius error)")
        print(f"  batched_eigh : {relerr(be):.2e}")
        print(f"  loop_higham  : {relerr(lh):.2e}")
        print(f"  batch_higham : {relerr(bh):.2e}")

        impls = {
            "loop_eigh": lambda: torch.stack([
                spd_frac_power_inv(H[i], gamma=0.5, eps=args.eps)
                for i in range(args.N)]),
            "batch_eigh": lambda: batched_eigh_inv_half(H, args.eps),
            "loop_higham": lambda: torch.stack([
                spd_inv_sqrt_higham(H[i], n_iters=args.higham_iters, eps=args.eps)
                for i in range(args.N)]),
            "batch_higham": lambda: spd_inv_sqrt_higham_batched(
                H, n_iters=args.higham_iters, eps=args.eps),
        }

        print(f"\n# r={r}: timing (ms)")
        print(f"  {'impl':<14} {'median':>9} {'min':>9} {'max':>9} {'speedup_vs_loop_eigh':>22}")
        loop_eigh_med = None
        results = {}
        for name, fn in impls.items():
            t = time_call(fn, args.n_warmup, args.n_reps, device)
            t_sorted = sorted(t)
            med = t_sorted[len(t) // 2]
            results[name] = med
            if name == "loop_eigh":
                loop_eigh_med = med
            speedup = (loop_eigh_med / med) if loop_eigh_med else float("nan")
            print(f"  {name:<14} {med:>9.3f} {t_sorted[0]:>9.3f} {t_sorted[-1]:>9.3f} "
                  f"{speedup:>21.2f}x")


if __name__ == "__main__":
    main()
