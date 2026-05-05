"""Microbench: Picard cross-coupling correction, looped vs batched.

For Picard iter k≥1 in `AdamPolarProductLoRA.step()`, the non-exact-chord
branch computes per-pair:

    u_A_eff = u_A + picard_alpha * (B^T @ dB_prev @ A) / lr
    u_B_eff = u_B + picard_alpha * (B @ dA_prev @ A^T) / lr

shapes:
    A      : (r, d_in)
    B      : (d_out, r)
    dA_prev: (r, d_in)
    dB_prev: (d_out, r)
    u_A    : (r, d_in)   — same shape as A.grad
    u_B    : (d_out, r)  — same shape as B.grad

Each line is a 3-matrix chain. With N pairs of identical shape this
batches into bmm chains.

Tested at the three real shape groups for OLMo-2-1B at r=16 — exactly
the cells that fire on every Picard correction iter when running the
coupled (k=3) optimizer.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def cross_coupling_loop(A_list, B_list, dA_list, dB_list, u_A_list, u_B_list,
                        lr, picard_alpha):
    """Per-pair loop, mirrors current optim.py:2676-2677."""
    out_uA = []
    out_uB = []
    for i in range(len(A_list)):
        A_f = A_list[i]
        B_f = B_list[i]
        u_A_eff = u_A_list[i] + picard_alpha * (B_f.T @ dB_list[i] @ A_f) / lr
        u_B_eff = u_B_list[i] + picard_alpha * (B_f @ dA_list[i] @ A_f.T) / lr
        out_uA.append(u_A_eff)
        out_uB.append(u_B_eff)
    return out_uA, out_uB


def cross_coupling_batched(A, B, dA, dB, u_A, u_B, lr, picard_alpha):
    """Batched. A: (N, r, d_in), B: (N, d_out, r), dA: (N, r, d_in),
    dB: (N, d_out, r), u_A: (N, r, d_in), u_B: (N, d_out, r).
    """
    BT_dB_A = B.transpose(-2, -1) @ dB @ A          # (N, r, d_in)
    B_dA_AT = B @ dA @ A.transpose(-2, -1)          # (N, d_out, r)
    u_A_eff = u_A + (picard_alpha / lr) * BT_dB_A
    u_B_eff = u_B + (picard_alpha / lr) * B_dA_AT
    return u_A_eff, u_B_eff


def equivalence_check(N, r, d_in, d_out, device, atol=1e-4):
    torch.manual_seed(0)
    A = torch.randn(N, r, d_in, device=device, dtype=torch.float32) * 0.01
    B = torch.randn(N, d_out, r, device=device, dtype=torch.float32) * 0.01
    dA = torch.randn(N, r, d_in, device=device, dtype=torch.float32) * 1e-4
    dB = torch.randn(N, d_out, r, device=device, dtype=torch.float32) * 1e-4
    u_A = torch.randn(N, r, d_in, device=device, dtype=torch.float32)
    u_B = torch.randn(N, d_out, r, device=device, dtype=torch.float32)
    lr = 1e-3
    picard_alpha = 1.0

    uA_loop, uB_loop = cross_coupling_loop(
        list(A), list(B), list(dA), list(dB), list(u_A), list(u_B),
        lr, picard_alpha)
    uA_loop_t = torch.stack(uA_loop)
    uB_loop_t = torch.stack(uB_loop)
    uA_b, uB_b = cross_coupling_batched(A, B, dA, dB, u_A, u_B,
                                        lr, picard_alpha)
    eA = (uA_loop_t - uA_b).abs().max().item()
    eB = (uB_loop_t - uB_b).abs().max().item()
    relA = eA / (uA_loop_t.abs().max().item() + 1e-30)
    relB = eB / (uB_loop_t.abs().max().item() + 1e-30)
    print(f"  equivalence (N={N}, r={r}, {d_in}×{d_out}): "
          f"uA rel={relA:.2e}, uB rel={relB:.2e}")
    assert relA < atol and relB < atol


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
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--n_reps", type=int, default=20)
    args = p.parse_args()
    device = torch.device(args.device)

    GROUPS = [
        ("group_64", 64, 16, 2048, 2048),
        ("group_32", 32, 16, 2048, 8192),
        ("group_16", 16, 16, 8192, 2048),
    ]

    print(f"# device={device}, n_reps={args.n_reps}")
    if device.type == "cuda":
        print(f"# gpu={torch.cuda.get_device_name(device)}")

    print("\n# === Equivalence ===")
    for name, N, r, d_in, d_out in GROUPS:
        equivalence_check(N, r, d_in, d_out, device)

    print("\n# === Timing (ms) ===")
    print(f"{'group':<10} {'N':>3} {'r':>3} {'d_in':>5} {'d_out':>6} "
          f"{'loop_ms':>10} {'batch_ms':>10} {'speedup':>8}")
    print("-" * 78)
    total_loop = 0.0
    total_batch = 0.0
    lr = 1e-3
    picard_alpha = 1.0
    for name, N, r, d_in, d_out in GROUPS:
        torch.manual_seed(0)
        A = torch.randn(N, r, d_in, device=device, dtype=torch.float32) * 0.01
        B = torch.randn(N, d_out, r, device=device, dtype=torch.float32) * 0.01
        dA = torch.randn(N, r, d_in, device=device, dtype=torch.float32) * 1e-4
        dB = torch.randn(N, d_out, r, device=device, dtype=torch.float32) * 1e-4
        u_A = torch.randn(N, r, d_in, device=device, dtype=torch.float32)
        u_B = torch.randn(N, d_out, r, device=device, dtype=torch.float32)
        A_l = list(A); B_l = list(B)
        dA_l = list(dA); dB_l = list(dB)
        uA_l = list(u_A); uB_l = list(u_B)

        loop_t = time_call(
            lambda: cross_coupling_loop(A_l, B_l, dA_l, dB_l, uA_l, uB_l,
                                         lr, picard_alpha),
            args.n_warmup, args.n_reps, device)
        batch_t = time_call(
            lambda: cross_coupling_batched(A, B, dA, dB, u_A, u_B,
                                           lr, picard_alpha),
            args.n_warmup, args.n_reps, device)
        loop_med = sorted(loop_t)[len(loop_t) // 2]
        batch_med = sorted(batch_t)[len(batch_t) // 2]
        sp = loop_med / batch_med
        total_loop += loop_med
        total_batch += batch_med
        print(f"{name:<10} {N:>3} {r:>3} {d_in:>5} {d_out:>6} "
              f"{loop_med:>10.3f} {batch_med:>10.3f} {sp:>7.2f}x")
    print("-" * 78)
    print(f"{'TOTAL':<10} {' ':>3} {' ':>3} {' ':>5} {' ':>6} "
          f"{total_loop:>10.3f} {total_batch:>10.3f} "
          f"{total_loop/total_batch:>7.2f}x")


if __name__ == "__main__":
    main()
