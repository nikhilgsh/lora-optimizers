"""Microbench: polar unwhiten + Frobenius rescale, looped vs batched.

The current `_polar_pipeline` (optim.py) inlines this block per-pair:

    geo_A = SB_half_inv @ P_A        # (r, r) @ (r, d_in)   = (r, d_in)
    geo_B = P_B @ SA_half_inv        # (d_out, r) @ (r, r)  = (d_out, r)
    uA_norm = u_A.norm()
    uB_norm = u_B.norm()
    gA_norm = geo_A.norm() + 1e-30
    gB_norm = geo_B.norm() + 1e-30
    dA = -lr * (uA_norm / gA_norm) * geo_A
    dB = -lr * (uB_norm / gB_norm) * geo_B

(Restricted to polar_norm_dir='frob' which is the default; other modes
add per-row/col normalization. They batch identically with elementwise
ops over the appropriate axis.)

Tests the looped vs single-call-on-stacked-batch pattern, on the three
real shape groups for OLMo-2-1B at r=16. Equivalence within fp32 noise.
"""
import argparse
import time

import torch


def unwhiten_rescale_loop(P_A_list, P_B_list, SA_half_inv_list,
                          SB_half_inv_list, u_A_list, u_B_list, lr=1e-3):
    """Per-pair loop, mirrors current _polar_pipeline behavior at frob mode."""
    dA_list = []
    dB_list = []
    for i in range(len(P_A_list)):
        geo_A = SB_half_inv_list[i] @ P_A_list[i]
        geo_B = P_B_list[i] @ SA_half_inv_list[i]
        uA_norm = u_A_list[i].norm()
        uB_norm = u_B_list[i].norm()
        gA_norm = geo_A.norm() + 1e-30
        gB_norm = geo_B.norm() + 1e-30
        dA_list.append(-lr * (uA_norm / gA_norm) * geo_A)
        dB_list.append(-lr * (uB_norm / gB_norm) * geo_B)
    return dA_list, dB_list


def unwhiten_rescale_batched(P_A, P_B, SA_half_inv, SB_half_inv, u_A, u_B,
                             lr=1e-3):
    """Batched: P_A: (N, r, d_in), P_B: (N, d_out, r),
    SA_half_inv: (N, r, r), SB_half_inv: (N, r, r),
    u_A: (N, r, d_in), u_B: (N, d_out, r). Returns dA, dB stacked.
    """
    geo_A = SB_half_inv @ P_A                      # (N, r, d_in)
    geo_B = P_B @ SA_half_inv                      # (N, d_out, r)
    # Per-pair Frobenius norms: one scalar per matrix in batch.
    uA_norm = u_A.flatten(-2).norm(dim=-1)         # (N,)
    uB_norm = u_B.flatten(-2).norm(dim=-1)         # (N,)
    gA_norm = geo_A.flatten(-2).norm(dim=-1) + 1e-30
    gB_norm = geo_B.flatten(-2).norm(dim=-1) + 1e-30
    sA = (-lr * uA_norm / gA_norm).unsqueeze(-1).unsqueeze(-1)
    sB = (-lr * uB_norm / gB_norm).unsqueeze(-1).unsqueeze(-1)
    return sA * geo_A, sB * geo_B


def equivalence_check(N, r, d_in, d_out, device, atol=1e-5):
    torch.manual_seed(0)
    P_A = torch.randn(N, r, d_in, device=device, dtype=torch.float32)
    P_B = torch.randn(N, d_out, r, device=device, dtype=torch.float32)
    SA = torch.randn(N, r, r, device=device, dtype=torch.float32)
    SB = torch.randn(N, r, r, device=device, dtype=torch.float32)
    SA = 0.5 * (SA + SA.transpose(-2, -1))
    SB = 0.5 * (SB + SB.transpose(-2, -1))
    u_A = torch.randn(N, r, d_in, device=device, dtype=torch.float32)
    u_B = torch.randn(N, d_out, r, device=device, dtype=torch.float32)

    dA_loop, dB_loop = unwhiten_rescale_loop(
        list(P_A), list(P_B), list(SA), list(SB), list(u_A), list(u_B))
    dA_loop_t = torch.stack(dA_loop)
    dB_loop_t = torch.stack(dB_loop)
    dA_b, dB_b = unwhiten_rescale_batched(P_A, P_B, SA, SB, u_A, u_B)
    eA = (dA_loop_t - dA_b).abs().max().item()
    eB = (dB_loop_t - dB_b).abs().max().item()
    print(f"  equivalence (N={N}, r={r}, d_in={d_in}, d_out={d_out}): "
          f"dA_err={eA:.2e}, dB_err={eB:.2e}")
    assert eA < atol and eB < atol, f"unwhiten_rescale loop/batched diverge"


def time_call(fn, n_warmup, n_reps, device):
    for _ in range(n_warmup):
        fn()
    if device.type == "cuda":
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

    # OLMo-2-1B r=16, all-linear → 3 shape groups (N pairs each):
    # the unwhiten_rescale acts on (N, r, d_in) and (N, d_out, r) jointly.
    GROUPS = [
        ("group_64", 64, 16, 2048, 2048),  # both d_in=d_out=2048
        ("group_32", 32, 16, 2048, 8192),  # d_in=2048, d_out=8192
        ("group_16", 16, 16, 8192, 2048),  # d_in=8192, d_out=2048
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
    for name, N, r, d_in, d_out in GROUPS:
        torch.manual_seed(0)
        P_A = torch.randn(N, r, d_in, device=device, dtype=torch.float32)
        P_B = torch.randn(N, d_out, r, device=device, dtype=torch.float32)
        SA = torch.randn(N, r, r, device=device, dtype=torch.float32)
        SB = torch.randn(N, r, r, device=device, dtype=torch.float32)
        SA = 0.5 * (SA + SA.transpose(-2, -1))
        SB = 0.5 * (SB + SB.transpose(-2, -1))
        u_A = torch.randn(N, r, d_in, device=device, dtype=torch.float32)
        u_B = torch.randn(N, d_out, r, device=device, dtype=torch.float32)
        P_A_l = list(P_A); P_B_l = list(P_B)
        SA_l = list(SA);   SB_l = list(SB)
        uA_l = list(u_A);  uB_l = list(u_B)

        loop_t = time_call(
            lambda: unwhiten_rescale_loop(P_A_l, P_B_l, SA_l, SB_l, uA_l, uB_l),
            args.n_warmup, args.n_reps, device)
        batch_t = time_call(
            lambda: unwhiten_rescale_batched(P_A, P_B, SA, SB, u_A, u_B),
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
