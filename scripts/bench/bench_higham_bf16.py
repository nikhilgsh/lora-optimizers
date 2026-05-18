"""Wall-time bench: fp32 vs bf16 spd_inv_sqrt_higham_batched.

Times the function on production-relevant (N, r, r) shapes under the
production `eps_relative=True, eps=1e-2` setting. Reports per-call
wall time and bf16/fp32 speedup.
"""
import sys, time
sys.path.insert(0, "/mnt/home/nghosh/lora")

import torch
from lora_playground.utils import spd_inv_sqrt_higham_batched

device = torch.device("cuda")
torch.cuda.synchronize()


def make_spd(N, r, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(N, r, 2 * r, generator=g)
    return (A @ A.transpose(-2, -1)).to(device).float()


def bench_one(H, compute_dtype, n_iters_call=10, n_repeats=30, warmup=5):
    for _ in range(warmup):
        Z = spd_inv_sqrt_higham_batched(
            H, n_iters=n_iters_call, eps=1e-2, eps_relative=True,
            compute_dtype=compute_dtype,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        Z = spd_inv_sqrt_higham_batched(
            H, n_iters=n_iters_call, eps=1e-2, eps_relative=True,
            compute_dtype=compute_dtype,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeats * 1000  # ms


print(f"{'shape (N,r,r)':>15s} {'fp32 ms':>10s} {'bf16 ms':>10s} {'speedup':>10s}")
print("-" * 50)
for N, r in [(112, 16), (112, 64), (112, 128), (224, 256)]:
    H = make_spd(N, r)
    t_fp32 = bench_one(H, compute_dtype=None)
    t_bf16 = bench_one(H, compute_dtype=torch.bfloat16)
    speedup = t_fp32 / t_bf16
    print(f"({N:4d},{r:4d},{r:4d}) {t_fp32:10.3f} {t_bf16:10.3f} {speedup:10.2f}x")
