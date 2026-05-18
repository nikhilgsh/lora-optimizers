"""Isolate the underlying matmul speed: naked bmm fp32 vs bf16."""
import sys, time
sys.path.insert(0, "/mnt/home/nghosh/lora")
import torch

device = torch.device("cuda")
print(f"PyTorch allow_tf32 (matmul): {torch.backends.cuda.matmul.allow_tf32}")
print(f"PyTorch float32_matmul_precision: {torch.get_float32_matmul_precision()}")
print(f"GPU: {torch.cuda.get_device_name(0)}\n")


def bench_bmm(N, r, dtype, n_repeats=200, warmup=20, tf32_setting=None):
    prev = torch.backends.cuda.matmul.allow_tf32
    if tf32_setting is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32_setting
    A = torch.randn(N, r, r, device=device, dtype=dtype)
    B = torch.randn(N, r, r, device=device, dtype=dtype)
    for _ in range(warmup):
        C = A @ B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        C = A @ B
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_repeats * 1000
    torch.backends.cuda.matmul.allow_tf32 = prev
    return elapsed


print(f"{'shape (N,r,r)':>15s} {'fp32-noTF32':>12s} {'fp32-TF32':>12s} {'bf16':>12s} "
      f"{'noTF32→bf16':>14s} {'TF32→bf16':>12s}")
print("-" * 80)
for N, r in [(112, 16), (112, 64), (112, 128), (224, 256)]:
    t_fp32_no = bench_bmm(N, r, torch.float32, tf32_setting=False)
    t_fp32_yes = bench_bmm(N, r, torch.float32, tf32_setting=True)
    t_bf16 = bench_bmm(N, r, torch.bfloat16)
    print(f"({N:4d},{r:4d},{r:4d}) {t_fp32_no:12.3f} {t_fp32_yes:12.3f} {t_bf16:12.3f} "
          f"{t_fp32_no/t_bf16:14.2f}x {t_fp32_yes/t_bf16:12.2f}x")
