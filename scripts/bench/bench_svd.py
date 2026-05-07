"""
Benchmark truncated SVD: full economy SVD vs torch.svd_lowrank on OLMo-2-0425-1B weight shapes.

Shapes: 112 target modules, rank=16.
  - 48 MLP matrices (gate/up/down_proj): 8192×2048 or 2048×8192
  - 64 attention matrices (q/o_proj: 2048×2048, k/v_proj: 512×2048)

Run on the target hardware (H100 PCIe) for valid timing.

Usage:
    python scripts/bench/bench_svd.py --device cuda
"""
import argparse
import time
import torch

RANK = 16
N_WARMUP = 2
N_REPS = 5

# (rows, cols, count) for each distinct shape in OLMo-2-0425-1B (16 layers)
SHAPES = [
    (2048, 2048, 2 * 16),   # q_proj, o_proj (16 layers × 2)
    (512,  2048, 2 * 16),   # k_proj, v_proj (GQA, 16 layers × 2)
    (8192, 2048, 2 * 16),   # gate_proj, up_proj (16 layers × 2)
    (2048, 8192, 1 * 16),   # down_proj (16 layers × 1)
]


def time_svd(matrices, fn, n_warmup, n_reps, device):
    torch.cuda.synchronize(device)
    for _ in range(n_warmup):
        for m in matrices:
            fn(m)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        for m in matrices:
            fn(m)
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / n_reps


def frob_error(approx, exact):
    return (approx - exact).norm().item() / exact.norm().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=RANK)
    args = parser.parse_args()

    device = args.device
    rank = args.rank
    print(f"device={device}  rank={rank}  n_reps={N_REPS}")
    print(f"GPU: {torch.cuda.get_device_name(device)}\n")

    matrices = []
    for rows, cols, count in SHAPES:
        for _ in range(count):
            matrices.append(torch.randn(rows, cols, dtype=torch.float32, device=device))
    total = sum(c for _, _, c in SHAPES)
    print(f"Total matrices: {total}")
    for rows, cols, count in SHAPES:
        print(f"  {rows}×{cols}  ×{count}")
    print()

    def exact_fn(m):
        U, S, Vh = torch.linalg.svd(m, full_matrices=False)
        return (U[:, :rank] * S[:rank]) @ Vh[:rank]

    def lowrank_fn(niter):
        def _fn(m):
            U, S, V = torch.svd_lowrank(m, q=rank, niter=niter)
            return (U * S) @ V.T
        return _fn

    # Skip exact SVD timing — it takes many hours for 112 large matrices.
    # We know from the H100 PCIe smoke test: 31.7 sec/step with exact SVD.
    # SVD is essentially all of that overhead (LoRA forward-backward ≈ 0.1 sec).
    EXACT_SVD_SEC_PER_PASS = 31.7  # from smoke test measurement

    print("=== svd_lowrank timing (seconds per full pass over all 112 matrices) ===")
    for niter in [0, 2, 4, 6]:
        t = time_svd(matrices, lowrank_fn(niter), N_WARMUP, N_REPS, device)
        speedup = EXACT_SVD_SEC_PER_PASS / t
        print(f"  svd_lowrank niter={niter}:  {t:.3f}s  ({speedup:.1f}× vs exact-SVD baseline)")
    print()

    print("=== Frobenius relative error vs exact (one pass, float32 random matrices) ===")
    for niter in [0, 2, 4, 6]:
        errors = []
        for m in matrices:
            approx = lowrank_fn(niter)(m)
            exact = exact_fn(m)
            errors.append(frob_error(approx, exact))
        mean_err = sum(errors) / len(errors)
        max_err = max(errors)
        print(f"  niter={niter}:  mean_rel_err={mean_err:.4f}  max_rel_err={max_err:.4f}")
    print()

    print("=== Projected per-step training time ===")
    print(f"  exact SVD baseline (smoke test, H100 PCIe): {EXACT_SVD_SEC_PER_PASS:.1f} sec/step")
    FWDBWD_SEC = 0.3  # forward+backward+Adam without SVD, estimated
    for niter in [2, 4]:
        t_lr = time_svd(matrices, lowrank_fn(niter), N_WARMUP, N_REPS, device)
        projected_step = FWDBWD_SEC + t_lr
        steps_in_2h = int(7200 / projected_step)
        print(f"  niter={niter}:  SVD={t_lr:.2f}s  proj_step≈{projected_step:.1f}s  "
              f"steps_in_2h≈{steps_in_2h}")


if __name__ == "__main__":
    main()
