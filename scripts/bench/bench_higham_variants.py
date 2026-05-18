"""Wall-time bench: three variants of the Higham inner loop, under
production damping (eps_abs=1e-6).

A. fp32 + TF32 enabled — original baddbmm polynomial, run on tensor
   cores via TF32. No precision lever in the algorithm; just allow
   the TC path that PyTorch defaults to OFF.

B. fp16 state for n_iters - 1 iters, then 1 fp32 polish iter at the
   end. Single cast pair (entry + exit) — total of ~3 cast launches,
   not per-iter.

C. bf16 state, same structure as B.

All three use the same `T = baddbmm(three_eye, Z, Y, beta=1, alpha=-1)`
polynomial — no algorithm change. Just different dtypes.

Inputs at varying log_cond. Damping (eps_abs=1e-6) and λ_max
computation are done ONCE outside the timed region, matching how
the production precond_refresh hot path works.
"""
import sys, time
sys.path.insert(0, "/mnt/home/nghosh/lora")

import torch

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")


def make_spd(N, r, log_cond=2.0, seed=0):
    """SPD batch with controlled cond. log_cond=2 → cond≈100 (typical
    production-Gram median); log_cond=3 → cond≈1000 (real worst-case
    from chord-tight r=64 snapshot audit); log_cond=6 → adversarial
    stress under eps=1e-6 absolute damping."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sing = torch.logspace(0, -log_cond / 2, r).pow(2)  # eigenvalues
    H_list = []
    for _ in range(N):
        M = torch.randn(r, r, generator=g)
        U, _ = torch.linalg.qr(M)
        H_list.append(U @ torch.diag(sing) @ U.transpose(-2, -1))
    return torch.stack(H_list).to(device).float()


def setup_iter_state(H, eps_abs=1e-6):
    """Damping (production: eps_abs=1e-6, absolute) + λ_max scaling.
    Done ONCE per refresh in production; outside the inner-loop timed
    region here. Returns (Y, Z, three_eye, s) for the iteration."""
    N, r, _ = H.shape
    eye_b = torch.eye(r, device=H.device, dtype=H.dtype).expand_as(H).contiguous()
    H = H + eps_abs * eye_b
    s = torch.linalg.eigvalsh(H)[..., -1].unsqueeze(-1).unsqueeze(-1)
    Y = H / s
    Z = eye_b.clone()
    three_eye = 3.0 * eye_b
    return Y, Z, three_eye, s


# ─── Variants: take pre-scaled (Y, Z, three_eye), return Z_final ────────────


def inner_A_fp32_tf32(Y0, Z0, three_eye, n_iters=10):
    """fp32 + TF32 enabled. Same baddbmm polynomial."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    Y, Z = Y0.clone(), Z0.clone()
    for _ in range(n_iters):
        T = torch.baddbmm(three_eye, Z, Y, beta=1.0, alpha=-1.0)
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)
    return Z


def inner_lowp_polish(Y0, Z0, three_eye, n_iters=10, lp_dtype=torch.bfloat16):
    """n_iters - 1 iters in lp_dtype, then 1 fp32 polish iter."""
    Y_lp = Y0.to(lp_dtype)
    Z_lp = Z0.to(lp_dtype)
    three_eye_lp = three_eye.to(lp_dtype)
    for _ in range(n_iters - 1):
        T_lp = torch.baddbmm(three_eye_lp, Z_lp, Y_lp, beta=1.0, alpha=-1.0)
        Y_lp = 0.5 * (Y_lp @ T_lp)
        Z_lp = 0.5 * (T_lp @ Z_lp)
    Y = Y_lp.float()
    Z = Z_lp.float()
    T = torch.baddbmm(three_eye, Z, Y, beta=1.0, alpha=-1.0)
    Y = 0.5 * (Y @ T)
    Z = 0.5 * (T @ Z)
    return Z


def inner_reference_fp32(Y0, Z0, three_eye, n_iters=10):
    """fp32, NO TF32 — current production path."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    Y, Z = Y0.clone(), Z0.clone()
    for _ in range(n_iters):
        T = torch.baddbmm(three_eye, Z, Y, beta=1.0, alpha=-1.0)
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)
    return Z


def bench(fn, n_repeats=50, warmup=10):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_repeats * 1000, out


def rel_err(Z, Z_ref):
    return float((Z.float() - Z_ref.float()).norm() / (Z_ref.float().norm() + 1e-30))


SHAPES = [(112, 16), (112, 64), (112, 128), (224, 256)]
COND_LEVELS = [2.0, 3.0, 6.0]

for log_cond in COND_LEVELS:
    print(f"\n=== cond ≈ 1e{int(log_cond)} (eps_abs=1e-6 damping) ===")
    print(f"{'shape':>14s} | {'ref fp32':>10s} | {'A: fp32+TF32':>20s} | "
          f"{'B: fp16+polish':>22s} | {'C: bf16+polish':>22s}")
    print(f"{'':>14s} | {'ms':>10s} | {'ms (×ref) / err':>20s} | "
          f"{'ms (×ref) / err':>22s} | {'ms (×ref) / err':>22s}")
    print("-" * 110)
    for N, r in SHAPES:
        H = make_spd(N, r, log_cond=log_cond)
        Y0, Z0, three_eye, s = setup_iter_state(H)

        Z_ref = inner_reference_fp32(Y0, Z0, three_eye)

        t_ref, _ = bench(lambda: inner_reference_fp32(Y0, Z0, three_eye))
        t_A, Z_A = bench(lambda: inner_A_fp32_tf32(Y0, Z0, three_eye))
        t_B, Z_B = bench(lambda: inner_lowp_polish(Y0, Z0, three_eye, lp_dtype=torch.float16))
        t_C, Z_C = bench(lambda: inner_lowp_polish(Y0, Z0, three_eye, lp_dtype=torch.bfloat16))

        eA = rel_err(Z_A, Z_ref)
        eB = rel_err(Z_B, Z_ref)
        eC = rel_err(Z_C, Z_ref)
        print(f"({N:3d},{r:3d},{r:3d}) | {t_ref:10.3f} | "
              f"{t_A:6.3f} ({t_ref/t_A:5.2f}×) / {eA:.0e} | "
              f"{t_B:7.3f} ({t_ref/t_B:5.2f}×) / {eB:.0e} | "
              f"{t_C:7.3f} ({t_ref/t_C:5.2f}×) / {eC:.0e}")
