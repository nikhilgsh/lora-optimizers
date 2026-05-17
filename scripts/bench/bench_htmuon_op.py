"""Phase 0 measurement gate for σ → σ^p (HTMuon) in the clean polar pipeline.

Two questions:
  (1) Accuracy: do candidate implementations of `(X X^T)^(p/2)` match
      SVD ground truth at production LoRA shapes?
  (2) Timing: how does the added op compare to the existing NS5 polar cost?

Candidates tested:
  - (A) batched eigh on (N, r, r) Gram
  - (B) iterated coupled-NS sqrt (project-standard, λ_max scaled) — preferred
  - (C) HTMuon Alg 5 ported with λ_max scaling (paper-faithful)

Decision gate (per plan):
  - accuracy < 1e-4 rel-err AND timing overhead < 30% of NS5 polar wall → proceed
  - 30–50% overhead → report to user before sweep
  - > 50% overhead OR ≥ 1e-4 rel-err → stop

Run on workergpu035 (Blackwell, SLURM job 6417423) OR locally (RTX A6000) —
absolute numbers won't transfer between A6000 and Blackwell but op-ratios
should. Hardware reported in the output table.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch

# Make `from lora_playground...` work when run from scripts/bench/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lora_playground.optim import _newton_schulz_batched
from lora_playground.utils import spd_inv_sqrt_higham_batched
from lora_playground.spectral import lambda_max_power_iter_psd_batched


# ─── Candidate implementations of (H)^p for SPD H, p ∈ (0, 1) ────────────────


def spd_power_eigh(H: torch.Tensor, p: float, eps: float = 1e-12) -> torch.Tensor:
    """Candidate (A): batched eigh-based reference.

    H: (..., n, n) SPD. Returns H^p of the same shape.
    """
    H = 0.5 * (H + H.transpose(-2, -1))
    L, V = torch.linalg.eigh(H)
    L = L.clamp_min(eps)
    Lp = L.pow(p)
    return (V * Lp.unsqueeze(-2)) @ V.transpose(-2, -1)


def _higham_sqrt_lambda_max(
    H: torch.Tensor, n_iters: int = 10, n_power_iter: int = 8, eps: float = 1e-6
) -> torch.Tensor:
    """One coupled-NS sqrt round on SPD H with λ_max scaling.

    Returns Y ≈ H^(1/2). Mirrors `spd_inv_sqrt_higham_batched` but keeps
    Y instead of returning Z. Adds δ·I damping for numerical safety.
    """
    n = H.shape[-1]
    H = 0.5 * (H + H.transpose(-2, -1))
    eye = torch.eye(n, dtype=H.dtype, device=H.device)
    H = H + eps * eye
    lam_max, _ = lambda_max_power_iter_psd_batched(H, n_iters=max(n_power_iter, 8))
    s = lam_max.unsqueeze(-1).unsqueeze(-1)
    Y = H / s
    Z = eye.expand_as(H).clone()
    three_eye = 3.0 * eye.expand_as(H)
    for _ in range(n_iters):
        T = three_eye - Z @ Y
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)
    # Y_∞ = (H/s)^(1/2) → H^(1/2) = Y · sqrt(s)
    return Y * s.sqrt()


def spd_power_iter_higham(H: torch.Tensor, p: float, n_iters: int = 10) -> torch.Tensor:
    """Candidate (B): iterated coupled-NS sqrt, project-standard scaling.

    For target p ∈ (0, 1), compute n = round(-log2(p)) successive sqrt rounds
    so H^(1/2^n) ≈ H^p. p must be ≈ a power-of-two reciprocal for exact match;
    other p values are snapped to nearest 2^-n.
    """
    n = max(1, round(-math.log2(p)))
    out = H
    for _ in range(n):
        out = _higham_sqrt_lambda_max(out, n_iters=n_iters)
    return out


def spd_power_htmuon_alg5(
    H: torch.Tensor, p: float, n_inner: int = 5, eps: float = 1e-6
) -> torch.Tensor:
    """Candidate (C): HTMuon Algorithm 5 (Pang et al. 2026) with λ_max scaling.

    Same coupled-NS sqrt outer loop as Algorithm 5 in the paper, but with the
    Frobenius normalization swapped for λ_max (project standard — Frobenius
    leaves eigenvalues far from 1 for spiky spectra and T=5 doesn't converge).

    Convention here: compute H^p (NOT the paper's H^(p/2) for input M^T M).
    Outer rounds n = round(-log2(p)) → final H^(1/2^n) = H^p (for p a power
    of 2 reciprocal; nearest power-of-two otherwise). Paper's
    n = ⌈log2(2/p)⌉ adds one extra round which yields H^(p/2) — correct for
    their NS_root convention, wrong here.

    n_inner = T (paper uses T=5). Faster than iter-Higham at n_iters=10
    because inner-loop count is half.
    """
    n_outer = max(1, round(-math.log2(p)))
    n = H.shape[-1]
    eye = torch.eye(n, dtype=H.dtype, device=H.device)
    X = 0.5 * (H + H.transpose(-2, -1))
    for _ in range(n_outer):
        # λ_max-normalize (not Frobenius as in paper).
        X = X + eps * eye
        lam_max, _ = lambda_max_power_iter_psd_batched(X, n_iters=8)
        s = lam_max.unsqueeze(-1).unsqueeze(-1)
        Xs = X / s
        Y = Xs.clone()
        Z = eye.expand_as(X).clone()
        three_eye = 3.0 * eye.expand_as(X)
        for _ in range(n_inner):
            T = three_eye - Z @ Y
            Y = 0.5 * (Y @ T)
            Z = 0.5 * (T @ Z)
        # Y is (X/s)^(1/2); unscale to X^(1/2)
        X = Y * s.sqrt()
        X = 0.5 * (X + X.transpose(-2, -1)) + eps * eye
    return X


# ─── Reference: SVD ground truth for U Σ^p V^T applied to rectangular X ──────


def htmuon_op_svd(X: torch.Tensor, p: float) -> torch.Tensor:
    """Ground-truth U Σ^p V^T for batched X."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return U @ torch.diag_embed(S.pow(p)) @ Vh


def htmuon_via_left_gram(
    X: torch.Tensor, p: float, spd_power_fn
) -> torch.Tensor:
    """(X X^T)^(p/2) · polar(X) via the given candidate."""
    G = X @ X.transpose(-2, -1)
    G_phalf = spd_power_fn(G, p / 2.0)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    polar = U @ Vh
    return G_phalf @ polar


# ─── Bench harness ───────────────────────────────────────────────────────────


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:7.2f} ms"


def time_op(fn, *args, n_warmup: int = 5, n_iters: int = 20, **kwargs) -> float:
    """Wall ms/call for a CUDA op, with sync."""
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-30))


def run_accuracy_synthetic(device: str) -> None:
    print(f"\n=== ACCURACY: synthetic batched SPD (rel-err vs SVD reference) ===")
    print(f"{'N':>3} {'r':>4} {'p':>7}  {'eigh':>10}  {'iter-Higham':>12}  {'Alg5-λmax':>10}")
    torch.manual_seed(0)
    for N, r in [(4, 16), (4, 64), (4, 256), (16, 256)]:
        for p in (0.5, 0.25, 0.125, 0.0625):
            X = torch.randn(N, r, 4096, device=device, dtype=torch.float32)
            H = X @ X.transpose(-2, -1) + 1e-4 * torch.eye(r, device=device, dtype=torch.float32)
            ref = spd_power_eigh(H, p / 2.0)
            e_eigh = 0.0  # reference; self-comparison is trivially 0
            e_iter = rel_err(spd_power_iter_higham(H, p / 2.0), ref)
            e_alg5 = rel_err(spd_power_htmuon_alg5(H, p / 2.0), ref)
            print(f"{N:>3d} {r:>4d} {p:>7.4f}  {e_eigh:>10.2e}  {e_iter:>12.2e}  {e_alg5:>10.2e}")


def run_accuracy_snapshot(device: str) -> None:
    print(f"\n=== ACCURACY: real snapshot u_A matrices (rel-err vs SVD reference) ===")
    SNAP_BASE = Path("/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell/task_0")
    if not SNAP_BASE.exists():
        print(f"snapshot dir not found: {SNAP_BASE} — skipping")
        return
    # Sweep T (inner iters) to find minimum that passes 1e-4.
    T_VALUES = (4, 5, 6, 7, 8, 10)
    print(f"{'step':>5} {'p':>7}  " + "  ".join(f'T={T:<2d}' for T in T_VALUES))
    for step_name in ("step_200", "step_1000", "step_4000"):
        snap = SNAP_BASE / step_name / "optimizer.pt"
        if not snap.exists():
            continue
        state = torch.load(snap, map_location="cpu", weights_only=False)
        u_A = state["pair_state"][0]["u_A"].float().to(device)
        if u_A.ndim == 2:
            u_A = u_A.unsqueeze(0)
        for p in (0.25, 0.125, 0.0625):
            X = u_A
            ref_op = htmuon_op_svd(X, p)
            errs = []
            for T in T_VALUES:
                cand = htmuon_via_left_gram(
                    X, p, lambda H, q: spd_power_htmuon_alg5(H, q, n_inner=T)
                )
                errs.append(rel_err(cand, ref_op))
            print(f"{int(step_name.split('_')[1]):>5d} {p:>7.4f}  "
                  + "  ".join(f'{e:.1e}' for e in errs))


def run_timing(device: str) -> None:
    print(f"\n=== TIMING: per-op wall (ms/call) at N=64, r=256, d=4096 ===")
    N, r, d = 64, 256, 4096
    torch.manual_seed(0)
    X = torch.randn(N, r, d, device=device, dtype=torch.float32)
    G = X @ X.transpose(-2, -1) + 1e-4 * torch.eye(r, device=device, dtype=torch.float32)

    t_polar = time_op(
        lambda Y: _newton_schulz_batched(Y, nsteps=5, dtype=torch.bfloat16).float(),
        X,
    )
    print(f"  NS5 polar (baseline)              : {fmt_ms(t_polar)}")

    t_higham_inv = time_op(spd_inv_sqrt_higham_batched, G, n_iters=10)
    print(f"  one Higham inv-sqrt call          : {fmt_ms(t_higham_inv)}")

    # Time at p=0.125 across T values (after accuracy sweep settled which T).
    print(f"  Alg5-λmax T-sweep at p=0.125:")
    for T in (4, 5, 6, 7, 8, 10):
        t = time_op(spd_power_htmuon_alg5, G, 0.125 / 2.0, n_inner=T)
        print(f"    T={T:>2d}  {fmt_ms(t)}")

    # Full htmuon insertion = polar + Gram + power + matmul.
    def full_htmuon(X_in, p):
        polar = _newton_schulz_batched(X_in, nsteps=5, dtype=torch.bfloat16).float()
        G_in = X_in @ X_in.transpose(-2, -1) + 1e-4 * torch.eye(
            X_in.shape[-2], device=X_in.device, dtype=X_in.dtype
        )
        Gp = spd_power_iter_higham(G_in, p / 2.0, n_iters=10)
        return Gp @ polar

    t_full = time_op(full_htmuon, X, 0.125)
    overhead = (t_full - t_polar) / t_polar
    print(f"\n  full htmuon insertion (polar + Gram + power + matmul) at p=0.125:")
    print(f"    total wall    : {fmt_ms(t_full)}")
    print(f"    baseline NS5  : {fmt_ms(t_polar)}")
    print(f"    overhead      : {overhead*100:+.1f}% of baseline polar")
    if overhead < 0.30:
        print(f"    → PASS (< 30% overhead). Proceed to implementation.")
    elif overhead < 0.50:
        print(f"    → 30–50%, REPORT TO USER before proceeding.")
    else:
        print(f"    → > 50%, STOP and revisit implementation.")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}")
        print(f"hostname: {os.uname().nodename}")
        slurm_id = os.environ.get("SLURM_JOB_ID", "(not in SLURM)")
        print(f"SLURM_JOB_ID: {slurm_id}")

    run_accuracy_synthetic(device)
    run_accuracy_snapshot(device)
    if device == "cuda":
        run_timing(device)
    else:
        print("\n(timing skipped — CPU only)")


if __name__ == "__main__":
    main()
