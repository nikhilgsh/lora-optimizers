"""Candidate inverse-sqrt methods for the protagonist's small-side Shampoo
`S_a^{-1/2}` — implement, stress-test, and pick the production method.

The protagonist (`CurvatureWhitenLoRA`) inverts the r×r curvature Gram `P_A`/`Q_B`
once per refresh, batched over all LoRA pairs. The relative-δ damping (`_rdinv`,
δ=1e-4) caps the *effective* condition number it ever sees at ~1/δ = 1e4, so the
discriminating regime is cond ∈ [1e1, 1e4] in fp32 AND bf16.

Three candidates, all batched over S: (N, r, r) SPD, all with the SAME relative-δ
damping so the comparison is apples-to-apples:

  eigh        — torch.linalg.eigh sandwich. Production default + accuracy truth
                (computed in fp64 for the reference). Sequential per-matrix on GPU.
  coupled_ns  — Iannazzo / Denman-Beavers coupled Newton-Schulz
                (`spd_inv_sqrt_higham_batched`), the Phase-1-wired baseline.
  gram_ns_pe  — Gram Newton-Schulz with the protagonist's OWN Polar-Express
                coefficients (Zhang/Amsel/Chen/Dao 2026 Alg 2; coefficient reuse
                with the polar map). Restart-stabilized, relative-δ. THE candidate.

Run:  python -m lora_playground.bench.inverse_sqrt_candidates [--device cuda]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from ..optim import _POLAR_EXPRESS_COEFFS
from ..spectral import lambda_max_power_iter_psd_batched
from ..utils import spd_inv_sqrt_higham_batched


# ---------------------------------------------------------------------------
# Candidate: Gram Newton-Schulz inverse-sqrt with Polar-Express coefficients.
# ---------------------------------------------------------------------------
def gram_ns_inv_sqrt(S, nsteps=6, eps=1e-4, eps_relative=True, restart_at=2,
                     compute_dtype=None, lam_max=None, safety_factor=1.05):
    """Batched (S + δ_eff·I)^{-1/2} for SPD S: (..., r, r) via Gram NS.

    Gram NS (Zhang/Amsel/Chen/Dao 2026, Alg 2): with R_0 = S_normed, Q_0 = I, and
    the degree-5 Polar-Express polynomial M_t = a_t I + b_t R + c_t R²,
        Q <- M_t Q,   R <- M_t R M_t,
    drives R -> I and Q -> S_normed^{-1/2}. Coefficients are `_POLAR_EXPRESS_COEFFS`
    — the SAME family used for the polar map (one coefficient surface for both).

    Normalization MUST match the production polar path (`_polar_express_gram_batched`,
    pre_norm="frob"): the coefficients (first = 8.237x-23.158x³+16.681x⁵) are
    calibrated for an input whose spectrum is COMPRESSED well below 1 — they then
    lift it toward 1. Frobenius-norm-of-the-factor == trace-of-the-Gram, so we
    normalize S by tr(S)·safety² (NOT by tight λ_max, which pins λ_max=1 with no
    headroom and makes the aggressive first coeff overshoot -> R<-MRM squares it ->
    blowup). With trace-norm, λ_max(R0) ≤ 1/safety² and the iteration is contractive.

    Stability (the load-bearing part, see CLAUDE.md spectral guardrail):
      - relative-δ floor (eps_relative): damp by δ·λ_max so λ_min is lifted off the
        bf16 noise floor and cond is capped at ~1/δ before normalization;
      - trace normalization (production-polar headroom) so eigs sit << 1;
      - restart at iter τ (reform R <- Q R0 Qᵀ) to cap growth of spurious eigenvalues;
      - default nsteps=6 (do NOT over-iterate in bf16; coeffs plateau by ~iter 6).

    compute_dtype: run the iteration body in this dtype (e.g. torch.bfloat16) with a
    final fp32 polish iter; None -> entire iteration in S.dtype.
    """
    S = 0.5 * (S + S.transpose(-2, -1))
    n = S.shape[-1]
    dev, dt = S.device, S.dtype
    eye = torch.eye(n, dtype=dt, device=dev).expand_as(S)

    if lam_max is None:
        lam_max, _ = lambda_max_power_iter_psd_batched(S, n_iters=8)
    lam = lam_max.reshape(*lam_max.shape[: S.dim() - 2], 1, 1)
    eps_eff = (eps * lam).clamp_min(1e-12) if eps_relative else torch.as_tensor(
        float(eps), device=dev, dtype=dt)
    S_d = S + eps_eff * eye
    # Trace normalization == production polar's Frobenius-norm-of-factor pre-norm.
    tr = S_d.diagonal(dim1=-2, dim2=-1).sum(-1).reshape(*lam.shape).clamp_min(1e-30)
    scale = tr * (safety_factor ** 2)          # S_d = scale · R0  =>  S_d^{-1/2} = Q/√scale
    R0 = S_d / scale                           # λ_max(R0) ≤ 1/safety² (headroom)

    coeffs = _POLAR_EXPRESS_COEFFS[:nsteps]
    if len(coeffs) < nsteps:
        coeffs = coeffs + [_POLAR_EXPRESS_COEFFS[-1]] * (nsteps - len(coeffs))

    iter_dt = compute_dtype or dt
    R = R0.to(iter_dt)
    R0_it = R0.to(iter_dt)
    Q = torch.eye(n, dtype=iter_dt, device=dev).expand_as(S).clone()
    eye_it = Q.clone()

    use_tf32_guard = (iter_dt == torch.float32) and S.is_cuda
    prev_tf32 = None
    if use_tf32_guard:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for t, (a, b, c) in enumerate(coeffs, start=1):
            if restart_at is not None and t == restart_at + 1:
                R = Q @ R0_it @ Q.transpose(-2, -1)   # re-anchor R to R0 via current Q
                R = 0.5 * (R + R.transpose(-2, -1))
            R2 = R @ R
            M = (b * R) + (c * R2)
            M.diagonal(dim1=-2, dim2=-1).add_(a)
            Q = M @ Q
            R = M @ R @ M
            R = 0.5 * (R + R.transpose(-2, -1))
    finally:
        if use_tf32_guard:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    if compute_dtype is not None and compute_dtype != dt:
        # one fp32 polish step on the near-converged Q (Newton-Schulz on R).
        Q = Q.float(); R = (Q @ R0 @ Q.transpose(-2, -1))
        R = 0.5 * (R + R.transpose(-2, -1))
        a, b, c = coeffs[-1]
        R2 = R @ R
        M = (b * R) + (c * R2)
        M.diagonal(dim1=-2, dim2=-1).add_(a)
        Q = M @ Q
    Q = Q.float()
    return Q / scale.sqrt()


# ---------------------------------------------------------------------------
# Reference + the wired baseline, both batched with matched relative-δ.
# ---------------------------------------------------------------------------
def eigh_inv_sqrt(S, eps=1e-4, eps_relative=True, out_dtype=torch.float32):
    """Batched (S + δ_eff·I)^{-1/2} via eigh. The production default; also the
    accuracy truth when called in fp64."""
    Sd = 0.5 * (S + S.transpose(-2, -1))
    evals, Q = torch.linalg.eigh(Sd)
    lam = evals.amax(dim=-1, keepdim=True).clamp_min(0)
    eps_eff = (eps * lam).clamp_min(1e-12) if eps_relative else eps
    inv = (evals + eps_eff).clamp_min(1e-30).pow(-0.5)
    out = (Q * inv.unsqueeze(-2)) @ Q.transpose(-2, -1)
    return out.to(out_dtype)


def coupled_ns_inv_sqrt(S, n_iters=6, eps=1e-4, eps_relative=True, compute_dtype=None):
    return spd_inv_sqrt_higham_batched(
        S, n_iters=n_iters, eps=eps, eps_relative=eps_relative,
        compute_dtype=compute_dtype)


# ---------------------------------------------------------------------------
# Test matrices.
# ---------------------------------------------------------------------------
def make_synthetic_spd(r, cond, n, device, *, decay="powerlaw", seed=0):
    """N SPD r×r matrices with a controlled condition number and a realistic
    decaying (power-law) spectrum, random orthogonal eigenbasis. fp64."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if decay == "powerlaw":
        x = torch.linspace(0, 1, r, dtype=torch.float64)
        base = cond ** (-x)                       # 1 .. 1/cond, smooth power law
    else:                                          # two-cluster (stiff) spectrum
        base = torch.ones(r, dtype=torch.float64)
        base[r // 2:] = 1.0 / cond
    eigs = base.unsqueeze(0).repeat(n, 1)
    out = torch.empty(n, r, r, dtype=torch.float64)
    for i in range(n):
        A = torch.randn(r, r, generator=g, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        out[i] = (Q * eigs[i].unsqueeze(0)) @ Q.transpose(-2, -1)
    return out.to(device)


def load_snapshot_grams_batched(snapshot_key, step, device, max_pairs=None):
    """Stack S_A = A Aᵀ and S_B = Bᵀ B over all pairs at one snapshot step ->
    (N, r, r) fp64."""
    from ..snapshot_analysis.snapshots import SNAP_ROOTS, load_snapshot
    root = SNAP_ROOTS[snapshot_key]
    snap = load_snapshot(step, root=root)
    mats = []
    for pi, p in snap["pair_state"].items():
        if max_pairs is not None and pi >= max_pairs:
            continue
        if "A" not in p or "B" not in p:
            continue
        A = p["A"].double(); B = p["B"].double()
        mats.append(A @ A.transpose(-2, -1))
        mats.append(B.transpose(-2, -1) @ B)
    return torch.stack(mats).to(device)


# ---------------------------------------------------------------------------
# Metrics + harness.
# ---------------------------------------------------------------------------
@dataclass
class MethodResult:
    name: str
    dtype: str
    rel_err_p50: float
    rel_err_p99: float
    residual_p99: float
    min_eig_Z: float          # smallest eigenvalue of the output across the batch
    n_nonfinite: int
    cond_eff_p99: float       # effective cond AFTER relative-δ damping
    ms_per_call: float = float("nan")


def _cond_eff(S, eps=1e-4):
    ev = torch.linalg.eigvalsh(0.5 * (S + S.transpose(-2, -1))).clamp_min(0)
    lam = ev.amax(dim=-1, keepdim=True)
    evd = ev + eps * lam
    c = (evd.amax(dim=-1) / evd.amin(dim=-1).clamp_min(1e-30))
    return c


def evaluate_methods(S_fp64, *, eps=1e-4, methods=None, gram_ns_steps=6,
                     coupled_iters=6):
    """S_fp64: (N, r, r) fp64 SPD. Returns list[MethodResult] + the fp64-eigh truth."""
    dev = S_fp64.device
    Z_truth = eigh_inv_sqrt(S_fp64, eps=eps, eps_relative=True, out_dtype=torch.float64)
    truth_norm = Z_truth.flatten(-2).norm(dim=-1).clamp_min(1e-30)
    cond_eff = _cond_eff(S_fp64, eps=eps)
    eye64 = torch.eye(S_fp64.shape[-1], dtype=torch.float64, device=dev)

    if methods is None:
        methods = [("eigh/fp32", lambda S: eigh_inv_sqrt(S.float(), eps=eps))]
        for ni in (10, 16, 24):
            methods.append((f"coupled_ns{ni}/fp32",
                            lambda S, ni=ni: coupled_ns_inv_sqrt(S.float(), n_iters=ni, eps=eps)))
        for ns in (6, 8, 10):
            methods.append((f"gram_ns_pe{ns}/fp32",
                            lambda S, ns=ns: gram_ns_inv_sqrt(S.float(), nsteps=ns, eps=eps)))
        # bf16 iteration body for the best iter count of each (tensor-core path).
        methods.append(("coupled_ns16/bf16",
                        lambda S: coupled_ns_inv_sqrt(S.float(), n_iters=16, eps=eps,
                                                      compute_dtype=torch.bfloat16)))
        methods.append(("gram_ns_pe8/bf16",
                        lambda S: gram_ns_inv_sqrt(S.float(), nsteps=8, eps=eps,
                                                   compute_dtype=torch.bfloat16)))

    results = []
    for name, fn in methods:
        Z = fn(S_fp64).double()
        nonfin = int((~torch.isfinite(Z)).any(dim=(-2, -1)).sum())
        Zf = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        rel = ((Zf - Z_truth).flatten(-2).norm(dim=-1) / truth_norm)
        Sd = S_fp64 + eps * torch.linalg.eigvalsh(S_fp64).amax(
            dim=-1, keepdim=True).unsqueeze(-1) * eye64
        resid = (Zf @ Sd @ Zf - eye64).flatten(-2).norm(dim=-1)
        # smallest eigenvalue of each output (must stay > 0 — SPD-ness preserved).
        try:
            min_eig = float(torch.linalg.eigvalsh(
                0.5 * (Zf + Zf.transpose(-2, -1))).amin())
        except Exception:
            min_eig = float("nan")
        dt = "fp64" if "/" not in name else name.split("/")[1]
        results.append(MethodResult(
            name=name.split("/")[0], dtype=dt,
            rel_err_p50=float(rel.median()),
            rel_err_p99=float(rel.quantile(0.99)),
            residual_p99=float(resid.quantile(0.99)),
            min_eig_Z=min_eig, n_nonfinite=nonfin,
            cond_eff_p99=float(cond_eff.quantile(0.99))))
    return results


def time_method_batched(fn, S_fp32, *, iters=50, warmup=10):
    """Median ms per batched call on the GPU."""
    dev = S_fp32.device
    for _ in range(warmup):
        fn(S_fp32)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(S_fp32)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def _print_table(title, results, timings=None):
    print(f"\n### {title}")
    print(f"{'method':<14}{'dtype':<6}{'rel_p50':>10}{'rel_p99':>10}"
          f"{'resid_p99':>11}{'min_eig(Z)':>12}{'nonfin':>8}", end="")
    print(f"{'ms/call':>10}" if timings else "")
    for r in results:
        key = f"{r.name}/{r.dtype}"
        ms = (f"{timings.get(key, float('nan')):>10.3f}" if timings else "")
        flag = "  <-- BLOWUP" if (r.n_nonfinite or r.min_eig_Z <= 0
                                  or r.rel_err_p99 > 1) else ""
        print(f"{r.name:<14}{r.dtype:<6}{r.rel_err_p50:>10.2e}{r.rel_err_p99:>10.2e}"
              f"{r.residual_p99:>11.2e}{r.min_eig_Z:>12.2e}{r.n_nonfinite:>8}"
              f"{ms}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--gram_steps", type=int, default=6)
    ap.add_argument("--coupled_iters", type=int, default=6)
    ap.add_argument("--time", action="store_true", help="also measure batched wall")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available()
                       or args.device == "cpu" else "cpu")
    print(f"device={dev}  eps(rel-δ)={args.eps}  gram_steps={args.gram_steps}  "
          f"coupled_iters={args.coupled_iters}")
    print(f"polar-express coeffs (first {args.gram_steps}): "
          f"{[tuple(round(x,3) for x in c) for c in _POLAR_EXPRESS_COEFFS[:args.gram_steps]]}")

    def timings_for(S):
        # Time only the fp32 accuracy-matched configs: bf16 blows up at the
        # production δ-floor (cond≈1/δ), so it is off the table. The matrices are
        # tiny (r×r) — the decision is pure batched wall, eigh vs the two NS.
        if not args.time or dev.type != "cuda":
            return None
        Sf = S.float()
        return {
            "eigh/fp32": time_method_batched(
                lambda s: eigh_inv_sqrt(s, eps=args.eps), Sf),
            "coupled_ns16/fp32": time_method_batched(
                lambda s: coupled_ns_inv_sqrt(s, n_iters=16, eps=args.eps), Sf),
            "gram_ns_pe6/fp32": time_method_batched(
                lambda s: gram_ns_inv_sqrt(s, nsteps=6, eps=args.eps), Sf),
            "gram_ns_pe8/fp32": time_method_batched(
                lambda s: gram_ns_inv_sqrt(s, nsteps=8, eps=args.eps), Sf),
        }

    # --- Real snapshot Grams (the realistic well-conditioned floor). ---
    from ..snapshot_analysis.snapshots import SNAP_ROOTS
    for key, label in [(('3e-2', 256, 5, 'chord-tight-k1'), "snapshot r256 k1 @ step2000"),
                       (('3e-2', 64, 5, 'chord-tight-k1'), "snapshot r64 k1 @ step2000")]:
        if not SNAP_ROOTS.get(key) or not SNAP_ROOTS[key].exists():
            continue
        S = load_snapshot_grams_batched(key, 2000, dev)
        res = evaluate_methods(S, eps=args.eps, gram_ns_steps=args.gram_steps,
                               coupled_iters=args.coupled_iters)
        _print_table(f"{label}  (N={S.shape[0]}, r={S.shape[-1]})", res, timings_for(S))

    # --- Synthetic stress: cond up to the δ-floor worst case (1/δ). ---
    for r in (64, 256):
        for cond in (1e2, 1e3, 1e4):
            S = make_synthetic_spd(r, cond, n=112, device=dev, decay="powerlaw")
            res = evaluate_methods(S, eps=args.eps, gram_ns_steps=args.gram_steps,
                                   coupled_iters=args.coupled_iters)
            _print_table(f"synthetic powerlaw r={r} cond={cond:.0e} (N=112)",
                         res, timings_for(S))


if __name__ == "__main__":
    main()
