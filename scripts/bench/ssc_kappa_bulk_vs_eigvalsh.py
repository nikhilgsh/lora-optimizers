"""Compare c_bulk (closed-form, μ-only) vs c_eigvalsh (full spectrum) on
real snapshots.

c_bulk formula (concave-Jensen-approximation to κ(c)):
    μ = ‖X‖²_F / r   (X is σ_max-rescaled to 1, so μ ∈ (1/r, 1])
    κ_target ∈ (μ, 1)  →  c_bulk² = μ (1-κ) / (κ-μ)

This collapses to one F-norm + a couple of scalar ops per pair. No eigvalsh,
no bisection, no Newton — exactly the kind of "free" diagnostic that, if
accurate enough, replaces the κ-adaptive eigvalsh entirely.

Caveat: κ(c) ≈ f_c(μ) only holds when the spectrum is concentrated. f_c is
concave in λ ⇒ Jensen ⇒ bulk overestimates the true average ⇒ c_bulk tends
to be slightly larger than c_true on spread spectra.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lora_playground.snapshot_analysis.snapshots import (
    SNAP_ROOTS, STEPS_BY_ROOT, RUN_A, RUN_B, RUN_C, RUN_D, load_snapshot,
)
from lora_playground.optim import _solve_c_from_kappa_batched
from lora_playground.utils import spd_frac_power_inv


def _spd_half_inv_loop(S, eps=1e-6):
    out = torch.empty_like(S)
    for i in range(S.shape[0]):
        out[i] = spd_frac_power_inv(S[i], gamma=0.5, eps=eps)
    return out


def _stack_group(group, device):
    A = torch.stack([g[1] for g in group]).to(device)
    B = torch.stack([g[2] for g in group]).to(device)
    u_A = torch.stack([g[3] for g in group]).to(device)
    u_B = torch.stack([g[4] for g in group]).to(device)
    return A, B, u_A, u_B


def _whiten_and_rescale(A, B, u_A, u_B):
    """Return X_A, X_B with σ_max(X) ≈ 1 per pair (matching §2.5)."""
    N, r, _ = A.shape
    SA = A @ A.transpose(-2, -1)
    SB = B.transpose(-2, -1) @ B
    WA = _spd_half_inv_loop(SA)
    WB = _spd_half_inv_loop(SB)
    X_A = WB @ u_A                                       # (N, r, d_in)
    X_B = u_B @ WA                                       # (N, d_out, r)
    # σ_max(X) per pair via matrix_norm.
    sigma_XA = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X_A])
    sigma_XB = torch.stack([torch.linalg.matrix_norm(x, ord=2) for x in X_B])
    X_A = X_A / (sigma_XA + 1e-30).view(N, 1, 1)
    X_B = X_B / (sigma_XB + 1e-30).view(N, 1, 1)
    return X_A, X_B


def c_eigvalsh(X, kappa):
    """X: (N, m, n) with σ_max(X)≈1. Returns c per pair."""
    if X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
    G = X @ X.transpose(-2, -1)
    lam = torch.linalg.eigvalsh(G).clamp_min(0.0)
    lam_max = lam.max(dim=-1, keepdim=True).values.clamp_min(1e-12)
    s_sq = lam / lam_max
    return _solve_c_from_kappa_batched(s_sq, kappa).cpu()


def c_two_moment(X, kappa, eps=1e-8):
    """Two-moment top-tail surrogate from μ₁=‖X‖²_F/r and μ₂=‖XXᵀ‖²_F/r.

    Spectrum approximation: mass p at λ=1, mass (1-p) at λ=a.
    Matching first two moments gives:
        a = (μ₁ - μ₂) / (1 - μ₁)
        p = (μ₁ - a) / (1 - a)
    Then κ(c) ≈ p + (1-p)·f_c(a) with f_c(λ) = λ(1+c²)/(λ+c²). Closed form:
        u = (κ - p) / (1 - p)
        c² = a(1-u)/(u-a)
    """
    if X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
    N, r, d = X.shape
    G = X @ X.transpose(-2, -1)                       # (N, r, r)
    mu1 = X.float().square().sum(dim=(-2, -1)) / r    # (N,)
    mu2 = G.float().square().sum(dim=(-2, -1)) / r    # (N,)
    mu1 = mu1.clamp(eps, 1.0 - eps)
    # μ₂ must lie in [μ₁², μ₁] (Cauchy-Schwarz + bounded spectrum).
    mu2 = torch.maximum(mu2, mu1 * mu1 + eps)
    mu2 = torch.minimum(mu2, mu1 - eps)
    a = (mu1 - mu2) / (1.0 - mu1).clamp_min(eps)
    a = a.clamp(eps, 1.0 - eps)
    p = (mu1 - a) / (1.0 - a).clamp_min(eps)
    p = p.clamp(0.0, 1.0 - eps)
    kappa_t = torch.full_like(mu1, float(kappa))
    kappa_t = torch.maximum(kappa_t, mu1 + eps).clamp_max(1.0 - eps)
    u = (kappa_t - p) / (1.0 - p).clamp_min(eps)
    u = torch.maximum(u, a + eps).clamp_max(1.0 - eps)
    c2 = a * (1.0 - u) / (u - a).clamp_min(eps)
    c = c2.clamp_min(1e-6).sqrt()
    return c.cpu()


def c_bulk(X, kappa, eps=1e-8):
    """Closed-form from μ = ‖X‖²_F / r (works on smaller side of X)."""
    if X.shape[-2] > X.shape[-1]:
        X = X.transpose(-2, -1)
    N, r, d = X.shape
    F2 = (X.float() ** 2).sum(dim=(-2, -1))           # (N,)  ‖X‖²_F
    mu = F2 / r                                       # (N,)  assumes σ_max(X)≈1
    mu = mu.clamp(eps, 1.0 - eps)
    k = torch.full_like(mu, float(kappa))
    # κ ∈ (μ, 1) needed for real c. If κ ≤ μ + eps, return c_lo (saturated).
    denom = (k - mu).clamp_min(eps)
    c2 = mu * (1.0 - k) / denom
    c = c2.clamp_min(1e-6).sqrt()
    return c.cpu()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--kappa', type=float, default=0.6)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    runs = [(RUN_A, 'r=64 low-lr'), (RUN_C, 'r=64 high-lr'),
            (RUN_D, 'r=256 low-lr'), (RUN_B, 'r=256 high-lr')]
    print(f"\n# c_bulk vs c_two_moment vs c_eigvalsh at κ={args.kappa}\n")

    for run_key, label in runs:
        if not SNAP_ROOTS[run_key].exists():
            continue
        for step in STEPS_BY_ROOT[run_key]:
            snap = load_snapshot(step, root=SNAP_ROOTS[run_key])
            from collections import defaultdict
            groups = defaultdict(list)
            for pi, p in snap['pair_state'].items():
                if 'A' not in p or 'u_A' not in p:
                    continue
                A, B = p['A'].float(), p['B'].float()
                u_A, u_B = p['u_A'].float(), p['u_B'].float()
                key = (A.shape[0], A.shape[1], B.shape[0])
                groups[key].append((pi, A, B, u_A, u_B))
            cb_A_all, c2_A_all, ct_A_all = [], [], []
            cb_B_all, c2_B_all, ct_B_all = [], [], []
            for shape_key, group in groups.items():
                A, B, u_A, u_B = _stack_group(group, args.device)
                X_A, X_B = _whiten_and_rescale(A, B, u_A, u_B)
                cb_A_all.append(c_bulk(X_A, args.kappa))
                c2_A_all.append(c_two_moment(X_A, args.kappa))
                ct_A_all.append(c_eigvalsh(X_A, args.kappa))
                cb_B_all.append(c_bulk(X_B, args.kappa))
                c2_B_all.append(c_two_moment(X_B, args.kappa))
                ct_B_all.append(c_eigvalsh(X_B, args.kappa))
            for side, cb, c2, ct in [
                ('A', torch.cat(cb_A_all).numpy(), torch.cat(c2_A_all).numpy(), torch.cat(ct_A_all).numpy()),
                ('B', torch.cat(cb_B_all).numpy(), torch.cat(c2_B_all).numpy(), torch.cat(ct_B_all).numpy()),
            ]:
                mask = ct > 0.002
                if not mask.any():
                    continue
                rel_bulk = np.abs(cb[mask] - ct[mask]) / np.maximum(ct[mask], 1e-30)
                rel_2mom = np.abs(c2[mask] - ct[mask]) / np.maximum(ct[mask], 1e-30)
                print(f"{label:>16}  {side:>4}  {step:>5}  {len(rel_bulk):>4}  "
                      f"true={np.median(ct[mask]):>7.4f}  "
                      f"bulk: p50={np.median(rel_bulk):>6.3f} p90={np.percentile(rel_bulk, 90):>6.3f}  "
                      f"2mom: p50={np.median(rel_2mom):>6.3f} p90={np.percentile(rel_2mom, 90):>6.3f} max={rel_2mom.max():>6.3f}")


if __name__ == '__main__':
    main()
