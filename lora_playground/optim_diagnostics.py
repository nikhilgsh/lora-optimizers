"""Shared LoRA optimizer diagnostics.

Tier-1 diagnostics that depend **only on the factor tensors (A, B)** — not on any
optimizer-internal state — so every LoRA optimizer can emit them identically.
Historically these lived inline inside ``AdamPolarProductLoRA.step()`` and were
therefore absent from every other optimizer (notably ``CurvatureWhitenLoRA`` /
kl-shampoo). They belong here.

Tier-2 diagnostics (whitened-direction spectra, AWC, the σ_max estimate gap)
need the optimizer's intermediate tensors and stay in the optimizer, fed through
an explicit call — there is no clean way to share those without a callback, and
the callback is the honest boundary.

PEFT factor convention (see docs/low_rank_peft_convention.md):
A ∈ (r, d_in), B ∈ (d_out, r), adapter ΔW = (α/r) B A.
"""

import torch

# Keys produced by ``factor_diagnostics`` — used to fill NaN consistently on the
# error path so per-pair records keep a stable key set (the aggregator in
# ``_emit_optim_diagnostics`` reads keys from the first record).
FACTOR_DIAG_KEYS = (
    "balance_resid",
    "norm_A", "norm_B",
    "nrank_A_1e3", "nrank_A_1e2", "nrank_B_1e3", "nrank_B_1e2",
    "stable_rank_A", "stable_rank_B",
    "sigma_max_A", "sigma_max_B",
)


def factor_diagnostics(A, B):
    """Tier-1 factor-only diagnostics for one LoRA pair (A, B).

    Pure function of the two factor tensors; returns a flat ``dict`` of
    float/int stats suitable for per-pair median/min/max aggregation by
    ``_emit_optim_diagnostics``. Costs one r×r matmul + one r×r ``eigvalsh``
    per factor — negligible at the diagnostic cadence even for r=256.

    Fields:

    - ``balance_resid`` — relative Frobenius gap ‖AAᵀ − BᵀB‖_F / max(‖AAᵀ‖,‖BᵀB‖).
      The GL(r) gauge (A, B) → (R⁻¹A, BR) leaves the adapter BA fixed but moves
      the factor conditioning; the *balanced* representative satisfies AAᵀ = BᵀB
      (BaLoRA, Castin et al. 2026, arXiv:2605.31484 — under the paper↔repo factor
      swap). 0 ⇒ on the balanced manifold. Larger values can indicate that the
      current factor representative is far from balanced, but the link to LR
      sensitivity is an empirical hypothesis that must be checked against runs.
    - ``norm_{A,B}`` = ‖A‖_F, ‖B‖_F — useful because PEFT's default zero-B init
      starts far from the balanced manifold, so the residual alone can saturate.
    - ``stable_rank_{A,B}`` = Σσ² / σ_max²  (computed as Σλ / λ_max of the r×r
      Gram, whose eigenvalues λ = σ²).
    - ``nrank_{A,B}_{1e3,1e2}`` = #{ σ² > τ · σ_max² }, τ ∈ {1e-3, 1e-2}
      (numerical rank at two thresholds).
    - ``sigma_max_{A,B}`` = σ_max(A), σ_max(B) — the very quantities the chord/
      Spectron rescale ρ = η/(σ_max(A)+σ_max(B)) divides by, so logging them lets
      the balance↔σ_max-fragility link be tested directly from logs.
    """
    A_f = A.detach().float()
    B_f = B.detach().float()
    rec = {
        "norm_A": float(A_f.norm()),
        "norm_B": float(B_f.norm()),
    }
    try:
        G_A = A_f @ A_f.T          # (r, r) = A Aᵀ   (eigs = σ²(A))
        G_B = B_f.T @ B_f          # (r, r) = Bᵀ B   (eigs = σ²(B))
        gA_fro = float(G_A.norm())
        gB_fro = float(G_B.norm())
        rec["balance_resid"] = float(
            (G_A - G_B).norm() / max(gA_fro, gB_fro, 1e-30))
        eigA = torch.linalg.eigvalsh(G_A).clamp_min(0.0)
        eigB = torch.linalg.eigvalsh(G_B).clamp_min(0.0)
        smax_A = float(eigA.max())     # = σ_max(A)²
        smax_B = float(eigB.max())
        rec["nrank_A_1e3"] = int((eigA > 1e-3 * smax_A).sum())
        rec["nrank_A_1e2"] = int((eigA > 1e-2 * smax_A).sum())
        rec["nrank_B_1e3"] = int((eigB > 1e-3 * smax_B).sum())
        rec["nrank_B_1e2"] = int((eigB > 1e-2 * smax_B).sum())
        rec["stable_rank_A"] = float(eigA.sum() / (smax_A + 1e-30))
        rec["stable_rank_B"] = float(eigB.sum() / (smax_B + 1e-30))
        rec["sigma_max_A"] = float(smax_A ** 0.5)
        rec["sigma_max_B"] = float(smax_B ** 0.5)
    except torch._C._LinAlgError:
        for k in FACTOR_DIAG_KEYS:
            rec[k] = float("nan")
    return rec
