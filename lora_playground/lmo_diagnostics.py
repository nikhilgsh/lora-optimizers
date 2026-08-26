"""Approximate-LMO scores for the pre-msign matrix of the polar-LoRA step.

The optimizer's direction step solves, per factor,

    max_{‖Z‖_2 ≤ 1} <H, Z>            with optimum  ‖H‖_*  attained at  Z = msign(H),

where H is the whitened momentum handed to ``_polar_ns_guarded`` — ``zA``/``zB``
at ``optim.py:2068-2073`` (batched path) and ``optim.py:1784-1787`` (per-pair
path). This module scores *cheaper* operators T against that exact optimum:

    rho(T; H) = <H, T(H)/‖T(H)‖_2> / ‖H‖_*        in [0, 1]

rho = 1 iff T(H) is (a positive multiple of) msign(H). It answers "how much of
the spectral LMO objective does T capture on the matrices this optimizer
actually forms" — cheaply, offline, before any optimizer variant is trained.

rho is invariant to positive scalar multiples of T(H) by construction, so a
magnitude rule downstream of T (REG's RMS rescale, this project's rho-radius)
cannot change it.

Operators
---------
``t_frob``          Z ∝ H. The Frobenius-LMO / preconditioned-gradient endpoint;
                    this is what the ``kl-diag-lora`` arm runs (identity
                    direction operator).
``t_reg_oneside``   REG Eq. (3), ``normal(M; p)``: normalize the SHORT side
                    (rows if m <= n, else columns) to unit ell_p norm.
                    reg_2510.03691.pdf.
``t_reg_alg1``      REG Algorithm 1 (appendix): row pass THEN column pass, t
                    times. NOTE this is NOT Eq. (3) even at t=1, and the paper
                    is internally inconsistent about which one the experiments
                    ran — Theorem 2's closed-form RMS = 1/sqrt(max(m,n)) holds
                    only for the one-sided Eq. (3), which is the operator Eq.
                    (5) consumes. Both are provided; they are different
                    operators and score differently.
``t_fisher_racs``   RACS Eq. (16) fixed point (racs_2502.07752.pdf,
                    Prop. 3): with P = H^(elementwise 2),
                        s <- P^T q / ‖q‖^2,    q <- P s / ‖s‖^2,
                    from q = 1, then Diag(q)^{-1/2} H Diag(s)^{-1/2}. The paper
                    uses 5 iterations and a 1-sample estimate of E[.], i.e. the
                    current gradient only. The fixed point is defined up to a
                    scale on q and on s independently; both are scalars on the
                    output, so rho is unaffected.
``t_polar_k``       The production PolarExpress used at K steps — the real
                    ``_polar_express_gram_batched``, not a re-implementation.
``t_msign``         Exact compact polar via SVD. rho == 1; the calibration check.

Precision and cost
------------------
Everything runs in float64 on the r x r Gram. ``‖H‖_*`` and ``eps_polar`` need
the FULL spectrum, so ``eigvalsh`` is correct here and is not the banned
"SVD for a scalar sigma_max" (CLAUDE.md) — that rule is about the optimizer's
hot path, where only the top value is wanted. This module is an offline
diagnostic: exactness of the rho denominator is the whole point, and the Gram is
r x r (r <= 256), not d x d.
"""

from __future__ import annotations

import torch

__all__ = [
    "gram_singular_values",
    "nuclear_norm",
    "eps_polar",
    "rho",
    "t_frob",
    "t_reg_oneside",
    "t_reg_alg1",
    "t_fisher_racs",
    "t_polar_k",
    "t_msign",
    "lmo_scores",
]

_EPS = 1e-300


def _as_wide(H):
    """Return (H_wide, was_transposed) with the SHORT side first.

    Every quantity here is transpose-covariant: <H^T, Z^T> = <H, Z>, and the
    spectral, nuclear and Frobenius norms are transpose-invariant, so scoring
    the wide orientation scores the original. REG Eq. (3) already picks its axis
    by dimension, so it is transpose-consistent by construction.
    """
    H = H.to(torch.float64)
    if H.shape[-2] > H.shape[-1]:
        return H.transpose(-2, -1), True
    return H, False


def gram_singular_values(H):
    """Singular values of H (descending), via eigvalsh of the short-side Gram.

    Full spectrum by design — ``nuclear_norm`` and ``eps_polar`` both consume
    all of it. Negative eigenvalues from round-off are clamped at 0 before the
    sqrt (the Gram is PSD by construction, so a negative value is noise).
    """
    Hw, _ = _as_wide(H)
    G = Hw @ Hw.transpose(-2, -1)
    lam = torch.linalg.eigvalsh(G).clamp_min(0.0)
    return lam.flip(-1).sqrt()


def nuclear_norm(H):
    """‖H‖_* = sum of singular values — the exact LMO optimum for ‖Z‖_2 <= 1."""
    return gram_singular_values(H).sum(-1)


def eps_polar(H):
    """‖Gbar - I‖_F / sqrt(r) with Gbar = (r / tr G) G, G = H H^T (short side).

    How far H already is from scaled-orthogonal, i.e. how much singular-value
    flattening msign has left to do. 0 <=> H's short-side rows are orthogonal
    with equal norms.
    """
    Hw, _ = _as_wide(H)
    r = Hw.shape[-2]
    G = Hw @ Hw.transpose(-2, -1)
    tr = G.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(_EPS)
    Gbar = G * (r / tr).unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(r, dtype=Gbar.dtype, device=Gbar.device).expand_as(Gbar)
    return (Gbar - I).flatten(-2).norm(dim=-1) / (r ** 0.5)


def rho(H, TH, nuc=None):
    """rho(T; H) = <H, T(H)/‖T(H)‖_2> / ‖H‖_*, in [0, 1].

    ``nuc`` lets a caller reuse an already-computed ‖H‖_* across many operators
    on the same H. Returns 0.0 when H or T(H) is numerically zero.
    """
    H64 = H.to(torch.float64)
    TH64 = TH.to(torch.float64)
    smax = gram_singular_values(TH64)[..., 0]
    nuc = nuclear_norm(H64) if nuc is None else nuc
    ip = (H64 * TH64).flatten(-2).sum(-1)
    denom = smax * nuc
    return torch.where(denom > _EPS, ip / denom.clamp_min(_EPS),
                       torch.zeros_like(ip))


# ─────────────────────────── candidate operators ───────────────────────────

def t_frob(H):
    """Z ∝ H — the Frobenius-LMO endpoint (the ``kl-diag-lora`` arm)."""
    return H.to(torch.float64)


def _norm_along(X, dim, p):
    n = X.norm(p=p, dim=dim, keepdim=True)
    # A zero row/column stays zero rather than becoming inf/nan: scale by 0.
    return torch.where(n > _EPS, n.reciprocal(), torch.zeros_like(n))


def t_reg_oneside(H, p=2.0):
    """REG Eq. (3) ``normal(M; p)``: unit-ell_p on the SHORT side only.

    For H of shape (m, n): rows if m <= n, columns if m > n. For the polar-LoRA
    A-side (r, d_in) with r << d_in this is row normalization, i.e. exactly
    ``diag(H H^T)^{-1/2} H`` at p=2 — the diagonal-Gram approximation to msign.
    """
    X = H.to(torch.float64)
    if X.shape[-2] <= X.shape[-1]:
        return X * _norm_along(X, -1, p)      # rows
    return X * _norm_along(X, -2, p)          # columns


def t_reg_alg1(H, iters=1, p=2.0):
    """REG Algorithm 1: ``iters`` rounds of (row pass, then column pass).

    Distinct from ``t_reg_oneside`` at every ``iters`` >= 1. The paper's
    experiments are described as ``iters=1`` of this, while Eq. (5) and
    Theorem 2 describe the one-sided Eq. (3); both are scored here rather than
    guessing which was deployed.
    """
    X = H.to(torch.float64)
    for _ in range(iters):
        X = X * _norm_along(X, -1, p)         # rows to unit ell_p
        X = X * _norm_along(X, -2, p)         # then columns
    return X


def t_fisher_racs(H, iters=5):
    """RACS Eq. (16) fixed point, then Diag(q)^{-1/2} H Diag(s)^{-1/2}.

    P = H elementwise-squared; s <- P^T q / ‖q‖^2 then q <- P s / ‖s‖^2, from
    q = 1 (the paper's initialization), ``iters`` times (5 in Algorithm 1).
    The per-step EMA on (s, q) in Algorithm 1 lines 6-7 is NOT applied — this is
    the fixed point on this H alone, i.e. an upper bound on what the deployed
    optimizer's stale, EMA'd scales achieve.
    """
    X = H.to(torch.float64)
    P = X * X
    m, n = X.shape[-2], X.shape[-1]
    q = X.new_ones(*X.shape[:-2], m)
    s = X.new_ones(*X.shape[:-2], n)
    for _ in range(iters):
        s = (P.transpose(-2, -1) @ q.unsqueeze(-1)).squeeze(-1) \
            / q.pow(2).sum(-1, keepdim=True).clamp_min(_EPS)
        q = (P @ s.unsqueeze(-1)).squeeze(-1) \
            / s.pow(2).sum(-1, keepdim=True).clamp_min(_EPS)
    qi = torch.where(q > _EPS, q.rsqrt(), torch.zeros_like(q))
    si = torch.where(s > _EPS, s.rsqrt(), torch.zeros_like(s))
    return X * qi.unsqueeze(-1) * si.unsqueeze(-2)


def t_polar_k(H, k):
    """The PRODUCTION PolarExpress at ``k`` steps.

    Calls ``optim._polar_express_gram_batched`` with ``pre_norm='frob'`` — the
    same routine and the same Frobenius pre-normalization the optimizer reaches
    through ``_polar_ns_guarded``, so rho_K measures the operator that would
    actually ship, not a re-derivation of it.
    """
    from .optim import _polar_express_gram_batched
    out = _polar_express_gram_batched(
        H.to(torch.float32), nsteps=k, dtype=torch.float32, pre_norm="frob")
    return out.to(torch.float64)


def t_msign(H):
    """Exact compact polar U_k V_k^T via SVD. rho == 1 up to round-off."""
    X = H.to(torch.float64)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    keep = S > max(1e-12, 1e-12 * float(S.max())) if S.numel() else S > 0
    Z = (U * keep.to(U.dtype).unsqueeze(-2)) @ Vh
    return Z


# ────────────────────────────── the score suite ─────────────────────────────

def lmo_scores(H, polar_ks=(1, 2, 4, 6, 8), racs_iters=5, p=2.0,
               reg_alg1_iters=(1, 2)):
    """All rho's plus eps_polar for one H, as a flat dict of python floats.

    H must be 2-D (a single LoRA factor's whitened momentum). Keys:
      ``eps_polar``, ``rho_frob``, ``rho_reg_oneside``,
      ``rho_reg_alg1_t{t}``, ``rho_fisher_racs``, ``rho_polar_k{k}``,
      ``rho_msign`` (the == 1 calibration), ``nuc``, ``smax``, ``fro``,
      ``stable_rank``, ``shape``.
    """
    if H.ndim != 2:
        raise ValueError(f"lmo_scores expects a 2-D factor, got {tuple(H.shape)}")
    X = H.to(torch.float64)
    sv = gram_singular_values(X)
    nuc = sv.sum()
    smax = sv[0]
    fro = X.norm()
    out = {
        "shape": tuple(X.shape),
        "nuc": nuc.item(),
        "smax": smax.item(),
        "fro": fro.item(),
        "stable_rank": (fro.pow(2) / smax.pow(2).clamp_min(_EPS)).item(),
        "eps_polar": eps_polar(X).item(),
        "rho_frob": rho(X, t_frob(X), nuc=nuc).item(),
        "rho_reg_oneside": rho(X, t_reg_oneside(X, p=p), nuc=nuc).item(),
        "rho_fisher_racs": rho(X, t_fisher_racs(X, iters=racs_iters), nuc=nuc).item(),
        "rho_msign": rho(X, t_msign(X), nuc=nuc).item(),
    }
    for t in reg_alg1_iters:
        out[f"rho_reg_alg1_t{t}"] = rho(X, t_reg_alg1(X, iters=t, p=p), nuc=nuc).item()
    for k in polar_ks:
        out[f"rho_polar_k{k}"] = rho(X, t_polar_k(X, k), nuc=nuc).item()
    return out
