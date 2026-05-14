import json
import sys
from dataclasses import dataclass

import torch
from torch import nn

from .spectral import lambda_max_power_iter_psd_batched


# Debug-only probe for the higham coupled-NS preconditioner. Default off;
# `train_lora.py --debug_higham_residual` flips it on. When enabled, every
# higham call computes ‖Z H Z − I‖_F per matrix (the actual quality of the
# inverse-square-root) and emits a JSON line with the residual + λ_max
# estimate + presence of non-finite output. Used for the integration test
# at high r where compound numerical drift / power-iteration underestimate
# can produce non-SPD output and downstream NaN.
HIGHAM_DEBUG = {"enabled": False, "step": 0, "call": 0}


def _higham_emit_residual(Z, H, lam_max, n_iters, eps, where):
    """Compute ‖Z H Z − I‖_F per matrix in batch (or scalar for non-batched)
    and emit a JSON line. Cheap; only fires when HIGHAM_DEBUG['enabled']."""
    if not HIGHAM_DEBUG["enabled"]:
        return
    with torch.no_grad():
        ZHZ = Z @ H @ Z
        n = Z.shape[-1]
        I = torch.eye(n, dtype=Z.dtype, device=Z.device)
        if Z.dim() > 2:
            I = I.expand_as(ZHZ)
        diff = ZHZ - I
        # Per-matrix Frobenius residual; flatten last 2 dims and norm.
        res = diff.flatten(-2).norm(dim=-1)
        finite_by_matrix = torch.isfinite(Z).flatten(-2).all(dim=-1)
        non_finite_Z = (~finite_by_matrix).any().item()
        bad_indices = (~finite_by_matrix).nonzero(as_tuple=False).flatten()[:16].tolist()
        res_for_argmax = torch.nan_to_num(res, nan=float("inf"), posinf=float("inf"))
        residual_argmax = int(res_for_argmax.reshape(-1).argmax())
        lam_flat = lam_max.reshape(-1)
        diag_max = H.diagonal(dim1=-2, dim2=-1).amax(dim=-1)
        diag_flat = diag_max.reshape(-1)
        rec = {
            "event": "higham_residual",
            "where": where,
            "step": HIGHAM_DEBUG["step"],
            "call": HIGHAM_DEBUG["call"],
            "n_iters": n_iters,
            "eps": eps,
            "n": int(n),
            "batch_size": int(Z.numel() // (n * n)),
            "residual_max": float(res.max()),
            "residual_median": float(res.median()),
            "residual_min": float(res.min()),
            "lam_max_min": float(lam_max.min()),
            "lam_max_max": float(lam_max.max()),
            "diag_max_min": float(diag_max.min()),
            "diag_max_max": float(diag_max.max()),
            "residual_argmax": residual_argmax,
            "lam_max_at_residual_argmax": float(lam_flat[residual_argmax]),
            "diag_max_at_residual_argmax": float(diag_flat[residual_argmax]),
            "non_finite_Z": bool(non_finite_Z),
            "non_finite_indices": [int(i) for i in bad_indices],
        }
        HIGHAM_DEBUG["call"] += 1
        print(json.dumps(rec, sort_keys=True), flush=True)


@dataclass
class TargetWeight:
    name: str
    module: nn.Module
    weight: torch.nn.Parameter
    base_weight: torch.Tensor

def spdify(M, eps):
    """
    Make a matrix symmetric positive definite (SPD).

    Args:
        M (Tensor): (N, N), any dtype.
        eps (float): Minimum eigenvalue to add to diag for PD.

    Returns:
        Tensor: (N, N), float32, symmetric and PD.
    """
    M = M.float()
    M = 0.5 * (M + M.T)
    M.diagonal().add_(eps)
    return M

def solve_spd(A, B):
    """
    Solve A X = B for SPD A using Cholesky decomposition.

    Args:
        A (Tensor): (N, N), float32 SPD.
        B (Tensor): (N, M), float32.

    Returns:
        X (Tensor): (N, M), float32 solution.
    """
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(B, L)

def solve_sylvester(SB, SA, RHS):
    """
    Solve for K in: K SA + SB K = RHS for SPD SB, SA.
    All shapes (r, r), float32.

    Args:
        SB (Tensor): (r, r), float32 SPD.
        SA (Tensor): (r, r), float32 SPD.
        RHS (Tensor): (r, r), float32.

    Returns:
        K (Tensor): (r, r), float32.
    """
    evalB, QB = torch.linalg.eigh(SB)
    evalA, QA = torch.linalg.eigh(SA)
    T = QB.T @ RHS @ QA                               # (r, r)
    denom = evalB[:, None] + evalA[None, :]           # (r, r)
    X = T / denom                                     # (r, r)
    return QB @ X @ QA.T                              # (r, r)


def spd_frac_power_inv(H, gamma, eps=1e-6, eps_relative=False):
    """
    Compute (H + eps_eff*I)^{-gamma} for SPD H via eigendecomposition.
    eps_eff = eps (absolute) or eps * λ_max(H) (relative, σ_max-scaled).
    H: (n, n) float32. Returns (n, n) float32.
    """
    H_sym = 0.5 * (H.float() + H.float().T)
    if eps_relative:
        # λ_max(H) via the eigh we're about to do anyway — read it from evals.
        evals_raw, Q = torch.linalg.eigh(H_sym)
        lam_max = float(evals_raw.abs().max())
        eps_eff = float(eps) * lam_max
        evals = (evals_raw + eps_eff).clamp(min=eps_eff if eps_eff > 0 else 1e-30)
        return Q @ torch.diag(evals.pow(-gamma)) @ Q.T
    H_d = spdify(H_sym, eps)
    evals, Q = torch.linalg.eigh(H_d)
    return Q @ torch.diag(evals.clamp(min=eps).pow(-gamma)) @ Q.T


def spd_inv_sqrt_higham_batched(H, n_iters=10, eps=1e-6, n_power_iter=4, eps_relative=False):
    """Batched Iannazzo / Denman-Beavers-Newton coupled NS for H^{-1/2}.

    Shape-broadcasting version of `spd_inv_sqrt_higham`. H: (..., n, n) SPD
    -> Z: (..., n, n) ≈ H^{-1/2}. Same iteration, same convergence properties,
    same regularization. Used by the batched precond_refresh path to compute
    SA^{-1/2}, SB^{-1/2} for all LoRA pairs in one bmm sequence rather than
    a per-pair Python loop.

    eps_relative: if True, effective damping is `eps * λ_max(H)` per batch
    (σ_max-relative damping, see docs/notes/polar_product/init_damping_math.md §5.3).
    Requires one extra power iter on raw H to estimate λ_max before damping.
    """
    n = H.shape[-1]
    H = 0.5 * (H + H.transpose(-2, -1))
    eye = torch.eye(n, dtype=H.dtype, device=H.device)
    batch_shape = H.shape[:-2]
    if eps_relative:
        # λ_max on RAW H (before damping) sets scale-invariant damping.
        # Use the shared PSD spectral primitive; a single H@ones start can
        # miss the top eigendirection and under-scale the Higham iteration.
        lam_max_raw, _ = lambda_max_power_iter_psd_batched(
            H, n_iters=max(n_power_iter, 8),
        )
        lam_max_raw = lam_max_raw.unsqueeze(-1).unsqueeze(-1)
        H = H + (eps * lam_max_raw).clamp_min(1e-12) * eye
    else:
        H = H + eps * eye
    eye_b = eye.expand_as(H)

    lam_max, _ = lambda_max_power_iter_psd_batched(
        H, n_iters=max(n_power_iter, 8),
    )

    s = lam_max.unsqueeze(-1).unsqueeze(-1)           # (..., 1, 1)
    Y = H / s
    Z = eye_b.clone()
    three_eye = 3.0 * eye_b
    for _ in range(n_iters):
        T = three_eye - Z @ Y
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)
    out = Z / s.sqrt()
    _higham_emit_residual(out, H, lam_max, n_iters, eps, where="batched")
    return out


def spd_inv_sqrt_higham(H, n_iters=10, eps=1e-6, n_power_iter=4, eps_relative=False):
    """
    Compute H^{-1/2} for SPD H via the coupled Newton-Schulz iteration
    (Iannazzo / Denman-Beavers-Newton). Replaces spd_frac_power_inv(H, 0.5)
    with O(n_iters) matrix multiplies — kernel-launch friendly, no eigh.

    The iteration:
        Y_0 = H_scaled = H / s,  Z_0 = I,    where s ≈ λ_max(H)
        T = 3I − Z_k Y_k
        Y_{k+1} = (1/2) Y_k T
        Z_{k+1} = (1/2) T Z_k
    converges quadratically to Y_∞ = H_scaled^{1/2}, Z_∞ = H_scaled^{-1/2}.
    Then H^{-1/2} = Z_∞ / sqrt(s).

    Quadratic convergence requires Y_0's eigenvalues clustered near 1, which means
    `s` must be a tight estimate of λ_max — Frobenius norm overshoots by √n on
    typical SPD matrices and kills convergence. We use a few power iterations to
    estimate λ_max directly.

    Args:
        H: (n, n) float32, SPD.
        n_iters: number of coupled NS iterations (default 7; quadratic convergence).
        eps: regularization added to H's diagonal before the iteration.
        n_power_iter: power-iteration steps for λ_max estimate (default 4 — cheap,
            each iter is one matvec). Increase for ill-conditioned H.

    Returns:
        Z: (n, n) float32 approximation of H^{-1/2}.
    """
    H_raw = H.float()
    H_raw = 0.5 * (H_raw + H_raw.T)
    n = H_raw.shape[0]
    if eps_relative:
        # λ_max estimate on raw H to set effective damping = eps * λ_max.
        lam_raw, _ = lambda_max_power_iter_psd_batched(
            H_raw.unsqueeze(0), n_iters=max(n_power_iter, 8),
        )
        eps_eff = max(float(eps) * float(lam_raw[0]), 1e-12)
    else:
        eps_eff = float(eps)
    H = spdify(H_raw, eps_eff)
    eye = torch.eye(n, dtype=H.dtype, device=H.device)

    # Estimate λ_max(H). Tight scaling is critical for quadratic convergence.
    lam_max, _ = lambda_max_power_iter_psd_batched(
        H.unsqueeze(0), n_iters=max(n_power_iter, 8),
    )
    lam_max = lam_max[0]

    # Y_0 = H / λ_max has spectrum in (0, 1]; iteration converges quadratically.
    s = lam_max
    Y = H / s
    Z = eye.clone()
    three_eye = 3.0 * eye
    for _ in range(n_iters):
        T = three_eye - Z @ Y
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)
    out = Z / s.sqrt()
    _higham_emit_residual(out, H, lam_max.unsqueeze(0), n_iters, eps, where="loop")
    return out


def _solve_ridge(A, B, eps=1e-6):
    """Solve (A + eps·I) X = B via Cholesky with fallback to dense solve.

    A is (r, r), assumed approximately SPD. Returns X with same shape as B.
    On singular fallback, escalate eps until cholesky succeeds (matches the
    reference's robustness — small numerical noise can knock the matrix
    just below the SPD boundary, but a slightly larger ridge always works).
    """
    n = A.shape[-1]
    Ar = A + torch.eye(n, dtype=A.dtype, device=A.device).mul(eps)
    L, info = torch.linalg.cholesky_ex(Ar)
    if int(info) == 0:
        return torch.cholesky_solve(B, L)
    # Cholesky said not-SPD. Bump eps and retry — float32 noise from large
    # contractions (e.g. gram of (16, d_in) with d_in ≫ 1) can produce
    # negative eigenvalues O(d_in · float32_eps · ||A||²) that exceed the
    # original ridge.
    for bump in (1e-4, 1e-2, 1.0):
        Ar = A + torch.eye(n, dtype=A.dtype, device=A.device).mul(eps + bump)
        L, info = torch.linalg.cholesky_ex(Ar)
        if int(info) == 0:
            return torch.cholesky_solve(B, L)
    # Last resort: dense solve with the heavily-bumped matrix.
    return torch.linalg.solve(Ar, B)


def lorsum(factors, coefficients, num_iters=1, lmbd=1e-2, start_turn="A"):
    """LoRSUM (Low-Rank Sum, paper eq. 10) via K-iteration proximal ALS.

    Find rank-r (A, B) — A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r} — such that B@A
    approximates the rank-r projection of Σᵢ cᵢ Lᵢ Rᵢᵀ in Frobenius norm,
    with proximal regularization toward A₀ = factors[0][0], B₀ = factors[0][1].

    factors: list of (factor_in_i, factor_out_i) tuples where
      factor_in_i  ∈ ℝ^{kᵢ × d_in}  (right factor, "input side")
      factor_out_i ∈ ℝ^{d_out × kᵢ} (left factor, "output side")
      The implied low-rank term is factor_out_i @ factor_in_i ∈ ℝ^{d_out × d_in}.
    coefficients: list of scalars same length as factors.
    The first (factor_in, factor_out) MUST be the current LoRA factors (A, B);
    the proximal regularizer pulls toward this pair.

    num_iters K: alternating Gauss-Seidel passes (paper recommends K=1 in practice).
    lmbd ρ: proximal regularizer strength.
    start_turn: "A" updates A first, "B" updates B first.

    Returns (A_new, B_new). All math in float32; outputs cast back to factors[0] dtype.

    Reference: ~/PSI-LoRA/src/oplora/utils.py:low_rank_sum (paper eq. 10).
    """
    assert len(factors) == len(coefficients) and len(factors) >= 1
    A0, B0 = factors[0]
    out_dtype = A0.dtype
    factors_f = [(L.float(), R.float()) for L, R in factors]
    A0f, B0f = factors_f[0]
    A_t, B_t = A0f.clone(), B0f.clone()
    r = A_t.shape[0]
    eye_r = torch.eye(r, dtype=A_t.dtype, device=A_t.device)

    # Each "turn" is a single A-update or B-update; one iter = one A and one B update.
    total_turns = max(1, 2 * num_iters)
    for t in range(total_turns):
        update_A = (start_turn == "A" and t % 2 == 0) or (start_turn == "B" and t % 2 == 1)
        if update_A:
            # In code-convention (A = paper's V transposed, shape r×d_in):
            #   A_new = (BᵀB + ρI)⁻¹ (Bᵀ Ŵ + ρ A₀)
            # where Ŵ = Σᵢ cᵢ Lᵢ Rᵢ. Build the RHS, shape (r, d_in).
            sum_A = A0f.mul(lmbd)
            for c, (factor_in, factor_out) in zip(coefficients, factors_f):
                sum_A = sum_A + ((B_t.T @ factor_out) @ factor_in).mul(c)
            A_t = _solve_ridge(B_t.T @ B_t, sum_A, eps=lmbd + 1e-6)  # (r, d_in)
        else:
            # B_new = (Ŵ A_tᵀ + ρ B₀) (A_t A_tᵀ + ρI)⁻¹
            # = ((A_t A_tᵀ + ρI)⁻¹ (A_t Ŵᵀ + ρ B₀ᵀ))ᵀ
            sum_B = B0f.mul(lmbd)
            for c, (factor_in, factor_out) in zip(coefficients, factors_f):
                sum_B = sum_B + (factor_out @ (factor_in @ A_t.T)).mul(c)
            B_t = _solve_ridge(A_t @ A_t.T, sum_B.T, eps=lmbd + 1e-6).T  # (d_out, r)

    return A_t.to(dtype=out_dtype), B_t.to(dtype=out_dtype)


def f_lorsum(factors, coefficients, D_U, D_V, num_iters=1, lmbd=1e-2,
             gamma=0.5, delta=1e-5, start_turn="A"):
    """F-LoRSUM (paper eq. 14): LoRSUM with K-FAC diagonal metrics baked in.

    Same shape conventions as `lorsum`. The metric tensors D_U, D_V are the EMA
    diagonal-K-FAC statistics:
      D_V ∈ ℝ^{d_in}  ≈ EMA of diag(XᵀX/B)   (input side)
      D_U ∈ ℝ^{d_out} ≈ EMA of diag(SᵀS/B)   (output side)

    Effective metrics applied in the projection are (D + δI)^γ; γ=0.5 is the
    paper default (square-root K-FAC, Shampoo-style).

    Update (B-update side):
        B_new = (ρ B₀ M_V + Σᵢ cᵢ M_U⁻¹ Lᵢ Rᵢ A_tᵀ) (A_t M_V A_tᵀ + ρI)⁻¹    (eq. 14)
    where M_V = (D_V + δ)^γ, M_U = (D_U + δ)^γ.

    Reference: ~/PSI-LoRA/src/oplora/utils.py:scaled_low_rank_sum.
    """
    assert len(factors) == len(coefficients) and len(factors) >= 1
    A0, B0 = factors[0]
    out_dtype = A0.dtype
    device = A0.device
    factors_f = [(L.float(), R.float()) for L, R in factors]
    A0f, B0f = factors_f[0]
    A_t, B_t = A0f.clone(), B0f.clone()
    r = A_t.shape[0]
    eye_r = torch.eye(r, dtype=A_t.dtype, device=device)

    # Diagonal metrics (γ-power, with damping)
    m_V = (D_V.float() + delta).pow(gamma)   # (d_in,)
    m_U = (D_U.float() + delta).pow(gamma)   # (d_out,)
    inv_m_V = m_V.reciprocal()
    inv_m_U = m_U.reciprocal()

    # Pre-condition non-prox factors (factors[1:]) by inverse metrics.
    # pfactor_in_i = inv_m_V ⊙ factor_in_iᵀ (column-scale)  → store as transposed shape
    pfactors = [None]
    for factor_in, factor_out in factors_f[1:]:
        # factor_in: (k, d_in), factor_out: (d_out, k)
        p_in = factor_in * inv_m_V.view(1, -1)            # scale columns
        p_out = inv_m_U.view(-1, 1) * factor_out          # scale rows
        pfactors.append((p_in, p_out))

    total_turns = max(1, 2 * num_iters)
    for t in range(total_turns):
        update_A = (start_turn == "A" and t % 2 == 0) or (start_turn == "B" and t % 2 == 1)
        if update_A:
            # A_new = (Bᵀ M_U B + ρI)⁻¹ (Bᵀ M_U B₀ A₀ · c₀ + Σⱼ≥₁ cⱼ Bᵀ Lⱼ pᵢⱼ + ρ A₀)
            sB_t = m_U.view(-1, 1) * B_t                  # (d_out, r)
            sum_A = A0f.mul(lmbd)
            sum_A = sum_A + ((sB_t.T @ B0f) @ A0f).mul(coefficients[0])
            for j in range(1, len(coefficients)):
                p_in_j, _ = pfactors[j]
                _, factor_out_j = factors_f[j]
                sum_A = sum_A + ((B_t.T @ factor_out_j) @ p_in_j).mul(coefficients[j])
            gram = sB_t.T @ B_t                            # (r, r) ≈ Bᵀ M_U B
            A_t = _solve_ridge(gram, sum_A, eps=lmbd + 1e-6)  # (r, d_in)
        else:
            # B_new computed via (gram⁻¹ rhsᵀ)ᵀ as in lorsum.
            sA_t = A_t * m_V.view(1, -1)                  # (r, d_in)
            sum_B = B0f.mul(lmbd)
            sum_B = sum_B + (B0f @ (A0f @ sA_t.T)).mul(coefficients[0])
            for j in range(1, len(coefficients)):
                _, p_out_j = pfactors[j]
                factor_in_j, _ = factors_f[j]
                sum_B = sum_B + (p_out_j @ (factor_in_j @ A_t.T)).mul(coefficients[j])
            gram = A_t @ sA_t.T                            # (r, r) ≈ A M_V Aᵀ
            B_t = _solve_ridge(gram, sum_B.T, eps=lmbd + 1e-6).T  # (d_out, r)

    return A_t.to(dtype=out_dtype), B_t.to(dtype=out_dtype)


def truncated_svd(matrix, rank, niter=4):
    """
    Return the Frobenius-optimal rank-r approximation of a matrix.

    niter: number of power iterations for the randomized algorithm (default 4).
           Use niter=None for exact economy SVD (slow on large matrices).
    """
    if matrix.ndim != 2:
        raise ValueError(f"truncated_svd expects a matrix, got shape {tuple(matrix.shape)}.")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    m = matrix.float()
    if niter is None:
        U, S, Vh = torch.linalg.svd(m, full_matrices=False)
        rank = min(rank, S.numel())
        return (U[:, :rank] * S[:rank]) @ Vh[:rank]
    rank = min(rank, min(m.shape))
    U, S, V = torch.svd_lowrank(m, q=rank, niter=niter)
    return (U * S) @ V.T


def effective_lora_delta(A, B, scale):
    """
    Compute PEFT-convention adapter displacement scale * B @ A.

    Docs: docs/low_rank_peft_convention.md and
    docs/plans/full_finetune_svd_low_rank_oracle.md.
    Canonical local collection function: collect_lora_pairs.
    """
    return scale * (B @ A)


def svd_to_lora_factors(delta, rank, scale=1.0, niter=4):
    """
    Convert a rank-r SVD projection into PEFT-convention LoRA factors.

    Returns A, B where scale * B @ A equals truncated_svd(delta, rank, niter),
    up to numerical precision.
    """
    if scale == 0:
        raise ValueError("scale must be nonzero.")
    proj = truncated_svd(delta, rank, niter=niter)
    U, S, Vh = torch.linalg.svd(proj.float(), full_matrices=False)
    rank = min(rank, int((S > 1e-10).sum().item()))
    A = Vh[:rank]
    B = U[:, :rank] * (S[:rank] / scale)
    return A, B


def module_has_matrix_weight(module):
    weight = module._parameters.get("weight")
    return isinstance(weight, torch.nn.Parameter) and weight.ndim == 2


def target_module_matches(name, module, target_modules):
    if target_modules == "all-linear":
        return isinstance(module, nn.Linear) and not name.endswith("lm_head")
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    return any(name == target or name.endswith(f".{target}") for target in target_modules)


def collect_dense_target_weights(model, target_modules, exclude_lm_head=True):
    """
    Collect trainable dense weights matching PEFT-style target module names.

    For explicit targets, any module with a direct 2D weight parameter may match;
    this supports Linear-like modules such as GPT-2 Conv1D. For all-linear, only
    torch.nn.Linear modules are selected.
    """
    targets = []
    seen = set()
    for name, module in model.named_modules():
        if not name:
            continue
        if exclude_lm_head and target_modules == "all-linear" and name.endswith("lm_head"):
            continue
        if not target_module_matches(name, module, target_modules):
            continue
        if not module_has_matrix_weight(module):
            continue
        weight = module.weight
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        targets.append(
            TargetWeight(
                name=name,
                module=module,
                weight=weight,
                base_weight=weight.detach().float().clone(),
            )
        )
    if not targets:
        raise ValueError(f"No dense 2D target weights matched target_modules={target_modules!r}.")
    return targets


def freeze_all_except_targets(model, targets):
    for param in model.parameters():
        param.requires_grad_(False)
    for target in targets:
        target.weight.requires_grad_(True)

def collect_ucv_triples(model):
    """
    Collect orthogonal-core LoRA (U, C, V) parameter triples from model.

    Walks named_modules() looking for the `ucv_U`, `ucv_C`, `ucv_V`
    attributes set by `lora_playground.ucv_layer.UCVLinear`.

    Each triple:
        U: (d_out, r), orthonormal columns
        C: (r, r)
        V: (d_in,  r), orthonormal columns

    Returns:
        List[Tuple[Tensor, Tensor, Tensor]]
    """
    triples = []
    for _, mod in model.named_modules():
        if hasattr(mod, "ucv_U") and hasattr(mod, "ucv_C") and hasattr(mod, "ucv_V"):
            triples.append((mod.ucv_U, mod.ucv_C, mod.ucv_V))
    return triples


def override_lora_init(model, mode, adapter_name=None):
    """Override the default PEFT LoRA init.

    mode ∈ {'zero', 'gaussian', 'symmetric'}:
      - 'zero':      no-op (PEFT default: A Kaiming, B=0).
      - 'gaussian':  Init[B] from Chen-Villar-Hayou — A=0, B~N(0, 1/in_features).
      - 'symmetric': Init[AB] from Li et al. — A keeps PEFT default, B sampled
                     at A's std so σ_A=σ_B. PiSSA-style residual: subtract the
                     initial (scaling)·B0@A0 contribution from base_layer.weight
                     so the merged weight at step 0 equals the pretrained weight.

    Returns the list of (layer_name, ΔW0_frob_norm) for diagnostics; empty under 'zero'.
    """
    import math
    if mode == "zero":
        return []
    if mode not in {"gaussian", "symmetric"}:
        raise ValueError(f"unknown lora init mode: {mode}")
    overrides = []
    for name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        keys = [adapter_name] if adapter_name else list(mod.lora_A.keys())
        for k in keys:
            if k not in mod.lora_A or k not in mod.lora_B:
                continue
            A = mod.lora_A[k].weight  # (r, in)
            B = mod.lora_B[k].weight  # (out, r)
            in_features = A.shape[1]
            with torch.no_grad():
                if mode == "gaussian":
                    A.zero_()
                    B.normal_(0.0, 1.0 / math.sqrt(in_features))
                    delta_w0_norm = 0.0
                elif mode == "symmetric":
                    a_std = float(A.detach().float().std())
                    B.normal_(0.0, a_std)
                    scaling = 1.0
                    if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and k in mod.scaling:
                        scaling = float(mod.scaling[k])
                    delta_w0 = scaling * (B.float() @ A.float())
                    delta_w0_norm = float(delta_w0.norm())
                    base = mod.base_layer.weight if hasattr(mod, "base_layer") else mod.weight
                    base.data.sub_(delta_w0.to(base.dtype).to(base.device))
            overrides.append((f"{name}.{k}", delta_w0_norm))
    return overrides


def collect_lora_pairs(model, adapter_name=None):
    """
    Collect LoRA (A, B) pairs from model.

    Each pair:
        A: (r, in)
        B: (out, r)

    Returns:
        List[Tuple[Tensor, Tensor]]
    """
    return [(A, B) for A, B, _ in collect_lora_pairs_named(model, adapter_name)]


def collect_lora_pairs_named(model, adapter_name=None):
    """Same as collect_lora_pairs but each entry is (A, B, name) with name
    being the LoRA module's dotted path. Used by per-pair NaN-trigger
    logging to identify which layer went non-finite."""
    triples = []
    for mod_name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            try:
                keys = [adapter_name] if adapter_name else list(mod.lora_A.keys())
                for k in keys:
                    if k in mod.lora_A and k in mod.lora_B:
                        A = mod.lora_A[k].weight
                        B = mod.lora_B[k].weight
                        triples.append((A, B, f"{mod_name}[{k}]"))
                continue
            except Exception:
                if hasattr(mod.lora_A, "weight") and hasattr(mod.lora_B, "weight"):
                    triples.append((mod.lora_A.weight, mod.lora_B.weight, mod_name))
    return triples
