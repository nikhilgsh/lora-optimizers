from dataclasses import dataclass

import torch
from torch import nn


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


def spd_frac_power_inv(H, gamma, eps=1e-6):
    """
    Compute (H + eps*I)^{-gamma} for SPD H via eigendecomposition.
    H: (n, n) float32. Returns (n, n) float32.
    """
    H = spdify(H, eps)
    evals, Q = torch.linalg.eigh(H)
    return Q @ torch.diag(evals.clamp(min=eps).pow(-gamma)) @ Q.T


def _solve_ridge(A, B, eps=1e-6):
    """Solve (A + eps·I) X = B via Cholesky with fallback to dense solve.

    A is (r, r), assumed approximately SPD. Returns X with same shape as B.
    """
    n = A.shape[-1]
    Ar = A + torch.eye(n, dtype=A.dtype, device=A.device).mul(eps)
    L, info = torch.linalg.cholesky_ex(Ar)
    if int(info) == 0:
        return torch.cholesky_solve(B, L)
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
            A_t = _solve_ridge(B_t.T @ B_t, sum_A, eps=lmbd)  # (r, d_in)
        else:
            # B_new = (Ŵ A_tᵀ + ρ B₀) (A_t A_tᵀ + ρI)⁻¹
            # = ((A_t A_tᵀ + ρI)⁻¹ (A_t Ŵᵀ + ρ B₀ᵀ))ᵀ
            sum_B = B0f.mul(lmbd)
            for c, (factor_in, factor_out) in zip(coefficients, factors_f):
                sum_B = sum_B + (factor_out @ (factor_in @ A_t.T)).mul(c)
            B_t = _solve_ridge(A_t @ A_t.T, sum_B.T, eps=lmbd).T  # (d_out, r)

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
            A_t = _solve_ridge(gram, sum_A, eps=lmbd)      # (r, d_in)
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
            B_t = _solve_ridge(gram, sum_B.T, eps=lmbd).T  # (d_out, r)

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

def collect_lora_pairs(model, adapter_name=None):
    """
    Collect LoRA (A, B) pairs from model.

    Each pair:
        A: (r, in)
        B: (out, r)

    Returns:
        List[Tuple[Tensor, Tensor]]
    """
    pairs = []
    for _, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            try:
                keys = [adapter_name] if adapter_name else list(mod.lora_A.keys())
                for k in keys:
                    if k in mod.lora_A and k in mod.lora_B:
                        A = mod.lora_A[k].weight  # (r, in), original dtype
                        B = mod.lora_B[k].weight  # (out, r), original dtype
                        pairs.append((A, B))
                continue
            except Exception:
                if hasattr(mod.lora_A, "weight") and hasattr(mod.lora_B, "weight"):
                    pairs.append((mod.lora_A.weight, mod.lora_B.weight))
    return pairs
