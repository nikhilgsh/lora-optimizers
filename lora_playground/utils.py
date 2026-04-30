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
