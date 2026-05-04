"""Custom orthogonal-core LoRA adapter (UCV^T).

Replaces PEFT's BA-LoRA with the SVD-like factorization
    ΔW = (alpha / r) * U C V^T,
where U is (d_out, r) with orthonormal columns, V is (d_in, r) with
orthonormal columns, and C is an explicit (r, r) core. C is initialized to
zero so the adapter contributes nothing at step 0.

Spec: docs/notes/polar_product/orthogonal_core_lora_2026_05_03.md.

Used in tandem with `AdamOrthogonalCoreLoRA` (lora_playground.optim) which
runs Adam on three states with polar/Muon retraction on U, V.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .utils import target_module_matches


def _orthonormal_init(rows: int, cols: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """QR of a Gaussian matrix → (rows, cols) with orthonormal columns. cols ≤ rows."""
    g = torch.randn(rows, cols, generator=generator)
    q, _ = torch.linalg.qr(g, mode="reduced")
    return q.contiguous()


class UCVLinear(nn.Module):
    """Wraps a frozen nn.Linear with an additive UCV^T adapter."""

    def __init__(self, base: nn.Linear, r: int, alpha: float | int):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"UCVLinear expects nn.Linear base, got {type(base).__name__}")
        d_out, d_in = base.weight.shape
        if r > min(d_in, d_out):
            raise ValueError(f"r={r} exceeds min(d_in, d_out)=({d_in}, {d_out}).")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        device = base.weight.device
        # store U, V in float32 for numerical stability of NS retractions; the
        # adapter output is cast back to base dtype in forward.
        u0 = _orthonormal_init(d_out, self.r).to(device=device, dtype=torch.float32)
        v0 = _orthonormal_init(d_in, self.r).to(device=device, dtype=torch.float32)
        self.ucv_U = nn.Parameter(u0)
        self.ucv_V = nn.Parameter(v0)
        self.ucv_C = nn.Parameter(torch.zeros(self.r, self.r, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # adapter branch in float32, cast result back to out.dtype
        x32 = x.to(torch.float32)
        # x @ V : (..., r);  @ C^T : (..., r);  @ U^T : (..., d_out)
        adapter = (x32 @ self.ucv_V) @ self.ucv_C.T @ self.ucv_U.T
        return out + (self.scaling * adapter).to(out.dtype)

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}"


def _split_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_ucv_adapters(
    model: nn.Module,
    target_modules,
    r: int,
    alpha: float | int,
) -> list[tuple[str, UCVLinear]]:
    """Replace nn.Linear modules matching `target_modules` with UCVLinear wrappers.

    Mirrors `collect_dense_target_weights` matching semantics: 'all-linear'
    excludes any module ending in 'lm_head'; explicit lists match by
    suffix-name. Freezes all model parameters except the new ucv_* ones.

    Returns list of (qualified_name, UCVLinear) for the injected modules.
    """
    targets = []
    for name, module in model.named_modules():
        if not name:
            continue
        if target_modules == "all-linear" and name.endswith("lm_head"):
            continue
        if not target_module_matches(name, module, target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        targets.append(name)

    if not targets:
        raise ValueError(f"inject_ucv_adapters: no nn.Linear matched target_modules={target_modules!r}.")

    for p in model.parameters():
        p.requires_grad_(False)

    injected: list[tuple[str, UCVLinear]] = []
    for name in targets:
        parent, attr = _split_parent(model, name)
        base = getattr(parent, attr)
        wrapper = UCVLinear(base, r=r, alpha=alpha)
        setattr(parent, attr, wrapper)
        injected.append((name, wrapper))

    # ucv_* params are leaf parameters on the wrapper; requires_grad is True by default.
    return injected
