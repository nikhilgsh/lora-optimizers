"""Damped SPD inverse-sqrt for whitening, plus the analysis-specific
`whitened_NS_input` that operates on a `pair_state` snapshot dict.

`spd_half_inv` is a one-line specialization of
`lora_playground.utils.spd_frac_power_inv(..., gamma=0.5)`.
"""
from __future__ import annotations

import torch

from lora_playground.utils import spd_frac_power_inv

# Absolute damping. Matches run B (--precond_delta 1e-6) but NOT run A
# (which used --precond_delta_relative --precond_delta 1e-2). Held fixed
# for the comparison cells; do not parameterize away without redoing those.
DELTA_ABS = 1e-6


def spd_half_inv(S: torch.Tensor, delta_abs: float = DELTA_ABS) -> torch.Tensor:
    """Damped S^{-1/2} via eigh — thin wrapper over `spd_frac_power_inv`."""
    return spd_frac_power_inv(S, gamma=0.5, eps=delta_abs)


def whitened_NS_input(pair: dict, side: str = 'A',
                      delta_abs: float = DELTA_ABS) -> torch.Tensor:
    """Whitened §9 NS input. Side A: X = S_B^{-1/2} u_A. Side B: X = u_B S_A^{-1/2}."""
    A = pair['A'].float()
    B = pair['B'].float()
    u = pair[f'u_{side}'].float()
    if side == 'A':
        return spd_half_inv(B.T @ B, delta_abs=delta_abs) @ u
    return u @ spd_half_inv(A @ A.T, delta_abs=delta_abs)
