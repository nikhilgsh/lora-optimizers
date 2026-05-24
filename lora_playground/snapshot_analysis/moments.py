"""Adam-moment helpers and spectrum summaries for snapshot analysis.

`stable_rank` lives in `lora_playground.utils` (production); re-exported here
for one-stop imports from the analysis notebooks.
"""
from __future__ import annotations

import numpy as np
import torch

from lora_playground.utils import stable_rank  # noqa: F401  (re-export)

EPS = 1e-8
BETA1 = 0.9
BETA2 = 0.999


def Mtilde(pair: dict, side: str = 'A', bias_correct: bool = False) -> torch.Tensor:
    """Paper's M̃_t (no bc) or our optimizer's u_A / u_B (with bc)."""
    m = pair[f'm_{side}']
    v = pair[f'v_{side}']
    if bias_correct:
        t = pair['step']
        m = m / (1 - BETA1 ** t)
        v = v / (1 - BETA2 ** t)
    return m / (v.sqrt() + EPS)


def normalized_sigmas(mat: torch.Tensor) -> np.ndarray:
    """Singular values, divided by ‖σ‖₂ (Frobenius normalization)."""
    s = torch.linalg.svdvals(mat.float()).numpy()
    return s / (np.linalg.norm(s) + 1e-30)


def normalized_sigmas_x(mat: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (x = i/(r-1), σ_i / ‖σ‖₂) for spectrum overlay plots."""
    s = torch.linalg.svdvals(mat.float()).numpy()
    s_n = s / (np.linalg.norm(s) + 1e-30)
    x = np.arange(len(s)) / max(len(s) - 1, 1)
    return x, s_n
