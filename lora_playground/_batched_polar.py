"""Batched polar-pipeline primitives.

Each of these is the launch-friendly counterpart of an op currently
inlined in `AdamPolarProductLoRA.step()` / `_polar_pipeline()`. They
take stacked tensors (..., m, n) instead of a per-pair loop.

Microbench provenance (all on RTX A6000, OLMo-2-1B at r=16):
- `_newton_schulz_batched`        — 5.3× total / 24× on dominant group
- `spd_inv_sqrt_higham_batched`   — 40-125× (eliminates eigh launch storm)
- `unwhiten_rescale_batched`      — 19× total / 33× on dominant group

NOT batched here because the standalone bench showed no win:
- Picard cross-coupling (`B^T dB A`, `B dA A^T` chains) — large-d contractions
  are compute-bound per-pair; batching adds bmm overhead without compressing
  real launches. Speedup measured at 0.97× (a wash). Keep per-pair loop.

`_newton_schulz_batched` lives in `optim.py` (with the loop NS for proximity);
`spd_inv_sqrt_higham_batched` lives in `utils.py` (with the loop higham). This
module collects the rest plus a NS re-export so integration code has one
import surface.
"""
from __future__ import annotations

import torch

from .optim import _newton_schulz_batched
from .utils import spd_inv_sqrt_higham_batched

__all__ = [
    "_newton_schulz_batched",
    "spd_inv_sqrt_higham_batched",
    "unwhiten_rescale_frob_batched",
]


def unwhiten_rescale_frob_batched(P_A, P_B, SA_half_inv, SB_half_inv,
                                  u_A, u_B, lr, lora_plus_multiplier=1.0):
    """Batched unwhiten + Muon+ frob-mode rescale.

    Mirrors the polar_norm_dir='frob' branch of `_polar_pipeline`'s tail
    (the default and only path that production uses). Other Muon+ modes
    (row, col, row_col, col_row) need axis-aware norms; not implemented
    here because no production sweep uses them.

    Inputs (all stacked over a shape-group of N pairs):
      P_A         : (N, r, d_in)         polar(c_A)
      P_B         : (N, d_out, r)        polar(c_B)
      SA_half_inv : (N, r, r)            (A A^T + δI)^{-1/2}
      SB_half_inv : (N, r, r)            (B^T B + δI)^{-1/2}
      u_A         : (N, r, d_in)         Adam direction on A side
      u_B         : (N, d_out, r)        Adam direction on B side
      lr          : float                learning rate
      lora_plus_multiplier : float       LoRA+ B-side scaling (default 1.0)

    Returns:
      dA : (N, r, d_in)
      dB : (N, d_out, r)

    Equivalence (vs per-pair loop): 1.4e-9 max abs err on real shapes.
    """
    geo_A = SB_half_inv @ P_A                          # (N, r, d_in)
    geo_B = P_B @ SA_half_inv                          # (N, d_out, r)
    uA_norm = u_A.flatten(-2).norm(dim=-1)             # (N,)
    uB_norm = u_B.flatten(-2).norm(dim=-1)             # (N,)
    gA_norm = geo_A.flatten(-2).norm(dim=-1) + 1e-30
    gB_norm = geo_B.flatten(-2).norm(dim=-1) + 1e-30
    sA = (-lr * uA_norm / gA_norm).unsqueeze(-1).unsqueeze(-1)
    sB = (-lora_plus_multiplier * lr * uB_norm / gB_norm).unsqueeze(-1).unsqueeze(-1)
    return sA * geo_A, sB * geo_B
