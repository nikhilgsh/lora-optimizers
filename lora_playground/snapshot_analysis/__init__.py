"""Snapshot-analysis helpers for `notebooks/snapshot_analysis/`.

This is a thin facade — the real implementations live where they belong:

  * Polar iterations (`_newton_schulz`, `_polar_express`) — `lora_playground.optim`
  * SSC primitives (`_ssc_svd`, `_ssc_misr_batched`)         — `lora_playground.optim`
  * SPD inverse-sqrt, truncated SVD                          — `lora_playground.utils`
  * Power iter for σ_max / λ_max                             — `lora_playground.spectral`

This package only carries the *snapshot-specific* glue:

  * snapshot registry + LRU-cached loader (`snapshots.py`)
  * adam-moment / spectrum summaries (`moments.py`)
  * damped eigh-based whitening (`whitening.py`)
  * pre-rescale + polar-UV^T conveniences (`ssc.py`)
  * heavy diagnostic sweeps (`calibration.py`)

Module-level constants (`SNAP_ROOTS`, `STEPS_BY_ROOT`, `RUNS`, `DELTA_ABS`,
`BETA1/BETA2`, `EPS`) are NOT refreshed by `%autoreload 2`; restart the kernel
after editing this package.
"""
from __future__ import annotations

# Re-export polar iterations from the main library under friendlier aliases.
from lora_playground.optim import _newton_schulz as newton_schulz_polar
from lora_playground.optim import _polar_express as polar_express

from .moments import (
    BETA1,
    BETA2,
    EPS,
    Mtilde,
    normalized_sigmas,
    normalized_sigmas_x,
    stable_rank,
)
from .snapshots import (
    RUN_A,
    RUN_B,
    RUN_C,
    RUN_D,
    RUN_E,
    RUN_F,
    RUNS,
    SNAP_ROOT,
    SNAP_ROOTS,
    STEPS,
    STEPS_BY_ROOT,
    clear_snapshot_cache,
    load_snapshot,
)
from .ssc import _polar_uvt, _prerescale_unit_op, _ssc_misr_batched, _ssc_svd
from .whitening import DELTA_ABS, spd_half_inv, whitened_NS_input

__all__ = [
    # snapshots / registry
    'SNAP_ROOT', 'SNAP_ROOTS', 'STEPS', 'STEPS_BY_ROOT',
    'RUNS', 'RUN_A', 'RUN_B', 'RUN_C', 'RUN_D', 'RUN_E', 'RUN_F',
    'load_snapshot', 'clear_snapshot_cache',
    # moments
    'Mtilde', 'normalized_sigmas', 'normalized_sigmas_x', 'stable_rank',
    'EPS', 'BETA1', 'BETA2',
    # whitening
    'spd_half_inv', 'whitened_NS_input', 'DELTA_ABS',
    # ssc / polar shape
    '_prerescale_unit_op', '_polar_uvt', '_ssc_svd', '_ssc_misr_batched',
    # polar iterations (aliases for `lora_playground.optim.*`)
    'newton_schulz_polar', 'polar_express',
]
