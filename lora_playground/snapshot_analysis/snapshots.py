"""Snapshot registry + cached loader for optimizer-state snapshots.

`load_snapshot` is LRU-cached because the same (step, root) is read from
multiple analysis cells; uncached this was the dominant wall-time cost.

Module-level constants (`SNAP_ROOTS`, `STEPS_BY_ROOT`, `RUNS`) are NOT
refreshed by `%autoreload 2` — restart the kernel after editing them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch

SNAP_ROOTS: dict[tuple, Path] = {
    # (lr, lora_r, ns_steps, optimizer-variant short tag)
    ('3e-2', 64, 5, 'chord-tight'):
        Path('/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell/task_0'),
    ('1e-1', 64, 5, 'chord-tight'):
        Path('/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_lr1e-1_blackwell/task_0'),
    ('1e-1', 256, 10, 'chord-tight-clean'):
        Path('/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_clean_snapshot_r256_k3_ns10_lr1e-1_blackwell/task_0'),
    ('1e-3', 256, 10, 'chord-tight-clean'):
        Path('/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_clean_snapshot_r256_k3_ns10_lr1e-3_blackwell/task_0'),
    # chord-tight k=1 (picard_iters_override=1, ns_steps=5) — OPC-1B
    # leaderboard best-η for the ns+polar baseline arm at r=64 and r=256.
    ('3e-2', 64, 5, 'chord-tight-k1'):
        Path('/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k1_snapshot_blackwell/task_0'),
    ('3e-2', 256, 5, 'chord-tight-k1'):
        Path('/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r256_k1_snapshot_blackwell/task_0'),
}

# Back-compat single-run entry: the lr=3e-2 r=64 run.
SNAP_ROOT: Path = SNAP_ROOTS[('3e-2', 64, 5, 'chord-tight')]

STEPS_BY_ROOT: dict[tuple, list[int]] = {
    key: sorted(int(p.name.split('_')[1]) for p in root.glob('step_*'))
    for key, root in SNAP_ROOTS.items() if root.exists()
}

STEPS: list[int] = STEPS_BY_ROOT.get(('3e-2', 64, 5, 'chord-tight'), [])

# Named handles for the four canonical runs (matching the original notebook).
RUN_A = ('3e-2', 64, 5, 'chord-tight')         # r=64  low  lr   (k=3)
RUN_C = ('1e-1', 64, 5, 'chord-tight')         # r=64  high lr   (k=3)
RUN_D = ('1e-3', 256, 10, 'chord-tight-clean')  # r=256 low  lr  (clean k=3)
RUN_B = ('1e-1', 256, 10, 'chord-tight-clean')  # r=256 high lr  (clean k=3)
RUN_E = ('3e-2', 64, 5, 'chord-tight-k1')      # r=64  best-η    (k=1)
RUN_F = ('3e-2', 256, 5, 'chord-tight-k1')     # r=256 best-η    (k=1)

RUNS: list[tuple] = [
    r for r in (RUN_A, RUN_C, RUN_D, RUN_B, RUN_E, RUN_F) if SNAP_ROOTS[r].exists()
]


@lru_cache(maxsize=32)
def _load_snapshot_cached(step: int, root_str: str) -> dict:
    root = Path(root_str)
    sd = torch.load(root / f'step_{step}' / 'optimizer.pt',
                    map_location='cpu', weights_only=False)
    meta = json.loads((root / f'step_{step}' / 'meta.json').read_text())
    return {
        'pair_state': sd['pair_state'],
        'group_state': sd.get('group_state', []),
        'meta': meta,
    }


def load_snapshot(step: int, root: Path | str = SNAP_ROOT) -> dict:
    """Load one snapshot. Cached by (step, root) — repeated calls are free."""
    return _load_snapshot_cached(int(step), str(root))


def clear_snapshot_cache() -> None:
    """Drop the LRU cache (memory pressure / sanity)."""
    _load_snapshot_cached.cache_clear()
