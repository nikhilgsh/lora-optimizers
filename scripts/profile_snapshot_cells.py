#!/usr/bin/env python
"""Ranked timing report for the heavy snapshot-analysis helpers.

Run from the repo root:

    conda run -n ffcv-pl python scripts/profile_snapshot_cells.py

Times each diagnostic against a small but realistic input (one step, all pairs)
and reports wall in a ranked table. Also confirms the `load_snapshot` LRU cache
buys what we think it buys (cold vs warm).

The script does NOT auto-poll or sweep — it's a one-shot measurement. Re-run
when the heavy helpers change to refresh the numbers in
`docs/notes/polar_product/` if needed.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import torch

from lora_playground.snapshot_analysis import (
    RUNS,
    SNAP_ROOTS,
    STEPS_BY_ROOT,
    clear_snapshot_cache,
    load_snapshot,
)
from lora_playground.snapshot_analysis.calibration import (
    collect_bcd_convergence,
    higham_residual_traj,
    kappa_calibration,
    ssc_calibration,
)


DEFAULT_STEP = 2000


@contextmanager
def timed(label: str, results: list):
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    results.append((label, wall))


def profile_load_snapshot_cache(step: int = DEFAULT_STEP):
    """Cold vs warm `load_snapshot` wall."""
    print('=== load_snapshot cold-vs-warm ===')
    for run_key in RUNS:
        root = SNAP_ROOTS[run_key]
        if step not in STEPS_BY_ROOT.get(run_key, []):
            continue
        clear_snapshot_cache()
        t0 = time.perf_counter()
        _ = load_snapshot(step, root=root)
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = load_snapshot(step, root=root)
        warm = time.perf_counter() - t0
        ratio = cold / max(warm, 1e-9)
        print(f'  {run_key} step={step}:  cold={cold * 1e3:7.1f} ms  '
              f'warm={warm * 1e6:7.1f} μs  ratio={ratio:.0f}x')


def profile_heavy_helpers(step: int = DEFAULT_STEP) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []

    # Pre-warm the snapshot cache so we time the compute, not the I/O.
    clear_snapshot_cache()
    for run_key in RUNS:
        if step in STEPS_BY_ROOT.get(run_key, []):
            load_snapshot(step, root=SNAP_ROOTS[run_key])

    print()
    print('=== heavy helpers (snapshot cache pre-warmed) ===')

    with timed('ssc_calibration (steps=(2000,), n_pairs=12, default run)', results):
        ssc_calibration(steps=(step,), n_pairs=12)

    with timed('kappa_calibration (steps=(2000,), n_pairs=12, all runs)', results):
        kappa_calibration(steps=(step,), n_pairs=12)

    with timed('collect_bcd_convergence (step=2000, K=10, all pairs)', results):
        collect_bcd_convergence(step, K=10)

    # Higham on every pair's S_B at one step.
    snap = load_snapshot(step)
    with timed(f'higham_residual_traj × {len(snap["pair_state"])} pairs', results):
        for pi, p in snap['pair_state'].items():
            B = p['B'].float()
            higham_residual_traj(B.T @ B, K_max=20)

    return results


def main():
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'active RUNS ({len(RUNS)}):')
    for r in RUNS:
        print(f'  {r}')

    profile_load_snapshot_cache(DEFAULT_STEP)
    results = profile_heavy_helpers(DEFAULT_STEP)

    print()
    print('=== ranked wall times (slowest first) ===')
    results.sort(key=lambda kv: -kv[1])
    width = max(len(lbl) for lbl, _ in results)
    for lbl, wall in results:
        print(f'  {lbl:<{width}}   {wall * 1e3:>9.1f} ms')


if __name__ == '__main__':
    main()
