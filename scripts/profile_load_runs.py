#!/usr/bin/env python3
"""Profile the loader's read path. Runs each notebook-style query and
reports phase-by-phase timings: pickle load, signature checks, JSONL fallback
parses, enrichment, exclusion-registry, end-to-end.

Usage:
    python scripts/profile_load_runs.py                 # default logs/
    python scripts/profile_load_runs.py --logs-root path
    python scripts/profile_load_runs.py --cprofile      # cumulative cProfile table
    python scripts/profile_load_runs.py --bench         # repeat to measure warm cache
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pickle
import pstats
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _fresh_imports():
    """Reload loader modules so timings reflect a cold first call (caches
    cleared, no JIT-cached argparse defaults)."""
    import importlib

    import lora_playground.run_cache as rc
    import lora_playground.plotting.loading as pl

    rc.reset()
    pl._LOAD_RUN_CACHE.clear()
    pl._LOAD_SWEEP_CACHE.clear()
    # Manifest cache lives at module level.
    import lora_playground.manifest as mf
    mf._LOAD_MANIFESTS_CACHE.clear()


def _pickle_stats(logs_root: str) -> None:
    path = Path(logs_root) / "_runs_cache.pkl"
    if not path.exists():
        print(f"[cache] {path} does not exist")
        return
    sz = path.stat().st_size
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        c = pickle.load(f)
    dt = time.perf_counter() - t0
    n_groups = len(c.get("groups", {}))
    n_runs = sum(len(e["runs"]) for e in c["groups"].values())
    n_evals = sum(
        len(evs)
        for e in c["groups"].values()
        for _, evs in e["runs"]
    )
    print(
        f"[cache] {path.name}: {sz/1e6:.1f} MB, "
        f"{n_groups} groups / {n_runs} runs / {n_evals} eval events. "
        f"pickle.load: {dt*1000:.0f} ms"
    )


def _phase_timings(logs_root: str) -> None:
    """Decompose load_runs into its measurable phases."""
    from lora_playground import run_cache
    from lora_playground.manifest import (
        live_manifests_newest_first,
        load_manifests,
    )
    from lora_playground.plotting import has_runs, load_sweep
    from lora_playground.plotting.loading import _LOAD_SWEEP_CACHE

    print("\n--- phase timings (cold) ---")

    t0 = time.perf_counter()
    manifests = load_manifests(logs_root, strict=False)
    groups = [m["group"] for m in live_manifests_newest_first(manifests)]
    print(f"load_manifests           : {(time.perf_counter()-t0)*1000:>7.0f} ms  ({len(groups)} live groups)")

    # Pickle is loaded lazily on first run_cache call. Force it.
    t0 = time.perf_counter()
    cache = run_cache._get_cache(logs_root)
    print(f"persistent pickle load   : {(time.perf_counter()-t0)*1000:>7.0f} ms  ({len(cache['groups'])} cached groups)")

    # Stat every group's source files to check freshness.
    t0 = time.perf_counter()
    sig_mismatches = 0
    for g in groups:
        stored = cache["groups"].get(g)
        current = run_cache.compute_group_sig(logs_root, g)
        if stored is None or stored["sig"] != current:
            sig_mismatches += 1
    dt = time.perf_counter() - t0
    print(f"sig check (all groups)   : {dt*1000:>7.0f} ms  ({sig_mismatches} mismatches → JSONL reparse needed)")

    # Time the actual sweep loop (this includes JSONL fallback for misses
    # but those should be 0 if the cache is warm).
    t0 = time.perf_counter()
    n_runs = 0
    for g in groups:
        if not has_runs(g, logs_root):
            continue
        runs = load_sweep(g, logs_root)
        n_runs += len(runs)
    print(f"load_sweep × all groups  : {(time.perf_counter()-t0)*1000:>7.0f} ms  ({n_runs} total runs)")


def _query_timings(logs_root: str) -> None:
    """Time the notebook's actual queries."""
    from lora_playground.loader import inventory_runs, load_runs

    print("\n--- end-to-end queries ---")

    queries = [
        ("inventory_runs()", lambda: inventory_runs(logs_root)),
        ("load_runs(packed_v1)", lambda: load_runs(
            where={"data_pipeline_version": "packed_v1"},
            logs_root=logs_root, warn_cross_commit=False,
        )),
        ("load_runs(packed_v1, optimizer=adamw)", lambda: load_runs(
            where={"data_pipeline_version": "packed_v1", "optimizer": "adamw"},
            logs_root=logs_root, warn_cross_commit=False,
        )),
        ("load_runs(chord-tight-clean, r=64, picard=3, ns=10)", lambda: load_runs(
            where={
                "optimizer": "adam-polar-product-lora-coupled-spectral-chord-tight-clean",
                "data_pipeline_version": "packed_v1",
                "lora_r": 64, "max_steps": 4000, "seed": 0,
                "effective_picard_iters": 3, "muon_ns_steps": 10,
            },
            logs_root=logs_root, warn_cross_commit=False,
        )),
    ]
    for label, fn in queries:
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        n = len(out) if hasattr(out, "__len__") else "n/a"
        print(f"{label:<55s} {dt*1000:>7.0f} ms  ({n} results)")


def _cprofile_full(logs_root: str) -> None:
    print("\n--- cProfile (cumulative) — load_runs(packed_v1) ---")
    from lora_playground.loader import load_runs
    pr = cProfile.Profile()
    pr.enable()
    load_runs(
        where={"data_pipeline_version": "packed_v1"},
        logs_root=logs_root, warn_cross_commit=False,
    )
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
    print(buf.getvalue())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", default=str(_REPO / "logs"))
    ap.add_argument("--cprofile", action="store_true",
                    help="also print a cumulative cProfile table")
    ap.add_argument("--bench", action="store_true",
                    help="repeat the queries to measure warm-cache speed")
    args = ap.parse_args()

    print(f"logs_root = {args.logs_root}")
    _pickle_stats(args.logs_root)
    _phase_timings(args.logs_root)
    _query_timings(args.logs_root)

    if args.bench:
        print("\n=== warm-cache repeat ===")
        _query_timings(args.logs_root)

    if args.cprofile:
        _fresh_imports()
        _cprofile_full(args.logs_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
