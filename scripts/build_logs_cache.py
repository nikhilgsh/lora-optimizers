#!/usr/bin/env python3
"""One-shot builder/refresher for `logs/_runs_cache.pkl`.

The cross-session pickle is normally populated lazily on the first
`load_runs()` call (every group whose persistent entry is missing or stale
gets re-parsed from JSONL and written back at flush time). Running this
script ahead of time pre-warms every group so the first notebook call is
near-instant.

Idempotent: re-running only re-parses groups whose source files have changed
since the last build.

Usage:
    python scripts/build_logs_cache.py                  # default logs/
    python scripts/build_logs_cache.py --logs-root path/to/logs
    python scripts/build_logs_cache.py --force          # rebuild every group
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from anywhere; resolve project root from this script's dir.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from lora_playground import run_cache
from lora_playground.manifest import load_manifests, live_manifests_newest_first
from lora_playground.plotting import has_runs, load_sweep
from lora_playground.plotting.loading import _LOAD_SWEEP_CACHE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs-root",
        default=str(_REPO / "logs"),
        help="path to logs/ directory (default: project logs/)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild every group even if the cached sig still matches",
    )
    args = ap.parse_args()

    logs_root = args.logs_root
    if not Path(logs_root).is_dir():
        print(f"error: {logs_root} is not a directory", file=sys.stderr)
        return 2

    if args.force:
        # Drop in-memory mirrors and the on-disk pickle so every group falls
        # back to the JSONL parser.
        run_cache.reset()
        _LOAD_SWEEP_CACHE.clear()
        cache_file = Path(logs_root) / "_runs_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()

    # Snapshot argparse defaults to a JSON sidecar so a fresh kernel can
    # skip the train.py import (torch + transformers, ~17 s).
    from lora_playground.loader import _argparse_defaults, _ARGPARSE_DEFAULTS_CACHE
    if args.force and _ARGPARSE_DEFAULTS_CACHE.exists():
        _ARGPARSE_DEFAULTS_CACHE.unlink()
    _argparse_defaults.cache_clear()
    t0 = time.perf_counter()
    n_defaults = len(_argparse_defaults())
    print(
        f"argparse defaults snapshot ({n_defaults} flags) -> "
        f"{_ARGPARSE_DEFAULTS_CACHE.name} in {time.perf_counter()-t0:.1f}s"
    )

    manifests = load_manifests(logs_root, strict=False)
    groups = [m["group"] for m in live_manifests_newest_first(manifests)]
    print(f"scanning {len(groups)} live groups under {logs_root} ...")

    t0 = time.perf_counter()
    rebuilt, skipped, empty = 0, 0, 0
    for i, group in enumerate(groups):
        if not has_runs(group, logs_root):
            empty += 1
            continue
        # Check if the persistent entry is already fresh — saves us the parse.
        current_sig = run_cache.compute_group_sig(logs_root, group)
        entry = run_cache._get_cache(logs_root)["groups"].get(group)
        if (
            not args.force
            and entry is not None
            and entry["sig"] == current_sig
        ):
            skipped += 1
            continue
        # Force a JSONL re-parse by clearing the in-process cache for this
        # group, then call load_sweep which will write through to run_cache.
        _LOAD_SWEEP_CACHE.pop((logs_root, group), None)
        load_sweep(group, logs_root)
        rebuilt += 1
        if rebuilt % 25 == 0:
            print(f"  rebuilt {rebuilt} groups so far ({i+1}/{len(groups)})")

    wrote = run_cache.flush(logs_root)
    dt = time.perf_counter() - t0
    print(
        f"done in {dt:.1f}s — rebuilt {rebuilt}, up-to-date {skipped}, "
        f"empty {empty}, wrote_cache={wrote}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
