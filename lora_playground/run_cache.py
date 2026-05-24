"""Persistent cross-session cache of parsed (cfg, evals) per sweep group.

Source-of-truth remains `logs/<group>/run_info/logs/log_*.out` (JSONL, written
by training). This module persists the already-parsed Python dicts to a single
pickle at `logs/_runs_cache.pkl`, keyed per group by `(name, mtime_ns, size)`
of every source log file (and resume segment). Identical invalidation logic to
the in-process `_LOAD_SWEEP_CACHE` in `plot_utils.py`, just carried across
kernel restarts.

Read path: `get_cached_sweep(group, logs_root)` returns the cached
`list[(cfg, evs)]` if every source file's stat matches the stored signature,
else returns None and the caller should re-parse via the JSONL parser.

Write path: `update_group(group, logs_root, runs, sig)` overwrites a group's
entry; `flush()` writes the whole cache to disk. Cache lives entirely in
memory between flushes — callers may dirty many groups, then flush once.
"""
from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

# Must match `plot_utils._TASK_FILE_RE` exactly. Mismatch → false invalidations.
# Original log file: `log_<idx>.out`; resume segment: `log_<idx>.out.resume_<K>`.
# Anything else (`log_0.out.bak`, `log_0.out~`, editor swap files) is ignored
# by the JSONL parser and must be ignored here too.
_TASK_FILE_RE = re.compile(r"^log_(\d+)\.out(?:\.resume_\d+)?$")

_CACHE_FILENAME = "_runs_cache.pkl"
_CACHE_VERSION = 1  # bump when entry schema changes

# Module-level singleton: maps logs_root abspath → loaded cache dict.
# Cache dict shape:
#   {
#     "version": int,
#     "groups": {
#       group_name: {
#         "sig": tuple[tuple[str, int, int], ...],   # per-file (name, mtime_ns, size)
#         "runs": list[tuple[dict, list[dict]]],     # (cfg, evals) per task
#       }
#     }
#   }
_CACHES: dict[str, dict[str, Any]] = {}
_DIRTY: dict[str, bool] = {}


def _cache_path(logs_root: str) -> Path:
    return Path(logs_root) / _CACHE_FILENAME


def _empty_cache() -> dict[str, Any]:
    return {"version": _CACHE_VERSION, "groups": {}}


def _get_cache(logs_root: str) -> dict[str, Any]:
    """Lazy-load the cache for this logs_root. Returns the module-level
    singleton dict; mutate it via update_group / flush."""
    key = str(Path(logs_root).resolve())
    if key in _CACHES:
        return _CACHES[key]
    path = _cache_path(logs_root)
    if path.exists():
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if not isinstance(cache, dict) or cache.get("version") != _CACHE_VERSION:
                cache = _empty_cache()
        except (pickle.PickleError, EOFError, OSError):
            cache = _empty_cache()
    else:
        cache = _empty_cache()
    _CACHES[key] = cache
    _DIRTY[key] = False
    return cache


def compute_group_sig(
    logs_root: str, group: str
) -> tuple[tuple[str, int, int], ...]:
    """Per-file (name, mtime_ns, size) over every `log_*.out` (incl. resume
    segments) in the group's log dir. Empty tuple if the dir is missing.
    Matches the signature used by `plot_utils._LOAD_SWEEP_CACHE`."""
    log_dir = Path(logs_root) / group / "run_info" / "logs"
    if not log_dir.exists():
        return ()
    files = []
    for p in log_dir.iterdir():
        if not _TASK_FILE_RE.match(p.name):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        files.append((p.name, st.st_mtime_ns, st.st_size))
    files.sort()
    return tuple(files)


def get_cached_sweep(
    group: str, logs_root: str
) -> list[tuple[dict, list[dict]]] | None:
    """Return the cached runs for `group` if its source-file signature
    matches the cache entry; otherwise None. Caller re-parses on None.

    Returned cfgs are deep-copied shallow (one level) so downstream mutation
    (`merge_runs` writing `log_group`, `_enrich_cfg` writing `_derived`)
    doesn't poison the cached entries. evs lists are shared — they are not
    mutated downstream.
    """
    cache = _get_cache(logs_root)
    entry = cache["groups"].get(group)
    if entry is None:
        return None
    current_sig = compute_group_sig(logs_root, group)
    if entry["sig"] != current_sig:
        return None
    return [(dict(cfg), evs) for cfg, evs in entry["runs"]]


def update_group(
    group: str,
    logs_root: str,
    runs: list[tuple[dict, list[dict]]],
    sig: tuple[tuple[str, int, int], ...] | None = None,
) -> None:
    """Replace this group's entry. Marks the cache dirty. Call flush() to
    persist to disk."""
    cache = _get_cache(logs_root)
    if sig is None:
        sig = compute_group_sig(logs_root, group)
    cache["groups"][group] = {
        "sig": sig,
        "runs": [(dict(cfg), evs) for cfg, evs in runs],
    }
    _DIRTY[str(Path(logs_root).resolve())] = True


def drop_group(group: str, logs_root: str) -> None:
    """Remove a group's entry (e.g., after deletion). Idempotent."""
    cache = _get_cache(logs_root)
    if group in cache["groups"]:
        del cache["groups"][group]
        _DIRTY[str(Path(logs_root).resolve())] = True


def is_dirty(logs_root: str) -> bool:
    key = str(Path(logs_root).resolve())
    return _DIRTY.get(key, False)


def flush(logs_root: str) -> bool:
    """Write the in-memory cache to disk if dirty. Returns True if a write
    happened. Atomic via tempfile + os.replace."""
    key = str(Path(logs_root).resolve())
    if not _DIRTY.get(key, False):
        return False
    cache = _CACHES[key]
    path = _cache_path(logs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write — never leave a partially-written cache on disk.
    fd, tmp = tempfile.mkstemp(
        prefix=".runs_cache.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _DIRTY[key] = False
    return True


def reset() -> None:
    """Clear the in-memory singleton (forces reload from disk on next use).
    Test helper; not used in normal flow."""
    _CACHES.clear()
    _DIRTY.clear()
