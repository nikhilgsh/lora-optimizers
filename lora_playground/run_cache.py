"""Persistent cross-session cache of parsed (cfg, evals) per sweep group.

Source-of-truth remains `logs/<group>/run_info/logs/log_*.out` (JSONL, written
by training). This module persists the already-parsed Python dicts to a
directory of per-group pickles at `logs/_runs_cache/<group>.pkl`, each keyed by
`(name, mtime_ns, size)` of every source log file (and resume segment).
Identical invalidation logic to the in-process `_LOAD_SWEEP_CACHE` in
`plot_utils.py`, just carried across kernel restarts.

**Why per-group files (not one monolith).** The cache holds hundreds of groups
and grows to hundreds of MB. A single in-flight sweep whose logs change between
two `load_runs()` calls invalidates only that group — but with a monolithic
pickle, re-writing it meant re-dumping the *entire* cache (~7 s for 0.5 GB) on
every analysis call. Per-group files mean `flush()` rewrites only the handful of
groups that actually changed (~ms), and a cold load reads only the groups a cell
touches (lazy), not all of them. The legacy monolith `_runs_cache.pkl` is
migrated to per-group files on first use and then removed.

Read path: `get_cached_sweep(group, logs_root)` returns the cached
`list[(cfg, evs)]` if every source file's stat matches the stored signature,
else returns None and the caller should re-parse via the JSONL parser.

Write path: `update_group(group, logs_root, runs, sig)` overwrites a group's
in-memory entry and marks that group dirty; `flush()` writes every dirty group's
file to disk. Cache lives in memory between flushes — callers may dirty many
groups, then flush once.
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

_CACHE_DIRNAME = "_runs_cache"          # per-group pickle directory
_LEGACY_CACHE_FILENAME = "_runs_cache.pkl"  # old monolithic pickle (migrated away)
_CACHE_VERSION = 1  # bump when entry schema changes

# Per-group entry schema (one pickle per group file):
#   {
#     "version": int,
#     "sig": tuple[tuple[str, int, int], ...],   # per-file (name, mtime_ns, size)
#     "runs": list[tuple[dict, list[dict]]],     # (cfg, evals) per task
#   }

# Module-level singletons keyed by resolved logs_root abspath.
#   _GROUPS[key]:  {group_name: {"sig": ..., "runs": ...}}  — lazily populated;
#                  a group is present only after it's been read or written.
#   _DIRTY[key]:   set of group names written since the last flush.
#   _MIGRATED[key]: True once the legacy-monolith migration has been attempted.
_GROUPS: dict[str, dict[str, dict[str, Any]]] = {}
_DIRTY: dict[str, set[str]] = {}
_MIGRATED: dict[str, bool] = {}


def _key(logs_root: str) -> str:
    return str(Path(logs_root).resolve())


def _cache_dir(logs_root: str) -> Path:
    return Path(logs_root) / _CACHE_DIRNAME


def _group_file(logs_root: str, group: str) -> Path:
    # Group names are themselves log/ subdirectory names, so they are already
    # filesystem-safe. Guard against a stray separator just in case.
    safe = group.replace("/", "__")
    return _cache_dir(logs_root) / f"{safe}.pkl"


def _maps(logs_root: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Return (groups, dirty) for this logs_root, creating empty ones on first
    use and migrating the legacy monolith into per-group files once."""
    key = _key(logs_root)
    if key not in _GROUPS:
        _GROUPS[key] = {}
        _DIRTY[key] = set()
        _MIGRATED[key] = False
    if not _MIGRATED[key]:
        _MIGRATED[key] = True
        _migrate_legacy_monolith(logs_root)
    return _GROUPS[key], _DIRTY[key]


def _migrate_legacy_monolith(logs_root: str) -> None:
    """One-time: split the old `_runs_cache.pkl` into per-group files, then
    delete it. No-op if the monolith is absent or unreadable. Preserves all
    previously-parsed groups so the first post-migration load doesn't re-parse
    every sweep from JSONL."""
    legacy = Path(logs_root) / _LEGACY_CACHE_FILENAME
    if not legacy.exists():
        return
    try:
        with open(legacy, "rb") as f:
            cache = pickle.load(f)
    except (pickle.PickleError, EOFError, OSError):
        return
    if not isinstance(cache, dict) or cache.get("version") != _CACHE_VERSION:
        # Stale schema — just drop the monolith; groups re-parse on demand.
        try:
            legacy.unlink()
        except OSError:
            pass
        return
    cache_dir = _cache_dir(logs_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for group, entry in cache.get("groups", {}).items():
        if not isinstance(entry, dict) or "sig" not in entry or "runs" not in entry:
            continue
        _write_group_file(logs_root, group, entry["sig"], entry["runs"])
    try:
        legacy.unlink()
    except OSError:
        pass


def _write_group_file(
    logs_root: str,
    group: str,
    sig: tuple[tuple[str, int, int], ...],
    runs: list[tuple[dict, list[dict]]],
) -> None:
    """Atomically write one group's pickle. Atomic via tempfile + os.replace —
    never leave a partially-written file on disk."""
    path = _group_file(logs_root, group)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"version": _CACHE_VERSION, "sig": sig, "runs": runs}
    fd, tmp = tempfile.mkstemp(
        prefix=".runs_cache.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_group(logs_root: str, group: str) -> dict[str, Any] | None:
    """Return this group's in-memory entry, lazily reading its per-group file
    from disk on first access. None if neither in memory nor on disk."""
    groups, _dirty = _maps(logs_root)
    if group in groups:
        return groups[group]
    path = _group_file(logs_root, group)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            entry = pickle.load(f)
    except (pickle.PickleError, EOFError, OSError):
        return None
    if (not isinstance(entry, dict) or entry.get("version") != _CACHE_VERSION
            or "sig" not in entry or "runs" not in entry):
        return None
    groups[group] = {"sig": entry["sig"], "runs": entry["runs"]}
    return groups[group]


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
    entry = _load_group(logs_root, group)
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
    """Replace this group's entry and mark the group dirty. Call flush() to
    persist to disk."""
    groups, dirty = _maps(logs_root)
    if sig is None:
        sig = compute_group_sig(logs_root, group)
    groups[group] = {
        "sig": sig,
        "runs": [(dict(cfg), evs) for cfg, evs in runs],
    }
    dirty.add(group)


def drop_group(group: str, logs_root: str) -> None:
    """Remove a group's entry (e.g., after deletion) from memory and disk.
    Idempotent."""
    groups, dirty = _maps(logs_root)
    groups.pop(group, None)
    dirty.discard(group)
    try:
        _group_file(logs_root, group).unlink()
    except OSError:
        pass


def is_dirty(logs_root: str) -> bool:
    return bool(_DIRTY.get(_key(logs_root)))


def flush(logs_root: str) -> bool:
    """Write every dirty group's per-group file to disk. Returns True if any
    write happened. Each file is written atomically."""
    key = _key(logs_root)
    dirty = _DIRTY.get(key)
    if not dirty:
        return False
    groups = _GROUPS[key]
    for group in list(dirty):
        entry = groups.get(group)
        if entry is None:
            continue
        _write_group_file(logs_root, group, entry["sig"], entry["runs"])
    dirty.clear()
    return True


def reset() -> None:
    """Clear the in-memory singletons (forces reload from disk on next use).
    Test helper; not used in normal flow."""
    _GROUPS.clear()
    _DIRTY.clear()
    _MIGRATED.clear()
