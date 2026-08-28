"""Per-task log loading: parse JSONL `.out` files into (cfg, evs) tuples,
select one physical segment per task, and cache the result in-process.

The two top-level entry points are `load_run(path)` (single file) and
`load_sweep(group, logs_root)` (all tasks in one group). `has_runs(group)` is
the cheap existence check used to skip empty groups in audit code.

Directory scanning goes through :func:`scan_group`, which reads a group's log
directory with ONE ``os.scandir`` and one ``stat`` per log file. Everything that
needs the file list or the freshness signature — ``has_runs`` and
``load_sweep`` — consumes that single scan instead of walking the directory
itself. Inside a :func:`scan_epoch` block the scan is memoised per
group, so a whole-tree pass costs one ``scandir`` per group rather than three,
and :func:`prescan_groups` issues them concurrently (metadata ops on GPFS are
latency-bound, and ``os.scandir``/``os.stat`` release the GIL).
"""
from __future__ import annotations

import os
import re
import shlex
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

from ..run_parsing import clear_run_file_cache, parse_run_file


@lru_cache(maxsize=None)
def _split_command_cached(command: str) -> tuple[str, ...]:
    """Memoized ``shlex.split(command)``.

    ``parse_flag`` is called many times (once per CLI kwarg) on the SAME
    ``command`` string — ``_backfill_optimizer_config`` alone calls it up to
    15x per run. ``shlex.split`` re-tokenizes the whole string from scratch
    each time, which measured as the dominant cost of `load_runs` (>1.5s of a
    ~3.5s call, profiled at 46k `shlex` token reads for one variant's worth of
    runs). Command strings are immutable log content, so caching by string
    value is exact — this changes no returned value, only how many times the
    tokenization work happens. Unbounded cache: distinct command strings
    across the whole `logs/` tree number in the low thousands, not enough to
    matter for memory.
    """
    return tuple(shlex.split(command))


def parse_flag(command: str, flag: str) -> str | None:
    """Extract --flag VALUE from a command string."""
    parts = _split_command_cached(command)
    for i, p in enumerate(parts):
        if p == flag and i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_run(log_path: Path) -> tuple[dict | None, list[dict]]:
    """Parse one physical task log without reconstructing configuration.

    optim_step diagnostic events (emitted by polar/lin/scaled-LoRA optimizers
    when --log_basic_diagnostics is on) are attached to the config dict as
    ``cfg["_optim_steps"]``. Both ``loader.RUNTIME_FIELDS`` and the
    ``RUNTIME_FIELDS`` re-exported from this package list it, so dedup
    ignores it.

    Neutral parsing is cached by (path, mtime, size). Missing producer fields
    remain missing: command strings and current defaults are not evidence about
    what the run executed.
    """
    parsed = parse_run_file(log_path)
    return parsed.raw_config(), parsed.mutable_evals()


_LOAD_SWEEP_CACHE: dict[tuple[str, str], tuple[tuple, list[tuple[dict, list[dict]]]]] = {}
_TASK_FILE_RE = re.compile(r"^log_(\d+)\.out(?:\.resume_\d+)?$")


# ─── directory scanning ───────────────────────────────────────────────────────

class GroupScan(NamedTuple):
    """One group's log directory, read once.

    exists:   the `run_info/logs` dir is there at all.
    tasks:    task index → filenames, `.out` first then `.resume_K` ascending.
    sig:      per-file `(name, mtime_ns, size)`, sorted — the freshness key for
              the in-process cache.
    nonempty: at least one matching file with size > 0 (what `has_runs` asks).
    """
    exists: bool
    tasks: dict[str, tuple[str, ...]]
    sig: tuple[tuple[str, int, int], ...]
    nonempty: bool


def _empty_scan() -> GroupScan:
    """A fresh "no such directory" scan. Fresh rather than a shared singleton
    so nothing can mutate one caller's `tasks` dict into another's."""
    return GroupScan(False, {}, (), False)


# Per-epoch scan memo, or None outside an epoch. Keyed (logs_root, group).
# Deliberately NOT a process-lifetime cache: a running sweep appends to a log
# file without changing its directory's mtime, so the only correct freshness
# key is the per-file stat itself. The epoch bounds how long a scan may be
# reused to a single top-level operation (one `load_runs` / `inventory_runs` /
# `merge_runs` call), inside which the tree is read exactly once.
_SCAN_EPOCH: dict[tuple[str, str], GroupScan] | None = None


def _segment_sort_key(name: str) -> tuple[int, int]:
    """`log_NN.out` before `log_NN.out.resume_K`, resumes by K ascending."""
    if name.endswith(".out"):
        return (0, 0)
    return (1, int(name.rsplit(".resume_", 1)[1]))


def _scan_group_uncached(group: str, logs_root: str) -> GroupScan:
    log_dir = os.path.join(logs_root, group, "run_info", "logs")
    tasks: dict[str, list[str]] = {}
    sig_items: list[tuple[str, int, int]] = []
    nonempty = False
    try:
        it = os.scandir(log_dir)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return _empty_scan()
    with it:
        for entry in it:
            m = _TASK_FILE_RE.match(entry.name)
            if not m:
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            sig_items.append((entry.name, st.st_mtime_ns, st.st_size))
            if st.st_size > 0:
                nonempty = True
            tasks.setdefault(m.group(1), []).append(entry.name)
    for names in tasks.values():
        names.sort(key=_segment_sort_key)
    sig_items.sort()
    return GroupScan(True, {k: tuple(v) for k, v in tasks.items()},
                     tuple(sig_items), nonempty)


@contextmanager
def scan_epoch():
    """Reuse each group's directory scan for the duration of the block.

    Nests harmlessly (an inner block joins the outer epoch). Outside any epoch
    `scan_group` re-reads the directory on every call, which is the behaviour
    every direct caller of `load_sweep` / `has_runs` had before.
    """
    global _SCAN_EPOCH
    if _SCAN_EPOCH is not None:
        yield
        return
    _SCAN_EPOCH = {}
    try:
        yield
    finally:
        _SCAN_EPOCH = None


def scan_group(group: str, logs_root: str = "../logs") -> GroupScan:
    """Read one group's log directory (memoised within a `scan_epoch`)."""
    epoch = _SCAN_EPOCH
    if epoch is None:
        return _scan_group_uncached(group, logs_root)
    key = (logs_root, group)
    hit = epoch.get(key)
    if hit is None:
        hit = epoch[key] = _scan_group_uncached(group, logs_root)
    return hit


def scan_workers() -> int:
    """Thread count for `prescan_groups`. `LORA_SCAN_WORKERS=1` disables the
    pool (useful when bisecting a scan-related failure).

    Kept modest on purpose: this runs inside notebooks and analysis scripts
    that sit next to torch processes, and the login/workstation cgroup caps
    total pids (512 here). The win is over an I/O latency of ~1 ms per
    directory, so eight in flight already recovers most of it.
    """
    try:
        return max(1, int(os.environ.get("LORA_SCAN_WORKERS", "8")))
    except ValueError:
        return 8


def parallel_map(fn, items: list):
    """``[fn(x) for x in items]``, concurrently when that is available.

    Runs serially for a single item and under ``LORA_SCAN_WORKERS=1``. If the
    interpreter refuses to start a thread — a shared box can sit at its pid cap
    — the items not yet submitted are done inline: the work already queued
    still runs on the threads that exist, so a mid-batch failure costs nothing
    beyond losing the concurrency. A metadata prefetch is never worth failing a
    load over, and it is never worth doing twice either.
    """
    n_workers = min(scan_workers(), len(items))
    if n_workers <= 1:
        return [fn(x) for x in items]
    try:
        ex = ThreadPoolExecutor(max_workers=n_workers,
                                thread_name_prefix="lora-scan")
    except RuntimeError:
        return [fn(x) for x in items]
    try:
        futures = []
        inline: list = []
        for i, item in enumerate(items):
            try:
                futures.append(ex.submit(fn, item))
            except RuntimeError:
                inline = items[i:]
                break
        return [f.result() for f in futures] + [fn(x) for x in inline]
    finally:
        ex.shutdown(wait=True)


def prescan_groups(groups, logs_root: str = "../logs") -> None:
    """Populate the current epoch's scans for `groups` concurrently.

    No-op outside a `scan_epoch` (there would be nowhere to put the results).
    Pure prefetch: every value it stores is what `scan_group` would have
    computed on demand, so skipping it changes only speed.
    """
    epoch = _SCAN_EPOCH
    if epoch is None:
        return
    todo = [g for g in groups if (logs_root, g) not in epoch]
    results = parallel_map(lambda g: _scan_group_uncached(g, logs_root), todo)
    for g, res in zip(todo, results):
        epoch[(logs_root, g)] = res


def _load_sweep_cached(group: str, logs_root: str) -> list[tuple[dict, list[dict]]]:
    """Shared implementation of `load_sweep` returning the CACHED list.

    The cfg dicts belong to the cache — callers must copy before mutating.
    `load_sweep` is the copying public wrapper; `iter_sweep_raw` is the
    no-copy one for callers that filter first and copy only survivors.
    """
    scan = scan_group(group, logs_root)
    if not scan.exists:
        return []
    sig = scan.sig
    cache_key = (logs_root, group)
    cached = _LOAD_SWEEP_CACHE.get(cache_key)
    if cached is not None and cached[0] == sig:
        return cached[1]

    log_dir = Path(logs_root) / group / "run_info" / "logs"
    runs = []
    for task_idx in sorted(scan.tasks):
        # Filename adjacency is not lineage evidence.  Keep this compatibility
        # API at one result per task, but select one physical file rather than
        # concatenating ``.resume_K`` siblings.  Explicit, versioned attempts
        # are the sole province of ``RunCatalog.resolve_lineages``.
        names = scan.tasks[task_idx]
        physical = []
        for name in names:
            cfg, evs = load_run(log_dir / name)
            if cfg is None or not evs:
                continue
            steps = [
                event.get("step")
                for event in evs
                if isinstance(event.get("step"), (int, float))
                and not isinstance(event.get("step"), bool)
            ]
            final_step = max(steps, default=-1)
            score = (
                final_step,
                len(evs),
                name.endswith(".out"),
                name,
            )
            physical.append((score, cfg, evs))
        if physical:
            _score, cfg, evs = max(physical, key=lambda item: item[0])
            runs.append((cfg, evs))
    _LOAD_SWEEP_CACHE[cache_key] = (sig, runs)
    return runs


def load_sweep(group: str, logs_root: str = "../logs") -> list[tuple[dict, list[dict]]]:
    """Load all runs for a sweep group. Returns list of (cfg, evs).

    A "run" is one disBatch task index. When a task has `log_NN.out` and
    `log_NN.out.resume_K` siblings, exactly one physical file is selected by
    final eval step, eval count, canonical `.out` preference, then filename.
    Histories are never concatenated from filename proximity. Consumers that
    need a multi-attempt history must use explicit, versioned lineage through
    :class:`lora_playground.run_catalog.RunCatalog`.

    Cached in the in-process module dict (`_LOAD_SWEEP_CACHE`), invalidated by
    the per-file (name, mtime, size) signature of the group's log files.

    Returned cfgs are shallow copies because compatibility consumers such as
    `merge_runs` attach view-local fields; the cached dicts must stay clean.
    """
    return [(dict(cfg), evs) for cfg, evs in _load_sweep_cached(group, logs_root)]


def iter_sweep_raw(group: str, logs_root: str = "../logs"
                   ) -> Iterator[tuple[dict, list[dict]]]:
    """`load_sweep` without the per-cfg copy — for callers that reject most
    runs and want to pay the copy only for the ones they keep.

    The yielded cfg dicts are the cache's own objects. **Copy before mutating**
    (`merge_runs` does exactly that, immediately after its pre-filter passes).
    """
    return iter(_load_sweep_cached(group, logs_root))


def clear_run_caches() -> None:
    """Clear both in-process loader caches (per-file + per-sweep).

    Useful as the first cell of an analysis notebook after a loader code
    change (e.g. new CLI-flag backfill) — file mtimes haven't changed, so
    the (mtime, size)-keyed entries are stale even though the parser logic
    has moved.
    """
    clear_run_file_cache()
    _LOAD_SWEEP_CACHE.clear()
    if _SCAN_EPOCH is not None:
        _SCAN_EPOCH.clear()


def has_runs(group: str, logs_root: str = "../logs") -> bool:
    """True if the group has at least one populated `log_NN.out` (or any
    `log_NN.out.resume_K` sibling)."""
    return scan_group(group, logs_root).nonempty


def load_optim_step_diagnostics(
    where: dict,
    fields: Sequence[str] | None = None,
    *,
    logs_root: str | Path | None = None,
    catalog=None,
):
    """Per-step optimizer diagnostics for the runs a predicate selects.

    Returns ``{run_id: {field: numpy array}}``, one entry per selected run that
    logged ``optim_step`` events, aligned on the rows that carry every
    requested field. ``fields=None`` returns every scalar field present, plus
    ``step``.

    Why this exists. ``run_catalog`` parses with ``include_optim_steps=False``
    (`run_catalog.py:337`) because the leaderboard and figure paths never read
    those rows and skipping them is the fast path, so ``load_runs`` never
    yields ``cfg["_optim_steps"]``. Notebook cells therefore reached past the
    loader entirely and re-implemented log discovery with ``glob``, JSON-line
    decoding, and per-learning-rate assembly -- three things the library
    already does -- keyed on HAND-TYPED group names, which drift from the
    manifest and silently miss data.

    Selection here is by predicate, exactly as `loader.load_runs` takes it, so
    a renamed or added sweep is picked up rather than missed. Only the selected
    logs are re-parsed with the diagnostic rows included, so the fast path
    stays fast for everything else.
    """
    import numpy as np

    from ..loader import load_runs

    runs = load_runs(where=where, catalog=catalog, quiet=True,
                     warn_cross_commit=False)
    root = Path(logs_root) if logs_root is not None else \
        Path(__file__).resolve().parents[2] / "logs"

    out = {}
    for cfg, _history in runs:
        group, filename = cfg.get("log_group"), cfg.get("_log_filename")
        if not group or not filename:
            continue
        path = root / group / "run_info" / "logs" / filename
        if not path.is_file():
            continue
        rows = parse_run_file(path, include_optim_steps=True).optim_steps
        if not rows:
            continue
        wanted = list(fields) if fields else sorted(
            {k for row in rows for k, v in row.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
        )
        if "step" not in wanted:
            wanted = ["step", *wanted]
        usable = [r for r in rows if all(r.get(f) is not None for f in wanted)]
        if not usable:
            continue
        out[cfg.get("run_id") or f"{group}/{filename}"] = {
            f: np.array([r[f] for r in usable], dtype=float) for f in wanted
        }
    return out
