"""Per-task log loading: parse JSONL `.out` files into (cfg, evs) tuples,
merge resume segments, and cache the result both in-process and persistently.

The two top-level entry points are `load_run(path)` (single file) and
`load_sweep(group, logs_root)` (all tasks in one group). `has_runs(group)` is
the cheap existence check used to skip empty groups in audit code.
"""
from __future__ import annotations

import json
import re
import shlex
from functools import lru_cache
from pathlib import Path


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


def _coerce_value(v: str):
    """String → int/float/bool when possible; otherwise leave as str.
    Used by parse_cli_command and CLI backfill so analysis code can filter
    on numeric values without per-flag handling."""
    if v in ("True", "true"): return True
    if v in ("False", "false"): return False
    if v in ("None", "none", "null"): return None
    try: return int(v)
    except ValueError: pass
    try: return float(v)
    except ValueError: pass
    return v


def parse_cli_command(command: str) -> dict:
    """Parse a logged ``--flag value`` (and ``--bool-flag``) command into a dict.

    Generic backfill source for the loader: turns the full launcher command
    line into ``{flag_name: typed_value}``. Boolean flags (``store_true`` /
    ``store_false`` argparse actions) become ``True`` when present. Values are
    coerced via :func:`_coerce_value`.
    """
    parts = shlex.split(command)
    out: dict = {}
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok.startswith("--"):
            key = tok[2:]
            nxt = parts[i + 1] if i + 1 < len(parts) else None
            if nxt is None or nxt.startswith("--"):
                out[key] = True
                i += 1
            else:
                out[key] = _coerce_value(nxt)
                i += 2
        else:
            i += 1
    return out


_LOAD_RUN_CACHE: dict[str, tuple[tuple[int, int], dict | None, list[dict]]] = {}


def load_run(log_path: Path) -> tuple[dict | None, list[dict]]:
    """Parse a single task .out file → (config dict, list of eval dicts).

    optim_step diagnostic events (emitted by polar/lin/scaled-LoRA optimizers
    when --log_basic_diagnostics is on) are attached to the config dict as
    ``cfg["_optim_steps"]``. Both ``loader.RUNTIME_FIELDS`` and the
    ``RUNTIME_FIELDS`` re-exported from this package list it, so dedup
    ignores it.

    Result is cached by (path, mtime, size); an in-flight file's growing size
    invalidates the entry so re-parses pick up new events. Returned cfg is
    shallow-copied per call because merge_runs mutates ``cfg["log_group"]``
    and downstream enrichment writes ``cfg["_derived"]``; the cached cfg
    must stay clean across callers.
    """
    path_str = str(log_path)
    try:
        st = log_path.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None
    cached = _LOAD_RUN_CACHE.get(path_str) if sig is not None else None
    if cached is not None and cached[0] == sig:
        cfg, evs = cached[1], cached[2]
        return (None if cfg is None else dict(cfg)), evs

    config, evals, optim_steps = None, [], []
    init_override_mode = None
    abort_event = None
    for line in Path(log_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = obj.get("event")
        if ev == "config":
            config = obj
        elif ev == "eval":
            evals.append(obj)
        elif ev == "optim_step":
            optim_steps.append(obj)
        elif ev == "lora_init_override":
            init_override_mode = obj.get("mode")
        elif isinstance(ev, str) and ev.startswith("abort_on_"):
            abort_event = obj
    if config is not None and evals:
        config["_log_filename"] = log_path.name
        config.setdefault("lr", evals[0]["lr"])
        cmd = config.get("command", "")
        lp = parse_flag(cmd, "--lora_plus_multiplier")
        config.setdefault("lora_plus_multiplier", float(lp) if lp else 1.0)
        rk = parse_flag(cmd, "--precond_refresh_every")
        config.setdefault("precond_refresh_every", int(rk) if rk else 1)
        config.setdefault("precond_method", parse_flag(cmd, "--precond_method"))
        if config.get("lora_init_b") is None:
            cli_init = parse_flag(cmd, "--lora_init_b")
            config["lora_init_b"] = cli_init or init_override_mode or "zero"

        # Generic CLI backfill — every flag in the launched command line
        # becomes a first-class cfg field via setdefault. Source order:
        #   1. explicit cfg event keys (newest, typed correctly by train.py)
        #   2. `_cli_args` blanket dump in the config event (typed)
        #   3. parsed `command` string (string-coerced fallback)
        cli_blob = config.get("_cli_args")
        if isinstance(cli_blob, dict):
            for k, v in cli_blob.items():
                config.setdefault(k, v)
        if cmd:
            for k, v in parse_cli_command(cmd).items():
                config.setdefault(k, v)
        config["_optim_steps"] = optim_steps
        if abort_event is not None:
            # Surface aborted runs to downstream analysis. Plotting primitives
            # (e.g. compare_variants_figure) treat aborted runs as completed-
            # but-diverged so they appear in the leaderboard table at their
            # last eval, instead of silently vanishing as partial runs.
            config["_aborted"] = abort_event
    if sig is not None:
        _LOAD_RUN_CACHE[path_str] = (sig, config, evals)
    return (None if config is None else dict(config)), evals


_LOAD_SWEEP_CACHE: dict[tuple[str, str], tuple[tuple, list[tuple[dict, list[dict]]]]] = {}
_TASK_FILE_RE = re.compile(r"^log_(\d+)\.out(?:\.resume_\d+)?$")


def load_sweep(group: str, logs_root: str = "../logs") -> list[tuple[dict, list[dict]]]:
    """Load all runs for a sweep group. Returns list of (cfg, evs).

    A "run" is one disBatch task index. Per task, events are merged across
    `log_NN.out` and any sibling `log_NN.out.resume_K` files written by
    submit.sh's pre-submit log rotation when a wall-killed run is resubmitted
    with checkpoint resume.

    Cached in two tiers, both invalidated by the per-file (name, mtime, size)
    signature of the group's log files:
      1. In-process module dict (`_LOAD_SWEEP_CACHE`) — fastest, dies on
         interpreter exit.
      2. Cross-session pickle cache at `<logs_root>/_runs_cache.pkl`
         (`lora_playground.run_cache`) — survives kernel restarts.
    """
    log_dir = Path(f"{logs_root}/{group}/run_info/logs")
    if not log_dir.exists():
        return []

    # Group log_NN.out and any log_NN.out.resume_K siblings by task index NN.
    # Original `.out` first, then `.resume_K` by K ascending so later-run
    # segments feed events through dedup last (segments are step-disjoint by
    # design, but tolerate overlap).
    tasks: dict[str, list[Path]] = {}
    for p in log_dir.iterdir():
        m = _TASK_FILE_RE.match(p.name)
        if not m:
            continue
        tasks.setdefault(m.group(1), []).append(p)
    def _sort_key(p: Path):
        if p.name.endswith(".out"):
            return (0, 0)
        return (1, int(p.name.rsplit(".resume_", 1)[1]))
    for files in tasks.values():
        files.sort(key=_sort_key)

    # Cache signature: every file's (name, mtime, size). The resume_K
    # suffixes get rotated atomically by submit.sh, which is a renames event
    # — invalidates the cache by changed name.
    all_files = sorted(
        (f for files in tasks.values() for f in files), key=lambda p: p.name
    )
    sig = tuple((f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in all_files)
    cache_key = (logs_root, group)
    cached = _LOAD_SWEEP_CACHE.get(cache_key)
    if cached is not None and cached[0] == sig:
        return [(dict(cfg), evs) for cfg, evs in cached[1]]

    from .. import run_cache as _run_cache
    persistent = _run_cache.get_cached_sweep(group, logs_root)
    if persistent is not None:
        _LOAD_SWEEP_CACHE[cache_key] = (sig, persistent)
        return [(dict(cfg), evs) for cfg, evs in persistent]

    runs = []
    for task_idx in sorted(tasks):
        # Merge per-file (cfg, evs) into a single (cfg, evs) for this task.
        # cfg: take first non-None (resumes re-emit the same config event).
        # evs: union sorted by step, deduping by step (latest segment wins).
        segments = [load_run(f) for f in tasks[task_idx]]
        cfg = next((c for c, _ in segments if c is not None), None)
        by_step: dict[int, dict] = {}
        for _, evs in segments:
            for ev in evs:
                by_step[int(ev["step"])] = ev
        merged = [by_step[s] for s in sorted(by_step)]
        if cfg is not None and merged:
            # Surface the canonical filename (newest segment) so the
            # exclusion / attestation layers find this task by its current name.
            cfg["_log_filename"] = tasks[task_idx][-1].name
            runs.append((cfg, merged))
    _LOAD_SWEEP_CACHE[cache_key] = (sig, runs)
    _run_cache.update_group(group, logs_root, runs, sig=sig)
    return [(dict(cfg), evs) for cfg, evs in runs]


def clear_run_caches() -> None:
    """Clear both in-process loader caches (per-file + per-sweep).

    Useful as the first cell of an analysis notebook after a loader code
    change (e.g. new CLI-flag backfill) — file mtimes haven't changed, so
    the (mtime, size)-keyed entries are stale even though the parser logic
    has moved. Cross-session pickle cache (run_cache.py) keys differently
    and refreshes via its own staleness check; this is the in-process tier.
    """
    _LOAD_RUN_CACHE.clear()
    _LOAD_SWEEP_CACHE.clear()


def has_runs(group: str, logs_root: str = "../logs") -> bool:
    """True if the group has at least one populated `log_NN.out` (or any
    `log_NN.out.resume_K` sibling)."""
    log_dir = Path(f"{logs_root}/{group}/run_info/logs")
    if not log_dir.exists():
        return False
    return any(
        _TASK_FILE_RE.match(p.name) and p.stat().st_size > 0
        for p in log_dir.iterdir()
    )
