"""Records-native run discovery with a deprecated mutable-tuple adapter.

New consumers use :func:`load_records` and :class:`RunCatalog`.  The
``load_runs(where=...)`` compatibility facade exposes mutable ``(cfg, history)``
copies containing only producer-recorded values; it does not reconstruct
missing defaults or silently apply admission policy.  ``inventory_runs``
reports the same neutral physical catalog.

Manifest scope tags are annotations, not discovery inputs. To remove an old
sweep from the physical catalog, delete its log directory.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from .manifest import SERIES_AXIS_FIELDS
from .plotting import (
    DIVERGE_THRESHOLD, OPTIM_COLORS, RUNTIME_FIELDS, prescan_groups, scan_epoch,
    scan_group,
)
from .run_catalog import RunCatalog, load_records
from .run_inventory import (
    CoverageRow,
    PINNING_ALL_DIVERGED,
    PINNING_HIGH,
    PINNING_INTERIOR,
    PINNING_LOW,
    PINNING_SINGLE,
    RunInventory,
    audit_run_catalog,
    render_inventory as _render_inventory,
)
from .run_records import RunRecord, thaw_value


class UncontrolledAxisError(RuntimeError):
    """Raised by ``load_runs(unique_on=...)`` when returned runs vary on a
    cfg axis the caller didn't list in ``unique_on``, runtime fields, or
    series-axis fields. Catches the pattern where a custom analysis cell
    buckets by a coarser key than the loader's dedup uses and silently
    collapses runs from an orthogonal ablation (e.g. ``htmuon_p``,
    ``ns_form``) into the canonical bucket."""


def _default_logs_root() -> str:
    """Repo-anchored ``logs/`` path, independent of caller cwd."""
    return str(Path(__file__).resolve().parent.parent / "logs")


# ─── temporary plotting-schema defaults ─────────────────────────────────────


_ARGPARSE_DEFAULTS_CACHE = Path(__file__).resolve().parent.parent / "logs" / "_argparse_defaults.json"
_TRAIN_PY_PATH = Path(__file__).resolve().parent / "train.py"


def _cache_is_fresh(cache: Path, *sources: Path) -> bool:
    """True iff ``cache`` exists AND is at least as new as every source file.
    Any edit to a source (adding a flag, changing a default, registering an
    optimizer) bumps its mtime and triggers a one-shot regeneration on the next
    call. Cheap: one ``stat()`` per path.

    A source that does not exist is skipped rather than treated as stale: these
    caches snapshot something reproducible from the codebase, so a missing
    source is a reason to trust the snapshot, not to re-import.
    """
    try:
        cache_mt = cache.stat().st_mtime_ns
    except OSError:
        return False
    for src in sources:
        try:
            src_mt = src.stat().st_mtime_ns
        except OSError:
            continue
        if src_mt > cache_mt:
            return False
    return True


def _argparse_defaults_cache_is_fresh() -> bool:
    """True iff the JSON sidecar exists AND is newer than train.py."""
    return _cache_is_fresh(_ARGPARSE_DEFAULTS_CACHE, _TRAIN_PY_PATH)


@lru_cache(maxsize=1)
def _argparse_defaults() -> dict[str, Any]:
    """Return ``{dest: default_value}`` for every CLI flag in train.py.

    This is a temporary plotting-schema helper used by ``plotting.arms`` and
    ``plotting.dedup``. Run discovery never calls it and never applies these
    current-code defaults to historical records.

    Persistent on-disk cache at ``logs/_argparse_defaults.json``: train.py
    imports torch + transformers (~17 s cold). Building the parser to read
    defaults is reproducible from the codebase, so we snapshot the result
    to JSON and consult that first. The JSON is rebuilt automatically
    whenever the loader runs in a fresh process and the file is missing.

    """
    if _argparse_defaults_cache_is_fresh():
        try:
            with open(_ARGPARSE_DEFAULTS_CACHE) as f:
                cached = json.load(f)
            if isinstance(cached, dict):
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    # Cold path: import train.py (torch + transformers) to introspect the
    # parser. Write the result back so subsequent processes skip the import.
    from .train import make_parser
    parser = make_parser()
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if action.dest in (None, "help"):
            continue
        defaults[action.dest] = action.default
    try:
        _ARGPARSE_DEFAULTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_ARGPARSE_DEFAULTS_CACHE, "w") as f:
            json.dump(defaults, f, sort_keys=True, indent=2, default=str)
    except OSError:
        pass
    return defaults


# ─── compatibility identity helpers ─────────────────────────────────────────
#
# RUNTIME_FIELDS is imported above from plot_utils, where it's the single
# canonical definition shared with merge_runs' hidden-axis collision check.
# The dedup key here and the collision check there cannot drift apart.
# To add a new runtime field, edit plot_utils.RUNTIME_FIELDS — only if it's
# instrumentation metadata that doesn't affect algorithm behavior. When in
# doubt, leave it out: false-positive collisions are loud and recoverable;
# false-negative collisions silently corrupt analysis.

# Allow-list dedup axes — preserved for callers that intentionally collapse
# across some axis (e.g. seed averaging). New code should prefer the
# deny-list default.
DEFAULT_KEY_AXES: tuple[str, ...] = (
    "optimizer", "lr", "lora_r", "lora_plus_multiplier", "seed",
)


def _hashable(v):
    """Recursive conversion to a hashable form for dedup keys.

    Dicts → frozenset of items; lists → tuples; everything else → as-is.
    Values that already are not hashable through this transform (custom
    objects, tensors) raise TypeError on first hash() call — that's the
    correct failure mode; the cfg should only contain JSON-serializable values
    by virtue of being read back from JSONL.
    """
    if isinstance(v, dict):
        return frozenset((k, _hashable(vv)) for k, vv in v.items())
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    return v


def _denylist_key(cfg: dict, runtime_fields: frozenset[str]) -> frozenset:
    """Dedup key = frozenset of (k, hashable(v)) for scalar source-of-truth
    cfg fields, excluding runtime metadata and derived/composite fields.

    Excluded:
      - keys in `runtime_fields` (git_commit, command, log_group, …)
      - underscore-prefixed keys (parser and runtime metadata)
      - dict-valued producer blocks (scalar effective values are the query
        surface)
      - None-valued fields (an absent value does not distinguish a run)

    Mirrors the exclusion rule used by `plot_utils.series_id` — the two
    must agree, otherwise the loader's dedup and the plot layer's
    series-identity contract drift apart.

    Two cfgs hashing to equal values mean they specify the same algorithm
    on the same data with the same hyperparameters. New behavioral fields
    that are scalar / non-derived automatically participate.
    """
    return frozenset(
        (k, _hashable(v)) for k, v in cfg.items()
        if k not in runtime_fields
        and not k.startswith("_")
        and not isinstance(v, dict)
        and v is not None
    )


def _check_unique_on(
    runs: list[tuple[dict, list[dict]]],
    unique_on: tuple[str, ...],
    runtime_fields: frozenset[str],
    allow_axes: tuple[str, ...],
) -> None:
    """Raise UncontrolledAxisError if any (unique_on)-bucket spans runs
    that differ on a non-allowed cfg axis. Allowed axes = ``unique_on ∪
    runtime_fields ∪ SERIES_AXIS_FIELDS ∪ allow_axes ∪ underscore-prefix``.
    Dict-valued cfg fields (e.g. ``optimizer_config``) are skipped — the
    same rule as ``_denylist_key``."""
    allowed: set[str] = set(unique_on) | set(runtime_fields) | set(SERIES_AXIS_FIELDS) | set(allow_axes)
    buckets: dict[tuple, list[dict]] = {}
    for cfg, _ in runs:
        bk = tuple(cfg.get(a) for a in unique_on)
        buckets.setdefault(bk, []).append(cfg)
    violations: list[str] = []
    for bk, cfgs in buckets.items():
        if len(cfgs) < 2:
            continue
        # For each non-allowed scalar cfg axis, gather the set of values.
        field_values: dict[str, set] = {}
        for cfg in cfgs:
            for k, v in cfg.items():
                if k in allowed or k.startswith("_") or isinstance(v, dict):
                    continue
                field_values.setdefault(k, set()).add(_hashable(v))
        differing = {k: sorted(vs, key=repr) for k, vs in field_values.items() if len(vs) > 1}
        if differing:
            bucket_repr = ", ".join(f"{a}={v!r}" for a, v in zip(unique_on, bk))
            diffs = ", ".join(f"{k}={list(vs)}" for k, vs in sorted(differing.items()))
            paths = [c.get("_log_filename", "?") for c in cfgs]
            violations.append(
                f"  bucket({bucket_repr}) has {len(cfgs)} runs differing on: {diffs}\n"
                f"    paths: {paths}"
            )
    if violations:
        raise UncontrolledAxisError(
            f"load_runs(unique_on={unique_on!r}) returned runs that vary on "
            f"axes outside `unique_on`, runtime, series, and `allow_axes`:\n"
            + "\n".join(violations)
            + "\n\nFix: either tighten `where=` to constrain the offending "
            "axis, OR pass `allow_axes=(...)` to acknowledge variation on "
            "that axis is expected (the consumer is responsible for "
            "filtering it before bucketing)."
        )


def _matches(spec: Any, value: Any) -> bool:
    """Predicate matcher for a single field.

    - callable                        → ``spec(value)`` truthy
    - list / set / tuple, scalar value → ``value in spec``   (membership)
    - list / set / tuple, list value   → ``value == spec``   (equality)
    - anything else (literal)         → ``value == spec``

    The two list branches exist because a Python list means two different things
    here: a SET OF ALLOWED VALUES for a scalar field (`ADAMW`'s
    ``precond_method=(None, "higham")``), and a LITERAL for a field that itself
    holds a list. Unconditional membership makes the second unpinnable —
    ``target_module_names=[]`` reads as "match nothing".

    `plotting.arms.field_matches` implements the same rule and
    `tests/test_matcher_agreement.py` pins that they agree. They must: a `where=`
    query goes through here while an arm predicate goes through `arms`, so a
    divergence means the SAME arm dict selects different runs depending on which
    loading path the caller took. Measured when they last diverged: a derived
    132-pin predicate matched 4 of its own 4 runs through `arms.pred_matches`
    and 0 of 4 through this function.
    """
    if callable(spec):
        return bool(spec(value))
    if isinstance(spec, (list, set, tuple, frozenset)):
        if isinstance(value, (list, set, tuple, frozenset)):
            return list(value) == list(spec)
        return value in spec
    return value == spec


# cfg-field aliases for `where=` filtering. Lets `_group` match the actual
# `log_group` cfg key (common confusion: `_group` looks like a derived/private
# field but the canonical key is `log_group`). Add new aliases here.
_WHERE_FIELD_ALIASES: dict[str, str] = {
    "_group": "log_group",
}


class LoggedFieldPredicate:
    """Callable explicitly safe to evaluate from a logged config header.

    Ordinary ``where`` callables stay residual filters because callers may
    depend on postprocessing or on fields unavailable until a full parse. This
    wrapper is for pure value predicates over fields recorded in the config
    event, allowing the catalog to reject nonmatches before reading histories.
    """

    __slots__ = ("predicate", "cache_key")

    def __init__(self, predicate: Callable[[Any], Any], *, cache_key: str):
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        if not isinstance(cache_key, str) or not cache_key:
            raise ValueError("cache_key must be a non-empty string")
        self.predicate = predicate
        self.cache_key = cache_key

    def __call__(self, value: Any) -> bool:
        return bool(self.predicate(value))

    def __repr__(self) -> str:
        return f"LoggedFieldPredicate({self.cache_key!r})"


def logged_field_predicate(
    predicate: Callable[[Any], Any], *, cache_key: str
) -> LoggedFieldPredicate:
    """Mark a pure logged-field predicate as safe for header pushdown."""
    return LoggedFieldPredicate(predicate, cache_key=cache_key)


def _resolve_where(where: dict[str, Any]) -> dict[str, Any]:
    return {_WHERE_FIELD_ALIASES.get(k, k): v for k, v in where.items()}


def _build_filter(where: dict[str, Any] | None) -> Callable[[dict], bool] | None:
    if not where:
        return None

    resolved = _resolve_where(where)

    def predicate(cfg: dict) -> bool:
        for field_name, spec in resolved.items():
            if field_name not in cfg:
                return False
            if not _matches(spec, cfg[field_name]):
                return False
        return True

    return predicate

def load_runs(
    where: dict[str, Any] | None = None,
    *,
    key_axes: tuple[str, ...] | None = None,
    runtime_fields: frozenset[str] = RUNTIME_FIELDS,
    cfg_postprocess: Callable[[dict, str], None] | None = None,
    logs_root: str | None = None,
    catalog: RunCatalog | None = None,
    warn_cross_commit: bool = True,
    unique_on: tuple[str, ...] | None = None,
    allow_axes: tuple[str, ...] = (),
    quiet: bool = True,
) -> list[tuple[dict, list[dict]]]:
    """Compatibility API returning mutable ``(cfg, history)`` tuples.

    New code should use :func:`load_records`. This adapter discovers physical
    records through :class:`RunCatalog`, resolves only explicitly recorded
    versioned lineages, and exposes mutable copies for older callers. It never
    reconstructs missing defaults, applies historical exclusion registries,
    infers resume lineage, or consults manifests for admission or ordering.

    Load all runs whose cfg matches every predicate in ``where``.

    Predicate types per field (see ``_matches``):
      - literal:               ``cfg[field] == value``
      - list/set/tuple:        ``cfg[field] in values``
      - callable:              ``predicate(cfg[field])`` truthy

    Omitted fields impose no constraint. A run missing a field referenced in
    ``where`` is excluded (treat absence as non-match).

    ``key_axes`` remains in the call signature but non-``None`` values raise:
    the compatibility layer no longer guesses which physical run should win a
    collision. Physical reruns stay visible by default; comparison code must
    select among them explicitly. ``runtime_fields`` still controls the optional
    ``unique_on`` guard below.

    Uncontrolled-axis guard (``unique_on``):
      When set, after the where-filter the loader buckets returned runs by
      ``tuple(cfg.get(a) for a in unique_on)`` and verifies that every bucket
      contains runs that agree on all cfg axes outside the allow-list
      (``unique_on ∪ runtime_fields ∪ SERIES_AXIS_FIELDS ∪ allow_axes ∪
      underscore-prefixed cfg fields``). If any axis varies within a bucket,
      raises ``UncontrolledAxisError`` listing the offending fields and
      values. Use this in custom analysis cells that manually bucket runs
      after ``load_runs`` — without it, orthogonal ablations (e.g.
      ``htmuon_p``, ``ns_form``) silently contaminate downstream dedup.
      Production plotting via ``standard_sweep_figure`` already gets this
      via ``assert_label_discriminates``; ``unique_on`` is the load-time
      analogue for ad-hoc cells.

    ``cfg_postprocess`` runs before ``where`` matching, preserving callers that
    use it to add a temporary alias consumed by their predicate. Every value
    exposed before that callback came from the log itself; missing values stay
    missing.
    """
    warnings.warn(
        "load_runs() is deprecated; use load_records() for immutable catalog "
        "records and explicit lineage resolution",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs = dict(
        key_axes=key_axes,
        runtime_fields=runtime_fields,
        cfg_postprocess=cfg_postprocess,
        logs_root=logs_root,
        warn_cross_commit=warn_cross_commit,
        unique_on=unique_on,
        allow_axes=allow_axes,
        quiet=quiet,
    )
    if catalog is not None:
        kwargs["catalog"] = catalog
    return _load_runs_compatibility(where, **kwargs)


def _compat_record_key(record: RunRecord) -> tuple[str, str | None, str]:
    """Stable lookup key shared with explicit-lineage terminal segments."""
    return record.group, record.log_filename, record.physical_id


def _compat_record_cfg(record: RunRecord) -> dict:
    """Mutable flat cfg containing only values logged by ``record``.

    The raw event remains the base so audit blocks stay available to legacy
    consumers. ``semantic_config`` then overlays the producer-recorded
    effective values (including values inside logged config blocks). No value
    is synthesized from current code or historical tables.
    """
    cfg = thaw_value(record.raw_config)
    cfg.update(thaw_value(record.semantic_config))
    cfg.setdefault("log_group", record.group)
    if record.log_filename is not None:
        cfg.setdefault("_log_filename", record.log_filename)
    cfg.setdefault("run_id", record.physical_id)
    return cfg


def _compat_pushdown_plan(
    where: dict[str, Any] | None,
    *,
    cfg_postprocess: Callable[[dict, str], None] | None,
    runtime_fields: frozenset[str],
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Any, ...]],
    dict[str, LoggedFieldPredicate],
]:
    """Extract only header-decidable scalar compatibility predicates.

    The full historical matcher still runs after records are parsed. This plan
    is only a conservative rejection stage. A caller postprocessor disables
    it because the callback may rewrite any queried field.
    """
    if not where or cfg_postprocess is not None:
        return {}, {}, {}

    physical_aliases = {
        "log_group": "group",
        "_log_filename": "log_filename",
        "run_id": "physical_id",
    }
    scalar_types = (str, int, float, bool, type(None))
    equals: dict[str, Any] = {}
    one_of: dict[str, tuple[Any, ...]] = {}
    predicates: dict[str, LoggedFieldPredicate] = {}
    for field, spec in _resolve_where(where).items():
        catalog_field = physical_aliases.get(field, field)
        if field not in physical_aliases and (
            field in runtime_fields
            or field in {"event", "semantic_revisions"}
            or field.startswith("_")
        ):
            continue
        if isinstance(spec, LoggedFieldPredicate):
            predicates[catalog_field] = spec
            continue
        if callable(spec):
            continue
        if isinstance(spec, (list, set, tuple, frozenset)):
            candidates = tuple(spec)
            if all(isinstance(value, scalar_types) for value in candidates):
                one_of[catalog_field] = candidates
            continue
        if isinstance(spec, scalar_types):
            equals[catalog_field] = spec
    return equals, one_of, predicates


def _warn_cross_commit_runs(runs: list[tuple[dict, list[dict]]]) -> None:
    """Report differing recorded commits without interpreting or gating them."""
    commits: dict[str, int] = {}
    for cfg, _ in runs:
        commit = cfg.get("git_commit") or "<missing>"
        commits[str(commit)] = commits.get(str(commit), 0) + 1
    if len(commits) <= 1:
        return
    summary = ", ".join(
        f"{commit[:7]} ({n} run{'s' if n != 1 else ''})"
        for commit, n in sorted(commits.items(), key=lambda item: -item[1])
    )
    warnings.warn(
        f"load_runs returned runs from {len(commits)} recorded commits: "
        f"{summary}. Commit values are audit provenance only; this compatibility "
        "loader does not interpret or gate them. Pass warn_cross_commit=False "
        "to silence.",
        UserWarning,
        stacklevel=3,
    )


def _load_runs_compatibility(
    where: dict[str, Any] | None = None,
    *,
    key_axes: tuple[str, ...] | None = None,
    runtime_fields: frozenset[str] = RUNTIME_FIELDS,
    cfg_postprocess: Callable[[dict, str], None] | None = None,
    logs_root: str | None = None,
    catalog: RunCatalog | None = None,
    warn_cross_commit: bool = True,
    unique_on: tuple[str, ...] | None = None,
    allow_axes: tuple[str, ...] = (),
    quiet: bool = True,
) -> list[tuple[dict, list[dict]]]:
    """Thin mutable-tuple adapter over the neutral physical run catalog."""
    if catalog is not None and logs_root is not None:
        raise ValueError("pass either catalog or logs_root, not both")
    if catalog is None:
        catalog = RunCatalog.discover(logs_root or _default_logs_root())
    elif not isinstance(catalog, RunCatalog):
        raise TypeError("catalog must be a RunCatalog")
    if key_axes is not None:
        raise NotImplementedError(
            "load_runs(key_axes=...) no longer performs implicit physical-run "
            "deduplication; load physical records and make the selection "
            "explicitly in the comparison consumer"
        )
    # ``quiet`` remains in the public signature while callers migrate, but
    # physical discovery has no implicit exclusion summary to silence.
    _ = quiet

    filter_fn = _build_filter(where)
    pushdown_equals, pushdown_one_of, pushdown_predicates = _compat_pushdown_plan(
        where,
        cfg_postprocess=cfg_postprocess,
        runtime_fields=runtime_fields,
    )
    if pushdown_equals or pushdown_one_of or pushdown_predicates:
        candidates = catalog.prefilter(
            equals=pushdown_equals,
            one_of=pushdown_one_of,
            predicates=pushdown_predicates,
        )
    else:
        candidates = catalog.records
    processed: dict[tuple[str, str | None, str], dict] = {}
    selected: list[RunRecord] = []

    for record in candidates:
        cfg = _compat_record_cfg(record)
        if cfg_postprocess is not None:
            cfg_postprocess(cfg, record.group)
        processed[_compat_record_key(record)] = cfg
        if filter_fn is None or filter_fn(cfg):
            selected.append(record)

    # Preserve the typo warning from header fields across the whole physical
    # pool. This does not parse nonmatching histories or treat current defaults
    # as evidence that a key exists.
    if where and catalog.groups:
        resolved_where = _resolve_where(where)
        if pushdown_equals or pushdown_one_of or pushdown_predicates:
            known_fields = set(catalog.logged_field_names)
        else:
            known_fields: set[str] = set()
            for cfg in processed.values():
                known_fields.update(cfg)
        unknown = sorted(
            field for field in resolved_where if field not in known_fields
        )
        if unknown:
            warnings.warn(
                f"load_runs: where-key(s) {unknown!r} do not appear in any "
                "logged config header in the physical candidate pool. Possible "
                "typo or filtering on an unrecorded field. Known cfg keys "
                f"(sample): {sorted(known_fields)[:20]}...",
                stacklevel=3,
            )

    runs: list[tuple[dict, list[dict]]] = []
    for resolved in catalog.resolve_lineages(selected):
        if isinstance(resolved, RunRecord):
            cfg = processed[_compat_record_key(resolved)]
            history = thaw_value(resolved.history)
        else:
            # RunCatalog.resolve_lineages returns MergedRunLineage here. Reuse
            # the already-postprocessed terminal cfg so cfg_postprocess runs
            # exactly once per physical record, then substitute the logical
            # attempt ID when the producer did not log its own run_id.
            terminal = resolved.segments[-1]
            terminal_key = (
                terminal.group,
                terminal.log_filename,
                terminal.physical_id,
            )
            cfg = processed.get(terminal_key)
            if cfg is None:
                cfg = _compat_record_cfg(terminal)
                processed[terminal_key] = cfg
            if "run_id" not in terminal.cfg:
                cfg["run_id"] = resolved.terminal_attempt_id
            history = thaw_value(resolved.history)
        runs.append((cfg, history))

    if unique_on is not None and runs:
        _check_unique_on(runs, unique_on, runtime_fields, allow_axes)
    if warn_cross_commit and runs:
        _warn_cross_commit_runs(runs)
    return runs


def logs_signature(logs_root: str | None = None,
                   groups: Iterable[str] | None = None) -> str:
    """Fingerprint of what ``load_runs`` reads off disk.

    Changes iff some group's manifest or some log file changed — including a
    running sweep appending to a ``log_NN.out``, which is why this stats every
    log file rather than trusting directory mtimes. It is the cheap half of a
    ``load_runs`` call: the same concurrent scan, without parsing, filtering,
    postprocessing or dedup.

    Intended use is a live-refresh memo, so a panel re-reads only when the tree
    actually moved::

        sig = logs_signature(LOGS)
        if key not in cache or cache[key][0] != sig:
            cache[key] = (sig, load_runs(where=..., logs_root=LOGS))
        runs = cache[key][1]

    Cheaper than the ``refresh=True`` pattern it replaces, which paid a full
    query per call to discover that nothing had changed. The manifest layer
    (one ``stat`` per ``meta.json``) is always included, so a newly submitted
    sweep changes the signature even before it has written a log line.

    ``groups`` narrows the log-file half to a subset — e.g. the groups a
    previous result came from. That makes an unrelated sweep's writes stop
    invalidating this query's memo, at the cost of missing a group that was
    already on disk, contributed nothing last time, and has since started
    producing matching runs. Leave it None (the default, whole tree) unless
    that trade is one you have thought about.
    """
    if logs_root is None:
        logs_root = _default_logs_root()
    from .manifest import _groups_with_run_info, _meta_stats
    h = hashlib.blake2b(digest_size=16)
    with scan_epoch():
        all_groups = _groups_with_run_info(logs_root)
        scanned = all_groups if groups is None else sorted(set(groups))
        prescan_groups(scanned, logs_root)
        for group, mtime, size in _meta_stats(logs_root, all_groups):
            h.update(f"{group}\0{mtime}\0{size}\0".encode())
        for group in scanned:
            h.update(f"|{group}|".encode())
            for name, f_mtime, f_size in scan_group(group, logs_root).sig:
                h.update(f"{name}\0{f_mtime}\0{f_size}\0".encode())
    return h.hexdigest()


# ─── inventory ────────────────────────────────────────────────────────────────

def inventory_runs(logs_root: str | None = None) -> RunInventory:
    """Return a catalog-backed audit of physical logs and recorded semantics.

    The inventory consumes the same immutable :class:`RunCatalog` records as
    new analysis code.  It does not apply exclusions, reconstruct defaults,
    infer resume relationships, query Git ancestry, or touch the legacy
    persistent cache.  Coverage is computed only from values present in each
    record's logged effective config.
    """
    if logs_root is None:
        logs_root = _default_logs_root()
    return _inventory_runs_inner(logs_root)


def _inventory_runs_inner(logs_root: str) -> RunInventory:
    """Compatibility wrapper around the neutral catalog audit."""
    return audit_run_catalog(
        logs_root,
        known_optimizers=OPTIM_COLORS,
        diverge_threshold=DIVERGE_THRESHOLD,
    )


def aggregate_by(
    runs: list[tuple[dict, list[dict]]],
    key: tuple[str, ...],
    reduce: Callable[[list[tuple[dict, list[dict]]]], Any] | None = None,
    *,
    runtime_fields: frozenset[str] = RUNTIME_FIELDS,
    allow_axes: tuple[str, ...] = (),
) -> dict[tuple, Any]:
    """Bucket ``runs`` (a list of ``(cfg, history)`` pairs as returned by
    ``load_runs``) by the cfg-field tuple ``key`` and apply ``reduce`` to
    each bucket. Raises ``UncontrolledAxisError`` if any bucket spans
    runs differing on cfg axes outside ``key``, ``runtime_fields``,
    ``SERIES_AXIS_FIELDS``, or ``allow_axes``.

    Motivation: post-load aggregators (``best_by_lr`` and friends) routinely
    re-bucket loaded runs on a coarser key than the loader's dedup uses.
    When the coarser key leaves cfg axes unaccounted for, the bucket
    silently collapses runs from orthogonal ablations (e.g. ``precond_delta_relative``,
    ``ns_form``, ``htmuon_p``) into one "best" or "mean" — biasing the
    output downward and hiding what's actually being compared. The
    loader exposes ``unique_on=`` at load time for the same check, but
    that doesn't help when bucketing happens downstream. This helper
    closes the loop.

    Args:
        runs: list of ``(cfg, history)`` pairs.
        key: tuple of cfg field names. Each bucket is indexed by the
            tuple of values ``(cfg[k] for k in key)``.
        reduce: callable applied to the list of ``(cfg, history)`` in each
            bucket. Default returns the list itself (no reduction).
        runtime_fields: cfg fields treated as metadata / runtime state
            and thus allowed to vary within a bucket.
        allow_axes: additional cfg fields the caller acknowledges may
            vary within a bucket.

    Returns:
        dict mapping bucket-key tuple → reduce(bucket_runs).

    Raises:
        UncontrolledAxisError: if any bucket spans runs that differ on an
            unaccounted-for cfg axis.

    Example::

        runs = load_runs(where={"optimizer": "adamw", "lora_r": 64})
        best_by_lr = aggregate_by(
            runs, key=("lr",),
            reduce=lambda bucket: min(h[-1]["eval_loss"] for _, h in bucket),
        )
    """
    if isinstance(key, str):
        key = (key,)
    _check_unique_on(runs, tuple(key), runtime_fields, allow_axes)

    buckets: dict[tuple, list[tuple[dict, list[dict]]]] = {}
    for cfg, hist in runs:
        bk = tuple(_hashable(cfg.get(a)) for a in key)
        buckets.setdefault(bk, []).append((cfg, hist))

    if reduce is None:
        return buckets
    return {bk: reduce(bucket) for bk, bucket in buckets.items()}


def render_inventory(inv: RunInventory) -> str:
    """Compatibility re-export of the neutral catalog report renderer."""
    return _render_inventory(inv)
