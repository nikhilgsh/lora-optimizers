"""Predicate-based sweep loader and inventory.

``load_runs(where=...)`` selects runs whose cfg matches a per-field predicate;
``inventory_runs(logs_root)`` returns a structural audit (orphans, unknown
optimizers, lr-pinning) for a notebook audit cell.

Scope tags on manifests are metadata only — they don't drive loading. To
remove an old sweep, delete its log dir.

Enrichment: every run returned by ``load_runs`` carries a ``cfg["_derived"]``
namespace with fields answering "what optimizer math actually ran", regardless
of how old the run is or how complete its raw cfg was. See ``_enrich_cfg``.
"""
from __future__ import annotations

import json
import subprocess
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .manifest import (
    SERIES_AXIS_FIELDS, live_manifests_newest_first, load_manifests, warn_untagged,
)
from .plotting import (
    DIVERGE_THRESHOLD, OPTIM_COLORS, RUNTIME_FIELDS, has_runs, load_sweep,
    max_loss, merge_runs, parse_flag,
)


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ─── commit-aware default resolution ──────────────────────────────────────────
#
# The motivating case: ``build_optimizer`` for ``adam-polar-product-lora-coupled``
# hardcoded its default ``picard_iters`` as 2 until commit ``dadea5d`` (May 3
# 2026), when it flipped to 3. Runs from before that commit that didn't pass
# ``--picard_iters_override`` actually ran with k=2, even though the *current*
# code's default is k=3. Backfilling with the current default would silently
# mislabel those runs.
#
# Each entry: (optimizer_type, kwarg_name) → list of (commit_sha, value, date)
# sorted oldest-first. The sentinel ``"<initial>"`` means "the value before any
# of the listed commits"; it always resolves to True for ``_is_ancestor`` so
# the chronologically-earliest entry is the fallback.
#
# The table is **manually curated** — we add entries as we discover historical
# default changes that affect interpretation. Defaults we don't know changed
# will be backfilled with the current value; the cross-commit warning in
# ``load_runs`` is the safety net for "you're comparing across commits, check
# whether anything changed".
HARDCODED_DEFAULT_HISTORY: dict[tuple[str, str], list[tuple[str, Any, str]]] = {
    ("adam-polar-product-lora-coupled", "picard_iters"): [
        ("<initial>", 2, "<original>"),
        ("dadea5d",   3, "2026-05-03"),
    ],
}


@lru_cache(maxsize=4096)
def _is_ancestor(commit: str, descendant: str = "HEAD") -> bool:
    """True iff ``commit`` is an ancestor of ``descendant`` in the repo at
    ``_repo_root()``. Sentinel ``"<initial>"`` is treated as an ancestor of
    everything (the chronologically-earliest registry entry).

    Cached because we query the same (commit, HEAD) pairs repeatedly when
    enriching many runs. Failures (commit not in repo, git not available)
    are caught and treated as "not an ancestor" so enrichment never crashes
    a load — at worst, a registry entry is skipped and we fall back to the
    next.
    """
    if commit == "<initial>":
        return True
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, descendant],
            cwd=str(_repo_root()),
            capture_output=True,
            check=False,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _resolve_default_at_commit(optimizer_type: str, kwarg: str,
                                run_commit: str | None) -> tuple[Any, bool]:
    """Look up the hardcoded default for ``(optimizer_type, kwarg)`` that was
    in effect at ``run_commit``. Returns (value, resolved_with_certainty).

    ``resolved_with_certainty=False`` when:
      - ``run_commit`` is missing or unparseable
      - the registry has no entry for this (optimizer_type, kwarg)
      - none of the registry's commit entries are ancestors of run_commit
        (only happens with rebased/orphan history)
    """
    history = HARDCODED_DEFAULT_HISTORY.get((optimizer_type, kwarg))
    if history is None:
        return (None, False)
    if not run_commit:
        # No commit info — return the latest entry, mark as uncertain.
        return (history[-1][1], False)
    # Walk newest-first; first ancestor wins.
    for commit, value, _date in reversed(history):
        if _is_ancestor(commit, run_commit):
            return (value, True)
    # No ancestor found — fall back to the latest entry, mark as uncertain.
    return (history[-1][1], False)


# ─── enrichment ───────────────────────────────────────────────────────────────


def _backfill_optimizer_config(cfg: dict) -> dict:
    """Reconstruct the equivalent of ``optimizer_config_dict(opt)`` for runs
    whose cfg lacks it (pre-b0baa4d, May 3 2026).

    Source priority for each kwarg, highest first:
      1. ``cfg[kwarg]`` if a top-level field exists (older runs put many
         kwargs at the top level alongside event="config" keys).
      2. ``parse_flag(cfg["command"], f"--{kwarg}")`` if the user passed it.
      3. Empty (the caller's derivations layer fills with class defaults).

    Always tagged ``_backfilled=True`` so downstream can tell.
    """
    cmd = cfg.get("command", "") or ""
    backfilled: dict[str, Any] = {"_backfilled": True}
    # All CLI flags currently surfaced by ``train.py`` that affect the
    # ``adam-polar-product-lora{,-coupled}`` algorithmic path. Other optimizers'
    # CLI flags are extracted opportunistically — same pattern, just unused.
    cli_kwargs = (
        "muon_ns_steps", "polar_method", "polar_sigma_power", "polar_norm_dir",
        "picard_iters_override", "picard_alpha", "anderson_m", "anderson_reg",
        "soap_beta", "soap_refresh_every", "beta1", "beta2",
        "lora_plus_multiplier", "precond_refresh_every", "precond_method",
        "higham_iters",
    )
    for kw in cli_kwargs:
        if kw in cfg and cfg[kw] is not None:
            backfilled[kw] = cfg[kw]
            continue
        flag_val = parse_flag(cmd, f"--{kw}")
        if flag_val is not None:
            backfilled[kw] = flag_val
    return backfilled


def _coerce(v, kind):
    """Best-effort cast for backfilled string CLI values. Returns ``v`` on
    failure rather than raising — derivations should not crash a load."""
    if v is None:
        return None
    if kind is float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    if kind is int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return v
    return v


def _derive_effective_inner_polar(cfg: dict, opt_cfg: dict) -> str | None:
    """Single derived field answering "what polar approximation actually ran".

    Precedence mirrors the short-circuit order in
    ``optim.py:_polar_pipeline``:
      ``polar_sigma_power == 0.0``       → "svd_exact"
      ``polar_sigma_power not None != 0`` → f"sigma_power(p={value})"
      ``polar_method``                    → that string ("ns" / "ns_hybrid" /
                                            "polar_express")
      polar-product family pre-feature   → "ns"  (the implicit pre-4b047f5
                                                   default — _polar_pipeline
                                                   unconditionally called
                                                   ``_newton_schulz`` before
                                                   ``polar_method`` existed)
      else (non-polar optimizer)          → None
    """
    psp_raw = opt_cfg.get("polar_sigma_power")
    psp = _coerce(psp_raw, float) if psp_raw not in (None, "None") else None
    if psp is not None:
        return "svd_exact" if psp == 0.0 else f"sigma_power(p={psp})"
    pm = opt_cfg.get("polar_method")
    if pm in {"ns", "ns_hybrid", "polar_express"}:
        return pm
    # Fallback for runs from before the polar_method param existed: any
    # optimizer whose name contains "polar-product" used ``_newton_schulz``
    # unconditionally inside ``_polar_pipeline``. The CLI flag wasn't
    # plumbed yet, so neither the cfg nor the command line carries it.
    optimizer = cfg.get("optimizer", "") or ""
    if "polar-product" in optimizer:
        return "ns"
    return None


def _derive_effective_polar_pre_norm(cfg: dict, opt_cfg: dict) -> str | None:
    """Forensic backfill for `effective_polar_pre_norm` on pre-fix runs.

    Tag values:
      "frob" — NS/polar-express called with Frobenius pre-norm (the historical
               default in `_newton_schulz*` and `_polar_express_gram_batched`).
               In the chord-tight-clean path this is REDUNDANT with §2.5's
               spec-norm and SHRINKS σ_max by 1/√(stable_rank) → 5-iter
               Schulz is incomplete (whitening_fraction ≈ 0.72 instead of 1.0).
               All pre-fix chord-tight-clean ns/polar_express runs sit in this
               regime; their nominal "ns-5 polar" was actually a graded,
               tail-truncated polar.
      "none" — NS/polar-express called with pre_norm="none" (post-fix). §2.5
               already spec-normed; no second divisor; iteration starts at
               σ=1. Whitening_fraction ≈ 1.0 (true polar).
      "ssc"  — SSC path; the Frob-pre-norm concern doesn't apply (SSC has no
               such pre-norm). Tagged distinctly so analysis can group by
               polar map family.
      None    — non-polar-product optimizer.

    Pre-fix means: cfg has no `optimizer_effective.effective_polar_pre_norm`
    field. After the polar-pre-norm fix landed, new runs emit the field
    directly and this fallback is skipped (see _enrich_cfg precedence).
    """
    optimizer = cfg.get("optimizer", "") or ""
    if "polar-product" not in optimizer:
        return None
    pm = opt_cfg.get("polar_method")
    if pm == "ssc":
        return "ssc"
    if pm in ("ns", "ns_hybrid", "polar_express"):
        # Universal pre-fix behavior: every NS/polar-express call hit the
        # function's default Frobenius pre-norm. (Chord-tight-clean's §2.5
        # didn't override it; the bug was that the redundant pre-norm
        # silently fired anyway.)
        return "frob"
    # Optimizer-class fallback for runs predating the polar_method flag.
    return "frob"


def _derive_effective_picard_iters(cfg: dict, opt_cfg: dict) -> tuple[Any, bool]:
    """Returns (k_value, resolved_with_certainty).

    Ground-truth precedence:
      1. ``opt_cfg["picard_iters"]`` — the constructor kwarg the optimizer
         actually ran with. Emitted by ``build_optimizer`` for every
         polar-product variant since the flag existed; if present, this is
         authoritative regardless of how the value was derived (CLI override
         vs. variant-default).
      2. ``opt_cfg["picard_iters_override"]`` / top-level
         ``cfg["picard_iters_override"]`` — the CLI flag itself. Useful for
         older cfg events that didn't pass ``picard_iters`` through to the
         optimizer_config dump.
      3. Commit-aware default for the optimizer slug (for cfg events that
         predate either field being recorded).
    """
    pi = opt_cfg.get("picard_iters")
    if pi not in (None, "None"):
        return (_coerce(pi, int), True)
    override = opt_cfg.get("picard_iters_override")
    if override in (None, "None"):
        override = cfg.get("picard_iters_override")
    if override not in (None, "None"):
        return (_coerce(override, int), True)
    # Fall through to commit-aware default lookup.
    optimizer_type = cfg.get("optimizer", "")
    return _resolve_default_at_commit(
        optimizer_type, "picard_iters", cfg.get("git_commit"),
    )


# Fields whose argparse default has CHANGED over time, where backfilling
# from the CURRENT argparse default would mislabel older runs. These get
# the historical default value instead. Add an entry here whenever a CLI
# flag's default changes and you need to keep loading older runs.
#
# Example: `--data_pipeline_version` was introduced 2026-05-08 with
# argparse default `packed_v1`. Older runs ran the previous (unpacked)
# pipeline and don't log the field. Without this override, our argparse
# backfill would tag every old AdamW run as packed_v1 → analysis filters
# on packed_v1 silently include unpacked_v0 runs from old sweeps.
HISTORICAL_DEFAULTS_WHEN_MISSING: dict[str, Any] = {
    "data_pipeline_version": "unpacked_v0",
}


_ARGPARSE_DEFAULTS_CACHE = Path(__file__).resolve().parent.parent / "logs" / "_argparse_defaults.json"
_TRAIN_PY_PATH = Path(__file__).resolve().parent / "train.py"


def _argparse_defaults_cache_is_fresh() -> bool:
    """True iff the JSON sidecar exists AND is newer than train.py. Any
    edit to train.py (adding a flag, changing a default) bumps its mtime
    and triggers a one-shot regeneration on the next call. Cheap: two
    `stat()` calls."""
    try:
        cache_mt = _ARGPARSE_DEFAULTS_CACHE.stat().st_mtime_ns
    except OSError:
        return False
    try:
        src_mt = _TRAIN_PY_PATH.stat().st_mtime_ns
    except OSError:
        # train.py missing (shouldn't happen in this repo); fall back to
        # the JSON regardless rather than re-import unnecessarily.
        return True
    return cache_mt >= src_mt


@lru_cache(maxsize=1)
def _argparse_defaults() -> dict[str, Any]:
    """Return ``{dest: default_value}`` for every CLI flag in train.py's
    parser. Used to backfill older cfgs that were logged before a flag
    existed: at the time, the runtime fell through to the same default we
    record here, so the run's effective value WAS the default. Without
    this backfill, the missing-vs-explicit-default asymmetry would split
    series_id across schema-growth boundaries even when the algorithm is
    identical (e.g. old run lacks `precond_delta`, new run logs
    `precond_delta=1e-6` — both ran with 1e-6).

    Persistent on-disk cache at ``logs/_argparse_defaults.json``: train.py
    imports torch + transformers (~17 s cold). Building the parser to read
    defaults is reproducible from the codebase, so we snapshot the result
    to JSON and consult that first. The JSON is rebuilt automatically
    whenever the loader runs in a fresh process and the file is missing;
    `scripts/build_logs_cache.py` rebuilds it as part of its workflow.

    Caveat: this assumes the flag's default has never changed. When it has,
    register the history in `HARDCODED_DEFAULT_HISTORY` and resolve via
    `_resolve_default_at_commit` instead. Today this matters for
    optimizer-kwarg defaults; CLI-flag defaults change rarely enough that
    the simple cache is acceptable.
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


def _enrich_cfg(cfg: dict) -> dict:
    """Add a ``_derived`` namespace and (if missing) a backfilled
    ``optimizer_config`` to ``cfg``. Mutates and returns ``cfg``.

    ``_derived`` is the analysis surface; raw fields are never overwritten.
    For run-comparison code, prefer ``cfg["_derived"][...]`` over the raw
    ``cfg["polar_method"]`` / ``cfg["picard_iters_override"]`` / etc.,
    because the raw fields can lie (``polar_method="ns"`` next to
    ``polar_sigma_power=0.0`` is the canonical example — the effective
    inner polar is "svd_exact", not "ns").
    """
    # Loader-assigned run_id: (log_group, log_filename) tuple. `log_group` is
    # mutated onto cfg by merge_runs; `_log_filename` is set by load_run. The
    # cfg event itself never carries run_id — it's derived from disk path so
    # the registry keys for RUN_EXCLUSIONS / DIRTY_ATTESTATIONS are stable
    # against any future field-name renaming in the cfg event.
    group = cfg.get("log_group")
    fname = cfg.get("_log_filename")
    if group is not None and fname is not None:
        cfg["run_id"] = (group, fname)
    if cfg.get("optimizer_config") is None:
        cfg["optimizer_config"] = _backfill_optimizer_config(cfg)
    opt_cfg = cfg["optimizer_config"]
    # Diagnostic knobs are emitted at two places: the top-level cfg event
    # (CLI names ``log_basic_diagnostics`` / ``optim_diagnostics_every``) and
    # the per-optimizer kwargs in ``optimizer_config`` (constructor names
    # ``log_basic_diagnostics`` / ``diagnostics_every``).
    #
    # Backward read-compat: cfg events from before the 2026-05-12 diagnostics
    # refactor used the OLD names — top-level ``log_optim_diagnostics`` and
    # constructor ``log_diagnostics``. Read both, prefer new. Same applies to
    # backfill from optimizer_config. Older cfg events also lack the top-level
    # CLI fields entirely (the flag was added later still) and only carry the
    # constructor names — also backfilled here.
    if cfg.get("log_basic_diagnostics") is None:
        cfg["log_basic_diagnostics"] = (
            cfg.get("log_optim_diagnostics")
            or opt_cfg.get("log_basic_diagnostics")
            or opt_cfg.get("log_diagnostics")
        )
    if cfg.get("log_heavy_diagnostics") is None:
        cfg["log_heavy_diagnostics"] = opt_cfg.get("log_heavy_diagnostics")
    if cfg.get("optim_diagnostics_every") is None:
        cfg["optim_diagnostics_every"] = opt_cfg.get("diagnostics_every")
    derived: dict[str, Any] = {}
    # Prefer the cfg-emitted `optimizer_effective` block (Phase 1 emit for new
    # runs, Phase 2 backfill for legacy). Fall back to the forensic _derive_*
    # path only when the field is missing (in-flight runs that started before
    # Phase 1 and haven't been re-backfilled). Once all in-flight runs are
    # done and re-backfilled, the forensic fallback is deletable.
    opt_eff = cfg.get("optimizer_effective") or {}
    if "effective_inner_polar" in opt_eff:
        derived["effective_inner_polar"] = opt_eff["effective_inner_polar"]
    else:
        derived["effective_inner_polar"] = _derive_effective_inner_polar(cfg, opt_cfg)
    if "effective_polar_pre_norm" in opt_eff:
        derived["effective_polar_pre_norm"] = opt_eff["effective_polar_pre_norm"]
    else:
        derived["effective_polar_pre_norm"] = _derive_effective_polar_pre_norm(cfg, opt_cfg)
    if "effective_picard_iters" in opt_eff:
        derived["effective_picard_iters"] = opt_eff["effective_picard_iters"]
        derived["effective_picard_iters_certain"] = True
    else:
        k, k_certain = _derive_effective_picard_iters(cfg, opt_cfg)
        derived["effective_picard_iters"] = k
        derived["effective_picard_iters_certain"] = k_certain
    k = derived["effective_picard_iters"]
    k_certain = derived["effective_picard_iters_certain"]
    # Promote canonical resolved-values to top-level scalars. `series_id`
    # and `_denylist_key` only see top-level non-underscore-prefixed
    # scalars; if the canonical lives only under `_derived`, two cfgs
    # that resolve to the same effective behavior but with different raw
    # override flags will incorrectly split. With these promoted, the raw
    # override flags can sit in RUNTIME_FIELDS (loader) /
    # SERIES_AXIS_FIELDS (plot) as redundant, and the effective value is
    # the single source of truth.
    cfg["effective_picard_iters"] = k
    if derived["effective_inner_polar"] is not None:
        cfg["effective_inner_polar"] = derived["effective_inner_polar"]
    if derived["effective_polar_pre_norm"] is not None:
        cfg["effective_polar_pre_norm"] = derived["effective_polar_pre_norm"]
    # Data pipeline version: explicit on new runs (>=2026-05-08), absent on
    # older runs which were all unpacked_v0 (legacy
    # DataCollatorForLanguageModeling, no prompt mask, dynamic shapes).
    # Default-fill so analysis can filter by version uniformly.
    pipeline_version = cfg.get("data_pipeline_version")
    if pipeline_version is None:
        pipeline_version = "unpacked_v0"
    derived["data_pipeline_version"] = pipeline_version
    cfg["_derived"] = derived
    # Schema-growth backfill: any train.py CLI flag absent or explicitly
    # logged as None in this cfg event must have fallen through to its
    # argparse default at runtime. Fill so series_id sees identical state
    # for runs logged before/after the flag existed AND for the
    # explicit-None case (newer schemas log unset flags as None instead
    # of omitting them). load_runs applies the same normalization BEFORE
    # merge_runs' dedup; this enrichment call is idempotent for cfgs that
    # went through load_runs, but exists for callers that build runs from
    # other paths (e.g. ad-hoc load_run usage).
    for k, v in _argparse_defaults().items():
        if cfg.get(k) is None:
            cfg[k] = HISTORICAL_DEFAULTS_WHEN_MISSING.get(k, v)
    return cfg

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
      - underscore-prefixed keys (loader enrichment: _derived, _cli_args,
        _optim_steps)
      - dict-valued fields (optimizer_config — derived backfill that mirrors
        scalar fields; older runs without it get reconstructed and the dict
        may not round-trip exactly, so it cannot be the source of truth)
      - None-valued fields (a newer schema may log not-provided flags as
        None; load_runs separately backfills these to argparse defaults
        before dedup, but the exclusion here is defense-in-depth)

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


# Pinning categories returned in CoverageRow.pinning.
PINNING_INTERIOR = "interior"
PINNING_LOW = "pinned_low"
PINNING_HIGH = "pinned_high"
PINNING_SINGLE = "single_lr"
PINNING_ALL_DIVERGED = "all_diverged"


def _matches(spec: Any, value: Any) -> bool:
    """Predicate matcher for a single field.

    - callable                  → ``spec(value)`` truthy
    - list / set / tuple        → ``value in spec``
    - anything else (literal)   → ``value == spec``
    """
    if callable(spec):
        return bool(spec(value))
    if isinstance(spec, (list, set, tuple, frozenset)):
        return value in spec
    return value == spec


# cfg-field aliases for `where=` filtering. Lets `_group` match the actual
# `log_group` cfg key (common confusion: `_group` looks like a derived/private
# field but the canonical key is `log_group`). Add new aliases here.
_WHERE_FIELD_ALIASES: dict[str, str] = {
    "_group": "log_group",
}


def _build_filter(where: dict[str, Any] | None) -> Callable[[dict], bool] | None:
    if not where:
        return None

    resolved = {_WHERE_FIELD_ALIASES.get(k, k): v for k, v in where.items()}

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
    warn_cross_commit: bool = True,
    unique_on: tuple[str, ...] | None = None,
    allow_axes: tuple[str, ...] = (),
    quiet: bool = True,
) -> list[tuple[dict, list[dict]]]:
    """Load all runs whose cfg matches every predicate in ``where``.

    Predicate types per field (see ``_matches``):
      - literal:               ``cfg[field] == value``
      - list/set/tuple:        ``cfg[field] in values``
      - callable:              ``predicate(cfg[field])`` truthy

    Omitted fields impose no constraint. A run missing a field referenced in
    ``where`` is excluded (treat absence as non-match).

    Dedup model:
      - ``key_axes=None`` (default, recommended): **deny-list** dedup. Two
        runs collapse iff their cfg fields are equal except for fields in
        ``runtime_fields`` (git_commit, command, log_group, etc.). New
        behavioral hyperparameters automatically become dedup axes.
      - ``key_axes=tuple(...)``: **allow-list** dedup. Used to intentionally
        collapse across some axis (e.g. seed averaging). Older mode; prefer
        the deny-list default for general analysis.

    ``merge_runs`` keeps longest-trajectory-wins; group priority is
    newest-first (by ``submitted_at``). The hidden-axis collision check
    still fires if two runs share the dedup key but differ on another cfg
    axis (most useful in allow-list mode; under deny-list it almost never
    fires by construction).

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
    """
    if logs_root is None:
        logs_root = _default_logs_root()
    manifests = load_manifests(logs_root, strict=False)
    groups = [m["group"] for m in live_manifests_newest_first(manifests)]
    filter_fn = _build_filter(where)

    if key_axes is None:
        def key_fn(cfg: dict) -> frozenset:
            return _denylist_key(cfg, runtime_fields)
    else:
        def key_fn(cfg: dict) -> tuple:
            return tuple(cfg.get(a) for a in key_axes)

    # Apply argparse-default backfill BEFORE merge_runs' dedup. Otherwise
    # two cfgs of the same algorithm at the same seed — one from an older
    # commit where a flag didn't exist, one from a newer commit logging
    # the same default explicitly — get different _denylist_key hashes
    # and both survive dedup. They then aggregate as two distinct "seed=N"
    # rows at the plot layer and inflate the seed-σ band. Backfilling
    # here makes the keys agree so the dedup tiebreak (newest group,
    # longest trajectory) picks one canonical run per seed.
    from .invariants import evaluate_invariants
    from .run_exclusions import is_run_excluded
    from .dirty_attestations import lookup_attestation
    from .commit_exclusions import (
        is_commit_excluded as _new_is_commit_excluded,
        is_buggy_eps_rel as _new_is_buggy_eps_rel,
    )
    _excluded_counts: dict[str, int] = {}
    # Per-reason example list: (group, log_filename) pairs, capped per reason.
    # Surfaces "which sweep got hit" in the loader summary; a user
    # investigating "why is my group empty?" can grep the summary print
    # for the group name instead of enumerating registries by hand.
    _excluded_examples: dict[str, list[tuple[str, str]]] = {}
    _EXAMPLES_PER_REASON = 3

    def _evaluate_new_layer(cfg: dict) -> tuple[bool, str | None]:
        """Code-correctness + run-quality exclusion. Three schema branches
        determine how dirty trees are resolved:

          Phase 4 cfgs (have `execution_source_sha`):
            content-hash auto-resolve. Searches descendants of cfg.git_commit
            for a commit C whose tree-restricted-to-execution_source_paths
            matches the recorded execution_source_sha. If found, invariants
            run against C. If not, exclude with reason.

          Phase 1 cfgs (have `git_dirty` / `git_diff_sha`, no execution_source_sha):
            legacy attestation policy. Dirty runs need a dirty_attestations
            entry keyed by (group, log_filename, diff_sha). Untracked files
            in this schema auto-exclude unless attested.

          Pre-Phase 1 cfgs (neither):
            treat as clean at cfg.git_commit.
        """
        # Per-run quality exclusion always takes precedence.
        log_group = cfg.get("log_group")
        log_filename = cfg.get("_log_filename")
        excluded, reason = is_run_excluded(log_group, log_filename)
        if excluded:
            return True, reason
        # Blanket-commit exclusion (JSON-backed commit_exclusions.json).
        excluded, reason = _new_is_commit_excluded(cfg.get("git_commit"))
        if excluded:
            return True, reason
        # ε_rel-specific buggy commits (eps_rel_buggy_commits.json).
        excluded, reason = _new_is_buggy_eps_rel(cfg)
        if excluded:
            return True, reason

        # ── Schema dispatch for dirty-tree resolution ────────────────────
        if cfg.get("execution_source_sha") is not None:
            # Phase 4 schema: content-hash auto-resolve.
            if not cfg.get("execution_source_dirty"):
                effective_commit = cfg.get("git_commit")
            else:
                from .execution_scope import auto_resolve_by_content, project_root
                resolved = auto_resolve_by_content(
                    base_commit=cfg["git_commit"],
                    paths=cfg["execution_source_paths"],
                    target_source_sha=cfg["execution_source_sha"],
                    project_root=project_root(),
                )
                if resolved is None:
                    # Phase 4 auto-resolve failed. Fall through to Phase-1
                    # manual attestation: a human can vouch that the dirty
                    # diff is non-load-bearing at runtime (e.g. opt-in CLI
                    # flags not invoked by this sweep), keyed by (group,
                    # log_filename, git_diff_sha). Without this fallback,
                    # Phase-4 runs are stuck whenever the at-submission tree
                    # is never committed — which is the common case for
                    # untracked-file dirt that gets cleaned up later.
                    attestation = lookup_attestation(
                        log_group, log_filename, cfg.get("git_diff_sha"),
                    )
                    if attestation is None:
                        return True, (
                            "execution source hash not found in descendant "
                            "commits within search bounds; either commit your "
                            "at-submission state, or add a dirty_attestation"
                        )
                    effective_commit = attestation.treat_as_commit
                else:
                    effective_commit = resolved
        elif cfg.get("git_dirty"):
            # Phase 1 schema: legacy attestation policy. NOTE: per the
            # Phase-4 policy revision (treat untracked files as audit-only,
            # not load-bearing), the legacy auto-exclude on untracked files
            # is removed for consistency. Attestation against
            # (group, log_filename, diff_sha) drives the resolution.
            # Attestation lookup also handles "manual relabel": entries
            # with null git_diff_sha match Phase-2-backfilled runs that have
            # no diff_sha — a human asserts the effective commit explicitly.
            attestation = lookup_attestation(
                log_group, log_filename, cfg.get("git_diff_sha"),
            )
            if attestation is not None:
                effective_commit = attestation.treat_as_commit
            elif cfg.get("git_diff_sha") is None:
                # Legacy-dirty, no attestation: accept at face value
                # (we have no information to do better).
                effective_commit = cfg.get("git_commit")
            else:
                return True, "unattested dirty tree"
        else:
            effective_commit = cfg.get("git_commit")

        # Code-correctness invariants.
        excluded, name, inv_reason = evaluate_invariants(
            cfg, effective_commit, _is_ancestor,
        )
        if excluded:
            return True, f"invariant {name}: {inv_reason}"
        return False, None

    # Apply the new exclusion layer (invariants + run_exclusions +
    # dirty_attestations + commit_exclusions). The Phase-3 dual-output mode
    # against the legacy Python registries is removed now that the
    # disagreement set has been reviewed; the JSON commit_exclusions
    # reproduces every legacy entry's effect by construction.
    _user_filter = filter_fn
    # Pool of cfg keys seen during filtering (used by where-key validation
    # post-merge: tracking BEFORE the user_filter rejects on missing keys
    # tells us "does this key exist anywhere in the candidate pool?").
    _seen_cfg_keys: set[str] = set()
    def _wrapped_filter(cfg: dict) -> bool:
        # Key-existence pool for the where-key typo warning below. Collect from
        # EVERY candidate cfg, independent of either filter, so a where-key that
        # legitimately narrows the pool to nothing still validates as a real
        # field. (`_evaluate_new_layer` is a pure read, so order vs. it is free.)
        _seen_cfg_keys.update(cfg.keys())
        # Cheap user predicate FIRST. `_evaluate_new_layer` can shell out to git
        # for dirty-tree content-hash resolution (~10s over a full multi-hundred
        # group pool, dominated by in-flight runs); paying that for runs the
        # caller's `where` rejects anyway was the bulk of cold-load latency.
        # Result set is unchanged — (not excluded) AND (user match) is the same
        # set either order; only the exclusion diagnostic now counts just the
        # where-matching runs, which is the relevant population for the query.
        if _user_filter is not None and not _user_filter(cfg):
            return False
        excluded, reason = _evaluate_new_layer(cfg)
        if excluded:
            _excluded_counts[reason] = _excluded_counts.get(reason, 0) + 1
            ex = _excluded_examples.setdefault(reason, [])
            if len(ex) < _EXAMPLES_PER_REASON:
                ex.append((cfg.get("log_group") or "?",
                           cfg.get("_log_filename") or "?"))
            return False
        return True
    filter_fn = _wrapped_filter

    _defaults = _argparse_defaults()
    def _wrapped_postprocess(cfg: dict, group: str) -> None:
        # Backfill argparse defaults. Treat explicit None and absent
        # identically: a newer schema may LOG `field: None` for a
        # not-provided flag while an older schema omitted the key
        # entirely; both indicate "ran with argparse default at runtime,"
        # so both should land at the default for dedup-key purposes.
        # `setdefault` alone would leave the explicit-None case in place
        # and the cfgs would still dedup to different keys.
        #
        # HISTORICAL_DEFAULTS_WHEN_MISSING overrides the argparse default
        # for fields whose default has changed over time — using the
        # current default for missing-field old runs would mislabel them
        # (the historical default is the value those runs actually ran with).
        for k, v in _defaults.items():
            if cfg.get(k) is None:
                cfg[k] = HISTORICAL_DEFAULTS_WHEN_MISSING.get(k, v)
        # Enrich BEFORE dedup so derived canonical fields
        # (`effective_picard_iters`, `effective_inner_polar`) are present
        # in the dedup key. Without this, two cfgs with the same
        # effective k but different raw `picard_iters_override` get
        # different dedup keys and both survive.
        _enrich_cfg(cfg)
        if cfg_postprocess is not None:
            cfg_postprocess(cfg, group)

    runs = merge_runs(
        groups,
        key_fn=key_fn,
        filter_fn=filter_fn,
        cfg_postprocess=_wrapped_postprocess,
        logs_root=logs_root,
    )

    # Enrichment: every cfg gains a `_derived` namespace and (if missing) a
    # backfilled `optimizer_config`. Done after merge_runs so dedup operates
    # on raw cfg fields (no risk of `_derived` differences hiding a
    # collision); analysis code reads enriched cfgs.
    for cfg, _ in runs:
        _enrich_cfg(cfg)

    if _excluded_counts:
        total = sum(_excluded_counts.values())
        parts: list[str] = []
        for reason, n in sorted(_excluded_counts.items()):
            ex = _excluded_examples.get(reason, [])
            ex_str = ", ".join(f"{g}/{lf}" for g, lf in ex)
            more = max(0, n - len(ex))
            tail = f" (e.g. {ex_str}" + (f", +{more} more" if more else "") + ")" if ex else ""
            parts.append(f"{n} for {reason!r}{tail}")
        if not quiet:
            print(f"  [loader] excluded {total} run(s): " + "; ".join(parts))

    # where-key validation: if the user filtered on a field that doesn't
    # appear in ANY non-excluded cfg in the loaded pool, the result is
    # silently empty. Warn so typos ('datset' for 'dataset_name') surface
    # loudly. We check against `_seen_cfg_keys` (collected pre-user-filter)
    # so legitimate value-misses don't fire the warning.
    if where and _seen_cfg_keys:
        resolved_keys = {_WHERE_FIELD_ALIASES.get(k, k) for k in where.keys()}
        unknown = sorted(k for k in resolved_keys if k not in _seen_cfg_keys)
        if unknown:
            warnings.warn(
                f"load_runs: where-key(s) {unknown!r} do not appear in any "
                f"cfg in the candidate pool ({len(_seen_cfg_keys)} unique "
                f"fields seen). Possible typo or filtering on a field that "
                f"doesn't exist for this dataset. Known cfg keys (sample): "
                f"{sorted(_seen_cfg_keys)[:20]}...",
                stacklevel=2,
            )


    if unique_on is not None and runs:
        _check_unique_on(runs, unique_on, runtime_fields, allow_axes)

    if warn_cross_commit and runs:
        commits: dict[str, int] = {}
        for cfg, _ in runs:
            c = cfg.get("git_commit") or "<missing>"
            commits[c] = commits.get(c, 0) + 1
        if len(commits) > 1:
            summary = ", ".join(
                f"{c[:7]} ({n} run{'s' if n != 1 else ''})"
                for c, n in sorted(commits.items(), key=lambda kv: -kv[1])
            )
            warnings.warn(
                f"load_runs returned runs from {len(commits)} commits: "
                f"{summary}. Behavior at default settings can differ across "
                f"commits; if comparing absolute losses, verify the relevant "
                f"defaults in HARDCODED_DEFAULT_HISTORY or pin the comparison "
                f"to a single commit. Pass warn_cross_commit=False to silence.",
                UserWarning,
                stacklevel=2,
            )

    # Persist any newly-parsed groups to the cross-session pickle cache.
    # Cheap when nothing changed (early-return on `not _DIRTY`).
    try:
        from . import run_cache as _run_cache
        _run_cache.flush(logs_root)
    except Exception:
        pass

    return runs


# ─── inventory ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CoverageRow:
    """One row per (optimizer, lora_r, lora_plus_multiplier) cell."""
    optimizer: str
    lora_r: int
    lora_plus_multiplier: float
    lrs_swept: tuple[float, ...]
    best_lr: float | None              # None when all runs diverged
    final_loss_at_best: float | None
    pinning: str
    source_groups: tuple[str, ...]


@dataclass(frozen=True)
class RunInventory:
    groups_on_disk: tuple[str, ...]            # populated logs/<group>/run_info dirs
    groups_loaded: tuple[str, ...]             # subset that contributes runs
    groups_orphaned: tuple[str, ...]           # populated, no manifest or empty scope
    groups_no_run_info: tuple[str, ...]        # logs/<group>/ exists with files but no run_info/ — invisible to load_manifests
    groups_all_excluded: tuple[tuple[str, str], ...]  # (group, dominant_reason) — manifested + populated, but every run is exclusion-dropped
    optimizers_unknown: tuple[str, ...]        # in logs but not in OPTIM_COLORS
    coverage: tuple[CoverageRow, ...]

    @property
    def pinned(self) -> tuple[CoverageRow, ...]:
        """Subset of coverage with pinning ∈ {pinned_low, pinned_high}."""
        return tuple(r for r in self.coverage
                     if r.pinning in (PINNING_LOW, PINNING_HIGH))


def _classify_pinning(lrs_swept: tuple[float, ...], best_lr: float | None) -> str:
    if best_lr is None:
        return PINNING_ALL_DIVERGED
    if len(lrs_swept) <= 1:
        return PINNING_SINGLE
    if best_lr == min(lrs_swept):
        return PINNING_LOW
    if best_lr == max(lrs_swept):
        return PINNING_HIGH
    return PINNING_INTERIOR


def inventory_runs(logs_root: str | None = None) -> RunInventory:
    """Walk all manifests + runs, return a structural audit.

    Each problem reported is a fact, not a threshold judgment:
      - groups_orphaned: populated dir without a valid scope-tagged manifest.
      - optimizers_unknown: optimizer present in some run's cfg but absent
        from ``OPTIM_COLORS`` — silently dropped from any cell that filters
        on color-map membership.
      - coverage: per (optimizer, lora_r, lora_plus_multiplier), the swept
        lrs, the best lr (lowest non-diverged final loss), and a pinning
        classification.
    """
    if logs_root is None:
        logs_root = _default_logs_root()
    manifests = load_manifests(logs_root, strict=False)
    on_disk = sorted(m["group"] for m in manifests)
    orphaned = sorted(warn_untagged(manifests))
    live = live_manifests_newest_first(manifests)
    live_groups = [m["group"] for m in live]

    # Single pass over all runs in live groups. We do NOT dedup here — the
    # inventory wants raw coverage across groups; downstream load_runs() does
    # the dedup for plotting.
    rows: dict[tuple[str, int, float], dict] = {}
    seen_optimizers: set[str] = set()
    contributing_groups: set[str] = set()
    # Per-group exclusion audit: re-run the exclusion chain on every
    # (group, run) and flag groups where ALL runs are exclusion-dropped.
    # This catches the "valid manifest + populated .out + every run on a
    # blanket-excluded commit" failure mode that's invisible to the
    # orphan / no_run_info / unknown-optimizer audits.
    from .invariants import evaluate_invariants
    from .run_exclusions import is_run_excluded
    from .commit_exclusions import (
        is_commit_excluded as _ic_excluded,
        is_buggy_eps_rel as _ic_buggy_eps_rel,
    )
    def _group_dominant_exclusion(cfgs: list[dict]) -> str | None:
        """Returns the most common exclusion reason if EVERY cfg is
        excluded, else None. Defensive against per-cfg variation: we want
        to surface a single representative reason in the inventory output.
        """
        reasons: list[str] = []
        for cfg in cfgs:
            ex, r = is_run_excluded(cfg.get("log_group"), cfg.get("_log_filename"))
            if ex:
                reasons.append(r); continue
            ex, r = _ic_excluded(cfg.get("git_commit"))
            if ex:
                reasons.append(r); continue
            ex, r = _ic_buggy_eps_rel(cfg)
            if ex:
                reasons.append(r); continue
            ex, name, inv_reason = evaluate_invariants(
                cfg, cfg.get("git_commit"), _is_ancestor)
            if ex:
                reasons.append(f"invariant {name}: {inv_reason}"); continue
            return None  # at least one run admitted — group is fine
        if not reasons:
            return None
        # Most-common reason (Counter would import, just use dict count).
        counts: dict[str, int] = {}
        for r in reasons:
            counts[r] = counts.get(r, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    groups_all_excluded: list[tuple[str, str]] = []

    for group in live_groups:
        if not has_runs(group, logs_root):
            continue
        # Capture the raw cfg list once so we can both feed the coverage
        # loop AND run the all-excluded audit without re-parsing.
        group_runs = load_sweep(group, logs_root)
        group_cfgs = [cfg for cfg, evs in group_runs if evs]
        if group_cfgs:
            dom = _group_dominant_exclusion(group_cfgs)
            if dom is not None:
                groups_all_excluded.append((group, dom))
        for cfg, evs in group_runs:
            if not evs:
                continue
            optimizer = cfg.get("optimizer", "?")
            lora_r = int(cfg.get("lora_r", 16))
            mult = float(cfg.get("lora_plus_multiplier", 1.0))
            try:
                lr = float(cfg["lr"])
            except (KeyError, TypeError, ValueError):
                continue
            seen_optimizers.add(optimizer)
            contributing_groups.add(group)
            key = (optimizer, lora_r, mult)
            row = rows.setdefault(key, {"lrs": {}, "groups": set()})
            row["groups"].add(group)
            final = evs[-1]["eval_loss"]
            diverged = max_loss(evs) >= DIVERGE_THRESHOLD
            existing = row["lrs"].get(lr)
            if existing is None or (not diverged and (existing[1] or final < existing[0])):
                row["lrs"][lr] = (final, diverged)

    coverage: list[CoverageRow] = []
    for (optimizer, lora_r, mult), info in sorted(rows.items()):
        lrs = tuple(sorted(info["lrs"].keys()))
        non_diverged = [(lr, fl) for lr, (fl, div) in info["lrs"].items() if not div]
        if non_diverged:
            best_lr, best_loss = min(non_diverged, key=lambda x: x[1])
        else:
            best_lr, best_loss = None, None
        coverage.append(CoverageRow(
            optimizer=optimizer,
            lora_r=lora_r,
            lora_plus_multiplier=mult,
            lrs_swept=lrs,
            best_lr=best_lr,
            final_loss_at_best=best_loss,
            pinning=_classify_pinning(lrs, best_lr),
            source_groups=tuple(sorted(info["groups"])),
        ))

    optimizers_unknown = tuple(sorted(o for o in seen_optimizers if o not in OPTIM_COLORS))

    # Groups on disk with files but no run_info/ subdir are invisible to
    # load_manifests (and therefore to load_runs). Surface them so they
    # don't silently disappear from analyses.
    manifest_groups = set(on_disk)
    no_run_info: list[str] = []
    root = Path(logs_root)
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in manifest_groups:
                continue
            if not (child / "run_info").exists() and any(child.iterdir()):
                no_run_info.append(child.name)

    # `inventory_runs` also touches load_sweep — flush any newly parsed
    # groups so the persistent pickle stays in sync.
    try:
        from . import run_cache as _run_cache
        _run_cache.flush(logs_root)
    except Exception:
        pass
    return RunInventory(
        groups_on_disk=tuple(on_disk),
        groups_loaded=tuple(sorted(contributing_groups)),
        groups_orphaned=tuple(orphaned),
        groups_no_run_info=tuple(no_run_info),
        groups_all_excluded=tuple(sorted(groups_all_excluded)),
        optimizers_unknown=optimizers_unknown,
        coverage=tuple(coverage),
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
    """Plain-text report for the notebook audit cell."""
    lines: list[str] = []
    lines.append(f"Loaded {len(inv.groups_loaded)} of {len(inv.groups_on_disk)} groups on disk.")

    if inv.groups_orphaned:
        lines.append("")
        lines.append(f"ORPHANED ({len(inv.groups_orphaned)}) — populated but no valid manifest, will not load:")
        for g in inv.groups_orphaned:
            lines.append(f"  {g}")

    if inv.groups_no_run_info:
        lines.append("")
        lines.append(f"NO run_info/ ({len(inv.groups_no_run_info)}) — files present but missing run_info/ dir, invisible to load_runs:")
        for g in inv.groups_no_run_info:
            lines.append(f"  {g}")

    if inv.groups_all_excluded:
        lines.append("")
        lines.append(f"ALL RUNS EXCLUDED ({len(inv.groups_all_excluded)}) — valid manifest + .out files, but every run dropped by exclusion chain:")
        for g, reason in inv.groups_all_excluded:
            lines.append(f"  {g}: {reason}")

    if inv.optimizers_unknown:
        lines.append("")
        lines.append(f"UNKNOWN OPTIMIZERS ({len(inv.optimizers_unknown)}) — in logs but missing from OPTIM_COLORS, "
                     f"will be dropped by any cell that filters on it:")
        for o in inv.optimizers_unknown:
            lines.append(f"  {o}")

    lines.append("")
    lines.append(f"Coverage: {len(inv.coverage)} (optimizer, rank, mult) cells")
    if inv.pinned:
        lines.append(f"PINNED at lr-range boundary ({len(inv.pinned)}) — extension sweep recommended:")
        for r in inv.pinned:
            mult = f" m={r.lora_plus_multiplier:g}" if r.lora_plus_multiplier != 1.0 else ""
            lines.append(
                f"  {r.optimizer:<32}  r={r.lora_r:<4}{mult:<6}  "
                f"best_lr={r.best_lr:.0e} (final={r.final_loss_at_best:.4f}) "
                f"  swept={[f'{x:.0e}' for x in r.lrs_swept]} → {r.pinning}"
            )
    else:
        lines.append("No (optimizer, rank, mult) cells pinned at lr-range boundary.")

    return "\n".join(lines)
