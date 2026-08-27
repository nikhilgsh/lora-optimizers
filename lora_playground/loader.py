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
import subprocess
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from .manifest import (
    SERIES_AXIS_FIELDS, live_manifests_newest_first, load_manifests, warn_untagged,
)
from .plotting import (
    DIVERGE_THRESHOLD, OPTIM_COLORS, RUNTIME_FIELDS, has_runs, load_sweep,
    max_loss, merge_runs, parse_flag, prescan_groups, scan_epoch, scan_group,
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

# Optimizers whose build_optimizer branch routes to ``CurvatureWhitenLoRA``.
# This class HARDCODED its polar step to ``ns_steps=muon_ns_steps`` and
# ``polar_method=polar_method`` (train.py defaults: ``muon_ns_steps=5``,
# ``polar_method="ns"``), but did NOT emit those two fields into the cfg
# event — so every existing run of these optimizers ran a PARTIAL polar
# (Newton-Schulz, ns=5) with no record of it in the log. The loader
# safety-net below backfills both fields from the train.py default so the
# canonical label can render partial-polar (ns=5) runs distinctly from any
# future full-polar (polar_express / ns>=8) runs of the same optimizer.
# This is the generic family-keyed extension of the existing schema-growth
# backfill — NOT a per-flag one-off. The ``-polar`` variants are the ones
# that actually run a polar step; the non-polar variants carry the fields
# harmlessly (use_polar=False, so the tag is irrelevant for them).
# DERIVED, not listed. `_precond_by_optimizer()` already enumerates exactly the
# `optim_specs.REGISTRY` specs whose class is `CurvatureWhitenLoRA`, behind the
# same JSON snapshot, so reading its keys costs no extra import and cannot go
# stale. The hardcoded nine-name frozenset this replaces had already gone
# stale: it omitted `kl-diag-flatout-lora`, so that optimizer's runs skipped
# the backfill below. Inert in practice — all 5 such runs on disk log
# `ns_steps`/`polar_method` explicitly, so the backfill had nothing to do — but
# a future non-logging run would have hit it silently.
def _curvature_whiten_optimizers() -> frozenset[str]:
    return frozenset(_precond_by_optimizer())
# The train.py CLI defaults the CurvatureWhitenLoRA constructor read at the
# time these runs launched (train.py: --muon_ns_steps default=5,
# --polar_method default="ns"). The constructor stores the step count under
# the kwarg name ``ns_steps`` (sourced from CLI ``muon_ns_steps``), so the
# optimizer_config block carries ``ns_steps``; older/top-level cfgs carry
# ``muon_ns_steps``. Both are normalized below. Keyed by field name so the
# backfill stays generic (one dict, not one clause per flag).
_CURVATURE_WHITEN_POLAR_NS_DEFAULT = 5
_CURVATURE_WHITEN_POLAR_METHOD_DEFAULT = "ns"


def _derive_effective_polar_iters(cfg: dict, opt_cfg: dict) -> int | None:
    """Step count of the polar map a CurvatureWhitenLoRA-backed run actually
    ran (the ``N`` in ns=N / PE=N). ``None`` for non-CurvatureWhitenLoRA
    optimizers, where the field is meaningless.

    Source precedence (highest first):
      1. ``opt_cfg["ns_steps"]`` — the constructor kwarg the optimizer stored
         (ground truth; present on every protagonist run logged to date).
      2. ``opt_cfg["muon_ns_steps"]`` / top-level ``cfg["muon_ns_steps"]`` —
         the CLI name, for cfgs that surfaced it there instead.
      3. train.py default (5) — the value the hardcoded polar step used for
         the runs that logged neither, since the class read ns_steps=
         muon_ns_steps and muon_ns_steps defaulted to 5.
    """
    if cfg.get("optimizer") not in _curvature_whiten_optimizers():
        return None
    for src in (opt_cfg.get("ns_steps"), opt_cfg.get("muon_ns_steps"),
                cfg.get("muon_ns_steps")):
        if src is not None:
            v = _coerce(src, int)
            if isinstance(v, int):
                return v
    return _CURVATURE_WHITEN_POLAR_NS_DEFAULT


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
    # Safety-net for CurvatureWhitenLoRA-backed optimizers: their polar step
    # was hardcoded to muon_ns_steps / polar_method but the CLI fields were
    # never logged. Fill from the train.py default (ns=5, method=ns) only when
    # still absent after the CLI/top-level passes above, so an explicitly-logged
    # value (a future full-polar run) is never clobbered. The authoritative
    # step count when present is the constructor kwarg ``ns_steps`` (handled by
    # ``_derive_effective_polar_iters``); this block only matters for the
    # rare cfgs lacking an optimizer_config block entirely.
    if cfg.get("optimizer") in _curvature_whiten_optimizers():
        if backfilled.get("muon_ns_steps") is None and cfg.get("muon_ns_steps") is None:
            backfilled["muon_ns_steps"] = _CURVATURE_WHITEN_POLAR_NS_DEFAULT
        if backfilled.get("polar_method") is None and cfg.get("polar_method") is None:
            backfilled["polar_method"] = _CURVATURE_WHITEN_POLAR_METHOD_DEFAULT
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
    optimizer = cfg.get("optimizer", "") or ""
    pm = opt_cfg.get("polar_method")
    if pm is None and "polar" in optimizer:
        # Older polar runs recorded polar_method only at the CLI/cfg level, not in
        # the optimizer sub-config; fall back to it so two kl-diag-polar-lora groups
        # that both ran polar_method=ns don't split on a missing-field artifact.
        pm = cfg.get("polar_method")
    if pm in {"ns", "ns_hybrid", "polar_express"}:
        return pm
    # Fallback for runs from before the polar_method param existed: any optimizer
    # whose name contains "polar-product" used ``_newton_schulz`` unconditionally
    # inside ``_polar_pipeline``; the CLI flag wasn't plumbed, so nothing carries it.
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
    whenever the loader runs in a fresh process and the file is missing.

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


# ─── precond-branch backfill ──────────────────────────────────────────────────
#
# Which of the three `precond` branches a pre-flag run was actually in. Before
# `--precond` existed the branch was implied by the optimizer's pinned
# `diag_metric`: the kl-diag / diag-shampoo specs pin it True (slots = the partner
# Grams B^T P B, A Q A^T = "product"), the kl-shampoo / curvature-whiten / flatout
# specs pin it False (slots = r x r EMAs of the factor's own gradients =
# "factorwise"). Mapping the old runs onto the new names is what lets one arm
# predicate pin ONE value: without it a pre-flag protagonist run carries no
# `precond` at all while a new one carries "product", and the two land in
# different arms of the same figure.
#
# The map is DERIVED from `optim_specs.REGISTRY`, not transcribed from it. A
# hand-written table duplicated the pins and had nothing asserting the two
# agreed, so an 11th CurvatureWhitenLoRA spec would have been absent from it,
# its runs would have carried `precond=None`, and every arm predicate that pins
# `precond` would have skipped them — rendering the arm as absent rather than
# raising. `tests/test_loader_precond_backfill.py` pins the derivation against
# the ten entries the table used to hold.
#
# There is no historical "one-sided": the retired `cw_no_rr_precond` put the
# identity in the direction only and left the p, q estimator whitening by the real
# C_A/C_B, so it is NOT the one-sided branch. It is dropped below rather than
# mapped onto a branch.
_PRECOND_CACHE = Path(__file__).resolve().parent.parent / "logs" / "_precond_by_optimizer.json"
_OPTIM_SPECS_PATH = Path(__file__).resolve().parent / "optim_specs.py"
_OPTIM_PATH = Path(__file__).resolve().parent / "optim.py"


def _derive_precond_by_optimizer() -> dict[str, str]:
    """``{optimizer_name: "product" | "factorwise"}`` for every registered
    ``CurvatureWhitenLoRA`` variant, read off the ``diag_metric`` its spec pins.

    Mirrors the resolution inside ``CurvatureWhitenLoRA.__init__``::

        self.precond = precond or ("product" if self.diag_metric else "factorwise")

    That line is the definition of the branch; this applies the same map to the
    value each spec pins, which is what a run with no ``--precond`` flag
    constructed with. A spec that does NOT pin ``diag_metric`` falls through to
    the constructor's own default — read by introspection, not transcribed — so
    a new variant is covered whether or not it states the pin.

    Imports ``optim_specs``, which imports torch (~2.5 s) and would otherwise be
    the loader's only heavy dependency; ``_precond_by_optimizer`` snapshots the
    result to JSON so the import is paid once per edit of the source, not once
    per process.
    """
    import inspect

    from . import optim as _optim
    from .optim_specs import REGISTRY
    cw = _optim.CurvatureWhitenLoRA
    class_default = inspect.signature(cw.__init__).parameters["diag_metric"].default
    return {
        name: ("product" if s.fixed.get("diag_metric", class_default) else "factorwise")
        for name, s in REGISTRY.items() if s.cls is cw
    }


@lru_cache(maxsize=1)
def _precond_by_optimizer() -> dict[str, str]:
    """:func:`_derive_precond_by_optimizer` behind a JSON snapshot.

    Same trade as :func:`_argparse_defaults`: the mapping is reproducible from
    the codebase but the import that produces it is expensive, and the loader is
    otherwise torch-free (it is imported by every plotting and notebook path).
    The snapshot at ``logs/_precond_by_optimizer.json`` is regenerated whenever
    ``optim_specs.py`` or ``optim.py`` is newer than it.

    Raises whatever the import raises when there is no usable snapshot AND
    ``optim_specs`` cannot be imported. That is deliberate: returning an empty
    map instead would put every curvature-whiten run back at ``precond=None``
    and silently empty the arms that pin it, which is the failure this
    derivation exists to prevent.
    """
    if _cache_is_fresh(_PRECOND_CACHE, _OPTIM_SPECS_PATH, _OPTIM_PATH):
        try:
            with open(_PRECOND_CACHE) as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached and all(
                    isinstance(k, str) and v in ("product", "factorwise")
                    for k, v in cached.items()):
                return cached
        except (json.JSONDecodeError, OSError):
            pass
    derived = _derive_precond_by_optimizer()
    try:
        _PRECOND_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_PRECOND_CACHE, "w") as f:
            json.dump(derived, f, sort_keys=True, indent=2)
    except OSError:
        pass
    return derived


# Cfg key recording what `_backfill_precond` synthesized or dropped on a run.
# Underscore-prefixed on purpose: every consumer that walks cfg keys skips those
# (`_denylist_key`, `_check_unique_on`, `dedup.series_id`,
# `dedup._label_collision_report`, `dedup._series_diff`,
# `dedup._baseline_values`, `merge`'s hidden-axis check), so the extra key
# cannot split a series or change a dedup key. `labels._residual_knobs` and the
# `arms.arm()` predicates iterate `arms.PINNED_FIELDS()` (OptimizerConfig
# fields), which it is not one of.
PRECOND_BACKFILL_MARKER = "_precond_backfilled"


def _backfill_precond(cfg: dict, opt_cfg: dict) -> None:
    """Give every curvature-whiten run the RESOLVED `precond` / `msign` branch.

    New runs log both directly (train.py records the optimizer's resolved
    attribute, not the CLI value, so a default run logs "product" rather than
    None). Pre-flag runs log neither; this derives them so old and new runs of the
    same arm carry the same value. Never overwrites a logged value.

    A cfg this touches no longer agrees with its own ``config.json`` on disk, so
    the keys it synthesized or dropped are recorded under
    ``PRECOND_BACKFILL_MARKER`` — a tuple of key names, absent entirely on a cfg
    that needed nothing. Read it to tell a synthesized branch from a logged one;
    ``tests/test_loader_precond_backfill.py`` counts it to decide when this shim
    has no runs left to serve and should be deleted.

    Idempotent: ``_enrich_cfg`` runs twice per run under ``load_runs``, and the
    second pass finds ``precond`` already set and the retired key already gone,
    so the record accumulates rather than resetting.
    """
    touched: list[str] = []
    if cfg.get("precond") is None:
        # `diag_metric` in the recorded optimizer_config wins over the derived
        # name map when present — it is what the run actually constructed with.
        dm = opt_cfg.get("diag_metric")
        if dm is not None:
            cfg["precond"] = "product" if dm else "factorwise"
            touched.append("precond")
        else:
            branch = _precond_by_optimizer().get(cfg.get("optimizer"))
            if branch is not None:
                cfg["precond"] = branch
                touched.append("precond")
    # `msign` needs no explicit backfill: the generic argparse-default loop below
    # fills every CLI flag that is None, and `--msign` defaults to "full", which
    # is what every pre-flag run did wherever it applied a matrix sign at all.
    #
    # RETIRED FIELD. `cw_no_rr_precond` was removed from OptimizerConfig, so no
    # arm pins it any more — but runs logged during its lifetime still carry
    # `False` while older runs carry nothing, and `merge_runs`' hidden-axis check
    # reads that False-vs-absent split as two distinct series under one label.
    # The three sweeps that set it True were deleted, so no surviving run means
    # anything by this key: drop it rather than let a dead flag split live series.
    if "cw_no_rr_precond" in cfg:
        del cfg["cw_no_rr_precond"]
        touched.append("cw_no_rr_precond")
    if touched:
        prior = cfg.get(PRECOND_BACKFILL_MARKER) or ()
        cfg[PRECOND_BACKFILL_MARKER] = tuple(sorted(set(prior).union(touched)))


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
    # cfg event itself never carries run_id — it is derived from the physical
    # disk path so provenance remains stable across cfg field-name changes.
    group = cfg.get("log_group")
    fname = cfg.get("_log_filename")
    if group is not None and fname is not None:
        cfg["run_id"] = (group, fname)
    if cfg.get("optimizer_config") is None:
        cfg["optimizer_config"] = _backfill_optimizer_config(cfg)
    opt_cfg = cfg["optimizer_config"]
    _backfill_precond(cfg, opt_cfg)
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
    # Polar step count (ns=N / PE=N). CurvatureWhitenLoRA-backed optimizers
    # hardcoded the polar step to ns_steps but only the optimizer_config block
    # records it; surface it so the canonical label can distinguish a partial
    # (ns=5) polar from a future full (polar_express / ns>=8) polar. Prefer the
    # cfg-emitted optimizer_effective field when newer runs add one.
    if "effective_polar_iters" in opt_eff:
        derived["effective_polar_iters"] = opt_eff["effective_polar_iters"]
    else:
        derived["effective_polar_iters"] = _derive_effective_polar_iters(cfg, opt_cfg)
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
    if derived["effective_polar_iters"] is not None:
        cfg["effective_polar_iters"] = derived["effective_polar_iters"]
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
