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

import subprocess
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .manifest import live_manifests_newest_first, load_manifests, warn_untagged
from .plot_utils import (
    DIVERGE_THRESHOLD, OPTIM_COLORS, RUNTIME_FIELDS, has_runs, load_sweep,
    max_loss, merge_runs, parse_flag,
)


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
    if cfg.get("optimizer_config") is None:
        cfg["optimizer_config"] = _backfill_optimizer_config(cfg)
    opt_cfg = cfg["optimizer_config"]
    # Diagnostic knobs are emitted at two places: the top-level cfg event
    # (CLI names ``log_optim_diagnostics`` / ``optim_diagnostics_every``) and
    # the per-optimizer kwargs in ``optimizer_config`` (constructor names
    # ``log_diagnostics`` / ``diagnostics_every``). Older cfg events lack the
    # top-level CLI fields entirely (the flag was added later) — they only
    # carry the constructor names in optimizer_config. Backfill from
    # optimizer_config so callers can read a single top-level field
    # consistently regardless of which generation of cfg event they have.
    if cfg.get("log_optim_diagnostics") is None:
        cfg["log_optim_diagnostics"] = opt_cfg.get("log_diagnostics")
    if cfg.get("optim_diagnostics_every") is None:
        cfg["optim_diagnostics_every"] = opt_cfg.get("diagnostics_every")
    derived: dict[str, Any] = {}
    derived["effective_inner_polar"] = _derive_effective_inner_polar(cfg, opt_cfg)
    k, k_certain = _derive_effective_picard_iters(cfg, opt_cfg)
    derived["effective_picard_iters"] = k
    derived["effective_picard_iters_certain"] = k_certain
    # Data pipeline version: explicit on new runs (>=2026-05-08), absent on
    # older runs which were all unpacked_v0 (legacy
    # DataCollatorForLanguageModeling, no prompt mask, dynamic shapes).
    # Default-fill so analysis can filter by version uniformly.
    pipeline_version = cfg.get("data_pipeline_version")
    if pipeline_version is None:
        pipeline_version = "unpacked_v0"
    derived["data_pipeline_version"] = pipeline_version
    cfg["_derived"] = derived
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
    """Dedup key = frozenset of (k, hashable(v)) for all non-runtime cfg fields.

    Two cfgs hashing to equal values means they specify the same algorithm
    on the same data with the same hyperparameters — modulo the explicitly
    excluded runtime/metadata. New behavioral fields automatically participate.
    """
    return frozenset(
        (k, _hashable(v)) for k, v in cfg.items() if k not in runtime_fields
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

    runs = merge_runs(
        groups,
        key_fn=key_fn,
        filter_fn=filter_fn,
        cfg_postprocess=cfg_postprocess,
        logs_root=logs_root,
    )

    # Enrichment: every cfg gains a `_derived` namespace and (if missing) a
    # backfilled `optimizer_config`. Done after merge_runs so dedup operates
    # on raw cfg fields (no risk of `_derived` differences hiding a
    # collision); analysis code reads enriched cfgs.
    for cfg, _ in runs:
        _enrich_cfg(cfg)

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
    for group in live_groups:
        if not has_runs(group, logs_root):
            continue
        for cfg, evs in load_sweep(group, logs_root):
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

    return RunInventory(
        groups_on_disk=tuple(on_disk),
        groups_loaded=tuple(sorted(contributing_groups)),
        groups_orphaned=tuple(orphaned),
        groups_no_run_info=tuple(no_run_info),
        optimizers_unknown=optimizers_unknown,
        coverage=tuple(coverage),
    )


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
