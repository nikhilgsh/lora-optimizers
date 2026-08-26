"""Sweep manifest helpers.

Every sweep submitted via ``slurm_scripts/submit.sh`` writes a ``meta.json``
into ``logs/<group>/run_info/``. This module reads those manifests.

Schema (written by submit.sh):

    {
      "group":         "adamuon_polar_product_2k",
      "submitted_at":  "2026-04-30T14:23:01-04:00",
      "slurm_job_id":  "6314009",
      "n_gpus":        6,
      "params_file":   "adamuon_polar_product_2k.json",
      "sweep_script":  "scripts/sweep/sweep_2k_r_diag.sh",
      "sbatch_script": "slurm_scripts/sbatch.sh",
      "git_commit":    "<sha>",
      "git_dirty":     false,
      "scope":         ["ext_compare", "polar_family"],
      "purpose":       "E2: AdaMuon-faithful + polar-product geometry",
      "data_pipeline_version": "packed_v1"
    }

The `data_pipeline_version` field marks which data path produced the runs:

  - "unpacked_v0": legacy DataCollatorForLanguageModeling, dynamic shapes,
                   no prompt-mask, full-text loss. All runs prior to
                   2026-05-08 are this version (backfilled automatically
                   by `scripts/data/backfill_pipeline_version.py`).
  - "packed_v1":   sequence-packed train side, pad-to-max eval side,
                   prompt-masked loss. New default from 2026-05-08.

Numbers from different versions are NOT directly comparable: prompt-mask
changes the loss objective; packing changes per-step token density. Use
`load_runs(where={"data_pipeline_version": "packed_v1"})` to filter.

Loading: ``lora_playground.loader.load_runs(where=...)`` — predicate-based,
no scope strings. To remove an old sweep from analysis, delete its log dir.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .plotting import RUNTIME_FIELDS, has_runs, parallel_map, prescan_groups


# Phase 3 deletion (2026-05-14): EXCLUDED_COMMITS, EXCLUDED_GROUPS,
# BUGGY_EPS_REL_COMMITS, is_commit_excluded, is_group_excluded, and
# is_buggy_eps_rel previously lived here. They have been migrated to:
#   - lora_playground/exclusions/commit_exclusions.json
#   - lora_playground/exclusions/eps_rel_buggy_commits.json
#   - lora_playground/exclusions/run_exclusions.json
# with thin JSON loaders in `commit_exclusions.py` and `run_exclusions.py`.
# All call sites updated. Edit JSON, not Python, when adding entries.

# Fields allowed to vary within a single analysis "series" (i.e. across
# seeds / lr-grid points / horizon extensions of the same algorithm at the
# same model config). Two runs whose cfgs disagree only on fields in this
# set are seeds of the same series and may be averaged. Two runs that
# disagree on ANY field outside this set are distinct series and MUST
# resolve to distinct display labels — `assert_label_discriminates` in
# `plot_utils` enforces that contract at every plotting entry point.
#
# Adding a new flag here means "this is a per-series axis, not a
# series-defining hyperparameter." Default for any new train.py CLI flag
# is series-defining — leave it OUT of this set unless you really mean
# "varying this within a series is fine, do not split."
SERIES_AXIS_FIELDS: frozenset[str] = frozenset({
    # Per-series axes — runs differing only on these are averaged together
    # as members of one series (seeds, lr-sweep points, horizon extensions).
    "seed", "lr", "lora_r", "lora_alpha",
    "max_steps", "eval_every",
    # CLI override flags whose canonical post-resolution value is
    # promoted by `_enrich_cfg` to a top-level scalar (e.g.
    # `effective_picard_iters`). The raw override is then redundant.
    # Series identity uses the effective value; the raw override is not
    # series-defining.
    "picard_iters_override",
}) | RUNTIME_FIELDS

# NOTE on what is intentionally NOT in SERIES_AXIS_FIELDS:
#   - `lora_plus_multiplier`: real algorithm parameter (LoRA+ B-multiplier);
#     different m = different setup, must split series.
#   - `precond_delta`, `precond_delta_relative`: damping scheme; different
#     ε_rel runs are distinct variants and must split series.
#   - `anderson_m`, `picard_alpha`: Picard inner-solver knobs — different
#     values change the algorithm.
#   - `polar_method`, `polar_sigma_power`, `polar_norm_dir`: polar/operator
#     choices — different values change the algorithm.


class UntaggedSweepError(RuntimeError):
    """A populated log group has no manifest or empty scope.

    Fix:
      1. New sweep that bypassed submit.sh — re-run via
         ``slurm_scripts/submit.sh`` with ``SWEEP_SCOPE`` set, or hand-
         write ``logs/<group>/run_info/meta.json`` with the schema in
         ``manifest.py``.
      2. Old log dir restored without metadata — re-derive scope and
         drop a meta.json next to the run logs.
    For ad-hoc exploratory work, ``load_manifests(strict=False)`` opts
    out of the check.
    """


_LOAD_MANIFESTS_CACHE: dict[tuple[str, bool], tuple[tuple, list[dict]]] = {}


def _groups_with_run_info(logs_root: str) -> list[str]:
    """Sorted group names under ``logs_root`` that have a ``run_info/`` dir.

    Replaces ``root.glob("*/run_info")``: one ``scandir`` of ``logs/`` plus one
    ``stat`` per child, and it never follows into the group directories.
    """
    try:
        children = sorted(e.name for e in os.scandir(logs_root) if e.is_dir())
    except OSError:
        return []
    return [g for g in children
            if os.path.isdir(os.path.join(logs_root, g, "run_info"))]


def _meta_stats(logs_root: str, groups: list[str]
                ) -> list[tuple[str, int, int]]:
    """``(group, meta.json mtime_ns, size)`` per group, missing file → (0, 0)."""
    def one(group: str) -> tuple[str, int, int]:
        try:
            st = os.stat(os.path.join(logs_root, group, "run_info", "meta.json"))
        except OSError:
            return (group, 0, 0)
        return (group, st.st_mtime_ns, st.st_size)

    return parallel_map(one, groups)


def load_manifests(logs_root: str = "../logs", strict: bool = True) -> list[dict]:
    """Return one manifest dict per sweep that has both a ``meta.json`` and
    at least one populated run output. Manifests without runs are skipped
    silently.

    With ``strict=True`` (default), raises ``UntaggedSweepError`` if any
    populated log dir lacks a manifest, has corrupt JSON, or empty scope.
    With ``strict=False``, those groups are returned with stub manifests
    flagged (`_untagged`, `_corrupt`, `_empty_scope`) so the caller can
    decide.

    Cached by signature ``(logs_root_mtime, sorted (group, meta_mtime,
    meta_size))`` so analysis loops that call this once per ``load_runs``
    don't re-walk 178+ directories. New groups bump logs_root mtime;
    edited meta.json bumps its own mtime.
    """
    root = Path(logs_root)
    if not root.exists():
        return []
    try:
        root_mtime = root.stat().st_mtime_ns
        groups = _groups_with_run_info(logs_root)
        # One `stat` per meta.json instead of the four the exists()+stat()+
        # exists()+stat() form cost, and issued concurrently: this runs on
        # every `load_runs` call (it is what decides the cache hit) over
        # 500+ groups on a shared parallel filesystem, where the syscall is
        # latency-bound and the GIL is released for its duration.
        meta_sig = tuple(_meta_stats(logs_root, groups))
        sig = (root_mtime, meta_sig)
        cache_key = (str(root.resolve()), strict)
        cached = _LOAD_MANIFESTS_CACHE.get(cache_key)
        if cached is not None and cached[0] == sig:
            return [dict(m) for m in cached[1]]
    except OSError:
        sig = None
        cache_key = None
        groups = _groups_with_run_info(logs_root)

    prescan_groups(groups, logs_root)
    manifests: list[dict] = []
    bad: list[tuple[str, str]] = []  # (group, reason)
    for group in groups:
        run_info = root / group / "run_info"
        if not has_runs(group, logs_root):
            continue
        meta_path = run_info / "meta.json"
        try:
            raw = meta_path.read_text()
        except OSError:
            raw = None
        if raw is not None:
            try:
                m = json.loads(raw)
            except json.JSONDecodeError:
                m = {"group": group, "scope": [], "purpose": "", "_corrupt": True}
                bad.append((group, "corrupt meta.json"))
        else:
            m = {"group": group, "scope": [], "purpose": "", "_untagged": True}
            bad.append((group, "no meta.json"))
        m.setdefault("group", group)
        # Empty scope on a present manifest also counts as bad — except
        # for explicitly tagged pilots, which get scope=[] by design.
        if not m.get("scope") and not m.get("_untagged") and not m.get("_corrupt"):
            # A pilot or explicitly-empty-scope group; allowed only when the
            # group name signals pilot via PILOT_SUFFIXES OR the manifest
            # carries an explicit "pilot" marker. Today we accept empty-scope
            # silently — analysis filters by scope membership and pilots fall
            # out automatically.
            pass
        manifests.append(m)
    if strict and bad:
        details = "\n".join(f"  {g}: {why}" for g, why in bad)
        raise UntaggedSweepError(
            f"{len(bad)} log group(s) without valid manifest:\n{details}\n"
            f"See lora_playground/manifest.py for fix instructions."
        )
    if cache_key is not None and sig is not None:
        _LOAD_MANIFESTS_CACHE[cache_key] = (sig, manifests)
    return [dict(m) for m in manifests]


def warn_untagged(manifests: list[dict]) -> list[str]:
    """Return group names that lack a manifest or have an empty scope.

    Print is left to the caller — notebooks typically print once near the
    top of a section so the warning is visible without polluting every cell.
    """
    bad = []
    for m in manifests:
        if m.get("_corrupt") or m.get("_untagged") or not m.get("scope"):
            bad.append(m["group"])
    return bad


def live_manifests_newest_first(manifests: list[dict]) -> list[dict]:
    """Filter to non-corrupt, scope-tagged manifests, sorted by
    ``submitted_at`` descending (newest first → wins ``merge_runs`` dedup).
    """
    live = [m for m in manifests
            if not m.get("_corrupt") and not m.get("_untagged")]
    live.sort(key=lambda m: m.get("submitted_at", ""), reverse=True)
    return live
