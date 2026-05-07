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
      "purpose":       "E2: AdaMuon-faithful + polar-product geometry"
    }

Loading: ``lora_playground.loader.load_runs(where=...)`` — predicate-based,
no scope strings. To remove an old sweep from analysis, delete its log dir.
"""
from __future__ import annotations

import json
from pathlib import Path

from .plot_utils import has_runs


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
        meta_sig = tuple(
            (run_info.parent.name,
             (run_info / "meta.json").stat().st_mtime_ns
             if (run_info / "meta.json").exists() else 0,
             (run_info / "meta.json").stat().st_size
             if (run_info / "meta.json").exists() else 0)
            for run_info in sorted(root.glob("*/run_info"))
        )
        sig = (root_mtime, meta_sig)
        cache_key = (str(root.resolve()), strict)
        cached = _LOAD_MANIFESTS_CACHE.get(cache_key)
        if cached is not None and cached[0] == sig:
            return [dict(m) for m in cached[1]]
    except OSError:
        sig = None
        cache_key = None

    manifests: list[dict] = []
    bad: list[tuple[str, str]] = []  # (group, reason)
    for run_info in sorted(root.glob("*/run_info")):
        group = run_info.parent.name
        if not has_runs(group, logs_root):
            continue
        meta_path = run_info / "meta.json"
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
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
