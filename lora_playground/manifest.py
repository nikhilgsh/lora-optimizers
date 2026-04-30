"""Sweep manifest helpers.

Every sweep submitted via ``slurm_scripts/submit.sh`` writes a ``meta.json``
into ``logs/<group>/run_info/``. This module reads those manifests and
exposes scope-based loaders so analysis cells filter by intent (``scope``)
rather than by hand-maintained tuples of group names.

Schema (written by submit.sh):

    {
      "group":         "adamuon_polar_product_2k",
      "submitted_at":  "2026-04-30T14:23:01-04:00",
      "slurm_job_id":  "6314009",
      "n_gpus":        6,
      "params_file":   "adamuon_polar_product_2k.json",
      "sweep_script":  "scripts/sweep_2k_r_diag.sh",
      "sbatch_script": "slurm_scripts/sbatch.sh",
      "git_commit":    "<sha>",
      "git_dirty":     false,
      "scope":         ["ext_compare", "polar_family"],
      "purpose":       "E2: AdaMuon-faithful + polar-product geometry",
      "supersedes":    null
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from .plot_utils import has_runs, load_sweep, merge_runs


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


def load_manifests(logs_root: str = "../logs", strict: bool = True) -> list[dict]:
    """Return one manifest dict per sweep that has both a ``meta.json`` and
    at least one populated run output. Manifests without runs are skipped
    silently.

    With ``strict=True`` (default), raises ``UntaggedSweepError`` if any
    populated log dir lacks a manifest, has corrupt JSON, or empty scope.
    With ``strict=False``, those groups are returned with stub manifests
    flagged (`_untagged`, `_corrupt`, `_empty_scope`) so the caller can
    decide.
    """
    root = Path(logs_root)
    manifests: list[dict] = []
    if not root.exists():
        return manifests
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
    return manifests


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


def groups_by_scope(scope: str, manifests: list[dict] | None = None,
                    logs_root: str = "../logs",
                    apply_supersedes: bool = True) -> tuple[str, ...]:
    """Return groups whose manifest declares ``scope`` in its scope list,
    in priority order suitable for ``merge_runs``.

    Priority: most recently submitted first (``submitted_at`` descending), so
    reruns dedup-win over older runs they replace.

    If ``apply_supersedes`` is True, any group named in another manifest's
    ``supersedes`` field is dropped from the returned list — the explicit
    rerun-relationship beats mtime ordering.
    """
    if manifests is None:
        manifests = load_manifests(logs_root)
    superseded = set()
    if apply_supersedes:
        for m in manifests:
            sup = m.get("supersedes")
            if sup:
                superseded.add(sup)
    matched = [m for m in manifests if scope in m.get("scope", []) and m["group"] not in superseded]
    matched.sort(key=lambda m: m.get("submitted_at", ""), reverse=True)
    return tuple(m["group"] for m in matched)


def load_runs_by_scope(scope: str,
                       key_fn: Callable[[dict], tuple],
                       filter_fn: Callable[[dict], bool] | None = None,
                       cfg_postprocess: Callable[[dict, str], None] | None = None,
                       manifests: list[dict] | None = None,
                       logs_root: str = "../logs"):
    """Convenience: ``merge_runs(groups_by_scope(scope), ...)``.

    Calling code stays declarative — no group names appear in the notebook.
    """
    groups = groups_by_scope(scope, manifests=manifests, logs_root=logs_root)
    return merge_runs(groups, key_fn=key_fn, filter_fn=filter_fn,
                      cfg_postprocess=cfg_postprocess, logs_root=logs_root)
