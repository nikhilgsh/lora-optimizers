"""Merge sweeps with dedup-key collision detection, plus diverged-run
filtering for the figure pipeline.

The dedup rule in :func:`merge_runs` is **longest trajectory wins, ties
broken by group priority** — handles the common case where an in-flight
rerun has the same key as a completed older run. Hidden-axis robustness
catches the silent-confounder failure mode where a cfg axis was forgotten
in the dedup key.
"""
from __future__ import annotations

from typing import Callable, Iterable

from .dedup import _hashable
from .loading import has_runs, load_sweep
from .style import DIVERGE_THRESHOLD


# Cfg fields that legitimately differ between otherwise-identical runs (run
# provenance, instrumentation knobs, file paths). Canonical definition;
# ``loader.RUNTIME_FIELDS`` re-exports this name so dedup-key construction
# and the hidden-axis collision check share one source of truth (drift
# between two parallel lists previously caused silent collisions on
# ``_optim_steps`` differences — see git log for 2026-05-06 fix).
RUNTIME_FIELDS: frozenset[str] = frozenset({
    "git_commit", "command", "log_group",
    # Provenance fields (Phase 1 cfg-event enrichment, 2026-05-14): dirty-tree
    # state captured at submission. Loader's invariants/dirty_attestations
    # layers consume them but dedup must not split otherwise-identical runs.
    "git_dirty", "git_diff_sha", "git_untracked_files",
    # Phase 4 (2026-05-14): execution-scope provenance. Drive loader exclusion
    # decisions but don't define the series.
    "execution_source_sha", "execution_source_paths",
    "execution_source_dirty", "execution_env", "execution_env_sha",
    # Loader-assigned per-run identifier; see loader._enrich_cfg.
    "run_id", "_log_filename",
    "wandb_project", "wandb_run_name",
    "device", "tf32", "no_tf32",
    # Diagnostic toggles (none affect optimizer math). Both current names
    # and legacy aliases from before the 2026-05-12 diagnostics refactor.
    "log_basic_diagnostics", "log_heavy_diagnostics",
    "log_optim_diagnostics", "no-log_optim_diagnostics",
    "optim_diagnostics_every",
    "diagnostics",   # canonical block (Phase 1, 2026-05-14)
    "profile_steps", "profile_dir",
    "_optim_steps",
    "train_file", "eval_file",
    # Per-task checkpoint plumbing injected by submit.sh / the disbatch template
    # (one dir per task). Two runs of the SAME config on different hardware (or a
    # resubmit) differ only in these paths; they are the same series, so dedup
    # must not split on them. Otherwise a cross-hardware re-run (e.g. the same
    # lr-sweep on both `_blackwell` and `_gpuxl_h200`) trips the label-collision
    # guard despite being one algorithm.
    "checkpoint_dir", "resume_from",
    # CLI override flags whose canonical resolved value is promoted by
    # `_enrich_cfg` to a top-level scalar (e.g. `effective_picard_iters`).
    "picard_iters_override",
})


def _hidden_axis_diffs(cfg_a: dict, cfg_b: dict) -> list[tuple]:
    """Non-runtime cfg fields where ``cfg_a`` and ``cfg_b`` disagree.

    Treats missing-vs-present as a difference — this is the asymmetric
    failure mode that previously hid collisions. An older cfg without a
    field (e.g. ``picard_iters_override``) coexisting with a newer cfg
    that has it would not trigger the prior fingerprint check, because
    that check only compared a fixed allow-list of axes and silently
    skipped fields outside it.

    Returns a list of ``(field, value_in_a, value_in_b)`` tuples; missing
    values surface as the literal string ``"<missing>"``.
    """
    _MISSING = object()
    # Mirror `_denylist_key` / `series_id` exclusion rules: ignore runtime
    # metadata, underscore-prefixed enrichment namespaces, and dict-valued
    # derived composites. Treat None and absent as equivalent.
    candidate_keys = set(cfg_a) | set(cfg_b)
    keys = {
        k for k in candidate_keys
        if k not in RUNTIME_FIELDS
        and not k.startswith("_")
        and not isinstance(cfg_a.get(k), dict)
        and not isinstance(cfg_b.get(k), dict)
    }
    diffs = []
    for k in sorted(keys):
        va = cfg_a.get(k, _MISSING)
        vb = cfg_b.get(k, _MISSING)
        if va is _MISSING:
            va = None
        if vb is _MISSING:
            vb = None
        if va is None and vb is None:
            continue
        if va != vb:
            diffs.append((
                k,
                "<missing>" if va is None else va,
                "<missing>" if vb is None else vb,
            ))
    return diffs


def merge_runs(group_priority: Iterable[str],
               key_fn: Callable[[dict], tuple],
               filter_fn: Callable[[dict], bool] | None = None,
               cfg_postprocess: Callable[[dict, str], None] | None = None,
               logs_root: str = "../logs",
               *, strict_hidden_axes: bool = True) -> list[tuple[dict, list[dict]]]:
    """Merge sweeps, deduplicating by ``key_fn(cfg)``.

    Dedup rule: **longest trajectory wins**, ties broken by group priority
    order (earlier group in ``group_priority`` wins). This handles the
    common case where an in-flight rerun has the same key as a completed
    older run — the completed run keeps its slot until the rerun catches
    up, and only takes over when the rerun reaches the same final step.

    Robustness check (``strict_hidden_axes=True`` default): if two runs
    collide on ``key_fn(cfg)`` but differ on a "hidden" cfg axis (e.g.
    ``lora_plus_multiplier``, ``muon_ns_steps``, ``training_mode``), raise
    a clear error pointing at which axis the dedup key is missing. This
    catches the silent-confounder failure mode where one cfg axis was
    forgotten in the key and runs are arbitrarily collapsed.

    Pass ``strict_hidden_axes=False`` only when you genuinely want axes
    collapsed (e.g., showing the best-of across m∈{1,4} as one curve).
    """
    prio = {g: i for i, g in enumerate(group_priority)}
    best: dict[tuple, tuple[int, int, dict, list[dict]]] = {}
    for group, idx in sorted(prio.items(), key=lambda kv: kv[1]):
        if not has_runs(group, logs_root):
            continue
        for cfg, evs in load_sweep(group, logs_root):
            cfg["log_group"] = group
            if cfg_postprocess is not None:
                cfg_postprocess(cfg, group)
            if filter_fn is not None and not filter_fn(cfg):
                continue
            k = key_fn(cfg)
            final_step = evs[-1]["step"] if evs else 0
            existing = best.get(k)
            if existing is None:
                best[k] = (final_step, idx, cfg, evs)
                continue
            ex_step, ex_idx, ex_cfg, _ex_evs = existing
            if strict_hidden_axes:
                diffs = _hidden_axis_diffs(ex_cfg, cfg)
                if diffs:
                    field_summary = ", ".join(
                        f"{f}={va!r}↔{vb!r}" for f, va, vb in diffs[:5]
                    )
                    if len(diffs) > 5:
                        field_summary += f", … (+{len(diffs)-5} more)"
                    raise ValueError(
                        f"merge_runs: dedup key collision on {k!r} between cfgs "
                        f"that differ on non-runtime field(s): {field_summary}. "
                        f"Two distinct runs would be silently collapsed. Either "
                        f"include these fields in key_fn (or use the deny-list "
                        f"default in load_runs), or pass strict_hidden_axes=False "
                        f"if collapsing is intended. "
                        f"Groups: {ex_cfg.get('log_group')!r} vs {cfg.get('log_group')!r}."
                    )
            if final_step > ex_step or (final_step == ex_step and idx < ex_idx):
                best[k] = (final_step, idx, cfg, evs)
    return [(cfg, evs) for _, _, cfg, evs in best.values()]


# ─── diverged-run filtering ───────────────────────────────────────────────────

def max_loss(evs: list[dict]) -> float:
    """Max eval_loss across `evs`, NaN-safe (NaN counts as +∞ → diverged)."""
    return max((float("inf") if (e["eval_loss"] != e["eval_loss"]) else e["eval_loss"])
               for e in evs)


def split_diverged(runs, threshold: float = DIVERGE_THRESHOLD,
                   hard_max: float = float("inf")):
    def _div(cfg, evs):
        final = evs[-1]["eval_loss"]
        if final != final or max_loss(evs) != max_loss(evs):  # NaN guard
            return True
        # Early-termination guard: run that died in the first 10% of training
        # counts as diverged. Catches NaN-killed runs whose pre-death eval
        # happened to be finite.
        max_steps = cfg.get("max_steps")
        if max_steps is not None and evs[-1]["step"] < max_steps * 0.1:
            return True
        return final >= threshold or max_loss(evs) >= hard_max

    keep = [(c, e) for c, e in runs if not _div(c, e)]
    drop = [(c, e) for c, e in runs if _div(c, e)]
    return keep, drop


def report_diverged(drop, label_fn: Callable[[dict], str]) -> None:
    for cfg, evs in sorted(drop, key=lambda x: label_fn(x[0])):
        print(f"  [filtered diverged] {label_fn(cfg):<24s} "
              f"max={max_loss(evs):.3f} final={evs[-1]['eval_loss']:.3f}")


def best_run(runs, filter_fn: Callable[[dict], bool]):
    """Return (cfg, evs) with lowest final eval_loss matching filter_fn, else None."""
    matches = [(c, e) for c, e in runs if filter_fn(c)]
    if not matches:
        return None
    return min(matches, key=lambda x: x[1][-1]["eval_loss"])
