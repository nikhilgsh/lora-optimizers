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

from ..run_records import RUNTIME_FIELDS
from .dedup import _hashable
from .loading import has_runs, iter_sweep_raw, prescan_groups, scan_epoch
from .style import DIVERGE_THRESHOLD


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


def _stitch_runs(runs: list[tuple[int, int, dict, list[dict]]]
                 ) -> tuple[dict, list[dict]]:
    """Stitch all same-key colliding runs into one step-monotonic trajectory.

    ``runs`` is a list of ``(final_step, idx, cfg, evs)`` sharing a dedup key.
    Evals are concatenated in order of their FIRST step (earliest segment
    first); for each run only the evals whose ``step`` exceeds the running max
    are appended. This is a strict generalization of "longest trajectory wins":

    - **Resume continuation** (the case this fixes): an original partial run
      (e.g. steps 250→8000) and its ``--resume_from`` continuation (8250→9000)
      collide on the same key. Stitching yields the full 250→9000 trajectory
      instead of dropping the pre-resume segment.
    - **Overlapping rerun** (e.g. in-flight 0→3000 vs completed 0→9000, or two
      identical 0→9000 reruns): same first step → ordered longest-first, so the
      longer run populates fully and the shorter contributes nothing new. The
      result equals the old longest-wins-then-priority behavior exactly.

    The representative cfg is the run with the greatest ``final_step`` (ties
    broken by group priority — lowest ``idx``), so leaderboard "final" reads the
    most-progressed run's config/last eval, matching the prior winner selection.
    """
    rep_final, rep_idx, rep_cfg, _ = max(runs, key=lambda r: (r[0], -r[1]))

    def _first_step(evs: list[dict]) -> int:
        return evs[0]["step"] if evs else 0

    # Earliest segment first; among equal first-steps, longer run first, then
    # higher group priority (lower idx) — so reruns reduce to longest-wins.
    ordered = sorted(runs, key=lambda r: (_first_step(r[3]), -r[0], r[1]))
    stitched: list[dict] = []
    max_step: int | None = None
    for _final, _idx, _cfg, evs in ordered:
        for e in evs:
            if max_step is None or e["step"] > max_step:
                stitched.append(e)
                max_step = e["step"]
    return rep_cfg, stitched


def merge_runs(group_priority: Iterable[str],
               key_fn: Callable[[dict], tuple],
               filter_fn: Callable[[dict], bool] | None = None,
               cfg_postprocess: Callable[[dict, str], None] | None = None,
               logs_root: str = "../logs",
               *, strict_hidden_axes: bool = True,
               pre_filter: Callable[[dict, str], bool] | None = None,
               group_filter: Callable[[str], bool] | None = None,
               ) -> list[tuple[dict, list[dict]]]:
    """Merge sweeps, deduplicating by ``key_fn(cfg)``.

    Dedup rule: **same-key runs are stitched into one step-monotonic
    trajectory** (see :func:`_stitch_runs`). For runs covering disjoint step
    ranges — an original run and its ``--resume_from`` continuation — this
    concatenates the segments into the full trajectory. For overlapping ranges
    (an in-flight rerun vs a completed older run, or duplicate reruns) it
    reduces to the prior behavior: **longest trajectory wins**, ties broken by
    group priority order (earlier group in ``group_priority`` wins).

    Robustness check (``strict_hidden_axes=True`` default): if two runs
    collide on ``key_fn(cfg)`` but differ on a "hidden" cfg axis (e.g.
    ``lora_plus_multiplier``, ``muon_ns_steps``, ``training_mode``), raise
    a clear error pointing at which axis the dedup key is missing. This
    catches the silent-confounder failure mode where one cfg axis was
    forgotten in the key and runs are arbitrarily collapsed.

    Pass ``strict_hidden_axes=False`` only when you genuinely want axes
    collapsed (e.g., showing the best-of across m∈{1,4} as one curve).

    Two optional early-rejection hooks let a caller keep ``cfg_postprocess``
    off runs it is going to discard anyway — the postprocess is the expensive
    part of a whole-tree pass, and running it on every run in ``logs/`` to
    return a few dozen was the dominant cost of ``loader.load_runs``:

    ``group_filter(group)``
        Consulted before the group's logs are touched at all. Only sound for a
        predicate that depends on the group NAME alone.
    ``pre_filter(raw_cfg, group)``
        Consulted on each run's cfg *as parsed from the log*, before
        ``log_group`` is written and before ``cfg_postprocess`` runs. Must
        return True whenever it cannot rule the run out; a False is a promise
        that ``filter_fn`` would also have rejected the postprocessed cfg.
        ``loader.load_runs`` derives one that only rejects on cfg fields its
        postprocess provably leaves alone (see ``loader._build_pushdown``).

    Both default to None, in which case every run takes the original path.
    """
    prio = {g: i for i, g in enumerate(group_priority)}
    ordered = sorted(prio.items(), key=lambda kv: kv[1])
    collected: dict[tuple, list[tuple[int, int, dict, list[dict]]]] = {}
    with scan_epoch():
        # One concurrent pass over the tree; `has_runs` and the per-group
        # freshness signature then read the same scan instead of re-walking.
        prescan_groups([g for g, _ in ordered
                        if group_filter is None or group_filter(g)], logs_root)
        for group, idx in ordered:
            if group_filter is not None and not group_filter(group):
                continue
            if not has_runs(group, logs_root):
                continue
            for cfg, evs in iter_sweep_raw(group, logs_root):
                if pre_filter is not None and not pre_filter(cfg, group):
                    continue
                # `iter_sweep_raw` yields the cache's own dicts; copy before
                # writing `log_group` / enriching. Copying here rather than
                # inside the loader means a rejected run costs no allocation.
                cfg = dict(cfg)
                cfg["log_group"] = group
                if cfg_postprocess is not None:
                    cfg_postprocess(cfg, group)
                if filter_fn is not None and not filter_fn(cfg):
                    continue
                k = key_fn(cfg)
                final_step = evs[-1]["step"] if evs else 0
                bucket = collected.get(k)
                if bucket is None:
                    collected[k] = [(final_step, idx, cfg, evs)]
                    continue
                if strict_hidden_axes:
                    # Compare against the bucket's first cfg; if all members
                    # agree with it on non-runtime fields they mutually agree
                    # (transitive).
                    ex_cfg = bucket[0][2]
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
                bucket.append((final_step, idx, cfg, evs))
    return [_stitch_runs(runs) for runs in collected.values()]


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
