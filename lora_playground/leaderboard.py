"""Speed-to-target leaderboard metrics.

For optimizer comparison we express a method's performance as the **fraction
of the training horizon** it needs to reach the best AdamW baseline's final
eval loss:

    perf = (step where eval_loss first crosses L*_adamw) / horizon

where ``L*_adamw`` is the lowest *final* eval loss over the AdamW lr sweep at
the same (model, dataset, rank). The crossing step is linearly interpolated
between the bracketing evals (see ``reach_fraction``) so the metric does not
inherit the eval grid's discretization. 1.0 means "needed AdamW's whole run to match
AdamW's final loss"; 0.5 means "reached it in half the steps"; NaN means the
method never reached it within the horizon. Lower is better.

Two summaries per method:
  - ``frac_best_lr``  — perf at the method's best lr (lowest final loss).
  - ``frac_lr_avg``   — mean perf over {best lr, one grid-step below, one
                        grid-step above}. NaN when the best lr sits at a swept
                        boundary (no lower or no higher swept lr) — i.e. the
                        method is lr-pinned — or when any of the three never
                        reaches the target.

This module is pure (no I/O); callers supply already-loaded
``(cfg, history)`` runs from ``lora_playground.loader.load_runs``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

# eval cadence varies (250) and some sweeps stopped at 8970 instead of 9000;
# treat a run as completed if its last eval is within this many steps of the
# horizon. Mirrors the notebooks' "final = run that reached max_steps" rule
# while tolerating the 8970/9000 split.
DEFAULT_COMPLETION_SLACK = 300


def is_final(last_step, horizon: int,
             completion_slack: int = DEFAULT_COMPLETION_SLACK) -> bool:
    """Whether a run's last eval step counts as having reached the horizon.

    Single source of truth for the completed-vs-in-flight decision, shared by the
    loader (`labeled_completed_runs`) and the plotter
    (`plotting.figures.compare_variants_figure`). A run is "final" if its last
    eval is within ``completion_slack`` of the horizon — so ~8970-step one-epoch
    runs (Tulu) count, but in-flight runs (e.g. step 250 of 9000) do NOT and must
    be rendered as partial. Keep these two callers in agreement via this helper;
    duplicating the inequality is exactly how partial runs leaked onto the
    final-vs-lr panel.
    """
    return last_step is not None and last_step >= horizon - completion_slack


def reach_fraction(history: list[dict], target: float, horizon: int) -> float:
    """Fraction of ``horizon`` at which ``eval_loss`` first drops to ``target``.

    The crossing is linearly interpolated between the last eval above the
    target and the first eval at-or-below it. Reading the crossing off the
    eval grid instead would always round it *up* to the next eval step,
    understating speedups by up to one grid cell (e.g. a true crossing at
    step 7319 on a 250-step grid reads as 7500 → 1.20× instead of 1.23×).
    When the very first eval is already at-or-below the target there is no
    bracketing point, so that eval's step is used as-is.

    Returns NaN if the target is never reached (or ``target`` is NaN).
    """
    if target is None or math.isnan(target):
        return math.nan
    prev_step = prev_loss = None
    for ev in sorted(history, key=lambda e: e.get("step", 0)):
        loss = ev.get("eval_loss")
        if loss is None or not math.isfinite(loss):
            continue
        if loss <= target:
            if prev_loss is None:
                return ev["step"] / horizon
            frac = (prev_loss - target) / (prev_loss - loss)
            return (prev_step + frac * (ev["step"] - prev_step)) / horizon
        prev_step, prev_loss = ev["step"], loss
    return math.nan


def speedup_from_frac(frac: float) -> float:
    """Speedup-vs-AdamW = ``1 / reach_fraction`` (higher is better).

    A fraction of 0.5 (reached AdamW's final loss in half the steps) becomes a
    2.0× speedup; AdamW's own ≈1.0 fraction becomes ≈1.0×. NaN (target never
    reached) passes through as NaN. This is the display-layer inverse of the
    internal fraction metric — the fraction stays canonical for
    ``performance_profile`` (which is direction-sensitive); presentation layers
    (the leaderboard doc, notebooks) call this to show speedup instead."""
    if frac is None or math.isnan(frac):
        return math.nan
    return 1.0 / frac


def labeled_completed_runs(
    runs: list[tuple[dict, list[dict]]],
    variant_key: Callable[[dict], str | None],
    *,
    horizon: int,
    completion_slack: int = DEFAULT_COMPLETION_SLACK,
) -> dict[str, dict[float, tuple[float, list[dict]]]]:
    """Bucket completed runs into ``{variant_label: {lr: (final_loss, history)}}``.

    A run is kept only if ``variant_key(cfg)`` is non-None (it belongs to a
    panel) and it completed (last eval step >= horizon - completion_slack).
    When several runs share a (label, lr) — e.g. a resub continuation — the
    longest-trajectory one wins, matching the notebooks' dedup.
    """
    by: dict[str, dict[float, tuple[float, list[dict], int]]] = defaultdict(dict)
    for cfg, hist in runs:
        if not hist:
            continue
        label = variant_key(cfg)
        if label is None:
            continue
        last = max(hist, key=lambda e: e.get("step", 0))
        last_step = last.get("step", 0)
        if not is_final(last_step, horizon, completion_slack):
            continue
        try:
            lr = float(cfg["lr"])
        except (KeyError, TypeError, ValueError):
            continue
        prev = by[label].get(lr)
        if prev is None or last_step > prev[2]:
            by[label][lr] = (last["eval_loss"], hist, last_step)
    return {v: {lr: (fl, h) for lr, (fl, h, _) in d.items()} for v, d in by.items()}


def merge_labeled(*labeled_dicts) -> dict[str, dict[float, tuple[float, list[dict]]]]:
    """Union several ``labeled_completed_runs`` outputs (multiple panels feeding
    one (model, rank) section). On a (label, lr) collision the longer history
    wins."""
    out: dict[str, dict[float, tuple[float, list[dict]]]] = defaultdict(dict)
    for d in labeled_dicts:
        for label, by_lr in d.items():
            for lr, (fl, hist) in by_lr.items():
                prev = out[label].get(lr)
                if prev is None or len(hist) > len(prev[1]):
                    out[label][lr] = (fl, hist)
    return dict(out)


def performance_profile(
    perf_matrix: dict[str, dict[str, float]],
    *,
    workloads: list[str] | None = None,
    max_tau: float = 4.0,
    n_tau: int = 64,
) -> list[dict]:
    """Cross-workload robustness ranking, AlgoPerf-style (performance profiles).

    `perf_matrix[variant][workload] = perf` where ``perf`` is the speed metric
    (fraction-of-steps-to-target; lower is better). Absent or NaN entries mean
    the variant was not run / never reached the target on that workload.

    For each workload ``w`` the best (smallest) perf over the variants present
    is ``b_w``; a variant's performance ratio is ``r = perf / b_w >= 1`` (1.0 ⇒
    fastest on that workload). The performance profile is
    ``rho(tau) = fraction of COVERED workloads with r <= tau``; the
    ``robustness_score`` is the normalised area under ``rho`` over
    ``tau in [1, max_tau]`` (1.0 ⇒ fastest on every workload it ran, 0 ⇒ always
    worse than ``max_tau x`` the best).

    The score is computed over the workloads the variant actually ran
    (``coverage``), NOT all workloads — so it measures "robust where tested",
    and a variant run on one easy cell is NOT spuriously ranked above a broadly
    tested one. Callers MUST read ``coverage`` alongside ``robustness_score``:
    a high score at low coverage is a hypothesis, not a result. Rows are sorted
    by coverage first, then score, to keep that honest.

    Returns one dict per variant: ``{variant, coverage, n_workloads,
    robustness_score, mean_ratio, ratios}`` where ``ratios`` maps workload →
    ratio-to-best (only covered workloads).
    """
    if workloads is None:
        workloads = sorted({w for d in perf_matrix.values() for w in d})

    def _ok(x):
        return x is not None and not (isinstance(x, float) and math.isnan(x))

    best_w = {}
    for w in workloads:
        vals = [perf_matrix[v][w] for v in perf_matrix
                if w in perf_matrix[v] and _ok(perf_matrix[v][w])]
        if vals:
            best_w[w] = min(vals)

    taus = [1.0 + (max_tau - 1.0) * i / (n_tau - 1) for i in range(n_tau)]
    rows = []
    for v, d in perf_matrix.items():
        ratios = {w: d[w] / best_w[w] for w in workloads
                  if w in d and _ok(d.get(w)) and w in best_w}
        cov = len(ratios)
        if cov == 0:
            continue
        rs = list(ratios.values())
        # area under rho(tau) = mean over tau-grid of (fraction with r <= tau)
        area = sum(sum(1 for r in rs if r <= t) / cov for t in taus) / len(taus)
        rows.append(dict(
            variant=v, coverage=cov, n_workloads=len(workloads),
            robustness_score=area, mean_ratio=sum(rs) / cov, ratios=ratios,
        ))
    rows.sort(key=lambda r: (-r["coverage"], -r["robustness_score"]))
    return rows


def _adjacent(sorted_lrs: list[float], lr: float) -> tuple[float | None, float | None]:
    i = sorted_lrs.index(lr)
    low = sorted_lrs[i - 1] if i > 0 else None
    high = sorted_lrs[i + 1] if i < len(sorted_lrs) - 1 else None
    return low, high


def leaderboard_rows(
    labeled: dict[str, dict[float, tuple[float, list[dict]]]],
    *,
    horizon: int,
    baseline_label: str = "AdamW",
    unreached_avg_value: float = 1.0,
) -> tuple[list[dict], float]:
    """Compute one leaderboard row per variant.

    Returns ``(rows, target)`` where ``target`` is the best AdamW final loss
    (the speed target) and each row is a dict::

        {variant, best_lr, final_at_best, frac_best_lr, frac_lr_avg,
         lr_avg_components, n_lrs}

    ``lr_avg_components`` is a list of ``(lr, frac)`` for the three averaged
    lrs (or None when lr-pinned).

    lr-average semantics: NaN (``—``) is reserved for the *pinned* case — the
    best lr sits at a swept boundary, so one of {3x-low, best, 3x-high} was
    never swept. When all three were swept but a neighbor never reaches the
    target within the horizon, that neighbor's fraction is clamped to
    ``unreached_avg_value`` (default 1.0 = "needed at least the full horizon")
    so the average stays defined and reflects lr-robustness. The per-lr
    ``frac_best_lr`` column is NOT clamped (it shows NaN if the best lr itself
    never reaches the target)."""
    base = labeled.get(baseline_label, {})
    target = min((fl for fl, _ in base.values()), default=math.nan)

    rows: list[dict] = []
    for variant, by_lr in labeled.items():
        if not by_lr:
            continue
        best_lr = min(by_lr, key=lambda lr: by_lr[lr][0])
        final_at_best = by_lr[best_lr][0]
        frac_best = reach_fraction(by_lr[best_lr][1], target, horizon)

        lrs_sorted = sorted(by_lr)
        low, high = _adjacent(lrs_sorted, best_lr)
        if low is None or high is None:
            frac_avg = math.nan
            comps = None
        else:
            comps = [(x, reach_fraction(by_lr[x][1], target, horizon))
                     for x in (low, best_lr, high)]
            clamped = [unreached_avg_value if math.isnan(f) else f for _, f in comps]
            frac_avg = sum(clamped) / len(clamped)

        rows.append(dict(
            variant=variant,
            best_lr=best_lr,
            final_at_best=final_at_best,
            frac_best_lr=frac_best,
            frac_lr_avg=frac_avg,
            lr_avg_components=comps,
            n_lrs=len(by_lr),
        ))
    return rows, target
