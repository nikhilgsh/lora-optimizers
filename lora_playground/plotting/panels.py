"""Single-axis panel renderers and the per-rank leaderboard.

`plot_eta_vs_final` renders the left panel (η vs final loss with diverged
markers and OOR clamping); `plot_best_eta_curves` renders the right panel
(best-η training trajectories). `plot_leaderboard_by_rank` is the
single-panel best-eval-vs-rank bar/line chart.
"""
from __future__ import annotations

import math
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .dedup import assert_label_discriminates
from .style import (
    AXIS_LABEL_FONTSIZE, BASELINE_COLOR, BASELINE_LS_CURVE,
    BASELINE_LW_CURVE, BASELINE_MARKER, BASELINE_ZORDER, LEGEND_KW,
    LINE_WIDTH, MARKER_SIZE, PANEL_TITLE_FONTSIZE, TICK_LABEL_FONTSIZE,
)


def _infer_min_step(runs) -> int | None:
    """Return the target horizon by reading `max_steps` from the input cfgs.

    Multiple max_steps in the same `runs` list (mixed-regime) → pick the
    MAX, since the left panel can only report "final" for runs that hit
    the highest target. Returns None if no cfg carries `max_steps`.
    """
    targets = {int(c["max_steps"]) for c, _ in runs if "max_steps" in c}
    return max(targets) if targets else None


def _non_divergent_finals(
    runs, lora_r: int | None, divergent_ratio: float,
) -> tuple[list[float], float | None]:
    """Final eval_losses for runs at `lora_r` (or all ranks if None) that
    are within `divergent_ratio * best`. NaN/inf are excluded.

    Returns `(converged_finals, best_final)` or `([], None)` when no finite
    final losses are present.
    """
    finite = (
        e[-1]["eval_loss"] for c, e in runs
        if e and (lora_r is None or c.get("lora_r") == lora_r)
        and isinstance(e[-1].get("eval_loss"), (int, float))
        and math.isfinite(e[-1]["eval_loss"])
    )
    finals = list(finite)
    if not finals:
        return [], None
    best = min(finals)
    cap = divergent_ratio * best
    return [v for v in finals if v <= cap], best


def auto_ylim_for_final_panel(
    runs,
    *,
    lora_r: int | None = None,
    divergent_ratio: float = 1.5,
    iqr_k: float = 1.5,
    lower_pad: float = 0.010,
    upper_pad: float = 0.012,
    fallback: tuple[float, float] = (0.505, 0.620),
) -> tuple[float, float]:
    """Y-axis bounds for an η-vs-final-loss panel — tight around the
    converged cluster so converged-region differences stay visible.

    Two-stage upper bound. (1) Hard divergence: a run whose final eval_loss
    exceeds `divergent_ratio × best_final` is dropped. (2) Tukey upper fence:
    among the survivors, drop runs above `Q3 + iqr_k × IQR`. Stage 2 matters
    because a fixed multiple of best is too loose when the loss range is
    compressed — at best ≈ 0.74, `1.5 × best ≈ 1.1` still admits clearly-worse
    0.9 runs; the IQR fence adapts to the cluster's own spread and clips them.

    The bound deliberately stays TIGHT around the converged cluster: a finite
    final above this top is not stretched into view — it simply exits the top
    of the box (matplotlib clips its marker; the connecting line runs off the
    top edge). Only genuinely NaN-aborted runs get a sentinel marker, pinned to
    the top edge by `clamp_for_hollow` so they read as "diverged," not as a
    real loss value.

    `lora_r=None` aggregates across all ranks. `fallback` is returned when
    no finite finals are present.
    """
    converged, best = _non_divergent_finals(runs, lora_r, divergent_ratio)
    if not converged:
        return fallback
    upper_val = max(converged)
    if len(converged) >= 4:
        import statistics
        q1, _q2, q3 = statistics.quantiles(converged, n=4, method="inclusive")
        fence = q3 + iqr_k * (q3 - q1)
        inliers = [v for v in converged if v <= fence]
        if inliers:
            upper_val = min(upper_val, max(inliers))
    return (best - lower_pad, upper_val + upper_pad)


def auto_ylim_for_trajectory_panel(
    runs,
    *,
    lora_r: int | None = None,
    warmup_frac: float = 0.2,
    divergent_ratio: float = 1.5,
    lower_pad: float = 0.010,
    upper_pad: float = 0.010,
    fallback: tuple[float, float] = (0.505, 0.620),
) -> tuple[float, float]:
    """Y-axis bounds for a per-step training-curve panel.

    Upper bound = max eval_loss across post-warmup eval events of
    non-divergent runs (final ≤ `divergent_ratio × best_final`). Excluding
    the early-training spike from the upper bound keeps late-training
    differences visible.
    """
    converged, best = _non_divergent_finals(runs, lora_r, divergent_ratio)
    if not converged:
        return fallback
    converged_cap = divergent_ratio * best
    post_warmup: list[float] = []
    for cfg, evs in runs:
        if not evs or (lora_r is not None and cfg.get("lora_r") != lora_r):
            continue
        last = evs[-1].get("eval_loss")
        if not (isinstance(last, (int, float)) and math.isfinite(last)):
            continue
        if last > converged_cap:
            continue
        max_step = max((ev.get("step") or 0) for ev in evs) or 1
        cutoff = warmup_frac * max_step
        for ev in evs:
            v = ev.get("eval_loss")
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                continue
            if (ev.get("step") or 0) < cutoff:
                continue
            post_warmup.append(v)
    if not post_warmup:
        return (best - lower_pad, max(converged) + upper_pad)
    return (best - lower_pad, max(post_warmup) + upper_pad)


def clamp_for_hollow(values, top: float | None):
    """Split a list of final-loss values into (ys, is_oor).

    Only a NON-FINITE final (NaN/inf — a NaN-aborted run) is a divergence
    sentinel: it has no real loss, so it is pinned to `top` (the box's top
    edge) and flagged hollow, so it reads as "diverged to NaN" rather than as a
    real loss value sitting just under the rim. A FINITE final is returned
    UNCHANGED (and `is_oor=False`, i.e. drawn solid) at its true height — if
    that height is above the deliberately-tight axis top it simply exits the top
    of the box (matplotlib clips the marker; the connecting line runs off the
    top edge), which is the intended "this lr is off-scale-worse" signal rather
    than a hollow marker pinned inside. With `top is None` (no axis bound)
    non-finite values are flagged but clamping is a no-op.
    """
    is_oor = [not math.isfinite(v) for v in values]
    if top is None:
        ys = [v for v in values]
    else:
        ys = [top if o else v for v, o in zip(values, is_oor)]
    return ys, is_oor


def draw_lr_series(ax, xs, ys_clamped, is_oor, *, color, marker, label=None,
                   lw=LINE_WIDTH, ls="-", ms=MARKER_SIZE, zorder=5, yerr=None,
                   hollow_ms_bump=4, hollow_edgewidth=2.2):
    """Render ONE lr-sweep series with the canonical diverged=hollow convention.

    Draws (1) a connecting line through `ys_clamped` (one continuous curve),
    (2) filled markers — with optional ±σ error bars — for in-range points,
    and (3) hollow markers (`markerfacecolor="none"`) for the OOR/diverged
    points clamped to the cap, so a NaN-aborted or off-axis lr never vanishes
    as a gap but stays connected to the rest of the curve. Used by both
    `plot_eta_vs_final` and `compare_variants_figure` so the two entry points
    can never diverge in how they show divergence.
    """
    ax.plot(xs, ys_clamped, color=color, lw=lw, ls=ls, zorder=zorder,
            label=label)
    in_idx = [i for i, o in enumerate(is_oor) if not o]
    if in_idx:
        if yerr is not None:
            ax.errorbar(
                [xs[i] for i in in_idx], [ys_clamped[i] for i in in_idx],
                yerr=[yerr[i] for i in in_idx], color=color, marker=marker,
                markersize=ms, ls="", zorder=zorder + 1,
                capsize=4, capthick=lw * 0.6, elinewidth=lw * 0.6,
            )
        else:
            ax.plot([xs[i] for i in in_idx], [ys_clamped[i] for i in in_idx],
                    color=color, marker=marker, markersize=ms, ls="",
                    zorder=zorder + 1)
    oor_idx = [i for i, o in enumerate(is_oor) if o]
    if oor_idx:
        ax.plot([xs[i] for i in oor_idx], [ys_clamped[i] for i in oor_idx],
                color=color, marker=marker, markersize=ms + hollow_ms_bump,
                markerfacecolor="none", markeredgewidth=hollow_edgewidth,
                ls="", zorder=zorder + 2)


def plot_eta_vs_final(ax, runs, group_key_fn: Callable[[dict], str],
                      color_map: dict, *, hlines: list[tuple] | None = None,
                      ref_eta_sweeps: list[tuple] | None = None,
                      title: str = "Final eval loss vs η, per group",
                      legend: bool = True,
                      adamw_group_keys: set[str] | None = None,
                      marker_map: dict | None = None,
                      linestyle_map: dict | None = None,
                      normalize_x_to_optimum: bool = False,
                      diverged_runs: list | None = None,
                      allow_label_collision: bool = False) -> None:
    """Left panel: η vs final eval loss, one line per group key.

    `hlines` is a list of (label, y, color, ls, lw) 5-tuples.
    `ref_eta_sweeps` is a list of (label, points, color, ls, lw, marker)
    6-tuples where `points` is [(lr, mean, std, n), …] for error-bar
    rendering, or a legacy [(lr, eval)] 2-tuple list.

    `normalize_x_to_optimum`: when True, each series is plotted at
    x = η/η⋆ where η⋆ is the lr giving the lowest mean eval for that
    series. Optimum-aligns at x=1.
    """
    adamw_group_keys = adamw_group_keys or set()
    marker_map = marker_map or {}
    linestyle_map = linestyle_map or {}

    # Defense in depth: enforce the series-id contract here too, in case
    # this is called outside standard_sweep_figure. Idempotent when the
    # caller already asserted.
    if not allow_label_collision and runs:
        assert_label_discriminates(runs, group_key_fn)

    def _maybe_norm(xs, ys):
        if not normalize_x_to_optimum or len(ys) < 1:
            return xs
        best_idx = min(range(len(ys)), key=lambda i: ys[i])
        eta_star = xs[best_idx]
        if eta_star <= 0:
            return xs
        return [x / eta_star for x in xs]

    all_losses = []
    if ref_eta_sweeps:
        for label, points, color, ls, lw, marker in ref_eta_sweeps:
            if not points:
                continue
            # points rows are (lr, mean, std, n); legacy (lr, eval) 2-tuples
            # still supported.
            if len(points[0]) == 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                stds = [p[2] for p in points]
                xs = _maybe_norm(xs, ys)
                ax.errorbar(xs, ys, yerr=stds, color=color, ls=ls, lw=lw,
                            marker=marker, markersize=MARKER_SIZE,
                            label=label, zorder=BASELINE_ZORDER,
                            capsize=6, capthick=lw, elinewidth=lw)
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                xs = _maybe_norm(xs, ys)
                ax.plot(xs, ys, color=color, ls=ls, lw=lw,
                        marker=marker, markersize=MARKER_SIZE,
                        label=label, zorder=BASELINE_ZORDER)
            all_losses.extend(ys)
    if hlines:
        for label, y, color, ls, lw in hlines:
            ax.axhline(y, color=color, ls=ls, lw=lw, label=label,
                       zorder=BASELINE_ZORDER)
            all_losses.append(y)

    in_range_losses = []
    groups = sorted({group_key_fn(c) for c, _ in runs})

    # ── PASS 1: aggregate per-group (no clamping yet) ─────────────────────
    # Two-pass design: we need to know `hi` (the visible top of the y-axis,
    # computed from in_range_losses) BEFORE we pick a y_cap for clamping
    # OOR / diverged points. With the old single-pass design y_cap was
    # set to `min(raw_losses) + 0.15` upfront, which could exceed `hi`,
    # pushing hollow markers off-screen.
    import statistics as _stats
    from collections import defaultdict as _dd
    group_agg = []
    for g in groups:
        by_lr: dict[float, list[float]] = _dd(list)
        for c, e in runs:
            if group_key_fn(c) != g:
                continue
            by_lr[float(c["lr"])].append(e[-1]["eval_loss"])
        # Fold diverged runs (NaN-final / max>thresh / early-killed) for
        # THIS group into the same aggregation, sentinel = +∞.
        if diverged_runs:
            for c, e in diverged_runs:
                if group_key_fn(c) != g:
                    continue
                by_lr[float(c["lr"])].append(float("inf"))
        if not by_lr:
            continue
        agg = sorted(
            (lr, sum(ys)/len(ys),
             _stats.stdev(ys) if len(ys) > 1 else 0.0,
             len(ys))
            for lr, ys in by_lr.items()
        )
        xs = [p[0] for p in agg]
        means = [p[1] for p in agg]
        stds = [p[2] for p in agg]
        is_nonfinite = [not math.isfinite(m) for m in means]
        in_range_losses.extend(m for m, nf in zip(means, is_nonfinite) if not nf)
        is_adamw = g in adamw_group_keys
        style = dict(
            color=BASELINE_COLOR if is_adamw else color_map.get(g, "grey"),
            marker=BASELINE_MARKER if is_adamw else marker_map.get(g, "o"),
            ls=BASELINE_LS_CURVE if is_adamw else linestyle_map.get(g, "-"),
            lw=BASELINE_LW_CURVE if is_adamw else LINE_WIDTH,
            zorder=BASELINE_ZORDER if is_adamw else 5,
            is_adamw=is_adamw,
        )
        group_agg.append((g, xs, means, stds, is_nonfinite, style))

    # ── compute y-axis bounds before pass 2 ──────────────────────────────
    if in_range_losses or all_losses:
        lo = min(in_range_losses + all_losses) - 0.005
        hi = (max(in_range_losses) if in_range_losses
              else max(all_losses)) + 0.01
        hi = min(hi + 0.005, lo + 0.16)
    else:
        lo, hi = 0.0, 1.0
    # y_cap = clamping height for OOR / diverged. Just inside the axis top,
    # but never below max(in_range) (else legit in-range points would clip).
    if in_range_losses:
        max_in_range = max(in_range_losses)
        y_cap = max(max_in_range + 0.003, hi - 0.006)
    else:
        y_cap = hi - 0.006

    # Surface in-range points clipped by y_cap (rare — only when a finite
    # mean is above the cap).
    clipped = []
    for g, xs, means, _stds, is_nonfinite, _style in group_agg:
        for lr, m, nf in zip(xs, means, is_nonfinite):
            if not nf and m > y_cap:
                clipped.append((g, lr, m))
    if clipped:
        print(f"  [clipped from left panel y_cap={y_cap:.3f}] {len(clipped)} run(s):")
        for g, lr, fl in sorted(clipped):
            print(f"    {g} η={lr:.0e} final={fl:.4f}")

    # ── PASS 2: render lines + markers ───────────────────────────────────
    for g, xs, means, stds, is_nonfinite, style in group_agg:
        ys_clamped = [y_cap if (nf or m > y_cap) else m
                      for m, nf in zip(means, is_nonfinite)]
        is_oor = [nf or m > y_cap for m, nf in zip(means, is_nonfinite)]
        if normalize_x_to_optimum and any(not o for o in is_oor):
            in_pairs = [(xs[i], means[i]) for i, o in enumerate(is_oor) if not o]
            eta_star = min(in_pairs, key=lambda p: p[1])[0]
            if eta_star > 0:
                xs = [x / eta_star for x in xs]
        color, marker, ls, lw, zorder, is_adamw = (
            style["color"], style["marker"], style["ls"],
            style["lw"], style["zorder"], style["is_adamw"])
        # Line + in-range (±σ error bars) + hollow OOR markers, via the
        # shared diverged=hollow renderer.
        draw_lr_series(
            ax, xs, ys_clamped, is_oor, color=color, marker=marker,
            label=f"{g} (baseline)" if is_adamw else g,
            lw=lw, ls=ls, ms=MARKER_SIZE, zorder=zorder, yerr=stds,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta / \eta^\star$ (log)" if normalize_x_to_optimum else "η (log)",
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Final eval loss", fontsize=AXIS_LABEL_FONTSIZE)
    if normalize_x_to_optimum:
        ax.axvline(1.0, color="grey", ls=":", lw=0.8, alpha=0.6, zorder=0)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25, lw=0.6, which="major")
    ax.grid(True, alpha=0.10, lw=0.5, which="minor")
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    if in_range_losses or all_losses:
        lo = min(in_range_losses + all_losses) - 0.005
        hi = (max(in_range_losses) if in_range_losses
              else max(all_losses)) + 0.01
        hi = min(hi + 0.005, lo + 0.16)
        ax.set_ylim(lo, hi)

    if legend:
        ax.legend(**LEGEND_KW)


def plot_best_eta_curves(ax, runs, group_key_fn: Callable[[dict], str],
                         color_map: dict, *, ref_curves: list[tuple] | None = None,
                         title: str = "Best η per group — training curves",
                         x_tick_step: int = 200,
                         legend: bool = True,
                         adamw_group_keys: set[str] | None = None,
                         marker_map: dict | None = None,
                         linestyle_map: dict | None = None) -> None:
    """Right panel: training curves for the best (lowest final loss) η per group.

    `ref_curves` is a list of 7-tuples
    (label, evs, color, ls, lw, marker, std_evs). When `std_evs` carries
    non-zero std (multi-seed), a ±σ shaded band is drawn behind the line.
    """
    adamw_group_keys = adamw_group_keys or set()
    marker_map = marker_map or {}
    linestyle_map = linestyle_map or {}

    if ref_curves:
        for label, evs, color, ls, lw, marker, std_evs in ref_curves:
            xs = [e["step"] for e in evs]
            ys = [e["eval_loss"] for e in evs]
            if std_evs is not None and any(s["eval_loss"] > 0 for s in std_evs):
                stds = [s["eval_loss"] for s in std_evs]
                lo = [y - s for y, s in zip(ys, stds)]
                hi = [y + s for y, s in zip(ys, stds)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.18,
                                zorder=BASELINE_ZORDER - 1, linewidth=0)
            ax.plot(xs, ys,
                    color=color, lw=lw, ls=ls, marker=marker,
                    markersize=MARKER_SIZE, label=label,
                    zorder=BASELINE_ZORDER)

    best = {}
    for cfg, evs in runs:
        g = group_key_fn(cfg)
        fl = evs[-1]["eval_loss"]
        if g not in best or fl < best[g][2]:
            best[g] = (cfg, evs, fl)
    for g, (cfg, evs, fl) in sorted(best.items(), key=lambda kv: kv[1][2]):
        is_adamw = g in adamw_group_keys
        label = (f"{g} (baseline, η={cfg['lr']:.0e}, final={fl:.4f})"
                 if is_adamw else f"{g} (η={cfg['lr']:.0e}, final={fl:.4f})")
        ax.plot([e["step"] for e in evs], [e["eval_loss"] for e in evs],
                color=BASELINE_COLOR if is_adamw else color_map.get(g, "grey"),
                marker=BASELINE_MARKER if is_adamw else marker_map.get(g, "o"),
                markersize=MARKER_SIZE,
                lw=BASELINE_LW_CURVE if is_adamw else LINE_WIDTH,
                ls=BASELINE_LS_CURVE if is_adamw else linestyle_map.get(g, "-"),
                zorder=BASELINE_ZORDER if is_adamw else 5,
                label=label)
    ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Eval loss", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    if legend:
        ax.legend(**LEGEND_KW)


def plot_leaderboard_by_rank(best: dict, baseline_optimizer: str = "adamw",
                              color_map: dict | None = None,
                              marker_map: dict | None = None,
                              linestyle_map: dict | None = None,
                              suptitle: str = "Best eval vs rank"):
    """Single-panel leaderboard: best-η eval loss vs rank, one line per optimizer.

    ``marker_map`` overrides the per-optimizer marker shape (default "o").
    ``linestyle_map`` overrides the per-optimizer linestyle (default "-").
    """
    color_map = color_map or {}
    marker_map = marker_map or {}
    linestyle_map = linestyle_map or {}
    baseline_floor = {r: best[(baseline_optimizer, r)][2]
                      for (opt, r) in best if opt == baseline_optimizer}
    ranks = sorted(baseline_floor)

    series = {}
    for (opt, r), (_cfg, _evs, fl) in best.items():
        series.setdefault(opt, []).append((r, fl))
    for opt in series:
        series[opt].sort()

    n_optimizers = len(series)
    legend_ncol = max(1, min(4, (n_optimizers + 7) // 8))
    legend_rows = (n_optimizers + legend_ncol - 1) // legend_ncol
    plot_height = 5.0
    legend_height = 0.30 * legend_rows + 0.5
    fig_height = plot_height + legend_height
    fig, ax = plt.subplots(figsize=(11, fig_height), constrained_layout=True)
    all_losses = []
    for opt, points in sorted(series.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        is_baseline = opt == baseline_optimizer
        ax.plot(xs, ys,
                color=BASELINE_COLOR if is_baseline else color_map.get(opt, "grey"),
                lw=BASELINE_LW_CURVE if is_baseline else LINE_WIDTH,
                ls=BASELINE_LS_CURVE if is_baseline else linestyle_map.get(opt, "-"),
                marker=BASELINE_MARKER if is_baseline else marker_map.get(opt, "o"),
                markersize=MARKER_SIZE,
                label=f"{opt} (baseline)" if is_baseline else opt,
                zorder=BASELINE_ZORDER if is_baseline else 5)
        all_losses.extend(ys)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Best-η final eval loss")
    ax.set_title(suptitle, fontsize=12, fontweight="bold")
    if all_losses:
        lo = min(all_losses) - 0.005
        hi = min(all_losses) + 0.05
        ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25, lw=0.6)
    legend_kw = dict(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=legend_ncol,
        fontsize=11 if n_optimizers > 8 else 13,
        frameon=True,
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.4,
        columnspacing=1.5,
    )
    ax.legend(**legend_kw)
    return fig, ax
