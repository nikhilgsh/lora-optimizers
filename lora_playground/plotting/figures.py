"""High-level 2-panel figure entry points used by every notebook cell.

`standard_sweep_figure(runs, group_key_fn, color_map, ...)` is THE single
entry point: it auto-splits multi-rank input, asserts the series-id
discrimination contract, builds the AdamW baseline overlay (locked styling),
and renders the 2-panel η-vs-final + best-η-training-curves figure.
"""
from __future__ import annotations

from typing import Callable, Iterable

import matplotlib.pyplot as plt

from .dedup import assert_label_discriminates
from .merge import report_diverged, split_diverged
from .overlays import baseline_overlay
from .panels import _infer_min_step, plot_best_eta_curves, plot_eta_vs_final
from .style import (
    CANONICAL_HORIZON, DEFAULT_FIGSIZE, DIVERGE_THRESHOLD, SUPTITLE_FONTSIZE,
)


# Sentinel: standard_sweep_figure / two_panel_sweep_figure should auto-infer
# min_step from the input runs (cfg["max_steps"]) rather than defaulting to
# CANONICAL_HORIZON=2000, which silently mis-fires under packed_v1's 4000-step
# regime.
_INFER_MIN_STEP: object = object()


def two_panel_sweep_figure(runs, group_key_fn, color_map, *,
                           suptitle: str = "",
                           hlines: list[tuple] | None = None,
                           ref_curves: list[tuple] | None = None,
                           ref_eta_sweeps: list[tuple] | None = None,
                           threshold: float = DIVERGE_THRESHOLD,
                           min_step: int | None | object = _INFER_MIN_STEP,
                           figsize: tuple = DEFAULT_FIGSIZE,
                           x_tick_step: int = 200,
                           left_title: str = "Final eval loss vs η, per group",
                           right_title: str = "Best η per group — training curves",
                           label_fn: Callable[[dict], str] | None = None,
                           adamw_group_keys: set[str] | None = None,
                           marker_map: dict | None = None,
                           linestyle_map: dict | None = None,
                           normalize_x_to_optimum: bool = False):
    """Build the standardized 2-panel sweep figure with diverged-run filtering.

    Returns (fig, axes, n_kept, n_dropped).

    Layout: left panel is η-vs-final-loss with legend suppressed (same colors
    as the right panel which carries descriptive labels). Right panel has
    best-η training curves with descriptive legend outside on the right.
    """
    keep, drop = split_diverged(runs, threshold)
    if drop:
        if label_fn is None:
            label_fn = lambda c: f"{group_key_fn(c)} η={c['lr']:.0e}"
        report_diverged(drop, label_fn)

    # Left panel implicitly assumes the run reached its final loss (plots
    # evs[-1]['eval_loss'] as "final"). In-flight runs would bias the
    # comparison. Exclude them from the LEFT panel only — the right panel
    # still shows the partial training curve.
    if min_step is _INFER_MIN_STEP:
        min_step = _infer_min_step(keep) or CANONICAL_HORIZON
        print(f"  [auto] min_step inferred from cfg['max_steps']: {min_step}")
    keep_for_right = keep
    keep_for_left = keep
    if min_step is not None:
        partial = [(c, e) for c, e in keep if e[-1]["step"] < min_step]
        keep_for_left = [(c, e) for c, e in keep if e[-1]["step"] >= min_step]
        if partial:
            print(f"  [filtered partial from left panel] {len(partial)} run(s) below min_step={min_step}:")
            if label_fn is None:
                label_fn = lambda c: f"{group_key_fn(c)} η={c['lr']:.0e}"
            for cfg, evs in sorted(partial, key=lambda x: label_fn(x[0])):
                print(f"    {label_fn(cfg):<30s} step={evs[-1]['step']} "
                      f"final-so-far={evs[-1]['eval_loss']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    effective_left_title = (
        r"Final eval loss vs $\eta/\eta^\star$, per group"
        if normalize_x_to_optimum and left_title == "Final eval loss vs η, per group"
        else left_title
    )
    plot_eta_vs_final(axes[0], keep_for_left, group_key_fn, color_map,
                      hlines=hlines, ref_eta_sweeps=ref_eta_sweeps,
                      title=effective_left_title, legend=False,
                      adamw_group_keys=adamw_group_keys,
                      marker_map=marker_map,
                      linestyle_map=linestyle_map,
                      normalize_x_to_optimum=normalize_x_to_optimum,
                      diverged_runs=drop)
    plot_best_eta_curves(axes[1], keep_for_right, group_key_fn, color_map,
                         ref_curves=ref_curves, title=right_title,
                         x_tick_step=x_tick_step,
                         adamw_group_keys=adamw_group_keys,
                         marker_map=marker_map,
                         linestyle_map=linestyle_map)
    if suptitle:
        fig.suptitle(suptitle, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    return fig, axes, len(keep), len(drop)


def standard_sweep_figure(runs, group_key_fn, color_map, *,
                          reference_runs,
                          suptitle: str = "",
                          extra_baselines: Iterable[tuple[str, str]] = (),
                          baseline_optimizer: str = "adamw",
                          same_value_axes: tuple = ("lora_plus_multiplier",),
                          allow_label_collision: bool = False,
                          **kwargs):
    """High-level uniform sweep figure — every section's single entry point.

    Guarantees on every call:
      - The baseline (AdamW) overlay: a single dotted floor hline (left
        panel), the AdamW η-sweep curve (left panel) so the LR grid that
        was tuned is visible, and the AdamW best-η training curve (right
        panel). All in distinguished long-dash black.
      - Identical layout, fonts, legend placement, grid, and figure size.

    Visual styling is locked at the library level — callers cannot override
    color, linestyle, linewidth, marker for the baseline. The only knobs
    are: which optimizer is the primary baseline, and whether to overlay
    secondary references via `extra_baselines`.

    Args:
        runs:            candidate sweeps to compare, list of (cfg, evs).
        group_key_fn:    cfg → legend group string.
        color_map:       group string → matplotlib color.
        reference_runs:  REQUIRED. Sweeps containing the baseline optimizer.
        suptitle:        figure suptitle (run-count tally is appended).
        extra_baselines: extra (optimizer, color) pairs to overlay as light
                         dotted references.
        **kwargs:        forwarded to two_panel_sweep_figure.

    Raises:
        ValueError: if `reference_runs` has no run for `baseline_optimizer`.
    """
    # Multi-rank input: auto-split into per-rank figures. Avoids the
    # silent-collision failure mode (runs at different ranks collapsing under
    # one group_key_fn(cfg)).
    ranks = sorted({int(c.get("lora_r", 16)) for c, _ in runs})
    if len(ranks) > 1:
        results = []
        for r in ranks:
            run_slice = [(c, e) for c, e in runs if int(c.get("lora_r", 16)) == r]
            ref_slice = [(c, e) for c, e in reference_runs
                         if int(c.get("lora_r", 16)) == r]
            if not any(c.get("optimizer") == baseline_optimizer for c, _ in ref_slice):
                ref_slice = reference_runs
            results.append(standard_sweep_figure(
                run_slice, group_key_fn, color_map,
                reference_runs=ref_slice,
                suptitle=suptitle,
                extra_baselines=extra_baselines,
                baseline_optimizer=baseline_optimizer,
                same_value_axes=same_value_axes,
                allow_label_collision=allow_label_collision,
                **kwargs,
            ))
        return results

    # Series-id discrimination contract: every (label, lora_r, lr) bucket
    # must contain exactly one series_id. Otherwise the per-label averaging
    # downstream silently mixes distinct algorithms.
    if not allow_label_collision and runs:
        assert_label_discriminates(runs, group_key_fn)

    # Robustness: every run in `runs` must agree on `same_value_axes`. Catches
    # the bug where a panel filter forgets to constrain an axis and runs with
    # mixed values silently collapse via group_key_fn into one group.
    if same_value_axes and runs:
        for axis in same_value_axes:
            values = {c.get(axis) for c, _ in runs if axis in c}
            if len(values) > 1:
                raise ValueError(
                    f"standard_sweep_figure: runs span multiple values of "
                    f"{axis!r}: {sorted(values)}. group_key_fn would silently "
                    f"collapse these into one group. Filter the runs by "
                    f"{axis} before calling, or pass same_value_axes=() if "
                    f"the multi-value comparison is intentional and the "
                    f"group_key_fn already encodes {axis}."
                )

    _marker_map = kwargs.get("marker_map") or {}

    # Filter partial-horizon runs out of reference_runs BEFORE the η-sweep is
    # built, so the left-panel AdamW η-sweep doesn't include in-flight runs.
    _min_step = kwargs.get("min_step", _INFER_MIN_STEP)
    if _min_step is _INFER_MIN_STEP:
        _min_step = _infer_min_step(runs) or CANONICAL_HORIZON
    if reference_runs is not None and _min_step is not None:
        n_before = len(reference_runs)
        complete = [(c, e) for c, e in reference_runs
                    if e and e[-1]["step"] >= _min_step]
        baseline_complete = [r for r in complete
                             if r[0].get("optimizer") == baseline_optimizer]
        if baseline_complete:
            reference_runs = complete
            n_dropped = n_before - len(reference_runs)
            if n_dropped:
                print(f"  [auto] filtered {n_dropped} partial reference run(s) below min_step={_min_step}")
        else:
            partial_baseline = [r for r in reference_runs
                                if r[0].get("optimizer") == baseline_optimizer]
            if partial_baseline:
                last = max(e[-1]["step"] for _, e in partial_baseline if e)
                print(f"  [auto] no {baseline_optimizer!r} reference reached "
                      f"min_step={_min_step}; using partial baseline (last step={last})")

    if reference_runs is None:
        hlines, ref_curves, eta_sweeps = [], [], []
    else:
        hlines, ref_curves, eta_sweeps = baseline_overlay(
            reference_runs, baseline_optimizer, is_primary=True,
            marker_map=_marker_map,
        )
        if not hlines:
            raise ValueError(
                f"No {baseline_optimizer!r} run found in reference_runs — "
                "every standard sweep figure requires the baseline.")

    # Library-enforced uniform suptitle: append " at r={N}" when single-rank
    # and rank isn't already in the title.
    if ranks and suptitle and "r=" not in suptitle:
        suptitle = f"{suptitle} at r={ranks[0]}"

    for opt, color in extra_baselines:
        if reference_runs is None:
            continue
        _h, r, e = baseline_overlay(
            reference_runs, opt, color=color, is_primary=False,
            marker_map=_marker_map,
        )
        # Secondary baselines contribute the right-panel reference curve
        # AND the left-panel η-sweep. The hline is suppressed — the primary
        # baseline (AdamW) owns the only floor hline.
        ref_curves.extend(r)
        eta_sweeps.extend(e)

    # If the candidate runs already contain the baseline (e.g. all-optimizers
    # or cross-investigation cells), the candidate's own η-sweep + best-η
    # curve already cover both panels; the overlay would double-draw. Detect
    # overlap, restyle the candidate to baseline style via adamw_group_keys,
    # and drop the overlay's eta_sweep + ref_curve (keep the floor hline).
    candidate_groups = {group_key_fn(c) for c, _ in runs}
    adamw_overlap_groups = candidate_groups & {baseline_optimizer}
    if adamw_overlap_groups:
        ref_curves = []
        eta_sweeps = []

    return two_panel_sweep_figure(
        runs, group_key_fn, color_map,
        suptitle=suptitle,
        hlines=hlines, ref_curves=ref_curves, ref_eta_sweeps=eta_sweeps,
        adamw_group_keys=adamw_overlap_groups,
        **kwargs,
    )
