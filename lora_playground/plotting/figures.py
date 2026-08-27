"""High-level 2-panel figure entry points used by every notebook cell.

`standard_sweep_figure(runs, group_key_fn, color_map, ...)` is THE single
entry point: it auto-splits multi-rank input, asserts the series-id
discrimination contract, builds the AdamW baseline overlay (locked styling),
and renders the 2-panel η-vs-final + best-η-training-curves figure.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import matplotlib.pyplot as plt

from lora_playground.comparison import ComparisonResult, VariantSpec
from lora_playground.leaderboard import is_final, mean_over_seeds
from .dedup import assert_label_discriminates
from .merge import report_diverged, split_diverged
from .overlays import baseline_overlay
from .panels import (
    _infer_min_step,
    auto_ylim_for_final_panel,
    auto_ylim_for_trajectory_panel,
    plot_best_eta_curves,
    plot_eta_vs_final,
)
from .style import (
    CANONICAL_HORIZON, DEFAULT_FIGSIZE, DIVERGE_THRESHOLD, LEGEND_BELOW_KW,
    SUPTITLE_FONTSIZE,
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


def sweep_figure_with_auto_ylim(
    runs,
    group_key_fn,
    color_map,
    *,
    reference_runs,
    suptitle: str = "",
    final_ylim_kwargs: dict | None = None,
    traj_ylim_kwargs: dict | None = None,
    **kwargs,
):
    """`standard_sweep_figure(...)` + post-process each returned panel pair
    with `auto_ylim_for_final_panel` (left) and `auto_ylim_for_trajectory_panel`
    (right).

    Wraps the common pattern: render the canonical 2-panel figure, then
    tighten each panel's y-axis around the non-divergent population so
    converged-region differences stay visible. `final_ylim_kwargs` and
    `traj_ylim_kwargs` are forwarded to the respective auto-ylim helpers.

    Returns the same shape as `standard_sweep_figure`: a single
    `(fig, axes, n_kept, n_dropped)` tuple for single-rank input, or a list
    of such tuples for multi-rank input.
    """
    from .panels import auto_ylim_for_final_panel, auto_ylim_for_trajectory_panel

    out = standard_sweep_figure(
        runs, group_key_fn, color_map,
        reference_runs=reference_runs, suptitle=suptitle, **kwargs,
    )
    results = out if isinstance(out, list) else [out]
    final_kw = final_ylim_kwargs or {}
    traj_kw = traj_ylim_kwargs or {}
    for fig, axes, _n_keep, _n_drop in results:
        # Recover the rank for this figure from the suptitle (single-rank
        # standard_sweep_figure auto-appends " at r=N"). Falls back to None
        # — auto_ylim_* treats lora_r=None as "all ranks".
        title = fig._suptitle.get_text() if fig._suptitle else ""
        rank = None
        for r_try in (256, 128, 64, 32, 16):
            if f"r={r_try}" in title:
                rank = r_try
                break
        axes[0].set_ylim(*auto_ylim_for_final_panel(runs, lora_r=rank, **final_kw))
        axes[1].set_ylim(*auto_ylim_for_trajectory_panel(runs, lora_r=rank, **traj_kw))
    return out


def compare_variants_figure(
    variants: dict | None = None,
    *,
    common_where: dict | None = None,
    ref_label: str | None = None,
    logs_root: str = "../logs",
    sigma_ref: float = 0.0007,
    suptitle: str | None = None,
    colors: dict | None = None,
    markers: dict | None = None,
    figsize: tuple[float, float] = (13, 5),
    max_steps: int = 4000,
    completion_slack: int = 300,
    allow_partial: bool = False,
    prefetched_runs: list | None = None,
    records: Iterable | None = None,
    variant_specs: Iterable[VariantSpec] | None = None,
    comparison: ComparisonResult | None = None,
    reference_id: str | None = None,
    target_id: str | None = None,
    variant_key=None,
    final_ylim: tuple[float, float] | None = None,
    traj_ylim: tuple[float, float] | None = None,
    auto_ylim: bool = True,
    divergent_ratio: float = 1.5,
    target_label: str | None = "AdamW",
    allow_label_collision: bool = False,
    allow_custom_labels: bool = False,
):
    """Compare named optimizer variants at a fixed config; 2-panel + tables.

    The records-native entry point is ``records`` plus explicit
    ``variant_specs``.  It delegates assignment, semantic-revision checks,
    completion classification, replicate aggregation, and best-LR selection
    to :func:`lora_playground.comparison.build_comparison`.  Stable
    ``VariantSpec.id`` values are the only identities in that path; labels and
    styles are presentation-only.  An already-built ``comparison`` can be
    passed instead when the caller owns selection upstream.  In both explicit
    paths, ``reference_id`` (and optional ``target_id``) names a stable variant
    ID.

    ``variants`` / ``common_where`` / ``ref_label`` and ``prefetched_runs``
    remain compatibility inputs. ``variants`` maps
    ``label -> extra_where_dict`` and, without prefetched input, each variant
    is loaded as ``load_runs(where={**common_where, **extra}, ...)``.

    Y-limits: by default (`auto_ylim=True`) both panels exclude runs whose
    final eval_loss exceeds `divergent_ratio × best_final` (default 1.5×) from
    the upper bound. On the LEFT (final-vs-lr) panel a diverged run still
    renders but goes off-axis at the top — the right visual signal for
    divergence — instead of compressing the converged region into a flat line.
    The RIGHT (trajectory) panel plots only the best-lr curve per variant, so
    the bound is the FULL-curve max of the non-divergent variants: the whole
    descent stays visible rather than clipping the early-training top off the
    curves. This is the robust replacement for hand-tuning `final_ylim` to crop
    diverged AdamW outliers. Passing an explicit `final_ylim`/`traj_ylim` always
    overrides the auto bound for that panel; `auto_ylim=False` falls back to
    matplotlib autoscale. Shared logic: `auto_ylim_for_final_panel` /
    `auto_ylim_for_trajectory_panel`.

    Returns
    -------
    (fig, table_df, summary_df) — `table_df` has lr as index and one column
    per variant; `summary_df` has ('variant', 'best_lr', 'final', 'delta',
    'delta_sigma').
    """
    if reference_id is not None and ref_label is not None:
        if reference_id != ref_label:
            raise ValueError(
                "reference_id and ref_label disagree; explicit comparisons "
                "must name one stable reference ID"
            )
    resolved_reference = reference_id or ref_label

    if records is not None and prefetched_runs is not None:
        raise ValueError("pass records or prefetched_runs, not both")

    def _render_explicit(result: ComparisonResult):
        if resolved_reference is None:
            raise ValueError(
                "reference_id is required for records-native comparisons"
            )
        from .render import render_comparison
        return render_comparison(
            result,
            reference_id=resolved_reference,
            target_id=target_id,
            horizon=max_steps,
            sigma_ref=sigma_ref,
            colors=colors,
            markers=markers,
            figsize=figsize,
            suptitle=suptitle,
            show_partials=allow_partial,
            final_ylim=final_ylim,
            traj_ylim=traj_ylim,
            auto_ylim=auto_ylim,
            divergent_ratio=divergent_ratio,
        )

    if comparison is not None:
        conflicts = []
        if variants is not None:
            conflicts.append("variants")
        if common_where is not None:
            conflicts.append("common_where")
        if prefetched_runs is not None or records is not None:
            conflicts.append("records/prefetched_runs")
        if variant_specs is not None:
            conflicts.append("variant_specs")
        if variant_key is not None:
            conflicts.append("variant_key")
        if conflicts:
            raise ValueError(
                "comparison already owns assignment and selection; do not "
                f"also pass {', '.join(conflicts)}"
            )
        return _render_explicit(comparison)

    if variant_specs is not None:
        if variants is not None or common_where is not None:
            raise ValueError(
                "variant_specs is the complete assignment registry; do not "
                "also pass variants or common_where"
            )
        if variant_key is not None:
            raise ValueError(
                "variant_specs and variant_key are mutually exclusive"
            )
        if records is None:
            raise ValueError(
                "variant_specs requires records; legacy prefetched_runs use "
                "the variants/variant_key compatibility path"
            )
        from lora_playground.comparison import build_comparison
        result = build_comparison(
            tuple(records),
            tuple(variant_specs),
            horizon=max_steps,
            completion_slack=completion_slack,
        )
        return _render_explicit(result)

    if records is not None:
        raise ValueError(
            "records requires explicit variant_specs so recorded runs are not "
            "classified through a display-label adapter"
        )
    if target_id is not None:
        raise ValueError(
            "target_id is for explicit comparison identities; compatibility "
            "callers should use target_label"
        )
    if variants is None:
        raise ValueError(
            "pass comparison, records plus variant_specs, or legacy variants"
        )
    if resolved_reference is None:
        raise ValueError("ref_label is required for the compatibility path")
    # From here down the compatibility implementation consistently uses its
    # historical name.  ``reference_id`` is accepted as a migration alias.
    ref_label = resolved_reference

    import pandas as pd
    from .labels import canonical_label, canonical_colors, order_labels
    from .dedup import assert_label_discriminates

    # Single source of truth: default to the canonical labeler so every cell
    # gets identical, fully-discriminating names without hand-rolling one.
    if variant_key is None:
        variant_key = canonical_label
    # Enforce AdamW-first + deterministic legend/plot order for ALL callers.
    variants = {l: variants[l] for l in order_labels(variants) if l in variants}

    # per_variant[label] = {lr: (final_or_last_loss, cfg, history)}
    # per_variant_partial[label] = same shape, only runs that haven't yet reached max_steps
    per_variant = {}
    per_variant_partial = {}

    if prefetched_runs is not None:
        # Single-load mode: the pure comparison core owns assignment,
        # completed-vs-partial classification, replicate averaging, and
        # best-LR selection.  This adapter keeps the established plotting
        # dictionaries intact so table construction and rendering below do
        # not acquire a second comparison implementation.
        from lora_playground.comparison import build_comparison

        known_labels = set(variants)
        # GUARD: a (label, lora_r, lr) bucket holding >1 distinct series_id means
        # the label silently merges distinct configs (the eps_rel ns5/ns8 bug).
        # Raise instead of keeping min-loss; opt out only with allow_label_collision.
        if not allow_label_collision:
            guarded = [(c, h) for c, h in prefetched_runs
                       if variant_key(c) in known_labels]
            assert_label_discriminates(guarded, variant_key,
                                       bucket_keys=("lora_r", "lr"))

        specs = tuple(
            VariantSpec(
                id=label,
                label=label,
                style_key=label,
                predicate=lambda cfg, expected=label: variant_key(cfg) == expected,
            )
            for label in variants
        )
        comparison = build_comparison(
            prefetched_runs,
            specs,
            horizon=max_steps,
            completion_slack=completion_slack,
        )
        if colors is None:
            colors = canonical_colors(list(variants))
        if markers is None:
            marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
            markers = {
                label: marker_cycle[index % len(marker_cycle)]
                for index, label in enumerate(variants)
            }
        from .render import render_comparison
        return render_comparison(
            comparison,
            reference_id=ref_label,
            target_id=(target_label if target_label in variants else None),
            horizon=max_steps,
            sigma_ref=sigma_ref,
            colors=colors,
            markers=markers,
            figsize=figsize,
            suptitle=suptitle,
            show_partials=allow_partial,
            final_ylim=final_ylim,
            traj_ylim=traj_ylim,
            auto_ylim=auto_ylim,
            divergent_ratio=divergent_ratio,
        )
    else:
        # Compatibility shim for callers that still ask this plotting helper
        # to perform one loader query per variant.  New callers should prefetch
        # once and use the comparison core above.
        from lora_playground.loader import load_runs
        for label, extra in variants.items():
            runs = load_runs(
                where={**(common_where or {}), **extra},
                logs_root=logs_root,
                warn_cross_commit=False,
                quiet=True,
            )
            # ENFORCE standard plot names: a hand-typed variant label must BE the
            # canonical_label of the runs it selects (single source of truth, so
            # method names are identical across every notebook). Catches ad-hoc
            # names like "geometric Gram (full polar)" that diverge from the
            # leaderboard vocabulary. Opt out only for a deliberate cross-config
            # grouping with allow_custom_labels=True.
            if not allow_custom_labels:
                canon = {canonical_label(c) for c, h in runs if h}
                canon.discard(None)
                if canon and label not in canon:
                    raise ValueError(
                        f"Non-standard variant label {label!r}: it is not the "
                        f"canonical_label of its runs (expected one of "
                        f"{sorted(canon)}). Plot names must come from "
                        f"canonical_label so they match every other notebook. "
                        f"Use the canonical label as the dict key (or pass "
                        f"allow_custom_labels=True for an intentional grouping)."
                    )
            # Seed repeats are COLLECTED per lr and averaged below, not reduced
            # pairwise as they arrive. This block remains only for the legacy
            # per-variant loader path; the prefetched path delegates the same
            # semantics to build_comparison above.
            by_lr: dict[float, list] = {}
            d, d_partial = {}, {}
            for c, h in runs:
                if not h:
                    continue
                lr = float(c["lr"])
                last_step = h[-1].get("step")
                last_loss = h[-1]["eval_loss"]
                is_aborted = c.get("_aborted") is not None
                if is_final(last_step, max_steps, completion_slack) or is_aborted:
                    by_lr.setdefault(lr, []).append((c, h, last_step))
                elif allow_partial:
                    if lr not in d_partial or last_step > d_partial[lr][3]:
                        d_partial[lr] = (last_loss, c, h, last_step)
            for lr, members in by_lr.items():
                longest = max(m[2] for m in members)
                finished = [m for m in members if m[2] == longest]
                mean_final, merged = mean_over_seeds(
                    [(h, h[-1], ls) for _c, h, ls in finished])
                d[lr] = (mean_final, finished[0][0], merged)
            per_variant[label] = d
            per_variant_partial[label] = d_partial

    # Final-loss table (rows = lr, columns = variant).
    all_lr = sorted({lr for v in per_variant.values() for lr in v})
    table_df = pd.DataFrame(
        {label: [per_variant[label].get(lr, (None,))[0] for lr in all_lr]
         for label in variants},
        index=pd.Index(all_lr, name="lr"),
    )

    # Best per variant + Δ vs ref. NaN-safe: a NaN-aborted (diverged) lr must
    # never win the argmin, else the table/Δ and the right-panel trajectory
    # would be driven by a blown-up run. Fall back to any lr only if EVERY
    # final is non-finite.
    def _best_lr(d):
        finite = [lr for lr in d if math.isfinite(d[lr][0])]
        pool = finite or list(d)
        return min(pool, key=lambda lr: d[lr][0])

    if ref_label not in per_variant or not per_variant[ref_label]:
        ref_best = None
    else:
        ref_best = per_variant[ref_label][_best_lr(per_variant[ref_label])][0]
    summary_rows = []
    for label, d in per_variant.items():
        if not d:
            continue
        best_lr = _best_lr(d)
        final = d[best_lr][0]
        delta = (final - ref_best) if ref_best is not None and label != ref_label else None
        summary_rows.append({
            "variant": label, "best_lr": best_lr, "final": final,
            "delta": delta,
            "delta_sigma": (delta / sigma_ref) if delta is not None else None,
        })
    summary_df = pd.DataFrame(summary_rows)

    if colors is None:
        # Canonical: AdamW → reserved black, others distinct. Replaces the old
        # tab10-by-dict-order default (which never reserved black for AdamW).
        colors = canonical_colors(list(variants))
    if markers is None:
        marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
        markers = {label: marker_cycle[i % len(marker_cycle)] for i, label in enumerate(variants)}

    from .panels import clamp_for_hollow, draw_lr_series

    fig, (ax_lr, ax_traj) = plt.subplots(1, 2, figsize=figsize,
                                         constrained_layout=True)
    # Compute the y-bound BEFORE plotting so diverged / NaN-aborted lr points
    # can be clamped just inside the top and drawn as HOLLOW markers connected
    # to the rest of the lr curve (the canonical convention — see
    # panels.draw_lr_series), instead of vanishing as a gap. Robust auto-clip:
    # drop diverged runs (final > divergent_ratio × best) out of the upper
    # bound. Explicit final_ylim always wins.
    if final_ylim is None and auto_ylim:
        final_runs = [(cfg, h) for d in per_variant.values()
                      for (_loss, cfg, h) in d.values()]
        if final_runs:
            final_ylim = auto_ylim_for_final_panel(
                final_runs, divergent_ratio=divergent_ratio)
    # `top`: NaN-aborted (diverged) lrs are pinned to the box's top edge and
    # drawn hollow (a sentinel, not a value just under the rim). Finite finals
    # above this deliberately-tight top are left at their true height and exit
    # the top of the box — they are NOT clamped/hollowed (see clamp_for_hollow).
    top = final_ylim[1] if final_ylim is not None else None
    for label, d in per_variant.items():
        if not d:
            continue
        lrs = sorted(d)
        finals = [d[lr][0] for lr in lrs]
        ys_clamped, is_oor = clamp_for_hollow(finals, top)
        draw_lr_series(ax_lr, lrs, ys_clamped, is_oor, color=colors[label],
                       marker=markers[label], label=label, lw=1.4, ms=6,
                       zorder=4)
    ax_lr.set_xscale("log")
    ax_lr.set_xlabel("lr")
    ax_lr.set_ylabel(f"final eval_loss @ {max_steps // 1000}k")
    ax_lr.set_title("final loss vs lr")
    ax_lr.grid(True, alpha=0.3)
    # No per-panel legend; one shared figure-level legend is added below after
    # the trajectory panel (both panels share the same variant→color mapping).
    if final_ylim is not None:
        ax_lr.set_ylim(*final_ylim)

    def _pick_traj(d, d_partial):
        """Best trajectory for a variant, preferring a FINITE curve.

        Order: finite completed > finite partial > diverged completed >
        diverged partial. The finite-first rule is the fix for the
        partial/aborted interaction: a variant whose ONLY completed run is a
        NaN-aborted high-lr makes `d` non-empty but all-diverged, so a plain
        `if d: ... elif d_partial:` would plot that blown-up curve and hide the
        variant's healthy still-in-flight lrs (which live in `d_partial`).
        Returns (cfg, history, best_lr, disp_loss, is_partial, last_step) or
        None when the variant has nothing to plot.
        """
        fin_d = {lr: v for lr, v in d.items() if math.isfinite(v[0])}
        fin_p = {lr: v for lr, v in d_partial.items() if math.isfinite(v[0])}
        if fin_d:
            lr = min(fin_d, key=lambda k: fin_d[k][0]); loss, cfg, h = fin_d[lr]
            return cfg, h, lr, loss, False, None
        if fin_p:
            lr = min(fin_p, key=lambda k: fin_p[k][0]); loss, cfg, h, st = fin_p[lr]
            return cfg, h, lr, loss, True, st
        if d:  # only diverged completed runs — still show divergence
            lr = _best_lr(d); loss, cfg, h = d[lr]
            return cfg, h, lr, loss, False, None
        if d_partial:  # only diverged partials
            lr = _best_lr({k: d_partial[k] for k in d_partial})
            loss, cfg, h, st = d_partial[lr]
            return cfg, h, lr, loss, True, st
        return None

    # Pick every arm's curve BEFORE drawing any, so the legend can carry a
    # STEP-MATCHED loss. When arms ran to different lengths their endpoints are
    # not comparable -- a 9000-step arm's final against a 2000-step arm's is a
    # horizon difference, not an effect -- and the reader would otherwise have to
    # reconstruct the matched number by eye off the curves.
    #
    # The match step is DERIVED, not configured: the largest step every plotted
    # arm reached. With all arms at one horizon it IS that horizon and the extra
    # legend field is suppressed as redundant; with a 2000-step arm in the panel
    # it is 2000. So the annotation appears exactly when it is needed.
    _picks = []
    for label in variants:
        d = per_variant.get(label, {})
        d_partial = per_variant_partial.get(label, {}) if allow_partial else {}
        pick = _pick_traj(d, d_partial)
        if pick is not None:
            _picks.append((label, pick))
    _ends = [max((e["step"] for e in pk[1]), default=0) for _l, pk in _picks]
    match_step = min(_ends) if _ends else None
    _mixed = bool(_ends) and max(_ends) != min(_ends)

    def _at(h, step):
        """eval_loss at the last eval on or before `step`, or None."""
        below = [e for e in h if e.get("step") is not None and e["step"] <= step]
        return max(below, key=lambda e: e["step"])["eval_loss"] if below else None

    for label, pick in _picks:
        _cfg, h, best_lr, disp_loss, is_partial, last_step = pick
        # Partial and complete runs are drawn IDENTICALLY; only the legend text
        # differs. An in-flight run used to get linestyle="--" at lw=1.2, which
        # carried no information the reader did not already have: the curve
        # visibly stops short of the horizon, each label owns a unique colour, and
        # the legend already says `partial @<step>`. What it did cost was the
        # figure's one linestyle convention -- a dashed curve then meant "still
        # running" here and "reference line" elsewhere in the same axes.
        note = (f"partial @{last_step}: {disp_loss:.4f}" if is_partial
                else f"final={disp_loss:.4f}")
        # A seed-averaged curve is drawn with its spread. `labeled_completed_runs`
        # averages runs that differ only on `manifest.SERIES_AXIS_FIELDS` and
        # stamps each record with n_seeds / eval_loss_sd / eval_loss_sem, so a
        # multi-seed mean must not render as a bare line indistinguishable from a
        # single run -- that reads as more authoritative while hiding the one
        # thing that says whether a gap between two arms is legible. Band is
        # +/-1 SEM (the uncertainty in the MEAN, which is what the curve IS);
        # n=1 arms get no band and say so in the legend.
        steps = [e["step"] for e in h]
        mean = [e["eval_loss"] for e in h]
        n_seeds = max((e.get("n_seeds", 1) for e in h), default=1)
        if n_seeds > 1:
            sem = [e.get("eval_loss_sem", 0.0) for e in h]
            ax_traj.fill_between(steps,
                                 [m - s for m, s in zip(mean, sem)],
                                 [m + s for m, s in zip(mean, sem)],
                                 color=colors[label], alpha=0.18, linewidth=0)
            note += f", n={n_seeds}"
        # Only when the arms disagree on length AND this arm ran past the match
        # step -- for an arm that ENDS there, `note` already reports that number
        # and a second copy just makes the legend wider.
        own_end = max((e["step"] for e in h), default=0)
        if _mixed and own_end > match_step and (mv := _at(h, match_step)) is not None:
            note += f" | @{match_step}: {mv:.4f}"
        ax_traj.plot(steps, mean,
                     marker=markers[label], ms=3, lw=1.4, color=colors[label],
                     label=f"{label}  (lr={best_lr:g}, {note})")
    # Dashed black line at the speed target: where each curve crosses it is the
    # fraction-of-steps-to-target metric. The target is the AdamW baseline by
    # default (the leaderboard metric is "steps to reach the opt-AdamW loss"),
    # NOT `ref_label` — `ref_label` is only the Δ/σ comparison reference and is
    # sometimes set to a non-AdamW arm (e.g. the ε_rel probe compares vs the
    # chord-tight baseline, which would otherwise pin the line to the floor).
    # Falls back to `ref_label` when the target arm isn't in the panel.
    target_src = target_label if (target_label in per_variant
                                  and per_variant[target_label]) else ref_label
    target_d = per_variant.get(target_src) or {}
    target_best = (min(target_d.values(), key=lambda v: v[0])[0]
                   if target_d else None)
    if target_best is not None:
        # Dashed reference line only — no legend entry (the target value is in
        # the summary table; a legend row for it is superfluous).
        ax_traj.axhline(target_best, color="black", ls="--", lw=1.2, alpha=0.8,
                        zorder=0)
    ax_traj.set_xlabel("step")
    ax_traj.set_ylabel("eval_loss")
    ax_traj.set_title("best-lr trajectory")
    # Pin the x-axis to the full horizon so in-flight (partial) curves render in
    # context instead of auto-scaling to a degenerate window (e.g. 240–260 when
    # only the first eval has landed). Completed runs already span this range.
    # Small right margin so a final marker at exactly `max_steps` isn't clipped
    # in half by the right spine (completed runs end on the horizon).
    # x-limit follows the longest curve actually drawn, not `max_steps`. When a
    # reference arm keeps its full history while the compared arms are truncated,
    # pinning to max_steps would cut the reference off mid-curve.
    _drawn = [max(x) for ln in ax_traj.get_lines()
              if len(x := ln.get_xdata()) and not str(ln.get_label()).startswith("_")]
    ax_traj.set_xlim(0, max(max_steps, max(_drawn, default=0)) * 1.015)
    ax_traj.grid(True, alpha=0.3)
    # Single figure-level legend below both panels (full width → long entries
    # fit across columns without colliding; never covers the data).
    fig.legend(*ax_traj.get_legend_handles_labels(),
               loc="outside lower center", **LEGEND_BELOW_KW)
    # Only best-lr-per-variant curves are plotted here, so the diverged high-lr
    # runs are already absent; auto_ylim_for_trajectory_panel drops any
    # fully-diverged variant (final > divergent_ratio × best). warmup_frac=0
    # makes the upper bound the full-curve max of the non-divergent variants —
    # so the entire descent is visible rather than clipping the early-training
    # top off the curves. (The left panel still clips diverged finals off-axis.)
    if traj_ylim is None and auto_ylim:
        traj_runs = []
        for label in variants:
            d = per_variant.get(label, {})
            d_partial = per_variant_partial.get(label, {}) if allow_partial else {}
            pick = _pick_traj(d, d_partial)
            if pick is not None:
                traj_runs.append((pick[0], pick[1]))
        if traj_runs:
            traj_ylim = auto_ylim_for_trajectory_panel(
                traj_runs, divergent_ratio=divergent_ratio, warmup_frac=0.0)
    if traj_ylim is not None:
        ax_traj.set_ylim(*traj_ylim)

    if suptitle:
        fig.suptitle(suptitle)
    return fig, table_df, summary_df


def leaderboard_panel(model, dataset, rank, suptitle, *,
                      label_filter=None, baselines=(), sigma_ref=None,
                      logs_root=None, **kwargs):
    """Render one leaderboard cell straight from the shared workload registry.

    Replaces the per-notebook `GROUPS = [...]` + hand-rolled dedup + render
    boilerplate: cells become a single call, and run membership comes from
    `lora_playground.workloads` (the same source the doc generator uses), so
    notebooks and the doc can never drift.

    `label_filter(canonical_label, cfg) -> bool` selects a sub-view (e.g. a
    picard or ns family, or the curvature/SOAP arms); default keeps every
    variant. `baselines` is an iterable of canonical labels to ALWAYS include as
    reference curves regardless of `label_filter` — the simple way to overlay an
    extra baseline (e.g. a non-curvature spectral arm) on a filtered sub-view
    without hand-editing the lambda. `sigma_ref` defaults to the workload's. The
    horizon passed to `compare_variants_figure` is the cell's achieved max step
    (so Tulu-3's ~8970-step one-epoch runs are treated as completed, not
    partial). Returns `(fig, table_df, summary_df)` from `compare_variants_figure`.
    """
    from lora_playground.workloads import find_workload, workload_runs
    from .labels import canonical_label
    from .dedup import dedup_by_canonical

    wl = find_workload(model, dataset, rank)
    runs = dedup_by_canonical(workload_runs(wl, logs_root=logs_root))
    keep = set(baselines)
    labeled = [(c, h) for c, h in runs
               if canonical_label(c) is not None
               and (canonical_label(c) in keep
                    or label_filter is None
                    or label_filter(canonical_label(c), c))]
    missing = keep - {canonical_label(c) for c, _ in labeled}
    if missing:
        import warnings
        warnings.warn(f"leaderboard_panel baselines not found for {wl.label}: "
                      f"{sorted(missing)} (label typo or run absent at this rank)")
    # Graceful degrade for in-flight cells: if no runs match, or runs are
    # launched but none has logged an eval yet, show a placeholder instead of
    # raising — so a notebook cell run before the first eval (step ~eval_every)
    # renders cleanly rather than crashing.
    if not labeled or not any(h for _, h in labeled):
        import matplotlib.pyplot as plt
        import pandas as pd
        n = len(labeled)
        msg = (f"no evals yet for {wl.label} — "
               + (f"{n} run(s) launched, awaiting first eval" if n
                  else "no runs found at this cell"))
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (13, 5)))
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        if suptitle:
            fig.suptitle(suptitle)
        return fig, pd.DataFrame(), pd.DataFrame()
    labels = {canonical_label(c) for c, _ in labeled}
    # Pass the workload HORIZON, not the achieved max step. In-flight runs must be
    # treated as partial (trajectory only); the achieved-max heuristic made a
    # step-250 in-flight run look "completed at 250" (wrong final-vs-lr points,
    # collapsed x-axis, spurious divergence). compare_variants_figure's
    # completion_slack still lets ~8970-step one-epoch (Tulu) runs count as final.
    return compare_variants_figure(
        variants={l: {} for l in labels}, common_where={},
        ref_label="AdamW", target_label="AdamW",
        sigma_ref=wl.sigma_ref if sigma_ref is None else sigma_ref,
        max_steps=wl.horizon, allow_partial=True, suptitle=suptitle,
        prefetched_runs=labeled, variant_key=canonical_label, **kwargs)
