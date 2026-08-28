"""Rendering-only adapter for :mod:`lora_playground.comparison` results.

The comparison core owns run assignment, completion classification, replicate
aggregation, and best-LR selection.  This module deliberately does none of
those things: it turns an already-built :class:`ComparisonResult` into the
established final-vs-LR table, summary table, and two-panel figure.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from lora_playground.comparison import (
    AggregatedCurve,
    ComparisonResult,
    VariantSpec,
)

from .panels import (
    auto_ylim_for_final_panel,
    auto_ylim_for_trajectory_panel,
    clamp_for_hollow,
    draw_lr_series,
)
from .style import LEGEND_BELOW_KW, STAR_MARKER_SIZE


# No "^"/"v": `draw_lr_series` reserves a filled upward triangle for "finite
# observation above the visible range", so a series that also uses one makes
# the two unreadable apart.
_MARKERS = ("o", "s", "D", "P", "X", "h", "p")
__all__ = ["render_comparison"]


def _style_value(
    values: Mapping[str, str] | None,
    spec: VariantSpec,
    default: str,
) -> str:
    if values is None:
        return default
    if spec.id in values:
        return values[spec.id]
    if spec.style_key is not None and spec.style_key in values:
        return values[spec.style_key]
    return default


def _display_labels(
    specs: tuple[VariantSpec, ...],
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    labels = {
        spec.id: overrides.get(spec.id, spec.label) if overrides else spec.label
        for spec in specs
    }
    duplicates = sorted({label for label in labels.values()
                         if list(labels.values()).count(label) > 1})
    if duplicates:
        raise ValueError(
            "display labels must be unique to form an unambiguous table: "
            f"{duplicates}"
        )
    return labels


def _selected_trajectory(
    result: ComparisonResult,
    variant_id: str,
    *,
    show_partials: bool,
) -> AggregatedCurve | None:
    """Choose between the two upstream-selected representatives.

    This preserves the established visibility order without searching either
    LR map: finite completed, finite partial, diverged completed, then diverged
    partial.  In particular, a NaN-aborted completed run cannot hide a healthy
    in-flight representative.
    """
    completed = result.best_completed.get(variant_id)
    partial = result.best_partial.get(variant_id) if show_partials else None
    if completed is not None and math.isfinite(completed.final_loss):
        return completed
    if partial is not None and math.isfinite(partial.final_loss):
        return partial
    return completed if completed is not None else partial


def _as_runs(curves: list[AggregatedCurve]):
    """Adapt curves to existing auto-ylim helpers without reducing them."""
    return [(dict(curve.cfg), [dict(event) for event in curve.history])
            for curve in curves]


def _horizon_label(horizon: int) -> str:
    if horizon % 1000 == 0:
        return f"{horizon // 1000}k"
    return str(horizon)


def render_comparison(
    result: ComparisonResult,
    *,
    reference_id: str,
    horizon: int,
    sigma_ref: float = 0.0007,
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    figsize: tuple[float, float] = (13, 6.2),
    suptitle: str | None = None,
    show_partials: bool = True,
    final_ylim: tuple[float, float] | None = None,
    traj_ylim: tuple[float, float] | None = None,
    auto_ylim: bool = True,
    divergent_ratio: float = 1.5,
    target_id: str | None = None,
) -> tuple[Figure, pd.DataFrame, pd.DataFrame]:
    """Render a precomputed comparison as ``(figure, LR table, summary)``.

    Presentation mappings are keyed by stable variant ID.  For migration from
    current registries, ``colors`` and ``markers`` may also contain a spec's
    ``style_key``; an explicit ID entry wins.  Neither labels nor style values
    participate in curve selection.

    ``result.best_completed`` drives the completed summary and trajectory.
    When partials are shown, ``result.best_partial`` is considered only by the
    finite/completed visibility precedence documented in
    :func:`_selected_trajectory`.  The per-LR maps are used solely for the
    completed LR table and left panel.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if sigma_ref <= 0:
        raise ValueError(f"sigma_ref must be positive, got {sigma_ref}")

    specs = tuple(result.variants)
    ids = tuple(spec.id for spec in specs)
    if len(set(ids)) != len(ids):
        raise ValueError("ComparisonResult contains duplicate variant IDs")
    if reference_id not in ids:
        raise KeyError(f"unknown reference_id {reference_id!r}")
    if target_id is not None and target_id not in ids:
        raise KeyError(f"unknown target_id {target_id!r}")

    display = _display_labels(specs, labels)
    cmap = plt.get_cmap("tab10")
    resolved_colors = {
        spec.id: _style_value(
            colors,
            spec,
            "black" if spec.id == reference_id else cmap(index % cmap.N),
        )
        for index, spec in enumerate(specs)
    }
    resolved_markers = {
        spec.id: _style_value(
            markers, spec, _MARKERS[index % len(_MARKERS)]
        )
        for index, spec in enumerate(specs)
    }
    all_lr = sorted({lr for spec in specs
                     for lr in result.completed.get(spec.id, {})})
    table_df = pd.DataFrame(
        {
            display[spec.id]: [
                (result.completed.get(spec.id, {}).get(lr).final_loss
                 if lr in result.completed.get(spec.id, {}) else float("nan"))
                for lr in all_lr
            ]
            for spec in specs
        },
        index=pd.Index(all_lr, name="lr"),
    )

    reference = result.best_completed.get(reference_id)
    reference_final = reference.final_loss if reference is not None else None
    summary_rows = []
    for spec in specs:
        curve = result.best_completed.get(spec.id)
        if curve is None:
            continue
        delta = None
        if reference_final is not None and spec.id != reference_id:
            delta = curve.final_loss - reference_final
        summary_rows.append({
            "variant": display[spec.id],
            "best_lr": curve.lr,
            "final": curve.final_loss,
            "delta": delta,
            "delta_sigma": delta / sigma_ref if delta is not None else None,
            # Carried here because the legend no longer spells it: the legend
            # names the arm and its eta, this table holds every number.
            "n": curve.n_replicates,
        })
    summary_df = pd.DataFrame(
        summary_rows,
        columns=("variant", "best_lr", "final", "delta", "delta_sigma", "n"),
    )

    fig, (ax_lr, ax_traj) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True
    )
    completed_curves = [
        curve
        for spec in specs
        for curve in result.completed.get(spec.id, {}).values()
    ]
    if final_ylim is None and auto_ylim and completed_curves:
        final_ylim = auto_ylim_for_final_panel(
            _as_runs(completed_curves), divergent_ratio=divergent_ratio
        )
    top = final_ylim[1] if final_ylim is not None else None

    for spec in specs:
        by_lr = result.completed.get(spec.id, {})
        if not by_lr:
            continue
        lrs = sorted(by_lr)
        finals = [by_lr[lr].final_loss for lr in lrs]
        ys_clamped, statuses = clamp_for_hollow(finals, top)
        draw_lr_series(
            ax_lr,
            lrs,
            ys_clamped,
            statuses,
            color=resolved_colors[spec.id],
            marker=resolved_markers[spec.id],
            label=display[spec.id],
            zorder=4,
        )

        best = min(
            (curve for curve in by_lr.values()
             if math.isfinite(curve.final_loss)),
            key=lambda curve: curve.final_loss,
            default=None,
        )
        if best is not None:
            # The one deliberate size override: the star marks the optimum ON
            # TOP of that lr's series marker, so it has to read as larger.
            ax_lr.plot(
                best.lr, best.final_loss, "*", ms=STAR_MARKER_SIZE,
                color=resolved_colors[spec.id],
                mec="white", mew=0.5, zorder=6,
            )

    ax_lr.set_xscale("log")
    ax_lr.set_xlabel(r"Learning rate $\eta$")
    ax_lr.set_ylabel(f"Final evaluation loss at {_horizon_label(horizon)} steps")
    if final_ylim is not None:
        ax_lr.set_ylim(*final_ylim)

    selected: dict[str, AggregatedCurve] = {}
    for spec in specs:
        curve = _selected_trajectory(
            result, spec.id, show_partials=show_partials
        )
        if curve is None:
            continue
        selected[spec.id] = curve
        steps = [event["step"] for event in curve.history]
        losses = [event["eval_loss"] for event in curve.history]
        if curve.n_replicates > 1:
            sem = [event.get("eval_loss_sem", 0.0) for event in curve.history]
            ax_traj.fill_between(
                steps,
                [mean - err for mean, err in zip(losses, sem)],
                [mean + err for mean, err in zip(losses, sem)],
                color=resolved_colors[spec.id],
                alpha=0.18,
                linewidth=0,
            )
        # Every other number that used to ride along here -- the final loss,
        # the replicate count -- is a column of `summary_df`, printed directly
        # under the figure. Repeating them tripled each entry's width, which is
        # what pushed the widest legends off both edges of the figure.
        eta = f"$\\eta$={curve.lr:g}"
        label = (
            f"{display[spec.id]}  ({eta}, partial @{curve.last_step})"
            if not curve.completed
            else f"{display[spec.id]}  ({eta})"
        )
        # No marker. This curve is ~37 eval samples joined by straight
        # segments, and a marker sitting on the stroke bulges it -- at line
        # width 2 and marker size 5 the lumps read as kinks in the loss, which
        # is a claim about the data. Markers belong on the left panel, where
        # each point IS one discrete measurement. Colour carries identity here.
        line, = ax_traj.plot(
            steps,
            losses,
            color=resolved_colors[spec.id],
            label=label,
        )
        line.set_gid(f"trajectory:{spec.id}")

    target_curve = result.best_completed.get(target_id or reference_id)
    if target_curve is None and target_id is not None:
        target_curve = reference
    if target_curve is not None and math.isfinite(target_curve.final_loss):
        ax_traj.axhline(
            target_curve.final_loss,
            color="black",
            ls="--",
            lw=1.2,
            alpha=0.8,
            zorder=0,
            label=f"{display[target_id or reference_id]} at "
                  f"{_horizon_label(horizon)} steps",
        )

    ax_traj.set_xlabel("Training step")
    ax_traj.set_ylabel(r"Evaluation loss at best $\eta$")
    longest = max(
        (max((event.get("step", 0) or 0 for event in curve.history), default=0)
         for curve in selected.values()),
        default=0,
    )
    ax_traj.set_xlim(0, max(horizon, longest) * 1.015)
    handles, legend_labels = ax_traj.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="outside lower center",
            **LEGEND_BELOW_KW,
        )

    if traj_ylim is None and auto_ylim and selected:
        traj_ylim = auto_ylim_for_trajectory_panel(
            _as_runs(list(selected.values())),
            divergent_ratio=divergent_ratio,
            warmup_frac=0.0,
        )
    if traj_ylim is not None:
        ax_traj.set_ylim(*traj_ylim)
    if suptitle:
        fig.suptitle(suptitle)
    return fig, table_df, summary_df
