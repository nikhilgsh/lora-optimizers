"""AdamW (and secondary) baseline overlays for the 2-panel sweep figure.

Returns `(hlines, ref_curves, eta_sweeps)` tuples consumed by the left and
right panel renderers. Primary AdamW overlay locks the visual register
(black, heavy line, dotted floor hline) — secondary references render as
ordinary candidate-styled lines.
"""
from __future__ import annotations

from .dedup import LabelCollisionError, _hashable, series_id
from .style import (
    BASELINE_COLOR, BASELINE_LS_CURVE, BASELINE_LS_HLINE,
    BASELINE_LW_CURVE, BASELINE_LW_HLINE, BASELINE_MARKER,
    LINE_WIDTH,
)


# Tuple shapes used by plot_eta_vs_final / plot_best_eta_curves:
#   hline:     (label, y, color, ls, lw)
#   ref_curve: (label, evs, color, ls, lw, marker, std_evs)  — std_evs may be None
#   eta_sweep: (label, points, color, ls, lw, marker)        — points = [(lr, mean, std, n)]


def _eta_sweep_points(reference_runs, optimizer: str,
                      *,
                      homogeneous_axes: tuple = ("max_steps", "lora_r"),
                      raise_on_mixed: bool = True):
    """Per-η (η, mean_final, std_final, n) points for `optimizer` in
    `reference_runs`. Multi-seed runs at the same η are aggregated.

    Robustness: ENFORCES that every lr-bucket is homogeneous on the
    `homogeneous_axes` (default: ``max_steps``, ``lora_r``). Mixing runs
    from different experimental regimes is a silent data-pollution failure
    — manifests as huge std on error bars and wildly inflated reference
    curves. On a mixed bucket, raise ``ValueError`` with the specific
    (lr, axis, values) detail so callers add the missing filter to their
    ``load_runs(where=...)`` call.
    """
    import statistics
    import sys
    from collections import defaultdict
    buckets: dict[float, list[tuple]] = defaultdict(list)
    for c, e in reference_runs:
        if c.get("optimizer") != optimizer:
            continue
        axis_sig = tuple(_hashable(c.get(k)) for k in homogeneous_axes)
        buckets[float(c["lr"])].append(
            (e[-1]["eval_loss"], axis_sig, c.get("log_group", "?"))
        )

    # Homogeneity audit: each lr must have one and only one axis signature.
    mixed = []
    for lr in sorted(buckets):
        sigs = {entry[1] for entry in buckets[lr]}
        if len(sigs) > 1:
            details = {}
            for sig in sigs:
                groups = {entry[2] for entry in buckets[lr] if entry[1] == sig}
                details[sig] = sorted(groups)
            mixed.append((lr, details))
    if mixed:
        lines = [
            f"_eta_sweep_points: cross-regime pollution detected for "
            f"optimizer={optimizer!r}, homogeneous_axes={list(homogeneous_axes)}:"
        ]
        for lr, det in mixed[:5]:
            lines.append(f"  lr={lr:.0e}:")
            for sig, groups in det.items():
                lines.append(
                    f"    {dict(zip(homogeneous_axes, sig))!r} ← "
                    f"log_groups={groups}"
                )
        lines.append(
            "Fix: add the missing axis to your load_runs(where=...) "
            "filter, e.g. `'max_steps': 4000`. To intentionally mix "
            "regimes, pass raise_on_mixed=False or filter "
            "homogeneous_axes=()."
        )
        msg = "\n".join(lines)
        if raise_on_mixed:
            raise ValueError(msg)
        print(msg, file=sys.stderr)

    points = []
    for lr in sorted(buckets):
        ys = [entry[0] for entry in buckets[lr]]
        mean = sum(ys) / len(ys)
        std = statistics.stdev(ys) if len(ys) > 1 else 0.0
        points.append((lr, mean, std, len(ys)))
    return points


def _multi_seed_curve(reference_runs, optimizer: str, best_lr: float):
    """For ``optimizer`` at ``best_lr`` in ``reference_runs``, aggregate
    per-step eval_loss across seeds into a mean-and-std trajectory.

    Returns (mean_evs, std_evs, n_seeds), where ``mean_evs`` and ``std_evs``
    are lists of ``{step, eval_loss}`` dicts sharing the same step axis.
    When only one seed is present, ``std_evs`` has ``eval_loss=0`` at each
    step (renders as a zero-width band).
    """
    import statistics
    from collections import defaultdict
    per_step: dict[int, list[float]] = defaultdict(list)
    seeds_at_lr: set = set()
    n_seeds_at_lr = 0
    averaged_cfgs: list[dict] = []
    for c, evs in reference_runs:
        if c.get("optimizer") != optimizer:
            continue
        if float(c["lr"]) != best_lr:
            continue
        seed = c.get("seed")
        if seed in seeds_at_lr:
            continue
        seeds_at_lr.add(seed)
        n_seeds_at_lr += 1
        averaged_cfgs.append(c)
        for ev in evs:
            per_step[int(ev["step"])].append(float(ev["eval_loss"]))

    # Series-id contract (defensive): the runs we're about to average must
    # all share the same series_id, otherwise we'd silently mix distinct
    # algorithms across "seeds." Reaches here only when the caller bypassed
    # the entry-point assertion (standard_sweep_figure / plot_eta_vs_final).
    if len(averaged_cfgs) > 1:
        ids = {series_id(c) for c in averaged_cfgs}
        if len(ids) > 1:
            raise LabelCollisionError(
                f"_multi_seed_curve({optimizer=!r}, {best_lr=}) would "
                f"average across {len(ids)} distinct series_ids. The caller "
                f"must filter reference_runs to a single algorithm + model "
                f"config before requesting seed-aggregated curves."
            )
    # Keep only steps present in every seed.
    common_steps = sorted(s for s, ys in per_step.items() if len(ys) == n_seeds_at_lr)
    mean_evs = []
    std_evs = []
    for s in common_steps:
        ys = per_step[s]
        mean_evs.append({"step": s, "eval_loss": sum(ys) / len(ys)})
        std_evs.append({"step": s,
                        "eval_loss": (statistics.stdev(ys) if len(ys) > 1 else 0.0)})
    return mean_evs, std_evs, n_seeds_at_lr


def baseline_overlay(reference_runs, optimizer: str, *,
                     label: str | None = None,
                     color: str | None = None,
                     is_primary: bool = False,
                     marker_map: dict | None = None,
                     ) -> tuple[list, list, list]:
    """Build (hlines, ref_curves, eta_sweeps) entries for overlaying the
    `optimizer` baseline from `reference_runs`.

    is_primary=True → primary baseline styling (heavier line, no markers,
    "(baseline)" label) for the AdamW reference. False → lighter dotted
    reference style for secondary baselines.

    The library guarantees a single visual idiom for the AdamW baseline
    — color, linewidth, linestyle, marker, hline style are all fixed
    constants and not configurable by callers. The floor hline uses a
    different linestyle than the training curve so the two don't visually
    merge.

    Returns:
        hlines:     [(label, y, color, ls, lw)]                       — left-panel floor
        ref_curves: [(label, mean_evs, color, ls, lw, marker, std_evs)] — right-panel curve;
                    if multi-seed at best η, std_evs is a list of
                    {step, eval_loss} dicts (per-step std) and the renderer
                    fills a ±σ band around mean_evs.
        eta_sweeps: [(label, points, color, ls, lw, marker)]          — left-panel η-sweep;
                    points is [(lr, mean, std, n), …]; empty if < 1 η point.
    """
    label = label or optimizer
    sweep_points = _eta_sweep_points(reference_runs, optimizer)
    if not sweep_points:
        return [], [], []
    # Pick the best η by mean final loss (multi-seed aware), NOT by best-seed.
    best_lr, _best_mean, best_std, _best_n = min(sweep_points, key=lambda p: p[1])
    mean_evs, std_evs, n_seeds = _multi_seed_curve(
        reference_runs, optimizer, best_lr,
    )
    if not mean_evs:
        return [], [], []
    fl = mean_evs[-1]["eval_loss"]

    # Always emit the η-sweep entry — even single-η references render as one
    # point-with-error-bar, which is more informative than silently dropping
    # the baseline from the left panel when only one η was tested at this rank.
    if is_primary:
        color = BASELINE_COLOR  # locked black
        hline = (f"{label} floor ({fl:.4f}{' ± ' + format(best_std, '.4f') if n_seeds > 1 else ''})",
                 fl, color, BASELINE_LS_HLINE, BASELINE_LW_HLINE)
        seed_tag = f", n_seed={n_seeds}" if n_seeds > 1 else ""
        curve = (f"{label} (baseline, η={best_lr:.0e}, final={fl:.4f}{seed_tag})",
                 mean_evs, color, BASELINE_LS_CURVE,
                 BASELINE_LW_CURVE, BASELINE_MARKER, std_evs)
        sweep_label = (f"{label} η-sweep ({len(sweep_points)} pts)"
                       if len(sweep_points) >= 2
                       else f"{label} @ η={best_lr:.0e} (only η tested)")
        eta_sweep = (
            sweep_label,
            sweep_points, color, BASELINE_LS_CURVE,
            BASELINE_LW_CURVE, BASELINE_MARKER,
        )
        return [hline], [curve], [eta_sweep]

    # Secondary reference: ordinary candidate styling. No hline.
    color = color or "#1f77b4"
    marker_map = marker_map or {}
    marker = marker_map.get(optimizer, "o")
    curve = (f"{label} (η={best_lr:.0e}, final={fl:.4f})",
             mean_evs, color, "-", LINE_WIDTH, marker, std_evs)
    sweep_label = (f"{label} η-sweep" if len(sweep_points) >= 2
                   else f"{label} @ η={best_lr:.0e}")
    eta_sweep = (
        sweep_label,
        sweep_points, color, "-", LINE_WIDTH, marker,
    )
    return [], [curve], [eta_sweep]
