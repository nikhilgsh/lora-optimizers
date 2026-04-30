"""Shared utilities for the sweep-analysis notebook.

Loads sweep results from per-task .out logs, filters diverged runs, and
draws the standardized 2-panel figure (η vs final loss + best-η training
curves) that all sections of the notebook share.

Design: every section of the notebook supplies a sequence of (cfg, evs) runs,
plus a callable `group_key(cfg) → str` that maps a run to a color/legend group.
Everything else (filtering, axes, legend placement, training-curve picking,
AdamW baseline overlay, strict-win bar, layout, fonts) is handled here.
The intended call site is `standard_sweep_figure(runs, group_key_fn,
color_map, suptitle=..., reference_runs=...)` — every figure produced this
way is uniform by construction.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DIVERGE_THRESHOLD = 1.5

# ─── per-section configs (kept here so cells are one-line library calls) ──────

# Optimizer color palette shared across the all-optimizer η-sweep and the
# cross-investigation sections. NOTE: pure "black" is reserved for the AdamW
# baseline overlay; no candidate group should use it.
OPTIM_COLORS = {
    "adamw":                       "#1a1a1a",  # near-black; baseline overlay uses pure black
    "scaled-lora":                 "#ff7f0e",
    "lin-lora":                    "#2ca02c",
    "adam-scaled-lora":            "#d62728",
    "adam-lin-lora":               "#9467bd",
    "muon-lora":                   "#e377c2",
    "diag-scaled-lora":            "#17becf",
    "kron-grad-lora":              "#bcbd22",
    "psi-lora":                    "#8c564b",
    "galore-adamw":                "#7f7f7f",
    "adam-lin-lora-post":          "#2ca02c",
    "adam-scaled-lora-post":       "#ff7f0e",
    "adam-lin-lora-matrix":        "#17becf",
    "adam-scaled-lora-matrix":     "#bcbd22",
    "adam-polar-product-lora":     "#8c564b",
    "polar-product-lora":          "#e377c2",
}


# ─── shared style constants ───────────────────────────────────────────────────

# Legend placement: outside the right edge of the panel.
LEGEND_KW_BASE = dict(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      fontsize=14, frameon=True, borderaxespad=0.0,
                      handlelength=2.4, handletextpad=0.7,
                      labelspacing=0.55)

# Title / axis-label sizes — bumped throughout for readability.
SUPTITLE_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 15
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 13

# Backwards-compat alias for any external code that still imports LEGEND_KW.
LEGEND_KW = LEGEND_KW_BASE

# Candidate-line styling.
MARKER_SIZE = 6
LINE_WIDTH = 2.0

# Reference / overlay styling for non-primary baselines (e.g. adam-lin-lora).
REF_LINE_WIDTH = 1.5

# Primary baseline (AdamW): visually distinct from every candidate.
BASELINE_COLOR = "black"
BASELINE_LW_HLINE = 1.5
BASELINE_LS_HLINE = (0, (1, 1.5))   # fine dotted — distinguished from the curve
BASELINE_LW_CURVE = 2.5
BASELINE_LS_CURVE = (0, (5, 2))     # long-dash; reads cleanly over data
BASELINE_MARKER = "s"               # square — distinguished from candidates' "o"
BASELINE_ZORDER = 2                 # behind candidates so it never covers a crossing

# Default figure size. Wide enough that an outside legend with long entries
# (e.g. "adam-polar-product-lora (η=3e-04, final=0.7546)") doesn't squish the
# axes. Constrained layout reserves the right margin for the legend.
DEFAULT_FIGSIZE = (20, 6.0)


# ─── data loading ─────────────────────────────────────────────────────────────

def parse_flag(command: str, flag: str) -> str | None:
    """Extract --flag VALUE from a command string."""
    parts = shlex.split(command)
    for i, p in enumerate(parts):
        if p == flag and i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_run(log_path: Path) -> tuple[dict | None, list[dict]]:
    """Parse a single task .out file → (config dict, list of eval dicts)."""
    config, evals = None, []
    for line in Path(log_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") == "config":
            config = obj
        elif obj.get("event") == "eval":
            evals.append(obj)
    if config is not None and evals:
        config.setdefault("lr", evals[0]["lr"])
        lp = parse_flag(config.get("command", ""), "--lora_plus_multiplier")
        config.setdefault("lora_plus_multiplier", float(lp) if lp else 1.0)
    return config, evals


def load_sweep(group: str, logs_root: str = "../logs") -> list[tuple[dict, list[dict]]]:
    """Load all runs for a sweep group. Returns list of (cfg, evs)."""
    log_dir = Path(f"{logs_root}/{group}/run_info/logs")
    runs = []
    for f in sorted(log_dir.glob("*.out")):
        cfg, evs = load_run(f)
        if cfg is not None and evs:
            runs.append((cfg, evs))
    return runs


def has_runs(group: str, logs_root: str = "../logs") -> bool:
    """True if the group has at least one populated .out file."""
    log_dir = Path(f"{logs_root}/{group}/run_info/logs")
    if not log_dir.exists():
        return False
    return any(f.stat().st_size > 0 for f in log_dir.glob("*.out"))


def merge_runs(group_priority: Iterable[str],
               key_fn: Callable[[dict], tuple],
               filter_fn: Callable[[dict], bool] | None = None,
               cfg_postprocess: Callable[[dict, str], None] | None = None,
               logs_root: str = "../logs") -> list[tuple[dict, list[dict]]]:
    """Merge sweeps in priority order, deduplicating by `key_fn(cfg)`."""
    out, seen = [], set()
    for group in group_priority:
        if not has_runs(group, logs_root):
            continue
        for cfg, evs in load_sweep(group, logs_root):
            if cfg_postprocess is not None:
                cfg_postprocess(cfg, group)
            if filter_fn is not None and not filter_fn(cfg):
                continue
            k = key_fn(cfg)
            if k in seen:
                continue
            seen.add(k)
            out.append((cfg, evs))
    return out


# ─── diverged-run filtering ───────────────────────────────────────────────────

def max_loss(evs: list[dict]) -> float:
    return max(e["eval_loss"] for e in evs)


def split_diverged(runs, threshold: float = DIVERGE_THRESHOLD,
                   hard_max: float = float("inf")):
    def _div(evs):
        return evs[-1]["eval_loss"] >= threshold or max_loss(evs) >= hard_max

    keep = [(c, e) for c, e in runs if not _div(e)]
    drop = [(c, e) for c, e in runs if _div(e)]
    return keep, drop


def report_diverged(drop, label_fn: Callable[[dict], str]) -> None:
    for cfg, evs in sorted(drop, key=lambda x: label_fn(x[0])):
        print(f"  [filtered diverged] {label_fn(cfg):<24s} "
              f"max={max_loss(evs):.3f} final={evs[-1]['eval_loss']:.3f}")


# ─── baseline / reference helpers ────────────────────────────────────────────

def best_run(runs, filter_fn: Callable[[dict], bool]):
    """Return (cfg, evs) with lowest final eval_loss matching filter_fn, else None."""
    matches = [(c, e) for c, e in runs if filter_fn(c)]
    if not matches:
        return None
    return min(matches, key=lambda x: x[1][-1]["eval_loss"])


# Tuple shapes used by plot_eta_vs_final / plot_best_eta_curves:
#   hline:     (label, y, color, ls, lw)
#   ref_curve: (label, evs, color, ls, lw, marker)
# Older 3-tuple / 4-tuple shapes are accepted for backward compatibility.

def _normalize_hline(entry):
    if len(entry) == 3:
        label, y, color = entry
        return (label, y, color, ":", REF_LINE_WIDTH)
    return entry  # already 5-tuple


def _normalize_ref_curve(entry):
    if len(entry) == 4:
        label, evs, color, ls = entry
        return (label, evs, color, ls, REF_LINE_WIDTH, None)
    return entry  # already 6-tuple


def _eta_sweep_points(reference_runs, optimizer: str):
    """All (η, final_eval) points for `optimizer` in `reference_runs`,
    sorted by η. Used to overlay the full LR-sweep curve of the baseline."""
    points = sorted(
        (float(c["lr"]), e[-1]["eval_loss"])
        for c, e in reference_runs if c.get("optimizer") == optimizer
    )
    return points


def baseline_overlay(reference_runs, optimizer: str, *,
                     label: str | None = None,
                     color: str | None = None,
                     is_primary: bool = False,
                     ) -> tuple[list, list, list]:
    """Build (hlines, ref_curves, eta_sweeps) entries for overlaying the
    `optimizer` baseline from `reference_runs`.

    is_primary=True → primary baseline styling (heavier line, long-dash,
    no markers, "(baseline)" label) for the AdamW reference. False →
    lighter dotted reference style for secondary baselines.

    The library guarantees a single visual idiom for the AdamW baseline
    — color, linewidth, linestyle, marker, hline style are all fixed
    constants (BASELINE_*) and not configurable by callers. The floor
    hline uses a different linestyle than the training curve so the two
    don't visually merge.

    Returns:
        hlines:     [(label, y, color, ls, lw)]            — left-panel floor
        ref_curves: [(label, evs, color, ls, lw, marker)]  — right-panel curve
        eta_sweeps: [(label, points, color, ls, lw, marker)] — left-panel
                    η-sweep so the full LR grid the baseline was tuned over
                    is visible. Empty if the baseline has < 2 η points.
    """
    ref = best_run(reference_runs, lambda c: c["optimizer"] == optimizer)
    if ref is None:
        return [], [], []
    cfg, evs = ref
    fl = evs[-1]["eval_loss"]
    label = label or optimizer
    sweep_points = _eta_sweep_points(reference_runs, optimizer)

    # Only the PRIMARY baseline (AdamW) gets the distinguished visual
    # register: long-dash, square markers, heavy line, black, floor hline.
    # Secondary references render as ordinary candidate-styled lines (solid,
    # circle markers, normal weight) — they're not baselines, just additional
    # comparisons pulled from reference_runs. This is enforced; callers
    # cannot upgrade a secondary to baseline styling.
    if is_primary:
        color = BASELINE_COLOR  # locked black
        hline = (f"{label} floor ({fl:.4f})",
                 fl, color, BASELINE_LS_HLINE, BASELINE_LW_HLINE)
        curve = (f"{label} (baseline, η={cfg['lr']:.0e}, final={fl:.4f})",
                 evs, color, BASELINE_LS_CURVE,
                 BASELINE_LW_CURVE, BASELINE_MARKER)
        eta_sweep = (
            f"{label} η-sweep ({len(sweep_points)} pts)",
            sweep_points, color, BASELINE_LS_CURVE,
            BASELINE_LW_CURVE, BASELINE_MARKER,
        ) if len(sweep_points) >= 2 else None
        return [hline], [curve], ([eta_sweep] if eta_sweep else [])

    # Secondary reference: ordinary candidate styling. No hline (only the
    # primary baseline owns one).
    color = color or "#1f77b4"
    curve = (f"{label} (η={cfg['lr']:.0e}, final={fl:.4f})",
             evs, color, "-", LINE_WIDTH, "o")
    eta_sweep = (
        f"{label} η-sweep",
        sweep_points, color, "-", LINE_WIDTH, "o",
    ) if len(sweep_points) >= 2 else None
    return [], [curve], ([eta_sweep] if eta_sweep else [])


# ─── per-rank leaderboard bar chart ──────────────────────────────────────────

def plot_leaderboard_by_rank(best: dict, baseline_optimizer: str = "adamw",
                              color_map: dict | None = None,
                              suptitle: str = "Best eval vs rank"):
    """Single-panel leaderboard: best-η eval loss vs rank, one line per optimizer."""
    color_map = color_map or {}
    baseline_floor = {r: best[(baseline_optimizer, r)][2]
                      for (opt, r) in best if opt == baseline_optimizer}
    ranks = sorted(baseline_floor)

    series = {}
    for (opt, r), (cfg, evs, fl) in best.items():
        series.setdefault(opt, []).append((r, fl))
    for opt in series:
        series[opt].sort()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    all_losses = []
    for opt, points in sorted(series.items()):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        is_baseline = opt == baseline_optimizer
        ax.plot(xs, ys,
                color=BASELINE_COLOR if is_baseline else color_map.get(opt, "grey"),
                lw=BASELINE_LW_CURVE if is_baseline else LINE_WIDTH,
                ls=BASELINE_LS_CURVE if is_baseline else "-",
                marker=BASELINE_MARKER if is_baseline else "o",
                markersize=MARKER_SIZE,
                label=f"{opt} (baseline)" if is_baseline else opt,
                zorder=BASELINE_ZORDER if is_baseline else 5)
        all_losses.extend(ys)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("LoRA rank r"); ax.set_ylabel("Best-η final eval loss")
    ax.set_title(suptitle, fontsize=12, fontweight="bold")
    if all_losses:
        lo = min(all_losses) - 0.005
        hi = min(all_losses) + 0.05
        ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.legend(**LEGEND_KW_BASE)
    return fig, ax


# ─── standardized 2-panel figure ──────────────────────────────────────────────

def _legend_kw(n_entries: int) -> dict:
    """Single-column legend always — 2-col looked busy and broke the
    visual hierarchy. Figure width carries the room instead."""
    return dict(LEGEND_KW_BASE)


def plot_eta_vs_final(ax, runs, group_key_fn: Callable[[dict], str],
                      color_map: dict, *, hlines: list[tuple] | None = None,
                      ref_eta_sweeps: list[tuple] | None = None,
                      title: str = "Final eval loss vs η, per group",
                      legend: bool = True,
                      adamw_group_keys: set[str] | None = None) -> None:
    """Left panel: η vs final eval loss, one line per group key.

    `ref_eta_sweeps`: list of (label, points, color, ls, lw, marker) tuples
    where `points` is a list of (η, final_loss) — e.g. the AdamW baseline's
    full LR-sweep curve. Drawn before candidates so candidate lines sit on top.
    """
    adamw_group_keys = adamw_group_keys or set()

    all_losses = []
    if ref_eta_sweeps:
        for entry in ref_eta_sweeps:
            label, points, color, ls, lw, marker = entry
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color=color, ls=ls, lw=lw,
                    marker=marker, markersize=MARKER_SIZE,
                    label=label, zorder=BASELINE_ZORDER)
            all_losses.extend(ys)
    if hlines:
        for entry in hlines:
            label, y, color, ls, lw = _normalize_hline(entry)
            ax.axhline(y, color=color, ls=ls, lw=lw, label=label,
                       zorder=BASELINE_ZORDER)
            all_losses.append(y)

    in_range_losses = []
    groups = sorted({group_key_fn(c) for c, _ in runs})
    raw_losses = [e[-1]["eval_loss"] for c, e in runs]
    y_cap = (min(raw_losses) + 0.15) if raw_losses else float("inf")

    for g in groups:
        rows = sorted([(c["lr"], e[-1]["eval_loss"]) for c, e in runs
                       if group_key_fn(c) == g])
        if not rows:
            continue
        xs = [r[0] for r in rows]
        ys = [r[1] if r[1] <= y_cap else float("nan") for r in rows]
        is_adamw = g in adamw_group_keys
        ax.plot(xs, ys,
                color=BASELINE_COLOR if is_adamw else color_map.get(g, "grey"),
                marker=BASELINE_MARKER if is_adamw else "o",
                markersize=MARKER_SIZE,
                lw=BASELINE_LW_CURVE if is_adamw else LINE_WIDTH,
                ls=BASELINE_LS_CURVE if is_adamw else "-",
                zorder=BASELINE_ZORDER if is_adamw else 5,
                label=f"{g} (baseline)" if is_adamw else g)
        in_range_losses.extend(y for y in ys if y == y)

    ax.set_xscale("log")
    ax.set_xlabel("η (log)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Final eval loss", fontsize=AXIS_LABEL_FONTSIZE)
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
        ax.legend(**_legend_kw(len(groups)))


def plot_best_eta_curves(ax, runs, group_key_fn: Callable[[dict], str],
                         color_map: dict, *, ref_curves: list[tuple] | None = None,
                         title: str = "Best η per group — training curves",
                         x_tick_step: int = 200,
                         legend: bool = True,
                         adamw_group_keys: set[str] | None = None) -> None:
    """Right panel: training curves for the best (lowest final loss) η per group."""
    adamw_group_keys = adamw_group_keys or set()

    if ref_curves:
        for entry in ref_curves:
            label, evs, color, ls, lw, marker = _normalize_ref_curve(entry)
            ax.plot([e["step"] for e in evs], [e["eval_loss"] for e in evs],
                    color=color, lw=lw, ls=ls, marker=marker,
                    markersize=MARKER_SIZE, label=label,
                    zorder=BASELINE_ZORDER)

    best = {}
    for cfg, evs in runs:
        g = group_key_fn(cfg)
        fl = evs[-1]["eval_loss"]
        if g not in best or fl < best[g][2]:
            best[g] = (cfg, evs, fl)
    n_lines = len(best) + (len(ref_curves) if ref_curves else 0)
    for g, (cfg, evs, fl) in sorted(best.items(), key=lambda kv: kv[1][2]):
        is_adamw = g in adamw_group_keys
        label = (f"{g} (baseline, η={cfg['lr']:.0e}, final={fl:.4f})"
                 if is_adamw else f"{g} (η={cfg['lr']:.0e}, final={fl:.4f})")
        ax.plot([e["step"] for e in evs], [e["eval_loss"] for e in evs],
                color=BASELINE_COLOR if is_adamw else color_map.get(g, "grey"),
                marker=BASELINE_MARKER if is_adamw else "o",
                markersize=MARKER_SIZE,
                lw=BASELINE_LW_CURVE if is_adamw else LINE_WIDTH,
                ls=BASELINE_LS_CURVE if is_adamw else "-",
                zorder=BASELINE_ZORDER if is_adamw else 5,
                label=label)
    ax.set_xlabel("Step", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Eval loss", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))
    ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    if legend:
        ax.legend(**_legend_kw(n_lines))


CANONICAL_HORIZON = 2000


def two_panel_sweep_figure(runs, group_key_fn, color_map, *,
                           suptitle: str = "",
                           hlines: list[tuple] | None = None,
                           ref_curves: list[tuple] | None = None,
                           ref_eta_sweeps: list[tuple] | None = None,
                           threshold: float = DIVERGE_THRESHOLD,
                           min_step: int | None = CANONICAL_HORIZON,
                           figsize: tuple = DEFAULT_FIGSIZE,
                           x_tick_step: int = 200,
                           left_title: str = "Final eval loss vs η, per group",
                           right_title: str = "Best η per group — training curves",
                           label_fn: Callable[[dict], str] | None = None,
                           adamw_group_keys: set[str] | None = None):
    """Build the standardized 2-panel sweep figure with diverged-run filtering.
    Returns (fig, axes, n_kept, n_dropped).

    Layout: left panel is η-vs-final-loss with legend suppressed (same colors
    as the right panel which carries descriptive labels). Right panel has
    best-η training curves with descriptive legend outside on the right.
    Constrained layout reserves room for the outside legend.
    """
    keep, drop = split_diverged(runs, threshold)
    if drop:
        if label_fn is None:
            label_fn = lambda c: f"{group_key_fn(c)} η={c['lr']:.0e}"
        report_diverged(drop, label_fn)

    # Left panel implicitly assumes the run reached its final loss (it
    # plots evs[-1]['eval_loss'] as "final"). In-flight runs at step 200
    # would bias the comparison. Exclude them from the LEFT panel only —
    # the right panel still shows the partial training curve so progress
    # is visible. Set min_step=None to disable.
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
    plot_eta_vs_final(axes[0], keep_for_left, group_key_fn, color_map,
                      hlines=hlines, ref_eta_sweeps=ref_eta_sweeps,
                      title=left_title, legend=False,
                      adamw_group_keys=adamw_group_keys)
    plot_best_eta_curves(axes[1], keep_for_right, group_key_fn, color_map,
                         ref_curves=ref_curves, title=right_title,
                         x_tick_step=x_tick_step,
                         adamw_group_keys=adamw_group_keys)
    if suptitle:
        fig.suptitle(suptitle, fontsize=SUPTITLE_FONTSIZE, fontweight="bold")
    return fig, axes, len(keep), len(drop)


def standard_sweep_figure(runs, group_key_fn, color_map, *,
                          reference_runs,
                          suptitle: str = "",
                          extra_baselines: Iterable[tuple[str, str]] = (),
                          baseline_optimizer: str = "adamw",
                          **kwargs):
    """High-level uniform sweep figure — every section's single entry point.

    The library guarantees, on every call:
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
        reference_runs:  REQUIRED. Sweeps containing the baseline optimizer
                         (typically `all_runs`). Pass `runs` itself when the
                         baseline is among the candidates.
        suptitle:        figure suptitle (run-count tally is appended).
        extra_baselines: extra (optimizer, color) pairs to overlay as light
                         dotted references.
        **kwargs:        forwarded to two_panel_sweep_figure.

    Raises:
        ValueError: if `reference_runs` has no run for `baseline_optimizer`.
    """
    hlines, ref_curves, eta_sweeps = baseline_overlay(
        reference_runs, baseline_optimizer, is_primary=True,
    )
    if not hlines:
        raise ValueError(
            f"No {baseline_optimizer!r} run found in reference_runs — "
            "every standard sweep figure requires the baseline.")

    for opt, color in extra_baselines:
        _h, r, e = baseline_overlay(
            reference_runs, opt, color=color, is_primary=False,
        )
        # Secondary baselines contribute the right-panel reference curve
        # AND the left-panel η-sweep (so the LR grid for the secondary
        # comparison is visible too). The hline is suppressed — the
        # primary baseline (AdamW) owns the only floor hline.
        ref_curves.extend(r)
        eta_sweeps.extend(e)

    # If the candidate runs already contain the baseline (e.g. all-optimizers
    # or cross-investigation cells), the candidate's own η-sweep + best-η
    # curve already cover the left + right panel; the overlay would
    # double-draw. Detect overlap, restyle the candidate to baseline style
    # via adamw_group_keys, and drop the overlay's eta_sweep + ref_curve
    # (keep the floor hline — the candidate doesn't draw that).
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
