"""Shared utilities for the sweep-analysis notebook.

Loads sweep results from per-task .out logs, filters diverged runs, and
draws the standardized 2-panel figure (η vs final loss + best-η training
curves) that all sections of the notebook share.

Design: every section of the notebook supplies a sequence of (cfg, evs) runs,
plus a callable `group_key(cfg) → str` that maps a run to a color/legend group.
Everything else (filtering, axes, legend placement, training-curve picking) is
handled here. Sections that need extras (reference lines, axis titles,
non-η x-axes) pass small dicts.
"""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DIVERGE_THRESHOLD = 1.5

# Legend placement: outside the right edge of the panel. fontsize tuned so
# 8-12 entries fit a (~6-inch-tall) panel without overflowing.
LEGEND_KW = dict(loc="center left", bbox_to_anchor=(1.02, 0.5),
                 fontsize=10, frameon=True)

# Shared marker/line sizes so all sections look uniform.
MARKER_SIZE = 4
LINE_WIDTH = 2.0
REF_LINE_WIDTH = 1.5

# Default figure size for the standardized 2-panel sweep figure. (18, 6) gives
# enough horizontal room for legends placed outside the right panel.
DEFAULT_FIGSIZE = (18, 6)


# ─── data loading ─────────────────────────────────────────────────────────────

def parse_flag(command: str, flag: str) -> str | None:
    """Extract --flag VALUE from a command string."""
    parts = shlex.split(command)
    for i, p in enumerate(parts):
        if p == flag and i + 1 < len(parts):
            return parts[i + 1]
    return None


def load_run(log_path: Path) -> tuple[dict | None, list[dict]]:
    """Parse a single task .out file → (config dict, list of eval dicts).
    Enriches config with 'lr' and 'lora_plus_multiplier' from the command
    string when not present in the config event.
    """
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
    """Merge sweeps in priority order, deduplicating by `key_fn(cfg)`.
    Earlier groups override later ones on key collisions.

    filter_fn: optional predicate; runs failing it are dropped silently.
    cfg_postprocess(cfg, group): hook for renames or metadata-injection.
    """
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
    """Split runs into (converged, diverged) by final eval loss.

    A run is considered diverged if EITHER:
      - its FINAL eval loss ≥ `threshold` (clearly failed to learn), OR
      - its MAX eval loss anywhere ≥ `hard_max` (extreme transient spike).

    Default behavior (hard_max=inf): final-loss-only filter. Keeps "ringing
    but recovered" runs whose late behavior is fine. Set hard_max=3 (say) to
    additionally filter runs that hit a really bad transient.
    """
    def _div(evs):
        return evs[-1]["eval_loss"] >= threshold or max_loss(evs) >= hard_max

    keep = [(c, e) for c, e in runs if not _div(e)]
    drop = [(c, e) for c, e in runs if _div(e)]
    return keep, drop


def report_diverged(drop, label_fn: Callable[[dict], str]) -> None:
    for cfg, evs in sorted(drop, key=lambda x: label_fn(x[0])):
        print(f"  [filtered diverged] {label_fn(cfg):<24s} "
              f"max={max_loss(evs):.3f} final={evs[-1]['eval_loss']:.3f}")


# ─── per-rank leaderboard bar chart ──────────────────────────────────────────

def plot_leaderboard_by_rank(best: dict, baseline_optimizer: str = "adamw",
                              color_map: dict | None = None,
                              strict_win_offset: float = 0.01,
                              suptitle: str = "Per-rank leaderboard"):
    """Visual leaderboard: horizontal bar chart per rank, sorted ascending by
    final eval. Baseline (default `adamw`) bars in black; strict-win bar
    (1% below baseline floor) overlaid.

    `best` is the dict returned by per-rank best-η aggregation:
        best[(optimizer, r)] = (cfg, evs, final_eval).
    """
    import matplotlib.pyplot as plt
    color_map = color_map or {}
    baseline_floor = {r: best[(baseline_optimizer, r)][2]
                      for (opt, r) in best if opt == baseline_optimizer}
    ranks = sorted(baseline_floor)
    fig, axes = plt.subplots(1, len(ranks), figsize=(9 * len(ranks), 5),
                              sharex=False)
    if len(ranks) == 1:
        axes = [axes]
    for ax, r_target in zip(axes, ranks):
        floor = baseline_floor[r_target]
        rows = sorted(
            ((opt, fl) for ((opt, r), (cfg, evs, fl)) in best.items()
             if r == r_target),
            key=lambda x: x[1], reverse=True,
        )
        opts = [r[0] for r in rows]
        losses = [r[1] for r in rows]
        colors = ["black" if o == baseline_optimizer else color_map.get(o, "grey")
                  for o in opts]
        ax.barh(range(len(opts)), losses, color=colors,
                edgecolor="black", linewidth=0.5)
        for i, (opt, fl) in enumerate(rows):
            delta = fl - floor
            flag = "✓" if delta < -0.005 else ("≈" if abs(delta) < 0.005 else "✗")
            ax.text(fl + 0.001, i, f"{fl:.4f} (Δ={delta:+.4f}) {flag}",
                    va="center", fontsize=10)
        ax.axvline(floor, color="black", lw=2, ls=":",
                   label=f"{baseline_optimizer} r={r_target} floor = {floor:.4f}")
        ax.axvline(floor - strict_win_offset, color="grey", lw=1.5, ls="--",
                   label=f"strict-win bar = {floor - strict_win_offset:.4f}  "
                         f"({strict_win_offset*100:.0f}% below floor)")
        ax.set_yticks(range(len(opts)))
        ax.set_yticklabels(opts, fontsize=10)
        ax.set_xlabel("Final eval loss")
        ax.set_title(f"r = {r_target}  —  best-η per optimizer")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        xmin = min(losses + [floor - 0.012])
        xmax = max(losses) + 0.025
        ax.set_xlim(xmin, xmax)
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


# ─── standardized 2-panel figure ──────────────────────────────────────────────

def plot_eta_vs_final(ax, runs, group_key_fn: Callable[[dict], str],
                      color_map: dict, *, hlines: list[tuple] | None = None,
                      title: str = "Final eval loss vs η, per group",
                      legend: bool = True) -> None:
    """Left panel: η vs final eval loss, one line per group key.

    `hlines`: optional list of (label, y, color) reference horizontal lines.
    """
    if hlines:
        for label, y, color in hlines:
            ax.axhline(y, color=color, ls=":", lw=REF_LINE_WIDTH, label=label)

    all_losses = []
    groups = sorted({group_key_fn(c) for c, _ in runs})
    for g in groups:
        rows = sorted([(c["lr"], e[-1]["eval_loss"]) for c, e in runs
                       if group_key_fn(c) == g])
        if rows:
            ax.plot([r[0] for r in rows], [r[1] for r in rows],
                    color=color_map.get(g, "grey"),
                    marker="o", markersize=MARKER_SIZE,
                    lw=LINE_WIDTH, label=g)
            all_losses.extend(r[1] for r in rows)
    ax.set_xscale("log")
    ax.set_xlabel("η (log)"); ax.set_ylabel("Final eval loss")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(title)
    # Auto-clip ylim to focus on the competitive region. Without this, runs
    # at extreme η whose final loss is e.g. 1.4 stretch the y-axis and squish
    # the 0.74-0.79 zone where actual differentiation lives.
    if all_losses:
        lo = min(all_losses) - 0.005
        hi = min(all_losses) + 0.10
        ax.set_ylim(lo, hi)
    if legend:
        ax.legend(**LEGEND_KW)


def plot_best_eta_curves(ax, runs, group_key_fn: Callable[[dict], str],
                         color_map: dict, *, ref_curves: list[tuple] | None = None,
                         title: str = "Best η per group — training curves",
                         x_tick_step: int = 200,
                         legend: bool = True) -> None:
    """Right panel: training curves for the best (lowest final loss) η per group.

    `ref_curves`: optional list of (label, evs, color, linestyle) reference
                  curves overlaid as dotted/dashed.
    """
    if ref_curves:
        for label, evs, color, ls in ref_curves:
            ax.plot([e["step"] for e in evs], [e["eval_loss"] for e in evs],
                    color=color, lw=REF_LINE_WIDTH, ls=ls,
                    label=f"{label} (final={evs[-1]['eval_loss']:.4f})")

    best = {}
    for cfg, evs in runs:
        g = group_key_fn(cfg)
        fl = evs[-1]["eval_loss"]
        if g not in best or fl < best[g][2]:
            best[g] = (cfg, evs, fl)
    for g, (cfg, evs, fl) in sorted(best.items(), key=lambda kv: kv[1][2]):
        ax.plot([e["step"] for e in evs], [e["eval_loss"] for e in evs],
                color=color_map.get(g, "grey"),
                marker="o", markersize=MARKER_SIZE, lw=LINE_WIDTH,
                label=f"{g} (η={cfg['lr']:.0e}, final={fl:.4f})")
    ax.set_xlabel("Step"); ax.set_ylabel("Eval loss")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_step))
    ax.set_title(title)
    if legend:
        ax.legend(**LEGEND_KW)


def two_panel_sweep_figure(runs, group_key_fn, color_map, *,
                           suptitle: str = "",
                           hlines: list[tuple] | None = None,
                           ref_curves: list[tuple] | None = None,
                           threshold: float = DIVERGE_THRESHOLD,
                           figsize: tuple = DEFAULT_FIGSIZE,
                           x_tick_step: int = 200,
                           left_title: str = "Final eval loss vs η, per group",
                           right_title: str = "Best η per group — training curves",
                           label_fn: Callable[[dict], str] | None = None):
    """Build the standardized 2-panel sweep figure with diverged-run filtering.
    Reports diverged runs on stdout, returns (fig, axes, n_kept, n_dropped).

    Layout: left panel has η-vs-final-loss (legend suppressed — same colors
    as the right panel which carries the descriptive labels). Right panel
    has best-η training curves with descriptive legend (group, η, final
    loss) placed outside on the right.
    """
    keep, drop = split_diverged(runs, threshold)
    if drop:
        if label_fn is None:
            label_fn = lambda c: f"{group_key_fn(c)} η={c['lr']:.0e}"
        report_diverged(drop, label_fn)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_eta_vs_final(axes[0], keep, group_key_fn, color_map,
                      hlines=hlines, title=left_title, legend=False)
    plot_best_eta_curves(axes[1], keep, group_key_fn, color_map,
                         ref_curves=ref_curves, title=right_title,
                         x_tick_step=x_tick_step)
    # Distinguished AdamW baseline: any line whose group label starts with
    # "adamw" gets bumped to thick black so the preferred baseline pops
    # visually in every section that uses this helper. Applied after both
    # subplots are drawn so it overrides their per-line color/lw choices.
    for ax in axes:
        for line in ax.get_lines():
            label = (line.get_label() or "").lower()
            if label.startswith("adamw"):
                line.set_linewidth(3.0)
                line.set_color("black")
                line.set_zorder(10)
    if suptitle:
        # Note total run count + filtered count in the suptitle.
        full_title = f"{suptitle} ({len(keep)}/{len(runs)} converged"
        if drop:
            full_title += f", {len(drop)} diverged"
        full_title += ")"
        fig.suptitle(full_title)
    fig.tight_layout()
    return fig, axes, len(keep), len(drop)
