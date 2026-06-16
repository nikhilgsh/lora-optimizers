"""Generate the paper's experiment figures + speedup table from the workload registry.

Design (locked with the user, conventions from Mousse/SPlus/LoRA+ in docs/papers/):
  fig1_hero.pdf          — Figure 1: loss-vs-steps showcase at Llama-3.2-1B openmath r256 (AdamW vs
                           PoLoRA vs iMuon when present), dashed line at AdamW's final
                           loss, drop-line at the interpolated crossing, speedup printed.
  tab1_speedup.tex       — Table 1: per-cell speedup-to-AdamW (best-lr), iMuon
                           rows where run. SPlus-style numeric companion to fig 1.
  figA_breadth.pdf       — appendix: per-setting loss curves behind tab_breadth (model
  figA_rank.pdf            families, code r256) and tab_rank (Llama Math rank ladder),
                           each excluding the cells already drawn as curves in the body.
  fig2_ablation.pdf      — E2 basins at Llama openmath r256: PoLoRA vs
                           "w/o curvature control" vs "w/o magnitude control"; ringed minima.
  fig3_lr_transfer.pdf   — E3 basins across the openmath rank ladder, shared log-x so
                           minima alignment across rank is comparable; diverged points
                           exit the frame top (no arrows).

All numbers flow through lora_playground.{workloads,leaderboard} — same source as
docs/notes/leaderboard.md. PNG previews written next to each PDF.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

# Headless (script/CI) → Agg; notebook → keep the inline backend so bpf.figN() renders
# in-cell instead of printing "<Figure ...>".
if "inline" not in matplotlib.get_backend().lower():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

from lora_playground.leaderboard import (
    labeled_completed_runs, leaderboard_rows, reach_fraction, speedup_from_frac,
)
from lora_playground.workloads import find_workload, iter_workloads, workload_runs

FIGS = ROOT / "paper" / "manuscript" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    # Professional, paper-matching look (STIX ~ Times; unified text+math) at readable sizes.
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

# Logical styling (Okabe-Ito, colorblind-safe):
#   ours        -> saturated blue, SOLID, thick (the protagonist stands out)
#   AdamW       -> neutral gray, SOLID, thin (the reference baseline)
#   Muon family -> distinct warm hues, DASHED (the spectral baselines: Muon, iMuon)
#   ablation    -> green/pink, DASHED (w/o curvature; w/o curvature + magnitude)
NAME_CURV = "w/o curvature"
NAME_MAGN = "w/o magnitude"
NAME_NAIVE = "Muon"            # raw per-factor polar = Muon on the factors (sec 3.1)
# Both-controls-removed ablation arm = the bare partner-Gram polar direction (iMuon's
# explicit decoupled update; the memoryless limit of LoRA-RITE; the compositional-Muon
# half-split). Named subtractively, NOT "LoRA-Muon step": (a) it reads as PoLoRA
# minus two controls, and (b) it does not over-credit the concurrent LoRA-Muon for an
# update that pre-dates it. The lineage citation lives in the fig2 caption.
NAME_LM = "w/o curvature + magnitude"
STYLE = {
    "AdamW":      dict(color="#666666", marker="o", ls="-",  lw=1.9),
    "PoLoRA": dict(color="#0072B2", marker="s", ls="-",  lw=2.2),
    NAME_NAIVE:   dict(color="#D55E00", marker="v", ls="--", lw=1.6),
    "iMuon":      dict(color="#E69F00", marker="^", ls="--", lw=1.6),
    NAME_LM:      dict(color="#CC79A7", marker="P", ls="--", lw=1.6),
    NAME_CURV:    dict(color="#009E73", marker="D", ls="--", lw=1.6),
    NAME_MAGN:    dict(color="#882255", marker="X", ls=":",  lw=1.6),
}


def _is_proto(cfg: dict) -> bool:
    return (
        cfg.get("optimizer") == "kl-diag-polar-lora"
        and cfg.get("cw_nesterov") is True
        and cfg.get("polar_method") == "polar_express"
        and cfg.get("beta1") == 0.9
        and cfg.get("precond_method") == "gram_ns"
        and cfg.get("precond_delta") == 1e-4   # pin the locked delta arm (never collapse delta sweeps)
    )


def paper_variant_key(cfg: dict) -> str | None:
    if cfg.get("optimizer") == "adamw":
        return "AdamW"
    if cfg.get("optimizer") == "imuon-lora":
        return "iMuon"
    if _is_proto(cfg) and cfg.get("cw_no_radius") is False and cfg.get("cw_no_diag_curv") is False:
        return "PoLoRA"
    return None


def ablation_variant_key(cfg: dict) -> str | None:
    if not _is_proto(cfg):
        return None
    if cfg.get("cw_no_diag_curv") is True:
        return NAME_CURV
    if cfg.get("cw_no_radius") is True:
        return NAME_MAGN
    if cfg.get("cw_no_radius") is False and cfg.get("cw_no_diag_curv") is False:
        return "PoLoRA"
    return None


def arm_key(cfg: dict) -> str | None:
    """Full comparison/ablation labeling for the hero (fig1) and the all-ablations
    basin figure (fig2): the incremental climb naive -> bare partner-Gram polar
    (w/o curvature + magnitude) -> +pin (w/o curvature) -> PoLoRA, plus the
    AdamW/iMuon references."""
    o = cfg.get("optimizer")
    if o == "adamw":
        return "AdamW"
    if o == "imuon-lora":
        return "iMuon"
    if o == "muon-lora" and cfg.get("polar_method") == "polar_express":
        return NAME_NAIVE
    if _is_proto(cfg):
        nc = cfg.get("cw_no_diag_curv") is True
        up = cfg.get("cw_unpinned") is True
        if not nc and not up:
            return "PoLoRA"
        if nc and not up:
            return NAME_CURV          # partner-Gram polar + pin, no curvature
        if nc and up:
            return NAME_LM            # bare partner-Gram polar: no pin, no curvature
    return None


def _hist_xy(hist):
    ev = sorted((e for e in hist if "eval_loss" in e and "step" in e),
                key=lambda e: e["step"])
    return [e["step"] for e in ev], [e["eval_loss"] for e in ev]


def _fmt_lr(lr: float) -> str:
    return f"{lr:g}"


def _cell_label(wl) -> str:
    return f"{wl.model_display} / {wl.dataset_display.split(' (')[0]} / r={wl.rank}"


def _paper_cells():
    """(workload, rows-by-variant, target) for every cell with PoLoRA + AdamW."""
    out = []
    for wl in iter_workloads():
        labeled = labeled_completed_runs(
            workload_runs(wl), paper_variant_key, horizon=wl.horizon)
        rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
        rows = {r["variant"]: r for r in rows}
        if "PoLoRA" in rows and "AdamW" in rows:
            out.append((wl, labeled, rows, target))
    out.sort(key=lambda c: -speedup_from_frac(c[2]["PoLoRA"]["frac_best_lr"]))
    return out


def _annotate_speedup(ax, cross, horizon, target, speed, color, drop_from):
    """Draw the step-speedup annotation on a loss-vs-step axis: the crossing dot, the
    drop-line down to PoLoRA's loss, the horizontal <-> arrow along the AdamW-final
    line, and the 'N x fewer steps' label.

    DURABLE RULE (no text overlap, no masking box): the label sits just above the
    <-> arrow, in the band between the AdamW target line and any higher curve. It is
    lifted to clear the tail that hugs the target line (AdamW), but NOT lifted above
    curves that run far above the band (e.g. iMuon) -- otherwise it floats up and
    reads as labeling that curve. So the scan ignores any curve more than `band` above
    the target. No bbox; never hand-place bare text or paper over an overlap."""
    ax.relim(); ax.autoscale_view()
    off = ax.get_ylim()[1] - ax.get_ylim()[0]
    ax.plot([cross], [target], marker="o", ms=5, color=color, zorder=6)
    ax.vlines(cross, ymin=drop_from, ymax=target, color=color, ls=(0, (2, 2)),
              lw=1.0, zorder=4)
    ax.annotate("", xy=(horizon, target + 0.02 * off),
                xytext=(cross, target + 0.02 * off),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0), zorder=5)
    # Clear only the near-target tail (AdamW); ignore curves > `band` above the target
    # (iMuon) so the label stays in the AdamW-to-iMuon gap, not up by iMuon.
    mid = (cross + horizon) / 2
    x0, x1 = ax.get_xlim(); half = 0.16 * (x1 - x0)
    band = 0.15 * off
    top = target + 0.02 * off                      # the arrow height
    for ln in ax.get_lines():
        ys = [y for x, y in zip(ln.get_xdata(), ln.get_ydata())
              if (mid - half) <= x <= (mid + half) and y == y and y <= target + band]
        if ys:
            top = max(top, max(ys))
    ax.text(mid, top + 0.03 * off, rf"${speed:.2f}\times$ fewer steps",
            ha="center", va="bottom", fontsize=9, zorder=7)


# ───────────────────────────── Figure 1: showcase ─────────────────────────────
def fig1():
    wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", 256)   # primary hero (PLAN.md)
    labeled = labeled_completed_runs(
        workload_runs(wl), arm_key, horizon=wl.horizon)
    rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
    rows = {r["variant"]: r for r in rows}

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    finals = {}
    # AdamW reference, the named spectral rival iMuon, raw Muon on the factors, and
    # ours. Muon ties AdamW but its dashed warm hue reads distinctly from the gray
    # AdamW line. (The bare partner-Gram polar -- which would overlap Muon -- stays in
    # the fig2 ablation, not here.)
    for v in ("AdamW", "iMuon", NAME_NAIVE, "PoLoRA"):
        if v not in rows:
            continue
        lr = rows[v]["best_lr"]
        xs, ys = _hist_xy(labeled[v][lr][1])
        ax.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                lw=STYLE[v].get("lw", 1.6), label=v)
        finals[v] = ys[-1]

    ax.axhline(target, color="#666666", ls=(0, (4, 3)), lw=1.0)
    frac = rows["PoLoRA"]["frac_best_lr"]
    cross = frac * wl.horizon
    speed = speedup_from_frac(frac)
    _annotate_speedup(ax, cross, wl.horizon, target, speed,
                      STYLE["PoLoRA"]["color"], finals["PoLoRA"])

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Loss")
    ax.set_xlim(0, 9300)
    ax.legend(frameon=False, loc="upper right" if "iMuon" in finals else "upper center",
              bbox_to_anchor=(0.52, 1.0) if "iMuon" not in finals else None)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_hero.pdf")
    fig.savefig(FIGS / "fig1_hero.png", dpi=150)
    print(f"── fig1: {_cell_label(wl)}  target {target:.4f}  crossing step {cross:.0f}"
          f"  speedup x{speed:.2f}  arms: {sorted(finals)}")
    return fig


def _model_short(disp: str) -> str:
    return disp.replace("Meta-Llama-3-8B", "Llama-3-8B")


def _data_short(disp: str) -> str:
    if "Math" in disp:
        return "Math"
    if "Bengali" in disp or "Aya" in disp:
        return "Bengali"
    if "opc" in disp or "Magicoder" in disp:
        return "Code"
    return disp.split(" (")[0]


def _x(s):
    return f"{s:.2f}$\\times$" if s == s else "---"


def _speedup_lookup():
    """(model_short, data_short, rank) -> (speedup_best_lr, speedup_lr_avg)."""
    out = {}
    print("── speedup table data ──")
    for wl, _labeled, rows, _target in _paper_cells():
        p = rows["PoLoRA"]
        key = (_model_short(wl.model_display), _data_short(wl.dataset_display), wl.rank)
        out[key] = (speedup_from_frac(p["frac_best_lr"]), speedup_from_frac(p["frac_lr_avg"]))
        print(f"  {key}  x{out[key][0]:.2f} / x{out[key][1]:.2f}")
        if "iMuon" in rows:
            m = rows["iMuon"]; sm = speedup_from_frac(m["frac_best_lr"])
            print(f"    iMuon @ {_cell_label(wl):38s} "
                  f"{('x%.2f' % sm) if sm == sm else '<1x (no crossing)'}")
    return out


def _write_tabular(path, header, rows):
    ncol = header.count("&") + 1
    spec = "l" + "r" * (ncol - 1)
    lines = [f"\\begin{{tabular}}{{{spec}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [r + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIGS / path).write_text("\n".join(lines) + "\n")


# ─────────────── Table 1: three focused speedup tables (not one mega-table) ───────────────
def table1():
    """Three focused speedup-to-AdamW tables, each making one point:
      tab_breadth.tex  — across model families (code, r=256): we win everywhere.
      tab_rank.tex     — rank ladder (Llama-3.2-1B, Math): larger speedup at higher rank.
      tab_ood.tex      — task pair (Qwen2.5-1.5B, r=256): larger speedup out-of-distribution.
    The rank/OOD axes are the two controlled "room to move" comparisons.
    """
    sp = _speedup_lookup()

    # Breadth: code @ r256 across model families, sorted by speedup (8B last = least room).
    breadth = sorted(
        [(m, sp[(m, "Code", 256)]) for m in
         ["OLMo-2-1B", "Qwen2.5-1.5B", "Llama-3.2-1B", "Llama-3-8B"] if (m, "Code", 256) in sp],
        key=lambda t: -t[1][0])
    _write_tabular(
        "tab_breadth.tex", r"model (code, $r{=}256$) & speedup",
        [f"{m} & {_x(b)}" for m, (b, _a) in breadth])

    # Rank ladder: Llama-3.2-1B Math, ascending rank.
    rank_rows = [(r, sp[("Llama-3.2-1B", "Math", r)]) for r in (32, 64, 128, 256)
                 if ("Llama-3.2-1B", "Math", r) in sp]
    _write_tabular(
        "tab_rank.tex", r"$r$ (Llama-3.2-1B, Math) & speedup",
        [f"{r} & {_x(b)}" for r, (b, _a) in rank_rows])

    # OOD pair: Qwen2.5-1.5B r256, in-distribution code vs out-of-distribution Bengali.
    ood = [("Code (in-dist.)", ("Qwen2.5-1.5B", "Code", 256)),
           ("Bengali (OOD)", ("Qwen2.5-1.5B", "Bengali", 256))]
    _write_tabular(
        "tab_ood.tex", r"corpus (Qwen2.5-1.5B, $r{=}256$) & speedup",
        [f"{label} & {_x(sp[k][0])}" for label, k in ood if k in sp])


# ─────────────────── Task-pair showcase: loss-vs-steps (OOD) ───────────────────
def fig_ood(figsize=(7.4, 3.3)):
    """Task-pair showcase (visual companion to tab_ood): loss-vs-steps for
    Qwen2.5-1.5B r256, AdamW vs PoLoRA, on in-distribution code and
    out-of-distribution Bengali. Same annotation as the hero (fig1): AdamW's
    final loss dashed, the interpolated crossing dotted, the step-speedup arrow
    printed. The OOD panel's gap is wider -> the speedup grows with room to move.
    Panels are NOT y-shared (the two corpora sit at different loss scales)."""
    specs = [("Qwen/Qwen2.5-1.5B", "opc", 256, "Code (in-distribution)"),
             ("Qwen/Qwen2.5-1.5B", "bengali", 256, "Bengali (out-of-distribution)")]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    print("── fig_ood (Qwen2.5-1.5B r256 task pair) ──")
    for ax, (model, data, rank, title) in zip(axes, specs):
        wl = find_workload(model, data, rank)
        labeled = labeled_completed_runs(
            workload_runs(wl), paper_variant_key, horizon=wl.horizon)
        rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
        rows = {r["variant"]: r for r in rows}
        finals = {}
        for v in ("AdamW", "PoLoRA"):
            if v not in rows:
                continue
            lr = rows[v]["best_lr"]
            xs, ys = _hist_xy(labeled[v][lr][1])
            ax.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                    lw=STYLE[v].get("lw", 1.6), label=v)
            finals[v] = ys[-1]
        ax.axhline(target, color="#666666", ls=(0, (4, 3)), lw=1.0)
        frac = rows["PoLoRA"]["frac_best_lr"]
        cross = frac * wl.horizon
        speed = speedup_from_frac(frac)
        _annotate_speedup(ax, cross, wl.horizon, target, speed,
                          STYLE["PoLoRA"]["color"], finals["PoLoRA"])
        ax.set_title(title)
        ax.set_xlabel("Training Step")
        ax.set_xlim(0, wl.horizon * 1.03)
        print(f"  {title:30s} target {target:.4f}  crossing step {cross:.0f}"
              f"  speedup x{speed:.2f}")
    axes[0].set_ylabel("Eval Loss")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_ood.pdf")
    fig.savefig(FIGS / "fig_ood.png", dpi=150)
    return fig


# ──────── Appendix: per-setting learning curves (the curves behind the tables) ────────
def _appendix_curves(specs, outname, figsize):
    """Loss-vs-step panels for settings the body reports only as a table speedup number.
    Same annotation as fig1/fig_ood (AdamW final dashed, interpolated crossing dotted,
    step-speedup arrow). Each figure is one logical axis (model breadth, or rank ladder)
    and EXCLUDES the settings already drawn as curves in the body -- the hero/ablation
    Llama-openmath-r256 (fig1/fig2) and the two Qwen cells (fig_ood) -- so the appendix
    adds the missing curves rather than redrawing ones the reader has already seen."""
    fig, axes = plt.subplots(1, len(specs), figsize=figsize)
    print(f"── {outname} ──")
    for ax, (model, data, rank, title) in zip(axes, specs):
        wl = find_workload(model, data, rank)
        labeled = labeled_completed_runs(
            workload_runs(wl), paper_variant_key, horizon=wl.horizon)
        rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
        rows = {r["variant"]: r for r in rows}
        finals = {}
        for v in ("AdamW", "PoLoRA"):
            if v not in rows:
                continue
            lr = rows[v]["best_lr"]
            xs, ys = _hist_xy(labeled[v][lr][1])
            ax.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                    lw=STYLE[v].get("lw", 1.6), label=v)
            finals[v] = ys[-1]
        ax.axhline(target, color="#666666", ls=(0, (4, 3)), lw=1.0)
        frac = rows["PoLoRA"]["frac_best_lr"]
        cross = frac * wl.horizon
        speed = speedup_from_frac(frac)
        _annotate_speedup(ax, cross, wl.horizon, target, speed,
                          STYLE["PoLoRA"]["color"], finals["PoLoRA"])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training Step")
        ax.set_xlim(0, wl.horizon * 1.03)
        print(f"  {title:16s} target {target:.4f}  crossing {cross:.0f}  x{speed:.2f}")
    axes[0].set_ylabel("Eval Loss")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / f"{outname}.pdf")
    fig.savefig(FIGS / f"{outname}.png", dpi=150)
    return fig


def figA_breadth(figsize=(9.8, 3.0)):
    """Model-breadth curves (code, r=256) behind tab_breadth -- the three model families
    NOT already shown as curves in the body (Qwen/code is in fig_ood)."""
    return _appendix_curves(
        [("allenai/OLMo-2-0425-1B", "opc", 256, "OLMo-2-1B"),
         ("meta-llama/Llama-3.2-1B", "opc", 256, "Llama-3.2-1B"),
         ("meta-llama/Meta-Llama-3-8B", "opc", 256, "Llama-3-8B")],
        "figA_breadth", figsize)


def figA_rank(figsize=(9.8, 3.0)):
    """Rank-ladder curves (Llama-3.2-1B, Math) behind tab_rank -- the three lower ranks
    NOT already shown as curves in the body (r=256 is the hero, fig1)."""
    return _appendix_curves(
        [("meta-llama/Llama-3.2-1B", "openmath", 32, "$r=32$"),
         ("meta-llama/Llama-3.2-1B", "openmath", 64, "$r=64$"),
         ("meta-llama/Llama-3.2-1B", "openmath", 128, "$r=128$")],
        "figA_rank", figsize)


# ─────────────────────────── Figs 2–3: lr basins ───────────────────────────
def _basin(ax, labeled, variants, star_minima=True, star_ms=12):
    for v in variants:
        by_lr = labeled.get(v)
        if not by_lr:
            continue
        lrs = sorted(by_lr)
        fls = [by_lr[lr][0] for lr in lrs]
        ax.plot(lrs, fls, label=v, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                marker=STYLE[v]["marker"], ms=4, lw=1.3)
        if star_minima:
            # mark the optimal lr with a same-color filled star (matches fig3 / the
            # transfer figure) -- never a ring/circle.
            b = min(by_lr, key=lambda lr: by_lr[lr][0])
            ax.plot([b], [by_lr[b][0]], "*", ms=star_ms, color=STYLE[v]["color"],
                    mec="white", mew=0.5, ls="none", zorder=5)
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)


def fig2():
    """Two-panel component ablation at the anchor (Llama openmath r256), subtractive
    from PoLoRA.
      (left)  loss curves for the three arms that separate as trajectories --
              -curvature+magnitude (bare partner-Gram polar) -> -curvature ->
              PoLoRA -- each at its best lr, AdamW's final loss as the dashed
              target (curve crossings = the speedups).
      (right) speedup-over-AdamW bars for ALL arms, including naive Muon, which ties
              AdamW (reads cleanly as a bar where it would overlap as a curve). Bar
              labels carry the numbers, so no separate table is needed.
    AdamW appears only as a reference (the target line / the 1.0x baseline), not as a
    competing trajectory (that is the hero's job)."""
    wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", 256)
    labeled = labeled_completed_runs(workload_runs(wl), arm_key, horizon=wl.horizon)
    rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
    rows = {r["variant"]: r for r in rows}
    labels = {NAME_NAIVE: "Muon", NAME_LM: "w/o curvature\n+ magnitude",
              NAME_CURV: "w/o curvature", "PoLoRA": "PoLoRA (ours)"}

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.4, 3.4), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ── left: loss curves of the three arms that separate (Muon ties AdamW and
    #    coincides with the bare partner-Gram polar -> moved to the appendix) ──
    print("── fig2 (left) ablation curves (openmath r256) ──")
    for v in (NAME_LM, NAME_CURV, "PoLoRA"):
        if v not in rows:
            continue
        lr = rows[v]["best_lr"]
        xs, ys = _hist_xy(labeled[v][lr][1])
        axL.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                 lw=STYLE[v].get("lw", 1.6), label=labels[v].replace("\n", " "))
        print(f"  {v:24s} best_lr {lr:g}  final {ys[-1]:.4f}")
    axL.axhline(target, color="#666666", ls=(0, (4, 3)), lw=1.0)
    axL.text(250, target, "AdamW", fontsize=8, va="bottom", ha="left", color="#666666")
    axL.set_xlabel("Training Step")
    axL.set_ylabel("Eval Loss")
    axL.set_xlim(0, 9300)
    axL.legend(frameon=False, fontsize=8.5, loc="upper right")
    axL.spines[["top", "right"]].set_visible(False)

    # ── right: speedup bars for the same three arms ──
    print("── fig2 (right) ablation speedup bars ──")
    arms = [NAME_LM, NAME_CURV, "PoLoRA"]   # ascending climb (bottom->top)
    for y, v in enumerate(arms):
        if v not in rows:
            continue
        s = speedup_from_frac(rows[v]["frac_best_lr"])
        crossed = s == s
        val = s if crossed else 1.0
        axR.barh(y, val, height=0.62, color=STYLE[v]["color"],
                 alpha=0.92 if crossed else 0.40,
                 hatch=None if crossed else "//", edgecolor=STYLE[v]["color"])
        axR.text(val + 0.015, y, f"{s:.2f}$\\times$" if crossed else "ties",
                 va="center", ha="left", fontsize=8.5)
        print(f"  {v:24s} speedup {('x%.2f' % s) if crossed else 'no crossing (ties)'}")
    axR.axvline(1.0, color="#666666", ls=(0, (4, 3)), lw=1.0)
    axR.text(1.0, len(arms) - 0.30, "AdamW", fontsize=8, ha="center", va="bottom",
             color="#666666")
    axR.set_yticks(range(len(arms)))
    axR.set_yticklabels([labels[v] for v in arms], fontsize=8.5)
    axR.set_xlabel("speedup over AdamW")
    axR.set_xlim(0, 1.9)
    axR.set_ylim(-0.6, len(arms) - 0.25)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig2_ablation.pdf")
    fig.savefig(FIGS / "fig2_ablation.png", dpi=150)
    return fig


def fig3(star_ms=11, figsize=(7.2, 3.0)):
    """lr basins on the openmath r>=32 ladder, two panels (PoLoRA | AdamW),
    one curve per rank (color = rank, reversed viridis), shared y windowed to the
    converged band. A star marks each curve's minimum, making the per-optimizer
    minimum-lr shift across rank directly readable: PoLoRA's holds at one lr,
    AdamW's drifts. Returns the figure (displays inline in a notebook); tweak
    `star_ms`/`figsize` from the cell, or edit here and re-run (autoreload)."""
    import numpy as np
    import matplotlib.cm as cm
    ranks = [32, 64, 128, 256]   # r16 excluded (flat/under-resolved basin top)
    rcol = {r: c for r, c in zip(ranks, cm.viridis_r(np.linspace(0.12, 0.92, len(ranks))))}
    arms = ["PoLoRA", "AdamW"]
    data = {a: {} for a in arms}                 # arm -> rank -> {lr: final_loss}
    allv = []
    print("── fig3 lr transfer (openmath r>=32 ladder) ──")
    for rank in ranks:
        wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", rank)
        lab = labeled_completed_runs(workload_runs(wl), paper_variant_key, horizon=wl.horizon)
        for a in arms:
            by_lr = lab.get(a, {})
            data[a][rank] = {lr: by_lr[lr][0] for lr in by_lr}
            allv += [by_lr[lr][0] for lr in by_lr]
            if by_lr:
                b = min(by_lr, key=lambda lr: by_lr[lr][0])
                print(f"  r{rank} {a:12s} best_lr {b:g}  final {by_lr[b][0]:.4f}")
    # robust y-window: keep the converged band readable; high-lr divergence exits the top
    med = float(np.median(allv)); conv = [v for v in allv if v < 3 * med]
    lo, hi = min(allv), max(conv); rng = hi - lo
    ylo, yhi = lo - 0.10 * rng, hi + 0.06 * rng
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, a in zip(axes, arms):
        for rank in ranks:
            d = data[a][rank]
            if not d:
                continue
            lrs = sorted(d)
            ax.plot(lrs, [d[lr] for lr in lrs], "o-", color=rcol[rank], ms=4, lw=1.3)
            b = min(d, key=lambda lr: d[lr])
            ax.plot([b], [d[b]], "*", ms=star_ms, color=rcol[rank], mec="white", mew=0.5, zorder=5)
        ax.set_xscale("log"); ax.set_xlabel("Learning Rate"); ax.set_title(a)
        ax.set_ylim(ylo, yhi)
    axes[0].set_ylabel("Eval Loss")
    handles = [plt.Line2D([], [], color=rcol[r], marker="o", lw=1.3, label=f"$r = {r}$") for r in ranks]
    # Legend outside the right panel: the r=32 divergence fills the in-panel space,
    # so place it in genuinely clear space rather than over any curve (no box).
    axes[1].legend(handles=handles, title="Rank", loc="center left",
                   bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_lr_transfer.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig3_lr_transfer.png", dpi=150, bbox_inches="tight")
    return fig


_GAUGE_GROUPS = {   # rank -> log group carrying the per-step op-norm gauge fields
    32:  "e1_kldiag_llama32_openmath_r16r32_bw",
    64:  "e1_kldiag_llama32_openmath_r64_bw",
    128: "e1_kldiag_llama32_openmath_r128_bw",
    256: "e1_kldiag_llama32_openmath_r256_bw",
}


def fig_gauge(figsize=(5.2, 3.4), lr_want=0.01):
    """Operator-norm ratio sigma_max(B)/sigma_max(A) over training for the
    protagonist (PoLoRA) at each rank on the Llama openmath ladder (best lr,
    0.01). The factors self-balance (ratio -> 1) and the balance tightens with rank.
    Reads the per-step gauge diagnostic (optim_step events) from the run logs."""
    import json
    import glob
    import numpy as np
    import matplotlib.cm as cm
    ranks = sorted(_GAUGE_GROUPS)
    rcol = {r: c for r, c in zip(ranks, cm.viridis_r(np.linspace(0.12, 0.92, len(ranks))))}

    def ratio_traj(group, rank):
        for f in sorted(glob.glob(str(ROOT / "logs" / group / "run_info" / "logs" / "log_*.out"))):
            lr = rk = None
            steps, ratio = [], []
            for line in open(f):
                if '"event": "config"' in line:
                    c = json.loads(line); lr = c.get("lr"); rk = c.get("lora_r")
                elif "sigma_max_A_median" in line and '"event": "optim_step"' in line:
                    e = json.loads(line)
                    a, b = e.get("sigma_max_A_median"), e.get("sigma_max_B_median")
                    if a and b:
                        steps.append(e["step"]); ratio.append(b / a)
            if lr is not None and abs(float(lr) - lr_want) < 1e-9 and rk == rank and steps:
                return np.array(steps), np.array(ratio)
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    print("── fig_gauge (op-norm ratio, Llama openmath, lr=%.3g) ──" % lr_want)
    for r in ranks:
        d = ratio_traj(_GAUGE_GROUPS[r], r)
        if d is None:
            print(f"  r{r}: NO lr={lr_want} gauge run in {_GAUGE_GROUPS[r]}"); continue
        ax.plot(d[0], d[1], color=rcol[r], lw=1.6, label=f"$r = {r}$")
        print(f"  r{r}: final ratio {d[1][-1]:.3f}")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(r"$\|B\|_2\,/\,\|A\|_2$")
    ax.legend(title="Rank", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_gauge.pdf")
    fig.savefig(FIGS / "fig_gauge.png", dpi=150)
    return fig


if __name__ == "__main__":
    fig1()
    table1()
    fig_ood()
    figA_breadth()
    figA_rank()
    fig2()
    fig3()
    fig_gauge()
    print(f"figures written to {FIGS}")
