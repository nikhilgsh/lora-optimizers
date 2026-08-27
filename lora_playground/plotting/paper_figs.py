"""Generate the paper's experiment figures + speedup table from the workload registry.

Design (locked with the user, conventions from Mousse/SPlus/LoRA+ in docs/papers/):
  fig1_hero.pdf          — Figure 1: loss-vs-steps showcase at Llama-3.2-1B openmath r256
                           (AdamW vs iMuon vs LoRA-RITE vs Muon vs PoLoRA), dashed line at
                           AdamW's final loss, drop-line at PoLoRA's interpolated crossing.
  tab_breadth.tex        — speedup-to-AdamW step tables: model breadth on code
  tab_breadth_math.tex     (tab_breadth) and math (tab_breadth_math), the Llama openmath
  tab_rank.tex             rank ladder (tab_rank), and the Qwen code-vs-Bengali pair
  tab_ood.tex              (tab_ood). The merged code+math tab:breadth in the manuscript
                           pairs these step numbers with bench-measured wall-clock.
  figA_curves.pdf        — appendix: ONE combined figure of per-setting loss curves behind
                           the table cells, three labelled rows -- (a) model breadth on code,
                           (b) on math, (c) the Llama math rank ladder -- each excluding the
                           cells already drawn as curves in the body.
  fig2_ablation.pdf      — E2 basins at Llama openmath r256: PoLoRA vs
                           "w/o curvature control" vs "w/o magnitude control"; ringed minima.
  fig3_lr_transfer.pdf   — E3 basins across the openmath rank ladder, shared log-x so
                           minima alignment across rank is comparable; diverged points
                           exit the frame top (no arrows).

Reviewed AdamW/PoLoRA comparisons flow through the checked-in publication archive,
the workload registry, and the records-native comparison core — the same sealed source
as docs/notes/leaderboard.md. Figures requiring external baselines not represented by
that archive retain their live-data compatibility path. PNG previews are written next
to each PDF.
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
    labeled_completed_runs, leaderboard_rows, leaderboard_rows_from_comparison,
    reach_fraction, speedup_from_frac,
)
from lora_playground.publication_paper import (
    labeled_completed,
    publication_view_panel,
    publication_workload_view_panel,
)
from lora_playground.workloads import find_workload, iter_workloads, workload_runs
from lora_playground.plotting import arms as _arms

FIGS = ROOT / "paper" / "manuscript" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    # Professional, paper-matching look (STIX ~ Times; unified text+math) at readable sizes.
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # Font sizes are the ON-PAGE point sizes: every figure is authored at its printed
    # width (\linewidth = 6.5in, or 0.6\linewidth = 3.9in for fig_gauge) so includegraphics
    # does NOT rescale it -- what you set here is what prints, sitting just below the 11pt
    # body. (fig2 pioneered this; the rest now follow. Do NOT author a figure wider than its
    # printed width or these sizes shrink by the downscale factor.)
    "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})

# Logical styling (Okabe-Ito, colorblind-safe):
#   ours        -> saturated blue, SOLID, thick (the protagonist stands out)
#   AdamW       -> neutral gray, SOLID, thin (the reference baseline)
#   Muon family -> distinct warm hues, DASHED (the spectral baselines: Muon, iMuon)
#   ablation    -> green/pink, SOLID thin (w/o curvature; w/o curvature or magnitude)
NAME_CURV = "w/o curvature"
NAME_MAGN = "w/o magnitude"
NAME_NAIVE = "Muon"            # raw per-factor polar = Muon on the factors (sec 3.1)
NAME_ONESIDED = "one-sided"      # precond=one-sided: C_B = C_A = I in direction AND estimator
NAME_FACTORWISE = "factorwise"   # precond=factorwise: C_B = P_A, C_A = Q_B
NAME_MSIGN_DIAG = "diagonal msign"  # msign=diag: rownorm(Z_A) / colnorm(Z_B)
# Both-controls-removed ablation arm = the decoupled update with identity metric and no
# spectral-norm magnitude rescale (iMuon's explicit decoupled update; the memoryless
# limit of LoRA-RITE; the compositional-Muon half-split). Named subtractively, NOT
# "LoRA-Muon step": (a) it reads as PoLoRA minus two controls, and (b) it does not
# over-credit the concurrent LoRA-Muon for an update that pre-dates it. The lineage
# citation lives in the fig2 caption.
NAME_LM = "w/o curvature or magnitude"
# The hero (fig1) stacks AdamW against three baselines that converge near AdamW's final
# loss. All curves are SOLID (SSO Fig 8); they separate by colour + weight: AdamW is the
# dark, thick reference (lw 2.0); the three baselines are THIN (lw 1.3) in clearly
# separated hues (gold / vermillion / purple); PoLoRA is the thickest blue. (The fig2
# ablation arms are SOLID too, separated by colour + weight -- they are not in fig1.)
STYLE = {
    "Adam":      dict(color="#333333", marker="o", ls="-",  lw=2.0),
    "PoLoRA": dict(color="#0072B2", marker="s", ls="-",  lw=2.4),
    NAME_NAIVE:   dict(color="#CB5A4C", marker="v", ls="-",  lw=1.7),  # muted brick (warm)
    "iMuon":      dict(color="#E0A33D", marker="^", ls="-",  lw=1.7),  # muted amber (warm)
    "LoRA-RITE":  dict(color="#8E6BAE", marker=">", ls="-",  lw=1.7),  # muted purple (cool) -- distinct from Muon
    NAME_LM:      dict(color="#CC79A7", marker="P", ls="-",  lw=1.6),
    NAME_CURV:    dict(color="#009E73", marker="D", ls="-",  lw=1.6),
    NAME_MAGN:    dict(color="#882255", marker="X", ls=":",  lw=1.6),
}


# ─── arm selection ───────────────────────────────────────────────────────────
# The predicates come from `arms.py`, which derives them from OptimizerConfig:
# `arm()` pins EVERY config field to its default and takes overrides only where an
# arm genuinely differs. The LABELS stay here, because what a curve is called in
# the paper is an editorial choice; only the "which runs are this arm" mechanism
# is shared.
#
# What this replaces: `_is_proto` plus three hand-written key functions, each an
# allowlist of the fields someone remembered. They checked 8 fields and never
# mentioned cw_solved_rho, curvature_beta, beta2, cw_no_radius or precond_method,
# so every sweep that set one of those was mislabeled. Measured on
# Llama-3.2-1B/openmath/r256 before this change: paper_variant_key raised
# LabelCollisionError with 4 collisions, ablation_variant_key with 6, arm_key
# with 6 — i.e. fig1/table1/fig2/fig3/figA_curves could not render.
#
# Order matters only as documentation: `arm()` pins cw_no_diag_curv, cw_no_radius,
# precond and msign on PROTO, so an ablation run cannot match the protagonist.
_PAPER_ARMS = {
    "Adam": _arms.ADAMW,
    "iMuon": _arms.IMUON,
    "PoLoRA": _arms.PROTO,
}

_ABLATION_ARMS = {
    NAME_MSIGN_DIAG: _arms.PROTO_DIAG,
    f"{NAME_ONESIDED} + {NAME_MSIGN_DIAG}": _arms.ONESIDED_DIAG,
    NAME_ONESIDED: _arms.ONESIDED,
    NAME_FACTORWISE: _arms.NOPRODUCT,
    NAME_CURV: _arms.NOSHAMPOO,
    NAME_MAGN: _arms.NAIVEMAG,
    "PoLoRA": _arms.PROTO,
}

_ARM_KEY_ARMS = {
    "Adam": _arms.ADAMW,
    "iMuon": _arms.IMUON,
    "LoRA-RITE": _arms.LORARITE,
    NAME_NAIVE: _arms.MUON,
    NAME_LM: _arms.DOUBLE,
    NAME_CURV: _arms.NOSHAMPOO,
    "PoLoRA": _arms.PROTO,
}

paper_variant_key = _arms.variant_key_fn({}, _PAPER_ARMS)
ablation_variant_key = _arms.variant_key_fn({}, _ABLATION_ARMS)
arm_key = _arms.variant_key_fn({}, _ARM_KEY_ARMS)


def _adam_polora_comparison(wl):
    panel = publication_workload_view_panel(
        "paper.adamw_polora.all_workloads.v1", wl
    )
    return panel, labeled_completed(panel)


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
        panel, labeled = _adam_polora_comparison(wl)
        if panel.comparison.best_completed[panel.variant_id("PoLoRA")] is None:
            continue
        rows, target = leaderboard_rows_from_comparison(
            panel.comparison,
            horizon=wl.horizon,
            baseline_id=panel.variant_id("Adam"),
        )
        rows = {r["variant"]: r for r in rows}
        if "PoLoRA" in rows and "Adam" in rows:
            out.append((wl, labeled, rows, target))
    out.sort(key=lambda c: -speedup_from_frac(c[2]["PoLoRA"]["frac_best_lr"]))
    return out


def _annotate_speedup(ax, cross, horizon, target, speed, color, drop_from=None,
                      proto_xy=None, show_speedup=True, box=True, fontsize=8.5, box_pad=0.3):
    """Mark the steps saved at the reference (Adam-final) loss, SPlus/SSO style: a dot
    where the protagonist first reaches that loss and a HORIZONTAL dashed line in the
    protagonist's colour from the crossing out to the horizon -- its length IS the steps
    saved. There is no full-width reference line; only this blue span is emphasized, so
    the eye reads the saved steps rather than the loss gap. With show_speedup, add a
    'N x fewer steps' callout beside the span (SSO Fig 8 convention). `box=True` draws it
    in the boxed style (tall body figures); `box=False` draws light coloured text with no
    box -- the same wording, sized for cramped multi-panel grids where a box reads oversized.

    drop_from / proto_xy are unused (kept for caller compatibility)."""
    ax.relim(); ax.autoscale_view()
    ax.plot([cross, horizon], [target, target], color=color, ls=(0, (2, 2)),
            lw=1.7, zorder=5)
    if show_speedup:
        # Anchor the callout in the clear region left of the crossing, but cap it at
        # 0.45*horizon so a LATE crossing does not drop it where the curve sits near the
        # reference loss (there the box would crowd the curve). Guarantee a pocket below the
        # reference so the box sits below the (higher) curve rather than being shoved up into
        # it. The final off-spine margin is applied post-layout by _place_speedup_boxes --
        # window extents are only correct once the layout (constrained_layout/subfigures) is
        # final, so the nudge cannot be done reliably here during panel construction.
        x_anchor = min(cross - 0.13 * horizon, 0.45 * horizon)
        ylo, yhi = ax.get_ylim()
        need = 0.20 * (yhi - ylo)
        if target - ylo < need:
            ax.set_ylim(bottom=target - need)
            ylo, yhi = ax.get_ylim()
        t = ax.text(x_anchor, target - 0.02 * (yhi - ylo),
                    rf"${speed:.2f}\times$ fewer steps",
                    ha="right", va="top", fontsize=fontsize, zorder=7,
                    color="black" if box else color,
                    bbox=(dict(boxstyle=f"round,pad={box_pad}", fc="white", ec=color, lw=0.8)
                          if box else None))
        t.set_gid("speedup_box")


def _place_speedup_boxes(fig, xfrac=0.035, yfrac=0.05):
    """Post-layout pass: push each speedup callout off the left/bottom spines by a margin so
    it never touches the frame. Run right before savefig -- axis positions (and hence the box
    window extents) are only final after the layout engine runs at draw time; nudging in data
    coordinates then keeps the margin stable across the savefig redraw."""
    fig.canvas.draw()
    for ax in fig.get_axes():
        for t in list(ax.texts):
            if t.get_gid() != "speedup_box":
                continue
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
            bb = t.get_window_extent()
            x0, y0 = ax.transData.inverted().transform((bb.x0, bb.y0))
            xm, ym = xfrac * (xmax - xmin), yfrac * (ymax - ymin)
            px, py = t.get_position()
            if x0 < xmin + xm:
                t.set_x(px + (xmin + xm - x0))
            if y0 < ymin + ym:
                t.set_y(py + (ymin + ym - y0))


# ───────────────────────────── Figure 1: showcase ─────────────────────────────
def _draw_hero(ax, show_speedup=True, legend=True):
    """Draw the hero trajectory panel (Llama openmath r256) onto `ax`. Returns the
    leaderboard summary so callers can print/compose; no figure creation or save."""
    wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", 256)   # primary hero (PLAN.md)
    labeled = labeled_completed_runs(
        workload_runs(wl), arm_key, horizon=wl.horizon)
    rows, target = leaderboard_rows(labeled, horizon=wl.horizon, baseline_label="Adam")
    rows = {r["variant"]: r for r in rows}

    finals = {}
    proto_xy = None
    # AdamW reference (dark, thick), the spectral rival iMuon, the adaptive baseline
    # LoRA-RITE, raw Muon on the factors (the three baselines thin solid in separated
    # hues so they stay legible where they converge near AdamW's final loss), and ours.
    # (The decoupled identity-metric/no-rescale arm -- which would overlap Muon -- stays
    # in the fig2 ablation, not here.)
    for v in ("Adam", "iMuon", "LoRA-RITE", NAME_NAIVE, "PoLoRA"):
        if v not in rows:
            continue
        lr = rows[v]["best_lr"]
        xs, ys = _hist_xy(labeled[v][lr][1])
        ax.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                lw=STYLE[v].get("lw", 1.6), label=v)
        finals[v] = ys[-1]
        if v == "PoLoRA":
            proto_xy = (xs, ys)

    frac = rows["PoLoRA"]["frac_best_lr"]
    cross = frac * wl.horizon
    speed = speedup_from_frac(frac)
    _annotate_speedup(ax, cross, wl.horizon, target, speed,
                      STYLE["PoLoRA"]["color"], finals["PoLoRA"], proto_xy=proto_xy,
                      show_speedup=show_speedup)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Loss")
    ax.set_xlim(0, 9300)
    if legend:
        ax.legend(frameon=False, loc="upper right" if "iMuon" in finals else "upper center",
                  bbox_to_anchor=(0.52, 1.0) if "iMuon" not in finals else None)
    return dict(wl=wl, target=target, cross=cross, speed=speed, finals=finals)


def fig1(show_speedup=True):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    info = _draw_hero(ax, show_speedup=show_speedup)
    fig.tight_layout()
    _place_speedup_boxes(fig)
    fig.savefig(FIGS / "fig1_hero.pdf")
    fig.savefig(FIGS / "fig1_hero.png", dpi=150)
    print(f"── fig1: {_cell_label(info['wl'])}  target {info['target']:.4f}"
          f"  crossing step {info['cross']:.0f}  speedup x{info['speed']:.2f}"
          f"  arms: {sorted(info['finals'])}")
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

    # Breadth: math @ r256 across the SAME model families -- the Math columns of the
    # merged tab:breadth in the manuscript (step speedup only; the wall-clock column
    # there divides each by the same per-(model,r256) overhead delta as the code column,
    # delta being dataset-independent at fixed tokens/step). Row order matches the code
    # table (code-descending), so the math values are read against their code partners.
    _write_tabular(
        "tab_breadth_math.tex", r"model (math, $r{=}256$) & speedup",
        [f"{m} & {_x(sp[(m, 'Math', 256)][0])}" for m, _ in breadth if (m, "Math", 256) in sp])

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
def fig_ood(figsize=(6.5, 2.9)):
    """Task-pair showcase (visual companion to tab_ood): loss-vs-steps for
    Qwen2.5-1.5B r256, AdamW vs PoLoRA, on code and Bengali. Same annotation
    as the hero (fig1): AdamW's final loss dashed, the interpolated crossing
    dotted, the step-speedup arrow printed. The Bengali panel's gap is wider
    -> the speedup grows with room to move.
    Panels are NOT y-shared (the two datasets sit at different loss scales)."""
    specs = [("Qwen/Qwen2.5-1.5B", "opc", 256, "Code"),
             ("Qwen/Qwen2.5-1.5B", "bengali", 256, "Bengali")]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    print("── fig_ood (Qwen2.5-1.5B r256 task pair) ──")
    for ax, (model, data, rank, title) in zip(axes, specs):
        wl = find_workload(model, data, rank)
        panel, labeled = _adam_polora_comparison(wl)
        rows, target = leaderboard_rows_from_comparison(
            panel.comparison,
            horizon=wl.horizon,
            baseline_id=panel.variant_id("Adam"),
        )
        rows = {r["variant"]: r for r in rows}
        finals = {}
        proto_xy = None
        for v in ("Adam", "PoLoRA"):
            if v not in rows:
                continue
            lr = rows[v]["best_lr"]
            xs, ys = _hist_xy(labeled[v][lr][1])
            ax.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                    lw=STYLE[v].get("lw", 1.6), label=v)
            finals[v] = ys[-1]
            if v == "PoLoRA":
                proto_xy = (xs, ys)
        frac = rows["PoLoRA"]["frac_best_lr"]
        cross = frac * wl.horizon
        speed = speedup_from_frac(frac)
        _annotate_speedup(ax, cross, wl.horizon, target, speed,
                          STYLE["PoLoRA"]["color"], finals["PoLoRA"], proto_xy=proto_xy)
        ax.set_title(title)
        ax.set_xlabel("Training Step")
        ax.set_xlim(0, wl.horizon * 1.03)
        print(f"  {title:30s} target {target:.4f}  crossing step {cross:.0f}"
              f"  speedup x{speed:.2f}")
    axes[0].set_ylabel("Eval Loss")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _place_speedup_boxes(fig)
    fig.savefig(FIGS / "fig_ood.pdf")
    fig.savefig(FIGS / "fig_ood.png", dpi=150)
    return fig


# ──────── Appendix: per-setting learning curves (the curves behind the tables) ────────
def _draw_loss_panel(ax, model, data, rank, title, xlabel=True, ylabel=False):
    """Adam vs PoLoRA loss-vs-step for one setting onto `ax`: both at their best lr, the
    dashed steps-saved span, and a small boxed 'N x fewer steps' callout. Returns
    (handles, labels) so the caller can build one shared legend."""
    wl = find_workload(model, data, rank)
    panel, labeled = _adam_polora_comparison(wl)
    rows, target = leaderboard_rows_from_comparison(
        panel.comparison,
        horizon=wl.horizon,
        baseline_id=panel.variant_id("Adam"),
    )
    rows = {r["variant"]: r for r in rows}
    finals = {}
    for v in ("Adam", "PoLoRA"):
        if v not in rows:
            continue
        lr = rows[v]["best_lr"]
        xs, ys = _hist_xy(labeled[v][lr][1])
        ax.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                lw=STYLE[v].get("lw", 1.6), label=v)
        finals[v] = ys[-1]
    frac = rows["PoLoRA"]["frac_best_lr"]
    cross = frac * wl.horizon
    speed = speedup_from_frac(frac)
    # Boxed "N x fewer steps" callout as in the body figures, scaled down so it stays
    # proportional in these narrow grid panels; sits in the clear pocket below-left of
    # the crossing.
    _annotate_speedup(ax, cross, wl.horizon, target, speed,
                      STYLE["PoLoRA"]["color"], finals["PoLoRA"], fontsize=7, box_pad=0.2)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel("Training Step")
    if ylabel:
        ax.set_ylabel("Eval Loss")
    ax.set_xlim(0, wl.horizon * 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    print(f"  {title:16s} target {target:.4f}  crossing {cross:.0f}  x{speed:.2f}")
    return ax.get_legend_handles_labels()


def figA_curves(figsize=(6.5, 7.0)):
    """Combined appendix learning curves in ONE figure so the section stays cohesive and
    tiles a page instead of three tall floats scattering across pages. Three labelled rows,
    each Adam vs PoLoRA at every setting's best lr, excluding settings already drawn in the
    body (fig1/fig2/fig_ood): (a) model families on code, (b) on math, (c) the rank ladder."""
    rows = [
        ("(a) Model families — code ($r=256$)",
         [("allenai/OLMo-2-0425-1B", "opc", 256, "OLMo-2-1B"),
          ("meta-llama/Llama-3.2-1B", "opc", 256, "Llama-3.2-1B"),
          ("meta-llama/Meta-Llama-3-8B", "opc", 256, "Llama-3-8B")]),
        ("(b) Model families — math ($r=256$)",
         [("allenai/OLMo-2-0425-1B", "openmath", 256, "OLMo-2-1B"),
          ("Qwen/Qwen2.5-1.5B", "openmath", 256, "Qwen2.5-1.5B"),
          ("meta-llama/Meta-Llama-3-8B", "openmath", 256, "Llama-3-8B")]),
        ("(c) Rank ladder — Llama-3.2-1B, math",
         [("meta-llama/Llama-3.2-1B", "openmath", 32, "$r=32$"),
          ("meta-llama/Llama-3.2-1B", "openmath", 64, "$r=64$"),
          ("meta-llama/Llama-3.2-1B", "openmath", 128, "$r=128$")]),
    ]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(len(rows), 1)
    print("── figA_curves (combined appendix learning curves) ──")
    handles = labels = None
    for i, (sf, (rowtitle, specs)) in enumerate(zip(subfigs, rows)):
        axes = sf.subplots(1, 3)
        last = (i == len(rows) - 1)
        for j, (ax, (model, data, rank, title)) in enumerate(zip(axes, specs)):
            h, l = _draw_loss_panel(ax, model, data, rank, title,
                                    xlabel=last, ylabel=(j == 0))
            if handles is None:
                handles, labels = h, l
        sf.suptitle(rowtitle, fontsize=9.5, fontweight="bold")
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    _place_speedup_boxes(fig)
    fig.savefig(FIGS / "figA_curves.pdf")
    fig.savefig(FIGS / "figA_curves.png", dpi=150)
    return fig


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
              -curvature+magnitude (identity metric, no magnitude rescale) ->
              -curvature -> PoLoRA -- each at its best lr, AdamW's final loss as the dashed
              target (curve crossings = the speedups).
      (right) speedup-over-AdamW bars for the same three arms. Bar labels carry the
              numbers, so no separate table is needed.
    AdamW appears only as a reference (the target line / the 1.0x baseline), not as a
    competing trajectory (that is the hero's job)."""
    panel = publication_view_panel("paper.fig2_ablation.v1")
    labeled = labeled_completed(panel)
    rows, target = leaderboard_rows_from_comparison(
        panel.comparison,
        horizon=panel.horizon,
        baseline_id=panel.target_id,
    )
    rows = {r["variant"]: r for r in rows}
    labels = {NAME_NAIVE: "Muon", NAME_LM: "w/o curvature\nor magnitude",
              NAME_CURV: "w/o curvature", "PoLoRA": "PoLoRA (ours)"}

    # Native width == printed \linewidth (6.5in, letterpaper 1in margins) so fonts
    # render at their true pt instead of being shrunk ~0.7x by includegraphics.
    LAB, TICK, LEG, ANN = 9.5, 8.5, 8.5, 8.0
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # ── left: loss curves of the three arms that separate (Muon ties AdamW and
    #    coincides with the identity-metric/no-rescale arm -> moved to the appendix) ──
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
    axL.text(250, target, "Adam", fontsize=ANN, va="bottom", ha="left", color="#666666")
    axL.set_xlabel("Training Step", fontsize=LAB)
    axL.set_ylabel("Eval Loss", fontsize=LAB)
    axL.set_xlim(0, 9300)
    axL.tick_params(labelsize=TICK)
    axL.legend(frameon=False, fontsize=LEG, loc="upper right")
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
        axR.text(val + 0.03, y, f"{s:.2f}$\\times$" if crossed else "ties",
                 va="center", ha="left", fontsize=ANN)
        print(f"  {v:24s} speedup {('x%.2f' % s) if crossed else 'no crossing (ties)'}")
    axR.axvline(1.0, color="#666666", ls=(0, (4, 3)), lw=1.0)
    axR.text(1.0, len(arms) - 0.30, "Adam", fontsize=ANN, ha="center", va="bottom",
             color="#666666")
    axR.set_yticks(range(len(arms)))
    axR.set_yticklabels([labels[v] for v in arms], fontsize=TICK)
    axR.set_xlabel("speedup over Adam", fontsize=LAB)
    axR.tick_params(axis="x", labelsize=TICK)
    axR.set_xlim(0, 1.9)
    axR.set_ylim(-0.6, len(arms) - 0.25)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGS / "fig2_ablation.pdf")
    fig.savefig(FIGS / "fig2_ablation.png", dpi=150)
    return fig


def fig3(
    star_ms=11,
    figsize=(6.5, 2.6),
    ranks=(32, 64, 128, 256),
):
    """LR basins on a requested openmath rank ladder (PoLoRA | AdamW),
    one curve per rank (color = rank, reversed viridis), shared y windowed to the
    converged band. A star marks each curve's minimum, making the per-optimizer
    minimum-lr shift across rank directly readable: PoLoRA's holds at one lr,
    AdamW's drifts. Returns the figure (displays inline in a notebook); tweak
    `star_ms`/`figsize` from the cell, or edit here and re-run (autoreload)."""
    import numpy as np
    import matplotlib.cm as cm
    ranks = tuple(ranks)
    if not ranks:
        raise ValueError("fig3 requires at least one rank")
    rcol = {r: c for r, c in zip(ranks, cm.viridis_r(np.linspace(0.12, 0.92, len(ranks))))}
    arms = ["PoLoRA", "Adam"]
    data = {a: {} for a in arms}                 # arm -> rank -> {lr: final_loss}
    allv = []
    print("── fig3 lr transfer (openmath rank ladder) ──")
    for rank in ranks:
        wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", rank)
        _panel, lab = _adam_polora_comparison(wl)
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
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)
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
    fig.savefig(FIGS / "fig3_lr_transfer.pdf")
    fig.savefig(FIGS / "fig3_lr_transfer.png", dpi=150)
    return fig


def fig_lr_tuning(figsize=(5.8, 3.3)):
    """LR-tuning provenance for the hero cell (Llama-3.2-1B openmath r256): final
    eval loss at each optimizer's tuned learning rate and its two grid neighbours
    (one ~3x step below and above the optimum). The x-axis is normalized to each
    optimizer's own optimum, so the basins overlay on a unified, symmetric grid and
    every reported number reads as a tuned interior optimum. The window stops one
    grid step each side: two steps above the optimum the LR-sensitive baselines
    (AdamW, Muon) diverge, which would break the converged y-window. The absolute
    optima differ by ~330x across optimizers, which is why the axis is normalized to
    each optimizer's optimum. Line weights/colours/styles follow the fig1 convention;
    one marker for all series (the colour, not the marker, identifies the optimizer)."""
    fig, ax = plt.subplots(figsize=figsize)
    _draw_lr_tuning(ax, legend=True)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_lr_tuning.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_lr_tuning.png", dpi=150, bbox_inches="tight")
    return fig


def _draw_lr_tuning(ax, legend=True):
    """Draw the LR-tuning basin panel (Llama openmath r256) onto `ax`; no figure
    creation or save. `legend=False` for composed figures that carry one shared key."""
    wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", 256)
    labeled = labeled_completed_runs(workload_runs(wl), arm_key, horizon=wl.horizon)
    order = ["Adam", NAME_NAIVE, "iMuon", "LoRA-RITE", "PoLoRA"]
    name = {NAME_NAIVE: "Muon"}          # other keys are already the display label
    xpos = [1 / 3, 1.0, 3.0]             # optimum and its two neighbours, shared by all

    print("── fig_lr_tuning (Llama openmath r256, optimum and +/-1 grid step) ──")
    vals = []
    for v in order:
        by_lr = labeled.get(v)
        if not by_lr:
            print(f"  {v:12s}: NONE"); continue
        lrs = sorted(by_lr)
        i = min(range(len(lrs)), key=lambda k: by_lr[lrs[k]][0])
        if i == 0 or i == len(lrs) - 1:
            print(f"  {v:12s}: optimum at grid edge -- skipped"); continue
        tri = [by_lr[lrs[i - 1]][0], by_lr[lrs[i]][0], by_lr[lrs[i + 1]][0]]
        vals += tri
        st = STYLE[v]
        ax.plot(xpos, tri, color=st["color"], ls=st["ls"], marker="o", ms=4.5,
                lw=st.get("lw", 1.6), label=name.get(v, v))
        print(f"  {v:12s}: lr* {lrs[i]:g}  floor {tri[1]:.4f}")

    lo, hi = min(vals), max(vals); pad = 0.10 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    ax.set_xlim(xpos[0] / 1.3, xpos[-1] * 1.3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["1/3", "1", "3"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Learning rate (× best)")
    ax.set_ylabel("Final Eval Loss")
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def fig_hero_with_tuning(figsize=(6.5, 2.7)):
    """The 2-panel Figure 1: loss-vs-step trajectory (left) + LR-tuning basin (right),
    one shared legend on the left panel (both panels share the 5 colours). Authored at
    the printed \\linewidth (6.5in) so its fonts render at true point size."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    info = _draw_hero(axL, legend=True)
    _draw_lr_tuning(axR, legend=False)
    print(f"── fig_hero_with_tuning: speedup x{info['speed']:.2f}")
    fig.tight_layout()
    _place_speedup_boxes(fig)
    fig.savefig(FIGS / "fig_hero_with_tuning.pdf")
    fig.savefig(FIGS / "fig_hero_with_tuning.png", dpi=150)
    return fig


_GAUGE_GROUPS = {   # rank -> log group carrying the per-step op-norm gauge fields
    32:  "e1_kldiag_llama32_openmath_r16r32_bw",
    64:  "e1_kldiag_llama32_openmath_r64_bw",
    128: "e1_kldiag_llama32_openmath_r128_bw",
    256: "e1_kldiag_llama32_openmath_r256_bw",
}


def fig_gauge(figsize=(3.9, 2.7), lr_want=0.01):
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
    ax.legend(title="Rank", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_gauge.pdf")
    fig.savefig(FIGS / "fig_gauge.png", dpi=150)
    return fig


if __name__ == "__main__":
    fig1()
    table1()
    fig_ood()
    figA_curves()
    fig2()
    fig3()
    fig_gauge()
    print(f"figures written to {FIGS}")
