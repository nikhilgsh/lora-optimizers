"""Generate the paper's experiment figures + speedup table from the workload registry.

Design (locked with the user, conventions from Mousse/SPlus/LoRA+ in docs/papers/):
  fig1_hero.pdf          — Figure 1: loss-vs-steps showcase at OLMo opc r256 (AdamW vs
                           Polar-LoRA vs iMuon when present), dashed line at AdamW's final
                           loss, drop-line at the interpolated crossing, speedup printed.
  tab1_speedup.tex       — Table 1: per-cell speedup-to-AdamW (best-lr + lr-avg), iMuon
                           rows where run. SPlus-style numeric companion to fig 1.
  figA_tailzoom.pdf      — appendix: 2x4 tail-zoom grid, one panel per cell, crossing
                           fraction printed per panel (Mousse Fig-4 pattern).
  fig2_ablation.pdf      — E2 basins at Llama openmath r256: Polar-LoRA vs
                           "w/o curvature control" vs "w/o magnitude control"; ringed minima.
  fig3_lr_transfer.pdf   — E3 basins across the openmath rank ladder, shared log-x so
                           minima alignment across rank is comparable; diverged points
                           exit the frame top (no arrows).

All numbers flow through lora_playground.{workloads,leaderboard} — same source as
docs/leaderboard.md. PNG previews written next to each PDF.
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

FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True)

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

# Protagonist saturated, baselines gray/desaturated (SPlus convention),
# Okabe-Ito hues (colorblind-safe).
NAME_CURV = "w/o curvature control"
NAME_MAGN = "w/o magnitude control"
STYLE = {
    "AdamW":      dict(color="#666666", marker="o", ls="-"),
    "Polar-LoRA": dict(color="#0072B2", marker="s", ls="-"),
    "iMuon":      dict(color="#E69F00", marker="^", ls="--"),
    NAME_CURV:    dict(color="#D55E00", marker="v", ls="--"),
    NAME_MAGN:    dict(color="#009E73", marker="D", ls=":"),
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
        return "Polar-LoRA"
    return None


def ablation_variant_key(cfg: dict) -> str | None:
    if not _is_proto(cfg):
        return None
    if cfg.get("cw_no_diag_curv") is True:
        return NAME_CURV
    if cfg.get("cw_no_radius") is True:
        return NAME_MAGN
    if cfg.get("cw_no_radius") is False and cfg.get("cw_no_diag_curv") is False:
        return "Polar-LoRA"
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
    """(workload, rows-by-variant, target) for every cell with Polar-LoRA + AdamW."""
    out = []
    for wl in iter_workloads():
        labeled = labeled_completed_runs(
            workload_runs(wl), paper_variant_key, horizon=wl.horizon)
        rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
        rows = {r["variant"]: r for r in rows}
        if "Polar-LoRA" in rows and "AdamW" in rows:
            out.append((wl, labeled, rows, target))
    out.sort(key=lambda c: -speedup_from_frac(c[2]["Polar-LoRA"]["frac_best_lr"]))
    return out


# ───────────────────────────── Figure 1: showcase ─────────────────────────────
def fig1():
    wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", 256)   # primary hero (PLAN.md)
    labeled = labeled_completed_runs(
        workload_runs(wl), paper_variant_key, horizon=wl.horizon)
    rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
    rows = {r["variant"]: r for r in rows}

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    finals = {}
    for v in ("AdamW", "iMuon", "Polar-LoRA"):
        if v not in rows:
            continue
        lr = rows[v]["best_lr"]
        xs, ys = _hist_xy(labeled[v][lr][1])
        ax.plot(xs, ys, color=STYLE[v]["color"], ls=STYLE[v]["ls"], lw=1.6,
                label=v)
        finals[v] = ys[-1]

    ax.axhline(target, color="#666666", ls=(0, (4, 3)), lw=1.0)
    frac = rows["Polar-LoRA"]["frac_best_lr"]
    cross = frac * wl.horizon
    speed = speedup_from_frac(frac)
    ax.plot([cross], [target], marker="o", ms=5, color=STYLE["Polar-LoRA"]["color"])
    ax.vlines(cross, ymin=finals["Polar-LoRA"], ymax=target,
              color=STYLE["Polar-LoRA"]["color"], ls=(0, (2, 2)), lw=1.0)
    ax.annotate(
        "", xy=(wl.horizon, target + 0.004), xytext=(cross, target + 0.004),
        arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.0))
    ax.text((cross + wl.horizon) / 2, target + 0.013,
            rf"${speed:.2f}\times$ fewer steps",
            ha="center", va="bottom", fontsize=9)

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


# ───────────────────────────── Table 1: speedups ─────────────────────────────
def table1():
    lines = [
        r"\begin{tabular}{l r r r r r}",
        r"\toprule",
        r" & \multicolumn{2}{c}{AdamW} & \multicolumn{3}{c}{\methodname} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"model / dataset / rank & lr & final & lr & speedup & speedup (lr-avg) \\",
        r"\midrule",
    ]
    imuon_lines = []
    print("── table1 rows ──")
    for wl, _labeled, rows, target in _paper_cells():
        a, p = rows["AdamW"], rows["Polar-LoRA"]
        s_best = speedup_from_frac(p["frac_best_lr"])
        s_avg = speedup_from_frac(p["frac_lr_avg"])
        savg = f"{s_avg:.2f}$\\times$" if s_avg == s_avg else "---"
        lines.append(
            f"{_cell_label(wl)} & {_fmt_lr(a['best_lr'])} & {target:.4f} & "
            f"{_fmt_lr(p['best_lr'])} & {s_best:.2f}$\\times$ & {savg} \\\\")
        print(f"  {_cell_label(wl):42s} x{s_best:.2f} / "
              f"{'x%.2f' % s_avg if s_avg == s_avg else '—'}")
        if "iMuon" in rows:
            m = rows["iMuon"]
            sm = speedup_from_frac(m["frac_best_lr"])
            sms = f"{sm:.2f}$\\times$" if sm == sm else "never reaches target"
            imuon_lines.append(
                f"iMuon @ {_cell_label(wl)} & & & {_fmt_lr(m['best_lr'])} & {sms} & --- \\\\")
    if imuon_lines:
        lines += [r"\midrule"] + imuon_lines
    lines += [r"\bottomrule", r"\end{tabular}"]
    (FIGS / "tab1_speedup.tex").write_text("\n".join(lines) + "\n")


# ──────────────────────── Appendix: tail-zoom grid ────────────────────────
def figA():
    cells = _paper_cells()
    n = len(cells)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow))
    axes = axes.ravel()
    for ax, (wl, labeled, rows, target) in zip(axes, cells):
        for v in ("AdamW", "Polar-LoRA"):
            lr = rows[v]["best_lr"]
            xs, ys = _hist_xy(labeled[v][lr][1])
            pts = [(x, y) for x, y in zip(xs, ys) if x >= 3000]
            ax.plot([x for x, _ in pts], [y for _, y in pts],
                    color=STYLE[v]["color"], ls=STYLE[v]["ls"], lw=1.3)
        ax.axhline(target, color="#666666", ls=(0, (4, 3)), lw=0.8)
        frac = rows["Polar-LoRA"]["frac_best_lr"]
        cross = frac * wl.horizon
        lo = rows["Polar-LoRA"]["final_at_best"]
        ax.vlines(cross, ymin=lo, ymax=target,
                  color=STYLE["Polar-LoRA"]["color"], ls=(0, (2, 2)), lw=0.9)
        ax.text(cross, lo, f" {speedup_from_frac(frac):.2f}x", fontsize=7, va="bottom",
                ha="left", color=STYLE["Polar-LoRA"]["color"])
        ax.set_title(_cell_label(wl), fontsize=7.5)
        ax.set_ylim(lo - 0.004, target + 0.035)
        ax.set_xlim(3000, 9200)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=STYLE[v]["color"], ls=STYLE[v]["ls"], label=v)
               for v in ("AdamW", "Polar-LoRA")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.supxlabel("training step", fontsize=9)
    fig.supylabel("eval loss", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.04, 1, 1))
    fig.savefig(FIGS / "figA_tailzoom.pdf")
    fig.savefig(FIGS / "figA_tailzoom.png", dpi=150)
    return fig


# ─────────────────────────── Figs 2–3: lr basins ───────────────────────────
def _basin(ax, labeled, variants, ring_minima=True):
    for v in variants:
        by_lr = labeled.get(v)
        if not by_lr:
            continue
        lrs = sorted(by_lr)
        fls = [by_lr[lr][0] for lr in lrs]
        ax.plot(lrs, fls, label=v, color=STYLE[v]["color"], ls=STYLE[v]["ls"],
                marker=STYLE[v]["marker"], ms=4, lw=1.3)
        if ring_minima:
            b = min(by_lr, key=lambda lr: by_lr[lr][0])
            ax.plot([b], [by_lr[b][0]], marker="o", ms=10, mfc="none",
                    mec=STYLE[v]["color"], ls="none", mew=1.5)
    ax.set_xscale("log")
    ax.spines[["top", "right"]].set_visible(False)


def fig2():
    wl = find_workload("meta-llama/Llama-3.2-1B", "openmath", 256)
    labeled = labeled_completed_runs(
        workload_runs(wl), ablation_variant_key, horizon=wl.horizon)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    _basin(ax, labeled, ["Polar-LoRA", NAME_CURV, NAME_MAGN])
    best = min(fl for d in labeled.values() for fl, _ in d.values())
    ax.axhspan(best, best + wl.sigma_ref, color="gray", alpha=0.18, lw=0)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("final eval loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_ablation.pdf")
    fig.savefig(FIGS / "fig2_ablation.png", dpi=150)
    print("── fig2 ablation (openmath r256, sigma=%.4f) ──" % wl.sigma_ref)
    for v, by_lr in sorted(labeled.items()):
        b = min(by_lr, key=lambda lr: by_lr[lr][0])
        print(f"  {v:24s} best_lr {b:g}  final {by_lr[b][0]:.4f}")
    return fig


def fig3(star_ms=11, figsize=(7.2, 3.0)):
    """lr basins on the openmath r>=32 ladder, two panels (Polar-LoRA | AdamW),
    one curve per rank (color = rank, reversed viridis), shared y windowed to the
    converged band. A star marks each curve's minimum, making the per-optimizer
    minimum-lr shift across rank directly readable: Polar-LoRA's holds at one lr,
    AdamW's drifts. Returns the figure (displays inline in a notebook); tweak
    `star_ms`/`figsize` from the cell, or edit here and re-run (autoreload)."""
    import numpy as np
    import matplotlib.cm as cm
    ranks = [32, 64, 128, 256]   # r16 excluded (flat/under-resolved basin top)
    rcol = {r: c for r, c in zip(ranks, cm.viridis_r(np.linspace(0.12, 0.92, len(ranks))))}
    arms = ["Polar-LoRA", "AdamW"]
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
    axes[1].legend(handles=handles, title="Rank", frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_lr_transfer.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig3_lr_transfer.png", dpi=150, bbox_inches="tight")
    return fig


def _gauge_traj(group):
    """Per-pair gauge stats (sigma_ratio, balance_resid: median/min/max over the 112 LoRA
    pairs = all layers) vs step, read from a diagnostic run's optim_step events (load_runs
    returns eval events, not per-step diagnostics)."""
    import glob
    import json
    import numpy as np
    for f in sorted(glob.glob(str(ROOT / "logs" / group / "run_info" / "logs" / "log_*.out"))):
        rows = []
        for line in open(f):
            if '"balance_resid_median"' in line:
                try:
                    e = json.loads(line)
                    if e.get("event") == "optim_step":
                        rows.append(e)
                except Exception:
                    pass
        if rows:
            g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
            return dict(step=g("step"),
                        SRm=g("sigma_ratio_median"), SRlo=g("sigma_ratio_min"), SRhi=g("sigma_ratio_max"),
                        BRm=g("balance_resid_median"), BRlo=g("balance_resid_min"), BRhi=g("balance_resid_max"))
    return None


def fig_gauge(proto_group="gauge_kldiag_r256_bw", muon_group="gauge_loramuon_r256_bw",
              figsize=(9.6, 3.6)):
    """Gauge balance during training (Llama-3.2-1B / OpenMathInstruct r256, per-pair over
    the 112 LoRA pairs = all layers). (A) radius gauge coordinate sigma_max(A)/sigma_max(B):
    our run self-balances toward 1. (B) normalized factor-Gram gap ||AAt - BtB|| / max(.):
    how far the factors sit from AAt = BtB. Reads the diagnostic runs' logs; returns the figure."""
    arms = [("Polar-LoRA (ours)", proto_group, "#2166ac"),
            ("LoRA-Muon step", muon_group, "#d6604d")]
    data = [(lab, _gauge_traj(grp), c) for lab, grp, c in arms]
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].axhline(1.0, color="gray", ls="--", lw=1)
    for lab, D, c in data:
        if D is None:
            continue
        # Plot ||B||_2/||A||_2 (well-behaved: 0 at B=0 init -> 1 balanced); the field logs
        # ||A||_2/||B||_2 per pair, so invert (median & band endpoints commute with 1/x).
        ax[0].fill_between(D["step"], 1.0 / D["SRhi"], 1.0 / D["SRlo"], color=c, alpha=0.15)
        ax[0].plot(D["step"], 1.0 / D["SRm"], color=c, lw=2, label=lab)
        ax[1].fill_between(D["step"], D["BRlo"], D["BRhi"], color=c, alpha=0.13)
        ax[1].plot(D["step"], D["BRm"], color=c, lw=2, label=lab)
    ax[0].set_xlabel("Step"); ax[0].set_ylabel(r"$\|B\|_2 / \|A\|_2$")
    ax[0].set_title("Operator Norm")
    ax[0].legend(frameon=False)
    ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("Step")
    ax[1].set_ylabel(r"$\|AA^\top - B^\top B\|_F$ (normalized)")
    ax[1].set_title("Frobenius Norm")
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_gauge.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_gauge.png", dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    fig1()
    table1()
    figA()
    fig2()
    fig3()
    fig_gauge()
    print(f"figures written to {FIGS}")
