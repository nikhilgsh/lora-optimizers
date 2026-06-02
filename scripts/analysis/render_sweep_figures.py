"""Render every sweep figure from notebooks/leaderboard_old/sweep_analysis.ipynb to PNG.

Drives the same plot_utils library code path the notebook uses, but headless
and reproducible — used to iterate on figure styling without re-executing the
notebook each pass.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent

from lora_playground.plotting import (
    OPTIM_COLORS, max_loss, merge_runs, parse_flag, load_sweep,
    plot_leaderboard_by_rank, standard_sweep_figure,
)

OUT_DIR = ROOT / "notebooks" / "_render"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_ROOT = str(ROOT / "logs")

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

# ─── Section-specific palettes ────────────────────────────────────────────────
M_COLORS = {1: plt.cm.plasma(0.15), 4: plt.cm.plasma(0.50), 16: plt.cm.plasma(0.85)}
MODE_COLORS = {"svd_step_oracle": "#ff7f0e", "svd_cumulative_oracle": "#2ca02c"}
MUON_VARIANT_COLORS = {
    "muon":              "#e377c2",
    "muon+m=4":          "#d62728",
    "muon+m=16":         "#8c2d04",
    "muon+ns=0":         "#bababa",
    "product-muon":      "#1f77b4",
    "product-muon+m=4":  "#0d3d66",
    "adam-muon":         "#2ca02c",
    "adam-muon+m=4":     "#0d4f0d",
}


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def _load_all_runs():
    SOURCE_GROUPS = (
        "psi_lora_2k", "muon_lowlr_2k",
        "boundary_extend3_2k", "boundary_extend2_2k", "boundary_extend_2k",
        "new_optimizers_high_eta_2k", "galore_fixed_2k",
        "optim_compare_high_eta_2k", "lr_sweep_2k",
        "new_optimizers_2k", "galore_2k",
    )
    RENAME_ALIAS = {"psi-lora": "diag-scaled-lora", "kfac-lora": "kron-grad-lora"}
    NEW_PSI_GROUPS = {"psi_lora_2k"}

    def _alias(cfg, group):
        if group not in NEW_PSI_GROUPS:
            cfg["optimizer"] = RENAME_ALIAS.get(cfg["optimizer"], cfg["optimizer"])

    return merge_runs(
        SOURCE_GROUPS,
        key_fn=lambda c: (c["optimizer"], float(c["lr"])),
        filter_fn=lambda c: c["optimizer"] in OPTIM_COLORS,
        cfg_postprocess=_alias,
        logs_root=LOGS_ROOT,
    )


def _load_ext_runs():
    EXT_SOURCES = (
        "polar_product_eta1e3_2k", "polar_product_2k",
        "h4_post_rmsalign_2k", "h4_post_2k",
        "h5_matrix_r64_2k", "h5_matrix_2k",
        "r64_eta_sweep_extension_2k", "h3_rsweep_2k",
        "optim_compare_high_eta_2k", "lr_sweep_2k",
    )
    EXT_OPTIMIZERS = {
        "adamw", "adam-lin-lora", "adam-scaled-lora",
        "adam-lin-lora-post", "adam-scaled-lora-post",
        "adam-lin-lora-matrix", "adam-scaled-lora-matrix",
        "polar-product-lora", "adam-polar-product-lora",
    }
    return merge_runs(
        EXT_SOURCES,
        key_fn=lambda c: (c["optimizer"], float(c["lr"]), int(c.get("lora_r", 16))),
        filter_fn=lambda c: c["optimizer"] in EXT_OPTIMIZERS,
        cfg_postprocess=lambda c, g: c.update(_lora_r=int(c.get("lora_r", 16))),
        logs_root=LOGS_ROOT,
    )


def _load_muon_runs():
    def _stamp(cfg, group):
        v = parse_flag(cfg.get("command", ""), "--muon_ns_steps")
        cfg["muon_ns_steps"] = int(v) if v is not None else 5

    MUON_SOURCES = (
        "muon_loraplus_lowlr_2k", "muon_loraplus_2k", "muon_nsoff_2k",
        "product_muon_2k", "adam_muon_2k",
        "muon_lowlr_2k", "new_optimizers_high_eta_2k",
    )
    return merge_runs(
        MUON_SOURCES,
        key_fn=lambda c: (c["optimizer"], float(c["lr"]),
                          float(c.get("lora_plus_multiplier", 1.0)),
                          c["muon_ns_steps"]),
        filter_fn=lambda c: c["optimizer"] in {
            "muon-lora", "product-muon-lora", "adam-muon-lora"},
        cfg_postprocess=_stamp,
        logs_root=LOGS_ROOT,
    )


def _muon_variant(cfg):
    base = {"muon-lora": "muon", "product-muon-lora": "product-muon",
            "adam-muon-lora": "adam-muon"}[cfg["optimizer"]]
    parts = [base]
    m = float(cfg.get("lora_plus_multiplier", 1.0))
    ns = cfg.get("muon_ns_steps", 5)
    if m != 1.0: parts.append(f"m={int(m)}")
    if ns == 0:  parts.append("ns=0")
    return "+".join(parts)


def _load_lp_runs():
    LP_SOURCES = (
        "loraplus_lowlr_2k", "loraplus_2k_1ep",
        "lr_sweep_2k", "optim_compare_high_eta_2k", "new_optimizers_2k",
    )
    return merge_runs(
        LP_SOURCES,
        key_fn=lambda c: (float(c.get("lora_plus_multiplier", 1.0)), float(c["lr"])),
        filter_fn=lambda c: c.get("optimizer") == "adamw",
        logs_root=LOGS_ROOT,
    )


def main():
    print("Loading runs ...")
    all_runs = _load_all_runs()
    ext_runs = _load_ext_runs()
    muon_runs = _load_muon_runs()
    svd_runs = load_sweep("svd_sweep_2k_1ep", logs_root=LOGS_ROOT)
    lp_runs = _load_lp_runs()
    print(f"  all={len(all_runs)} ext={len(ext_runs)} muon={len(muon_runs)} "
          f"svd={len(svd_runs)} lp={len(lp_runs)}")

    print("Rendering figures ...")

    # Cross-investigation leaderboard.
    best = {}
    for cfg, evs in ext_runs:
        if evs[-1]["step"] < 2000:
            continue
        k = (cfg["optimizer"], cfg["_lora_r"])
        fl = evs[-1]["eval_loss"]
        if k not in best or fl < best[k][2]:
            best[k] = (cfg, evs, fl)
    fig, _ = plot_leaderboard_by_rank(
        best, baseline_optimizer="adamw", color_map=OPTIM_COLORS,
        suptitle="Best eval loss vs LoRA rank")
    save(fig, "00_leaderboard")

    runs_r16 = [(c, e) for c, e in ext_runs if c["_lora_r"] == 16]
    fig, *_ = standard_sweep_figure(
        runs_r16, lambda c: c["optimizer"], OPTIM_COLORS,
        reference_runs=runs_r16,
        suptitle="Optimizer comparison at r=16")
    save(fig, "01_cross_r16")

    runs_r64 = [(c, e) for c, e in ext_runs if c["_lora_r"] == 64]
    fig, *_ = standard_sweep_figure(
        runs_r64, lambda c: c["optimizer"], OPTIM_COLORS,
        reference_runs=runs_r64,
        suptitle="Optimizer comparison at r=64")
    save(fig, "02_cross_r64")

    fig, *_ = standard_sweep_figure(
        all_runs, lambda c: c["optimizer"], OPTIM_COLORS,
        reference_runs=all_runs,
        suptitle="All optimizers at r=16")
    save(fig, "03_all_optimizers")

    fig, *_ = standard_sweep_figure(
        muon_runs, _muon_variant, MUON_VARIANT_COLORS,
        reference_runs=all_runs,
        suptitle="Muon-LoRA variants vs AdamW")
    save(fig, "04_muon")

    fig, *_ = standard_sweep_figure(
        svd_runs, lambda c: c["training_mode"], MODE_COLORS,
        reference_runs=all_runs,
        extra_baselines=(("galore-adamw", "#1f77b4"),),
        suptitle="SVD oracle and GaLore vs AdamW (r=16)")
    save(fig, "05_svd")

    LP_COLOR_MAP = {f"m={m}": c for m, c in M_COLORS.items()}
    def _lp_group(cfg):
        return f"m={int(round(cfg.get('lora_plus_multiplier', 1.0)))}"
    fig, *_ = standard_sweep_figure(
        lp_runs, _lp_group, LP_COLOR_MAP,
        reference_runs=all_runs,
        suptitle="AdamW with LoRA+ B-multiplier (r=16)")
    save(fig, "06_loraplus")

    print(f"\nDone. PNGs in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
