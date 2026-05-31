"""Aggregate the per-(model, dataset, rank) leaderboard notebooks into one doc.

Performance metric = fraction of the training horizon a method needs to reach
the best AdamW baseline's final eval loss (see lora_playground.leaderboard).
Reported per method at (a) its best lr and (b) averaged over {best lr, one
grid-step below, one grid-step above} (NaN when lr-pinned).

Panel definitions (log_groups + variant_key) mirror the notebooks so the
method rows match the rendered panels:
  notebooks/olmo2_1b_opc_leaderboard.ipynb            OLMo-2-1B  x opc
  notebooks/olmo2_1b_openmath_leaderboard.ipynb       OLMo-2-1B  x OpenMathInstruct-2
  notebooks/olmo2_1b_tulu3_leaderboard.ipynb          OLMo-2-1B  x Tulu-3
  notebooks/llama32_1b_opc_leaderboard.ipynb    Llama-3.2-1B x opc
  notebooks/llama32_1b_openmath_leaderboard.ipynb Llama-3.2-1B x OpenMathInstruct-2

NOTE on dataset identity: the packed-data-dir runs set the dataset via
`--data_dir`, leaving the cfg `dataset_name` field at its argparse default
(Magicoder). The true dataset is therefore the notebook the group belongs to,
not `dataset_name`. The horizon is taken per section as the longest completed
run's step (tulu3 exhausts at ~8970 steps = one epoch, not 9000).

Run:  python scripts/analysis/build_leaderboard_doc.py
Out:  docs/notes/leaderboard.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from lora_playground.loader import load_runs
from lora_playground.leaderboard import (
    labeled_completed_runs, merge_labeled, leaderboard_rows,
    performance_profile, DEFAULT_COMPLETION_SLACK,
)

ROOT = Path(__file__).resolve().parents[2]
LOGS = str(ROOT / "logs")
OUT = ROOT / "docs" / "notes" / "leaderboard.md"

OPT_ADAMW = "adamw"
OPT_CT = "adam-polar-product-lora-coupled-spectral-chord-tight"
OPT_CT_CLEAN = "adam-polar-product-lora-coupled-spectral-chord-tight-clean"


# ── variant_key functions (mirror the notebooks) ─────────────────────────────

def vk_olmo_opc(cfg):
    """olmo2_1b_opc_leaderboard cell 3 — k=1 and k=2 families."""
    opt = cfg.get("optimizer")
    if opt == "adamw":
        return "AdamW"
    eff_k = cfg.get("_derived", {}).get("effective_picard_iters", cfg.get("picard_iters_override"))
    pm = cfg.get("polar_method")
    pre_norm = cfg.get("_derived", {}).get("effective_polar_pre_norm")
    ctc_polar_suffix = ""
    if opt == OPT_CT_CLEAN and pm in ("ns", "polar_express") and pre_norm == "none":
        ctc_polar_suffix = " [post-fix]"
    if opt == OPT_CT and pm == "polar_express" and eff_k == 1:
        return "chord-tight k=1 polar_express"
    if opt == OPT_CT and eff_k == 1:
        return "chord-tight k=1 (ns, polar)"
    if opt == OPT_CT_CLEAN and cfg.get("ssc_kappa") == 0.75:
        return f"chord-tight-clean κ_sr=0.75 k={eff_k}"
    if opt == OPT_CT_CLEAN and cfg.get("ssc_c") == 0.2:
        return f"chord-tight-clean c=0.2 k={eff_k}"
    if (opt == OPT_CT_CLEAN and eff_k == 2
            and cfg.get("ssc_kappa") is None and cfg.get("ssc_c") is None):
        if pm == "polar_express":
            return f"chord-tight-clean k=2 polar_express{ctc_polar_suffix}"
        return f"chord-tight-clean k=2 (ns){ctc_polar_suffix}"
    return None


def vk_olmo_opc_epsrel(cfg):
    """olmo2_1b_opc_leaderboard cell 12 — only the ε_rel-specific arms."""
    if cfg.get("optimizer") != OPT_CT:
        return None
    ns = cfg.get("muon_ns_steps")
    if cfg.get("precond_delta_relative"):
        return f"chord-tight ε_rel 1e-2 ns={ns}"
    return None


def _ns_of(cfg):
    oc = cfg.get("optimizer_config") or {}
    return cfg.get("muon_ns_steps", oc.get("ns_steps"))


def vk_ns(cfg):
    """Robustness notebooks: label chord-tight by NS-iteration count."""
    opt = cfg.get("optimizer")
    if opt == OPT_ADAMW:
        return "AdamW"
    if opt == OPT_CT:
        ns = _ns_of(cfg)
        return f"chord ns={ns}" if ns is not None else "chord ns=?"
    return None


def vk_llama_opc(cfg):
    """Llama opc: basic + ns ablation + picard ablation, merged."""
    opt = cfg.get("optimizer")
    if opt == OPT_ADAMW:
        return "AdamW"
    if opt == OPT_CT:
        ns = _ns_of(cfg)
        return f"chord ns={ns}" if ns is not None else "chord ns=?"
    if opt == OPT_CT_CLEAN:
        return "clean ns8 picard=2"
    return None


def vk_canonical(cfg):
    """Mechanistic cross-setting label — same string ⇒ same algorithm across
    every (model, dataset, rank). Axes: family | polar-completeness | picard-k
    | damping. Used only for the cross-setting robustness aggregate, NOT the
    per-section tables (those keep the notebooks' display labels).

      family : ct (chord-tight) | clean (chord-tight-clean)
      polar  : ns{N} (N Newton-Schulz iters) | PE (polar_express ≈ full polar)
      k      : picard iterations (1 | 2)
      damp   : abs (eps=1e-6) | epsrel (relative) | ssckap{κ} | sscc{c}
    """
    opt = cfg.get("optimizer")
    if opt == "adamw":
        return "AdamW"
    if opt not in (OPT_CT, OPT_CT_CLEAN):
        return None
    fam = "clean" if opt == OPT_CT_CLEAN else "ct"
    pm = cfg.get("polar_method")
    ns = cfg.get("muon_ns_steps")
    # polar_express and ns>=8 both approximate a complete polar map, so they
    # collapse to one "full" bucket (they are not run on the same cell, so this
    # only unions coverage — it never hides a within-cell comparison).
    polar = "full" if (pm == "polar_express" or (ns is not None and ns >= 8)) else f"ns{ns}"
    k = cfg.get("_derived", {}).get("effective_picard_iters",
                                    cfg.get("picard_iters_override")) or 1
    if cfg.get("precond_delta_relative"):
        damp = "epsrel"
    elif cfg.get("ssc_kappa") is not None:
        damp = f"ssckap{cfg['ssc_kappa']}"
    elif cfg.get("ssc_c") is not None:
        damp = f"sscc{cfg['ssc_c']}"
    else:
        damp = "abs"
    return f"{fam}|{polar}|k{k}|{damp}"


# ── sections: (model, dataset, rank) -> list of (groups, variant_key) ────────

SECTIONS: list[dict] = [
    dict(model="OLMo-2-1B", dataset="opc-sft-stage2 (Magicoder)", rank=64, panels=[
        ([
            "adamw_phase_L_lrsweep_r64_blackwell",
            "chord_tight_phase_L_lrsweep_r64_blackwell",
            "adamw_phase_L_lrsweep_r64_repack_blackwell",
            "chord_tight_phase_L_lrsweep_r64_repack_blackwell",
            "adamw_phase_L_lrsweep_r64_extension_blackwell",
            "chord_tight_phase_L_lrsweep_r64_extension_blackwell",
            "chord_tight_clean_ssc_kappa075_phase_L_lrsweep_r64_blackwell",
            "chord_tight_clean_ssc_fixedc_c0p2_phase_L_lrsweep_r64_gpuxl_h200",
            "chord_tight_polar_express_phase_L_lrsweep_r64_gpuxl_h200",
            "chord_tight_polar_express_phase_L_lrsweep_r64_gpuxl_h200_resub",
            "chord_tight_polar_express_phase_L_lrsweep_r64_lr1e-3_blackwell",
            "chord_tight_clean_ssc_kappa075_phase_L_lrsweep_r64_extension_blackwell",
            "chord_tight_clean_ns_phase_L_lrsweep_r64_k2_blackwell",
            "chord_tight_clean_polar_express_phase_L_lrsweep_r64_k2_gpuxl_h200",
            "chord_tight_clean_ssc_kappa075_k1_phase_L_lrsweep_r64_blackwell",
            "chord_tight_clean_ssc_fixedc_c0p2_k1_phase_L_lrsweep_r64_blackwell",
        ], vk_olmo_opc),
    ]),
    dict(model="OLMo-2-1B", dataset="opc-sft-stage2 (Magicoder)", rank=256, panels=[
        ([
            "adamw_phase_L_lrsweep_r256_blackwell",
            "chord_tight_phase_L_lrsweep_r256_blackwell",
            "chord_tight_clean_ssc_kappa075_phase_L_lrsweep_r256_blackwell",
            "chord_tight_clean_ssc_fixedc_c0p2_phase_L_lrsweep_r256_gpuxl_h200",
            "chord_tight_polar_express_phase_L_lrsweep_r256_gpuxl_h200",
            "chord_tight_polar_express_phase_L_lrsweep_r256_blackwell",
            "chord_tight_polar_express_phase_L_lrsweep_r256_lr1e-3_blackwell",
            "chord_tight_clean_ssc_kappa075_phase_L_lrsweep_r256_extension_blackwell",
            "chord_tight_clean_ns_phase_L_lrsweep_r256_k2_blackwell",
            "chord_tight_clean_polar_express_phase_L_lrsweep_r256_k2_blackwell",
            "chord_tight_clean_ssc_kappa075_k1_phase_L_lrsweep_r256_blackwell",
            "chord_tight_clean_ssc_fixedc_c0p2_k1_phase_L_lrsweep_r256_blackwell",
            "chord_tight_phase_L_r256_epsrel_ns_probe_blackwell_v2",
        ], vk_olmo_opc),
        (["chord_tight_phase_L_r256_epsrel_ns_probe_blackwell_v2"], vk_olmo_opc_epsrel),
    ]),
    dict(model="OLMo-2-1B", dataset="OpenMathInstruct-2", rank=64, panels=[
        ([
            "adamw_robustness_openmath_1b_r64_gpuxl",
            "adamw_robustness_openmath_1b_r64_ext_right_gpuxl",
            "chord_tight_robustness_openmath_1b_r64_gpuxl",
            "chord_tight_robustness_openmath_1b_r64_ext_right_gpuxl",
            "chord_tight_robustness_openmath_1b_r64_ns8_blackwell",
            "chord_tight_robustness_openmath_1b_r64_ns8_ext_right_blackwell",
        ], vk_ns),
    ]),
    dict(model="OLMo-2-1B", dataset="OpenMathInstruct-2", rank=256, panels=[
        ([
            "adamw_robustness_openmath_1b_r256_gpuxl",
            "chord_tight_robustness_openmath_1b_r256_gpuxl",
            "chord_tight_robustness_openmath_1b_r256_ext_right_gpuxl",
            "chord_tight_robustness_openmath_1b_r256_ns8_blackwell",
            "chord_tight_robustness_openmath_1b_r256_ns8_ext_right_blackwell",
        ], vk_ns),
    ]),
    dict(model="OLMo-2-1B", dataset="Tulu-3 SFT mixture", rank=64, panels=[
        ([
            "adamw_robustness_tulu3_1b_r64_blackwell",
            "adamw_robustness_tulu3_1b_r64_gpuxl",
            "chord_tight_robustness_tulu3_1b_r64_gpuxl",
        ], vk_ns),
    ]),
    dict(model="OLMo-2-1B", dataset="Tulu-3 SFT mixture", rank=256, panels=[
        ([
            "adamw_robustness_tulu3_1b_r256_blackwell",
            "adamw_robustness_tulu3_1b_r256_gpuxl",
            "chord_tight_robustness_tulu3_1b_r256_gpuxl",
            "chord_tight_robustness_tulu3_1b_r256_ext_right_gpuxl",
        ], vk_ns),
    ]),
    dict(model="Llama-3.2-1B", dataset="opc-sft-stage2 (Magicoder)", rank=64, panels=[
        ([
            "adamw_robustness_llama32_1b_opc_r64_blackwell",
            "chord_tight_robustness_llama32_1b_opc_r64_blackwell",
            "chord_tight_robustness_llama32_1b_opc_r64_extend_left_blackwell",
            "chord_tight_robustness_llama32_1b_opc_r64_ns3_blackwell",
            "chord_tight_robustness_llama32_1b_opc_r64_ns8_blackwell",
            "chord_tight_clean_ns8_picard_ablation_llama32_1b_opc_r64_blackwell",
            "chord_tight_clean_ns8_picard_ablation_llama32_1b_opc_r64_extend_left_blackwell",
        ], vk_llama_opc),
    ]),
    dict(model="Llama-3.2-1B", dataset="opc-sft-stage2 (Magicoder)", rank=256, panels=[
        ([
            "adamw_robustness_llama32_1b_opc_r256_blackwell",
            "adamw_robustness_llama32_1b_opc_r256_extend_left_blackwell",
            "chord_tight_robustness_llama32_1b_opc_r256_blackwell",
            "chord_tight_robustness_llama32_1b_opc_r256_ns8_blackwell",
            "chord_tight_clean_ns8_picard_ablation_llama32_1b_opc_r256_blackwell",
        ], vk_llama_opc),
    ]),
    dict(model="Llama-3.2-1B", dataset="OpenMathInstruct-2", rank=64, panels=[
        ([
            "adamw_robustness_llama32_1b_openmath_r64_blackwell",
            "chord_tight_robustness_llama32_1b_openmath_r64_ns8_blackwell",
            "chord_tight_robustness_llama32_1b_openmath_r64_ns8_ext_left_blackwell",
        ], vk_ns),
    ]),
    dict(model="Llama-3.2-1B", dataset="OpenMathInstruct-2", rank=256, panels=[
        ([
            "adamw_robustness_llama32_1b_openmath_r256_blackwell",
            "chord_tight_robustness_llama32_1b_openmath_r256_ns8_blackwell",
            "chord_tight_robustness_llama32_1b_openmath_r256_ns8_ext_left_blackwell",
        ], vk_ns),
    ]),
]


def _fmt_frac(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def _load_section(sec: dict):
    """Load a section once. Returns (panel_runs, all_runs, horizon)."""
    panel_runs, all_runs = [], []
    for groups, vk in sec["panels"]:
        runs = load_runs(where={"log_group": groups}, logs_root=LOGS,
                         warn_cross_commit=False, quiet=True)
        panel_runs.append((runs, vk))
        all_runs.extend(runs)
    last_steps = [max(h, key=lambda e: e.get("step", 0)).get("step", 0)
                  for _, h in all_runs if h]
    horizon = max(last_steps) if last_steps else 9000
    return panel_runs, all_runs, horizon


def _section_label(sec: dict) -> str:
    return f"{sec['model'].split('-')[0]}/{sec['dataset'].split()[0]}/r{sec['rank']}"


def build_section(sec: dict, loaded=None) -> str:
    panel_runs, _all, horizon = loaded if loaded is not None else _load_section(sec)

    labeled_dicts = [labeled_completed_runs(runs, vk, horizon=horizon)
                     for runs, vk in panel_runs]
    labeled = merge_labeled(*labeled_dicts)
    rows, target = leaderboard_rows(labeled, horizon=horizon)

    title = f"### {sec['model']} × {sec['dataset']} × r={sec['rank']}"
    if not rows:
        return f"{title}\n\n_No completed runs found._\n"

    def sort_key(r):
        if r["variant"] == "AdamW":
            return (0, 0.0)
        f = r["frac_best_lr"]
        return (1, math.inf if (f is None or math.isnan(f)) else f)
    rows.sort(key=sort_key)

    lines = [title, ""]
    tgt = "—" if math.isnan(target) else f"{target:.4f}"
    lines.append(f"AdamW speed target (best-lr final loss): **{tgt}**  ·  horizon {horizon} steps")
    lines.append("")
    lines.append("| method | best lr | final@best | perf @ best lr | perf (lr-avg) |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['best_lr']:.0e} | {r['final_at_best']:.4f} "
            f"| {_fmt_frac(r['frac_best_lr'])} | {_fmt_frac(r['frac_lr_avg'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_aggregate(perf_matrix: dict, workloads: list) -> str:
    """Cross-setting robustness ranking (AlgoPerf-style performance profiles).

    `perf_matrix[canonical_variant][workload] = frac_best_lr`. Ranks variants by
    coverage-then-robustness so coverage-starved rows can't masquerade as
    broadly-validated winners.
    """
    rows = performance_profile(perf_matrix, workloads=workloads)
    lines = [
        "## Cross-setting robustness ranking",
        "",
        "AlgoPerf-style performance profile across the "
        f"{len(workloads)} (model, dataset, rank) workloads "
        "(method: `lora_playground.leaderboard.performance_profile`). Variants "
        "use the **canonical mechanistic key** `family|polar|k|damp` "
        "(`vk_canonical`), so a row means the same algorithm in every setting. "
        "For each workload the fastest variant present has ratio 1.0; "
        "`robustness_score` is the normalised area under the profile over "
        "τ∈[1,4] (1.0 ⇒ fastest everywhere it ran). **`coverage` = how many "
        "workloads the variant has actually been run on** — a high score at low "
        "coverage is a hypothesis, not a result, so rows are sorted by coverage "
        "first. Most non-`abs` variants are coverage-starved (OLMo-opc only).",
        "",
        "| canonical variant | coverage | robustness_score | mean ratio-to-best |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['variant']}` | {r['coverage']}/{r['n_workloads']} "
            f"| {r['robustness_score']:.3f} | {r['mean_ratio']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_doc() -> str:
    """Build the full leaderboard markdown from the live logs. Pure (no I/O)."""
    header = [
        "# Optimizer leaderboard — speed-to-AdamW-target",
        "",
        "**Metric (lower = better).** Performance = fraction of the training "
        "horizon a method needs to reach the best AdamW baseline's *final* eval "
        "loss at the same (model, dataset, rank). `1.0` ⇒ needs AdamW's whole "
        "run to match AdamW's final loss; `0.5` ⇒ reaches it in half the steps; "
        "`—` ⇒ never reached it within the horizon.",
        "",
        "- **perf @ best lr** — at the method's best (lowest-final-loss) lr.",
        "- **perf (lr-avg)** — mean over {best lr, one grid-step below, one "
        "grid-step above}. `—` when the best lr is at a swept boundary "
        "(lr-pinned) or any of the three never reaches the target.",
        "",
        "Generated by `scripts/analysis/build_leaderboard_doc.py` from "
        "`lora_playground.loader.load_runs` over the same log_groups the "
        "leaderboard notebooks render (metric in `lora_playground/leaderboard.py`). "
        "AdamW's own row is ≈1.0 by construction. Horizon is the longest "
        "completed run per section (tulu3 exhausts at ~8970 steps = one epoch). "
        "Dataset identity comes from each run's `--data_dir` / notebook, not the "
        "cfg `dataset_name` field (which stays at its Magicoder argparse default "
        "for packed-data-dir runs).",
        "",
    ]
    # One load per section, feeding both the per-section table and the
    # cross-setting aggregate matrix (canonical key, frac_best_lr per workload).
    sections, perf_matrix, workloads = [], {}, []
    for sec in SECTIONS:
        loaded = _load_section(sec)
        sections.append(build_section(sec, loaded))
        wl = _section_label(sec)
        workloads.append(wl)
        panel_runs, _all, horizon = loaded
        canon_runs = [r for runs, _vk in panel_runs for r in runs]
        labeled = labeled_completed_runs(canon_runs, vk_canonical, horizon=horizon)
        rows, _ = leaderboard_rows(labeled, horizon=horizon)
        for r in rows:
            perf_matrix.setdefault(r["variant"], {})[wl] = r["frac_best_lr"]

    aggregate = build_aggregate(perf_matrix, workloads)
    return ("\n".join(header) + "\n" + aggregate + "\n"
            + "\n".join(sections) + "\n")


def _logs_present() -> bool:
    """True if the logs tree exists and holds at least one group dir."""
    p = Path(LOGS)
    return p.is_dir() and any(p.iterdir())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="regenerate in memory and compare to the committed doc; exit 1 on "
             "drift without writing (for staleness detection / CI gating).",
    )
    args = ap.parse_args(argv)

    if not _logs_present():
        # Clean checkout / machine without data: never blank the doc.
        print(f"no logs under {LOGS} — leaving {OUT} untouched")
        return 0

    fresh = render_doc()

    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if fresh != current:
            print(f"STALE: {OUT} differs from a fresh regeneration — "
                  f"run `python scripts/analysis/build_leaderboard_doc.py`")
            return 1
        print(f"up to date: {OUT}")
        return 0

    OUT.write_text(fresh)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
