"""Render docs/notes/polar_product/ns_polar_express_bench.md from JSONL output of
`scripts/bench/bench_polar_orthog.py`.

Reads:
  logs/bench/polar_orthog_blackwell.jsonl  (canonical hardware)
  logs/bench/polar_orthog_a6000.jsonl       (local workstation)

Emits a single Markdown report with:
  1. Wall-time table per shape, K — both hardware columns side by side.
  2. Accuracy table per shape, K — random-input residual (for the same shape)
     plus min/median/max residual across the snapshot u_A pairs.
  3. Cost-matched callout: NS_rect K=10 vs PE K=8 on the same shape.

Run after both benches have written their JSONL files. Idempotent — pure
function of the JSONL inputs.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs" / "bench"
OUT = ROOT / "docs" / "notes" / "polar_product" / "ns_polar_express_bench.md"

HARDWARE = ["blackwell", "a6000"]
VARIANTS = ["ns_rect", "ns_gram_fp16", "polar_express"]
VARIANT_LABEL = {
    "ns_rect": "NS_rect",
    "ns_gram_fp16": "NS_gram (fp16+restart)",
    "polar_express": "PolarExpress",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def fmt_ms(x):
    if x is None:
        return "—"
    return f"{x:6.3f}"


def fmt_resid(x):
    if x is None:
        return "—"
    if x < 1e-6:
        return f"{x:.1e}"
    return f"{x:.2e}"


def render() -> str:
    rows_by_hw = {hw: load_jsonl(LOGS / f"polar_orthog_{hw}.jsonl")
                  for hw in HARDWARE}
    have = [hw for hw, r in rows_by_hw.items() if r]
    if not have:
        return ("# NS / Polar Express bench\n\n"
                "No JSONL inputs found. Run scripts/bench/bench_polar_orthog.py "
                "first.\n")

    # Pull commit / gpu_name from any available row.
    any_row = next(iter(rows_by_hw[have[0]]))
    sha = any_row.get("git_commit", "unknown")[:10]

    lines: list[str] = []
    lines.append("# Newton-Schulz vs Polar Express bench")
    lines.append("")
    lines.append(f"Source: `scripts/bench/bench_polar_orthog.py` "
                 f"(commit `{sha}`). Snapshot u_A from "
                 f"`chord_tight_r64_k3_snapshot_blackwell/step_2000`.")
    lines.append("")
    lines.append("Variants timed (all in `lora_playground.optim`):")
    lines.append("")
    lines.append("- `_newton_schulz` — cubic, per-matrix, fp32 (canonical Muon).")
    lines.append("- `_newton_schulz_gram_batched` — Dao 2026 Algorithm 3 "
                 "with fp16 iteration + restart at iter 2 (production default).")
    lines.append("- `_polar_express` — Amsel 2025, degree-5 with optimal "
                 "coefficients for σ ∈ [1e-3, 1].")
    lines.append("")
    for hw in have:
        rows = rows_by_hw[hw]
        gpu = rows[0].get("gpu_name", "unknown")
        n_reps = next((r["n_reps"] for r in rows if r.get("n_reps")), None)
        lines.append(f"- **{hw}** = `{gpu}` (n_reps={n_reps})")
    lines.append("")

    # ------------------------------------------------------------------
    # Wall-time table: rows are (shape_kind, K), columns are variant × hardware.
    lines.append("## Wall time (ms / call)")
    lines.append("")
    lines.append("Random fp32 inputs at production LoRA shapes "
                 "(A-side: `(r, d_in)`). Mean over n_reps CUDA-event "
                 "samples, after warmup.")
    lines.append("")

    # Header
    col_headers = ["shape", "K"]
    for hw in have:
        for v in VARIANTS:
            col_headers.append(f"{VARIANT_LABEL[v]} ({hw})")
    lines.append("| " + " | ".join(col_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(col_headers)) + "|")

    # Aggregate: key = (shape_kind, K) -> hardware -> variant -> ms_mean
    timings: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for hw, rows in rows_by_hw.items():
        for r in rows:
            if r["source"] != "random_fp32":
                continue
            timings[(r["shape_kind"], r["K"])][hw][r["variant"]] = r["ms_mean"]

    for (shape, K) in sorted(timings.keys()):
        row = [shape, str(K)]
        for hw in have:
            for v in VARIANTS:
                row.append(fmt_ms(timings[(shape, K)].get(hw, {}).get(v)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    # Accuracy table: residual to true polar.
    lines.append("## Accuracy: max|σ_i − 1| after K iterations")
    lines.append("")
    lines.append("Lower = closer to true polar. Two input sources:")
    lines.append("")
    lines.append("- **random**: fp32 Gaussian at the shape (single sample).")
    lines.append("- **snapshot u_A**: real bias-corrected Adam direction from "
                 "the chord-tight r=64 production snapshot, aggregated across "
                 "the loaded pairs as `[min, median, max]`.")
    lines.append("")

    # Random-input accuracy (same across hardware up to fp16 noise — use blackwell
    # if available, else any).
    src_hw = "blackwell" if "blackwell" in have else have[0]
    rand_rows = [r for r in rows_by_hw[src_hw] if r["source"] == "random_fp32"]
    snap_rows = [r for r in rows_by_hw[src_hw] if r["source"] == "snapshot_u_A"]

    # Random accuracy: (shape, K) -> variant -> residual
    rand_acc: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rand_rows:
        rand_acc[(r["shape_kind"], r["K"])][r["variant"]] = r["sigma_residual_max"]

    # Snapshot accuracy: bucket by (r, d, K) -> variant -> list[residuals across pairs]
    snap_acc: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in snap_rows:
        key = (r["r"], r["d"], r["K"])
        snap_acc[key][r["variant"]].append(r["sigma_residual_max"])

    lines.append("### Random fp32 inputs")
    lines.append("")
    hdr = ["shape", "K"] + [VARIANT_LABEL[v] for v in VARIANTS]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for (shape, K) in sorted(rand_acc.keys()):
        row = [shape, str(K)]
        for v in VARIANTS:
            row.append(fmt_resid(rand_acc[(shape, K)].get(v)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if snap_acc:
        lines.append("### Snapshot u_A inputs (real production tensors)")
        lines.append("")
        hdr = ["(r, d)", "K", "n_pairs"] + [VARIANT_LABEL[v] for v in VARIANTS]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
        for (r, d, K) in sorted(snap_acc.keys()):
            row = [f"({r}, {d})", str(K)]
            # n_pairs from any variant column (same per cell)
            any_v = next(iter(snap_acc[(r, d, K)]))
            row.append(str(len(snap_acc[(r, d, K)][any_v])))
            for v in VARIANTS:
                vals = snap_acc[(r, d, K)].get(v, [])
                if not vals:
                    row.append("—")
                else:
                    row.append(f"[{min(vals):.2e}, {median(vals):.2e}, {max(vals):.2e}]")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # ------------------------------------------------------------------
    # Cost-matched callout
    lines.append("## Cost-matched: NS_rect K=10 vs PolarExpress K∈{6,7,8}")
    lines.append("")
    lines.append("Question raised in `notebooks/muon_squared_snapshot_analysis.ipynb`: "
                 "given that the leaderboard shows NS j=10 > j=5, does PE-j=k do "
                 "better than NS-j=10 at comparable wall? The Polar Express "
                 "schedule is fully exhausted by iter 7 (iter 8 onward uses plain "
                 "NS-deg5), so K∈{6, 7, 8} bracket the candidate replacements.")
    lines.append("")
    hdr = ["shape", "hw",
           "NS K=10 ms", "PE K=6 ms", "PE K=7 ms", "PE K=8 ms",
           "NS K=10 resid", "PE K=6 resid", "PE K=7 resid", "PE K=8 resid"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    for hw in have:
        rows = rows_by_hw[hw]
        by_key = {(r["shape_kind"], r["variant"], r["K"]): r
                  for r in rows if r["source"] == "random_fp32"}
        shapes_seen = sorted({r["shape_kind"] for r in rows
                              if r["source"] == "random_fp32"})
        for shape in shapes_seen:
            ns = by_key.get((shape, "ns_rect", 10))
            pe6 = by_key.get((shape, "polar_express", 6))
            pe7 = by_key.get((shape, "polar_express", 7))
            pe8 = by_key.get((shape, "polar_express", 8))
            if ns is None:
                continue
            row = [shape, hw, fmt_ms(ns["ms_mean"])]
            row += [fmt_ms(x["ms_mean"]) if x else "—" for x in (pe6, pe7, pe8)]
            row.append(fmt_resid(ns["sigma_residual_max"]))
            row += [fmt_resid(x["sigma_residual_max"]) if x else "—"
                    for x in (pe6, pe7, pe8)]
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ------------------------------------------------------------------
    lines.append("## Notes on interpretation")
    lines.append("")
    lines.append("- `NS_gram_fp16` residual plateaus at ~1–6e-3 (half-precision "
                 "noise floor on the Gram iterates) — that's the cost of the "
                 "production fp16+restart path. Whether this matters downstream "
                 "is the same empirical question the K=5 vs K=10 leaderboard "
                 "comparison answered: small but not zero.")
    lines.append("- PolarExpress is designed to reach the σ_min=1e-3 region in ~7 "
                 "iterations. The synthetic-random shapes here have wider "
                 "cond(X) than typical u_A inputs, so the random-input residual "
                 "column is a worst-case bound; the snapshot u_A column is what "
                 "the production optimizer actually sees.")
    lines.append("- NS_rect K=10 reaches fp32 noise floor only at small r; at "
                 "r=256 it still has ~1e-3 residual on random inputs. PE K=10 "
                 "reaches fp32 noise floor across all r.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```")
    lines.append("# Blackwell (canonical):")
    lines.append("#   submit slurm_pending/bench_polar_orthog_blackwell.sbatch")
    lines.append("# A6000 (local):")
    lines.append("python scripts/bench/bench_polar_orthog.py \\")
    lines.append("    --n_warmup 5 --n_reps 30 --n_pairs 8 --Ks 3 5 6 7 8 10 \\")
    lines.append("    --hardware a6000 --out logs/bench/polar_orthog_a6000.jsonl")
    lines.append("# Then re-render:")
    lines.append("python scripts/analysis/render_polar_orthog_table.py")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
