"""Aggregate the per-(model, dataset, rank) leaderboard cells into one doc.

Performance metric = speedup-vs-AdamW = horizon / (steps a method needs to reach
the best AdamW baseline's final eval loss) — the reciprocal of the
fraction-of-horizon (see lora_playground.leaderboard; higher is better).
Reported per method at (a) its best lr and (b) the reciprocal of the mean
fraction over {best lr, one grid-step below, one grid-step above} (NaN when
lr-pinned).

Cells and their run membership come from the shared registry
`lora_playground.workloads` — the SAME source the leaderboard notebooks consume,
so the doc and notebooks can never drift. Each cell is every completed
long-horizon run at a fixed (model_name, lora_r) whose `--data_dir` resolves to
the cell's dataset; there are no hand-maintained log-group lists here.

Run:  python scripts/analysis/build_leaderboard_doc.py
Out:  docs/notes/leaderboard.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from lora_playground.leaderboard import (
    labeled_completed_runs, leaderboard_rows, performance_profile,
    speedup_from_frac,
)
from lora_playground.plotting.labels import canonical_label
from lora_playground.workloads import iter_workloads, workload_runs

# Cross-setting ranking shows only variants run on >= AGG_MIN_COVERAGE workloads;
# scores are only comparable between variants that span a similar set of cells.
AGG_MIN_COVERAGE = 5

ROOT = Path(__file__).resolve().parents[2]
LOGS = str(ROOT / "logs")
OUT = ROOT / "docs" / "notes" / "leaderboard.md"


def _fmt_speedup(x):
    """Format the fraction-to-target as a speedup-vs-AdamW (``1/frac``, higher
    is better). 0.5 → ``2.00×``; AdamW's ≈1.0 → ``1.00×``; NaN → ``—``."""
    s = speedup_from_frac(x)
    return "—" if isinstance(s, float) and math.isnan(s) else f"{s:.2f}×"


def build_section(wl, rows: list[dict], target: float) -> str:
    """Format one cell's table. `rows`/`target` come from leaderboard_rows."""
    title = f"### {wl.title}"
    if not rows:
        return f"{title}\n\n_No completed runs found._\n"

    def sort_key(r):
        if r["variant"] == "AdamW":
            return (0, 0.0)
        f = r["frac_best_lr"]
        return (1, math.inf if (f is None or math.isnan(f)) else f)
    rows = sorted(rows, key=sort_key)

    lines = [title, ""]
    tgt = "—" if math.isnan(target) else f"{target:.4f}"
    lines.append(f"AdamW speed target (best-lr final loss): **{tgt}**  ·  horizon {wl.horizon} steps")
    lines.append("")
    lines.append("| method | best lr | final@best | speedup @ best lr | speedup (lr-avg) |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['best_lr']:.0e} | {r['final_at_best']:.4f} "
            f"| {_fmt_speedup(r['frac_best_lr'])} | {_fmt_speedup(r['frac_lr_avg'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_aggregate(perf_matrix: dict, workloads: list) -> str:
    """Cross-setting robustness ranking (AlgoPerf-style performance profiles).

    `perf_matrix[variant][workload] = frac_best_lr`, keyed by the fully
    discriminating `canonical_label` so one row is exactly one algorithm in
    every setting. A variant's score is computed only over the workloads it
    ran, so scores are comparable only between variants of similar coverage —
    the ranking therefore shows only variants with coverage >= AGG_MIN_COVERAGE
    and lists the rest (unranked) in a footnote.
    """
    rows = performance_profile(perf_matrix, workloads=workloads)
    n = len(workloads)
    shown = [r for r in rows if r["coverage"] >= AGG_MIN_COVERAGE]
    hidden = [r for r in rows if r["coverage"] < AGG_MIN_COVERAGE]
    lines = [
        "## Cross-setting robustness ranking",
        "",
        "AlgoPerf-style performance profile across the "
        f"{n} (model, dataset, rank) workloads "
        "(method: `lora_playground.leaderboard.performance_profile`). Rows are "
        "keyed by the fully discriminating `canonical_label` "
        "(`lora_playground.plotting.labels`), so **one row is exactly one "
        "algorithm** — `ns=5`, `ns=8`, and `PE=10` (polar_express) are kept "
        "distinct and never merged. For each workload the fastest variant "
        "present has ratio 1.0; `robustness_score` is the normalised area under "
        "the profile over τ∈[1,4] (1.0 ⇒ fastest everywhere it ran). A "
        "variant's score is computed only over the workloads it actually ran, "
        "so scores are comparable only between variants of similar coverage. "
        f"**This ranking therefore shows only variants run on ≥{AGG_MIN_COVERAGE}/{n} "
        "workloads**; lower-coverage variants are validated on too few settings "
        "to rank cross-setting and appear in the per-section tables below "
        "(listed, unranked, underneath).",
        "",
        "| canonical variant | coverage | robustness_score | mean ratio-to-best |",
        "|---|---|---|---|",
    ]
    for r in shown:
        lines.append(
            f"| `{r['variant']}` | {r['coverage']}/{r['n_workloads']} "
            f"| {r['robustness_score']:.3f} | {r['mean_ratio']:.3f} |"
        )
    if hidden:
        listed = "; ".join(
            f"`{r['variant']}` ({r['coverage']}/{n})" for r in hidden
        )
        lines += [
            "",
            f"_Coverage-starved (<{AGG_MIN_COVERAGE}/{n}, not ranked — see the "
            f"per-section tables): {listed}._",
        ]
    lines.append("")
    return "\n".join(lines)


def render_doc() -> str:
    """Build the full leaderboard markdown from the live logs. Pure (no I/O)."""
    header = [
        "# Optimizer leaderboard — speed-to-AdamW-target",
        "",
        "**Metric (higher = better).** Speedup-vs-AdamW = the multiple of "
        "AdamW's training horizon a method saves to reach the best AdamW "
        "baseline's *final* eval loss at the same (model, dataset, rank) — i.e. "
        "`horizon / (steps the method needs)`, the reciprocal of the "
        "fraction-of-horizon. The crossing step is linearly interpolated "
        "between the bracketing evals (`leaderboard.reach_fraction`), so the "
        "metric does not round the crossing up to the next point of the "
        "250-step eval grid. `1.0×` ⇒ no speedup (needs AdamW's whole run to "
        "match its final loss); `2.0×` ⇒ reaches it in half the steps; `—` ⇒ "
        "never reached it within the horizon.",
        "",
        "- **speedup @ best lr** — at the method's best (lowest-final-loss) lr.",
        "- **speedup (lr-avg)** — reciprocal of the mean fraction over {best lr, "
        "one grid-step below, one grid-step above}. `—` when the best lr is at a "
        "swept boundary (lr-pinned) or any of the three never reaches the target.",
        "",
        "Generated by `scripts/analysis/build_leaderboard_doc.py` from the shared "
        "workload registry (`lora_playground.workloads`) — the same source the "
        "leaderboard notebooks consume, so the doc and notebooks cannot drift. "
        "Each cell is every completed long-horizon run at a fixed "
        "(model_name, lora_r) whose `--data_dir` resolves to the cell's dataset "
        "(the cfg `dataset_name` field is the stale Magicoder argparse default "
        "and is not used). AdamW's own row is ≈1.0 by construction. Horizon is "
        "9000 steps (Tulu-3 exhausts at ~8970 = one epoch, absorbed by the "
        "completion slack).",
        "",
    ]
    # One labeling pass per cell (canonical_label) feeds BOTH the per-section
    # table and the cross-setting aggregate matrix (frac_best_lr per workload).
    sections, perf_matrix, workloads = [], {}, []
    for wl in iter_workloads():
        runs = workload_runs(wl, logs_root=LOGS)
        labeled = labeled_completed_runs(runs, canonical_label, horizon=wl.horizon)
        rows, target = leaderboard_rows(labeled, horizon=wl.horizon)
        sections.append(build_section(wl, rows, target))
        workloads.append(wl.label)
        for r in rows:
            perf_matrix.setdefault(r["variant"], {})[wl.label] = r["frac_best_lr"]

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
