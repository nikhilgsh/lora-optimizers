"""Aggregate the per-(model, dataset, rank) leaderboard cells into one doc.

Performance metric = speedup-vs-AdamW = horizon / (steps a method needs to reach
the best AdamW baseline's final eval loss) — the reciprocal of the
fraction-of-horizon (see lora_playground.leaderboard; higher is better).
Reported per method at (a) its best lr and (b) the reciprocal of the mean
fraction over {best lr, one grid-step below, one grid-step above} (NaN when
lr-pinned).

Cells come from the shared `lora_playground.workloads` registry. Their run
membership and publication identities come from the checked-in records-native
archive, so regeneration never reconstructs executed optimizer semantics from
today's loader or labeling defaults.

Run:  python scripts/analysis/build_leaderboard_doc.py
Out:  docs/notes/leaderboard.md
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from lora_playground.comparison import build_comparison
from lora_playground.leaderboard import (
    UncertifiedBaselineError,
    leaderboard_rows_from_comparison,
    performance_profile,
    speedup_from_frac,
)
from lora_playground.leaderboard_variants import publication_variant_specs
from lora_playground.publication_archive import (
    PublicationArchiveError,
    load_publication_archive,
)
from lora_playground.workloads import iter_workloads, resolve_record_dataset

# Cross-setting ranking shows only variants run on >= AGG_MIN_COVERAGE workloads;
# scores are only comparable between variants that span a similar set of cells.
AGG_MIN_COVERAGE = 5

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "publication" / "legacy_leaderboard_v1.json"
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

    `perf_matrix[variant][workload] = frac_best_lr`, keyed by the stable
    publication label so one row is exactly one algorithm in
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
        f"{n} pipeline-scoped (model, dataset, rank) workloads "
        "(method: `lora_playground.leaderboard.performance_profile`). Rows are "
        "keyed by the archive's stable publication variant identity, so **one "
        "row is exactly one "
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


def _workload_archive_runs(archive_runs, wl) -> tuple:
    """Select one declared workload from immutable archived records."""
    selected = []
    for index, run in enumerate(archive_runs):
        cfg = run.effective_config
        max_steps = cfg.get("max_steps")
        if cfg.get("model_name") != wl.model_name or cfg.get("lora_r") != wl.rank:
            continue
        if not isinstance(max_steps, int) or max_steps < wl.min_completed_steps:
            continue
        if resolve_record_dataset(run, index=index) != wl.dataset:
            continue
        pipeline = cfg.get("data_pipeline_version")
        if not isinstance(pipeline, str) or not pipeline:
            raise PublicationArchiveError(
                f"archived run {run.physical_id!r} has no recorded "
                "config.data_pipeline_version"
            )
        if pipeline != wl.data_pipeline_version:
            continue
        selected.append(run)
    return tuple(selected)


def _unique_variant_id(variants, label: str) -> str:
    matches = [variant.id for variant in variants if variant.label == label]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one publication variant labeled {label!r}; "
            f"found {matches!r}"
        )
    return matches[0]


def _unscored_section(wl, issue: str) -> str:
    return (
        f"### {wl.title}\n\n"
        f"_Not scored: {issue}; no speed-to-AdamW target is published for this "
        "cell._\n"
    )


def render_doc(archive_path: str | Path = ARCHIVE) -> str:
    """Build the full leaderboard markdown from the checked-in archive."""
    archive = load_publication_archive(archive_path)
    variants = publication_variant_specs(archive.runs)
    baseline_id = _unique_variant_id(variants, "AdamW")
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
        "Generated by `scripts/analysis/build_leaderboard_doc.py` from the "
        "checked-in records-native publication archive and the shared workload "
        "registry (`lora_playground.workloads`). Stable archived variant IDs, "
        "not current loader or labeling defaults, define optimizer identity. "
        "Each cell selects archived long-horizon records at a fixed "
        "(model_name, dataset, lora_r, data_pipeline_version). AdamW's own "
        "row is ≈1.0 by construction. A cell is withheld from scoring when the "
        "best recorded AdamW LR is sweep-boundary-pinned, because that does not "
        "establish a best-LR target. Horizon is "
        "9000 steps (Tulu-3 exhausts at ~8970 = one epoch, absorbed by the "
        "completion slack).",
        "",
    ]
    # One records-native comparison feeds BOTH the per-section table and the
    # cross-setting aggregate matrix (frac_best_lr per workload).
    sections, perf_matrix, workloads = [], {}, []
    for wl in iter_workloads():
        runs = _workload_archive_runs(archive.runs, wl)
        comparison = build_comparison(runs, variants, horizon=wl.horizon)
        try:
            rows, target = leaderboard_rows_from_comparison(
                comparison,
                horizon=wl.horizon,
                baseline_id=baseline_id,
            )
        except UncertifiedBaselineError as exc:
            sections.append(_unscored_section(wl, exc.reason))
            continue
        sections.append(build_section(wl, rows, target))
        workloads.append(wl.label)
        for r in rows:
            perf_matrix.setdefault(r["variant"], {})[wl.label] = r["frac_best_lr"]

    aggregate = build_aggregate(perf_matrix, workloads)
    return ("\n".join(header) + "\n" + aggregate + "\n"
            + "\n".join(sections) + "\n")


def _archive_present(archive_path: str | Path = ARCHIVE) -> bool:
    """True when the publication archive path names a non-empty file."""
    path = Path(archive_path)
    return path.is_file() and path.stat().st_size > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check", action="store_true",
        help="regenerate in memory and compare to the committed doc; exit 1 on "
             "drift without writing (for staleness detection / CI gating).",
    )
    ap.add_argument(
        "--archive", default=str(ARCHIVE),
        help=(
            "publication archive to read (default: "
            "publication/legacy_leaderboard_v1.json)."
        ),
    )
    ap.add_argument(
        "--output", default=str(OUT),
        help="generated markdown path (default: docs/notes/leaderboard.md).",
    )
    ap.add_argument(
        "--require-archive", action="store_true",
        help="fail instead of leaving output untouched when the archive is absent.",
    )
    args = ap.parse_args(argv)
    archive_path = Path(args.archive).resolve()
    output = Path(args.output).resolve()

    if not _archive_present(archive_path):
        print(
            f"no publication archive at {archive_path} — "
            f"leaving {output} untouched"
        )
        return 2 if args.require_archive else 0

    fresh = render_doc(archive_path)

    if args.check:
        current = output.read_text() if output.exists() else ""
        if fresh != current:
            print(f"STALE: {output} differs from a fresh regeneration — "
                  "run `./scripts/analysis/update_leaderboard.sh --stage`")
            return 1
        print(f"up to date: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fresh)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
