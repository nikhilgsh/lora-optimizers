"""E2 derivation-ablation table from sealed publication evidence.

The report selects one registered workload and stable publication variant IDs.
Historical defaults are never reconstructed and live logs are never scanned.

Usage:  python scripts/ablation_table.py [--step N] [--all-steps]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lora_playground.publication_ablation import (
    ABLATION_ARMS,
    ABLATION_HORIZON,
    DEFAULT_PUBLICATION_ARCHIVE,
    PROTAGONIST_ID,
    build_ablation_comparison,
    comparison_curves,
    eval_trajectory,
    load_ablation_evidence,
)

SIGMA = 0.00050  # protagonist seeds 0-3 at step 9000 in this archived cell


def render_table(comparison, *, step: int | None, all_steps: bool) -> str:
    """Render step-matched best-LR rows from a records-native comparison."""
    series: dict[str, dict[float, dict[int, float]]] = {}
    depth: dict[str, int] = {}
    for arm in ABLATION_ARMS:
        if arm.variant_id is None:
            continue
        trajectories = {
            lr: eval_trajectory(curve.history)
            for lr, curve in comparison_curves(
                comparison, arm.variant_id
            ).items()
        }
        trajectories = {
            lr: trajectory
            for lr, trajectory in trajectories.items()
            if trajectory
        }
        if trajectories:
            series[arm.variant_id] = trajectories
            depth[arm.variant_id] = max(
                max(trajectory) for trajectory in trajectories.values()
            )

    lines: list[str] = []
    if all_steps:
        for arm in ABLATION_ARMS:
            arm_depth = depth.get(arm.variant_id, 0)
            lines.append(f"  {arm.label:26s} deepest step {arm_depth}")
        lines.append("")

    if PROTAGONIST_ID not in series:
        lines.append("no protagonist data; nothing to anchor on")
        return "\n".join(lines)

    report_step = step if step is not None else min(depth.values())
    at: dict[str, tuple[float, float]] = {}
    for variant_id, by_lr in series.items():
        values = {
            lr: trajectory[report_step]
            for lr, trajectory in by_lr.items()
            if report_step in trajectory
        }
        if values:
            best_lr = min(values, key=lambda lr: (values[lr], lr))
            at[variant_id] = (best_lr, values[best_lr])

    if PROTAGONIST_ID not in at:
        lines.append(f"protagonist has no eval at step {report_step}")
        return "\n".join(lines)
    base = at[PROTAGONIST_ID][1]

    lines.extend([
        f"step-matched at {report_step}/{ABLATION_HORIZON}   "
        f"sigma={SIGMA} (protagonist multiseed, packed_v1.1)",
        "",
        f"{'structure removed':28s} {'best lr':>8s} {'eval':>8s} "
        f"{'delta':>9s} {'sigma':>7s}  verdict",
    ])
    for arm in ABLATION_ARMS:
        result = at.get(arm.variant_id)
        if result is None:
            arm_depth = depth.get(arm.variant_id, 0)
            note = f"only to step {arm_depth}" if arm_depth else "no data yet"
            lines.append(
                f"{arm.label:28s} {'-':>8s} {'-':>8s} {'-':>9s} "
                f"{'-':>7s}  {note}"
            )
            continue
        best_lr, value = result
        delta = value - base
        sigma_units = delta / SIGMA
        if arm.variant_id == PROTAGONIST_ID:
            verdict = "reference"
        elif abs(sigma_units) < 1:
            verdict = "within noise"
        else:
            verdict = f"{sigma_units:.1f} sigma"
        lines.append(
            f"{arm.label:28s} {float(best_lr):8g} {value:8.4f} "
            f"{delta:+9.4f} {sigma_units:7.1f}  {verdict}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="report at this step; default = deepest step every arm has reached",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="also print how deep each arm currently is",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_PUBLICATION_ARCHIVE,
        help="sealed publication archive to query",
    )
    args = parser.parse_args(argv)

    evidence = load_ablation_evidence(args.archive)
    comparison = build_ablation_comparison(
        evidence, horizon=ABLATION_HORIZON
    )
    print(render_table(comparison, step=args.step, all_steps=args.all_steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
