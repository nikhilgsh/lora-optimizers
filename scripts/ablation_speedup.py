"""E2 derivation-ablation speed report from sealed publication evidence.

A 0.003 loss gap is hard to interpret on its own. The question the paper asks is
how many steps each arm needs to reach tuned Adam's FINAL loss, and the speedup is
Adam's horizon over that. So each ablation says how much of the protagonist's
speedup that piece of structure is responsible for.

Uses lora_playground.leaderboard.reach_fraction / speedup_from_frac -- the same
crossing-interpolation the leaderboard and paper figures use -- rather than a
hand-rolled crossing, which would round up to the eval grid and understate.

The report selects one registered workload and stable publication variant IDs.
Historical defaults are never reconstructed and live logs are never scanned.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from lora_playground.leaderboard import reach_fraction, speedup_from_frac
from lora_playground.publication_ablation import (
    ABLATION_ARMS,
    ABLATION_HORIZON,
    ADAMW_ID,
    DEFAULT_PUBLICATION_ARCHIVE,
    PROTAGONIST_ID,
    build_ablation_comparison,
    eval_trajectory,
    load_ablation_evidence,
)


def _best_exact_step(comparison, variant_id: str, step: int):
    """Lowest-loss completed LR carrying an eval at exactly ``step``."""
    candidates = []
    for lr, curve in comparison.completed.get(variant_id, {}).items():
        trajectory = eval_trajectory(curve.history)
        if step in trajectory:
            candidates.append((trajectory[step], lr, curve))
    if not candidates:
        return None
    final_loss, lr, curve = min(candidates, key=lambda item: (item[0], item[1]))
    return lr, final_loss, curve


def render_speedup(comparison, *, horizon: int) -> str:
    """Render best-LR speed-to-Adam rows from a records-native comparison."""
    # Adam's tuned final loss is the target every arm is timed against.
    adam_best = _best_exact_step(comparison, ADAMW_ID, horizon)
    if not adam_best:
        return "no completed AdamW run at the horizon; nothing to time against"
    adam_lr, target, _adam_curve = adam_best
    lines = [
        f"target = tuned Adam final loss at step {horizon}: {target:.4f} "
        f"(lr={adam_lr})",
        "",
        f"{'structure removed':28s} {'best lr':>8s} {'final':>7s} "
        f"{'steps-to-Adam':>14s} {'speedup':>8s} "
        f"{'% of PoLoRA gain':>17s}",
    ]
    rows = []
    for arm in ABLATION_ARMS:
        best = (
            None
            if arm.variant_id is None
            else _best_exact_step(comparison, arm.variant_id, horizon)
        )
        if best is None:
            rows.append((arm, None))
            continue
        lr, final_loss, curve = best
        fraction = reach_fraction(curve.history, target, horizon)
        rows.append((arm, (lr, final_loss, fraction)))

    proto = next(
        (
            best
            for arm, best in rows
            if arm.variant_id == PROTAGONIST_ID and best is not None
        ),
        None,
    )
    proto_speed = speedup_from_frac(proto[2]) if proto else None
    for arm, best in rows:
        if best is None:
            lines.append(
                f"{arm.label:28s} {'-':>8s} {'-':>7s} {'-':>14s} "
                f"{'-':>8s} {'-':>17s}"
            )
            continue
        lr, final_loss, fraction = best
        speedup = speedup_from_frac(fraction)
        steps = fraction * horizon
        # fraction of the protagonist's excess speedup over Adam (1.0x) retained
        if proto_speed and proto_speed > 1.0:
            share = 100.0 * (speedup - 1.0) / (proto_speed - 1.0)
            share_s = f"{share:.0f}%"
        else:
            share_s = "-"
        lines.append(
            f"{arm.label:28s} {float(lr):8g} {final_loss:7.4f} "
            f"{steps:14.0f} {speedup:7.2f}x {share_s:>17s}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=ABLATION_HORIZON)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_PUBLICATION_ARCHIVE,
        help="sealed publication archive to query",
    )
    args = parser.parse_args(argv)

    evidence = load_ablation_evidence(args.archive)
    comparison = build_ablation_comparison(evidence, horizon=args.horizon)
    print(render_speedup(comparison, horizon=args.horizon))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
