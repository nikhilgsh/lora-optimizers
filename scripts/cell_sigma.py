"""Across-seed spread in the sealed Llama/openmath/r256 publication cell.

Stable publication variant IDs select the arms. Historical optimizer defaults
are never reconstructed and live logs are never scanned.

sigma is step-specific: an early-step spread does not transfer to step 9000, so the
step is printed with every row and must be quoted alongside the number.
"""
import argparse
import statistics
from pathlib import Path

from lora_playground.publication_ablation import (
    DEFAULT_PUBLICATION_ARCHIVE,
    KL_SHAMPOO_ID,
    ONE_SIDED_ID,
    PROTAGONIST_ID,
    load_ablation_evidence,
    seed_trajectories,
)


ARMS = (
    ("PoLoRA (protagonist)", PROTAGONIST_ID, 0.01),
    ("w/o rxr metric contents", KL_SHAMPOO_ID, 0.01),
    ("w/o rxr preconditioner", ONE_SIDED_ID, 0.003),
)
BORROWED = 0.0017


def render_sigma(evidence, *, requested_step: int | None) -> str:
    """Render per-arm sample spread at one shared archived eval step."""
    per_arm = {
        label: seed_trajectories(
            evidence.runs,
            variant_id=variant_id,
            lr=lr,
        )
        for label, variant_id, lr in ARMS
    }

    lines = [
        f"{'arm':26s} {'lr':>7s} {'seeds':>14s} {'step':>6s} "
        f"{'mean':>8s} {'sigma':>9s} {'2sigma':>8s}"
    ]
    sigmas = {}
    for label, _variant_id, lr in ARMS:
        seeds = per_arm[label]
        if len(seeds) < 2:
            lines.append(
                f"{label:26s} {float(lr):7g} {str(sorted(seeds)):>14s} "
                f"{'-':>6s} {'-':>8s} {'-':>9s} {'-':>8s}"
            )
            continue
        common = set.intersection(*(set(trajectory) for trajectory in seeds.values()))
        step = (
            requested_step
            if requested_step is not None and requested_step in common
            else max(common)
        )
        vals = [seeds[s][step] for s in sorted(seeds)]
        mean, sigma = statistics.mean(vals), statistics.stdev(vals)
        sigmas[label] = (sigma, step)
        lines.append(
            f"{label:26s} {float(lr):7g} {str(sorted(seeds)):>14s} "
            f"{step:6d} {mean:8.4f} {sigma:9.5f} {2*sigma:8.5f}"
        )
        lines.append(f"{'':26s} values: {['%.4f' % value for value in vals]}")

    lines.extend([
        "",
        f"historical cross-workload anchor (not used by ablation_table.py): "
        f"{BORROWED}",
    ])
    if sigmas:
        worst = max(s for s, _ in sigmas.values())
        lines.append(
            f"largest measured cell sigma so far: {worst:.5f} "
            f"({'above' if worst > BORROWED else 'below'} the borrowed anchor)"
        )
    lines.extend([
        "sigma is step-specific -- quote the printed step with it; do not",
        "transfer a spread measured at one step to another.",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_PUBLICATION_ARCHIVE,
        help="sealed publication archive to query",
    )
    args = parser.parse_args(argv)

    evidence = load_ablation_evidence(args.archive)
    print(render_sigma(evidence, requested_step=args.step))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
