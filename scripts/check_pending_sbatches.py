#!/usr/bin/env python
"""Refuse pending sbatches whose `--ntasks=N` undershoots `# CELLS: M`
**only when M ≤ QOS_CAP** (default 24) and there's no QOS-clamp reason
to leave parallelism on the table.

Three failure modes vs three responses:

  1. Small grid, ntasks < CELLS (M ≤ QOS_CAP):  REFUSE.
     Mistake-shaped: inherited --ntasks from a prior sbatch; the marginal
     QOS cost of scaling up is zero but the wall-time win is real
     (CELLS/ntasks rounds → 1 round).

  2. Large grid, ntasks < CELLS (M > QOS_CAP):  WARN (don't refuse).
     Legitimate QOS clamp: a 40-cell grid genuinely needs --ntasks≤24
     and queues remaining cells through later rounds. Surfaces the
     wall-time / round count so the user can sanity-check.

  3. ntasks > CELLS (any size):                  REFUSE.
     Reserves idle GPU slots. Always a mistake.

Override: FORCE_NTASKS_MISMATCH=1. QOS cap: QOS_CAP env var (default 24).

Convention: every pending sbatch declares `# CELLS: N` near `# ETA: H`.
Sbatches missing the header are flagged (not refused) so the convention
spreads to new templates without breaking submission on legacy ones.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PENDING = Path("slurm_pending")
NTASKS_PATTERN = re.compile(r"^\s*#SBATCH\s+--ntasks=(\d+)\s*$", re.MULTILINE)
CELLS_PATTERN = re.compile(r"^\s*#\s*CELLS:\s*(\d+)\s*$", re.MULTILINE)
# Body line `PARAM_FILE="params/<sweep>.json"` (the disBatch sweep spec). Two
# pending sbatches sharing a PARAM_FILE are almost always the same sweep staged
# twice under different group names — submitting both burns GPUs on identical
# cells and collides in the loader. (Real failure, 2026-05-31: a curvature
# sweep got staged a second time without first checking slurm_pending/.)
PARAM_FILE_PATTERN = re.compile(
    r"""^\s*PARAM_FILE=["']?([^"'\s]+)["']?\s*$""", re.MULTILINE
)


def _duplicate_param_file_msgs() -> list[str]:
    """REFUSE lines for any params file referenced by >1 pending sbatch."""
    by_param: dict[str, list[str]] = {}
    for sbatch in sorted(PENDING.glob("*.sbatch")):
        m = PARAM_FILE_PATTERN.findall(sbatch.read_text())
        # Array-driven multi-group sbatches assign PARAM_FILE="${PARAMS[$i]}" in
        # a loop; the literal token is an unresolved shell expansion, not a real
        # path, so comparing it across sbatches is meaningless and false-collides
        # any two such sbatches. Skip unresolvable values — dup detection still
        # works for the single-`PARAM_FILE="params/x.json"` case it was built for.
        m = [v for v in m if "${" not in v]
        if m:
            by_param.setdefault(m[-1], []).append(sbatch.name)
    return [
        f"  {param_file}  <-  {', '.join(names)}"
        for param_file, names in sorted(by_param.items())
        if len(names) > 1
    ]


def main() -> int:
    if not PENDING.exists():
        return 0
    qos_cap = int(os.environ.get("QOS_CAP", "24"))
    refuse_msgs: list[str] = []
    warn_msgs: list[str] = []
    missing_header: list[str] = []

    dup_param_msgs = _duplicate_param_file_msgs()

    for sbatch in sorted(PENDING.glob("*.sbatch")):
        text = sbatch.read_text()
        ntasks_matches = NTASKS_PATTERN.findall(text)
        if not ntasks_matches:
            continue
        ntasks = int(ntasks_matches[-1])
        cells_matches = CELLS_PATTERN.findall(text)
        if not cells_matches:
            missing_header.append(
                f"  {sbatch.name}: --ntasks={ntasks}, no `# CELLS: N` header"
            )
            continue
        cells = int(cells_matches[-1])

        if ntasks == cells:
            continue
        if ntasks > cells:
            refuse_msgs.append(
                f"  {sbatch.name}: --ntasks={ntasks} > # CELLS: {cells} "
                f"({ntasks - cells} idle GPU slot{'s' if ntasks - cells != 1 else ''} reserved)"
            )
            continue
        # ntasks < cells:
        rounds = -(-cells // ntasks)  # ceil-div
        if cells <= qos_cap:
            refuse_msgs.append(
                f"  {sbatch.name}: --ntasks={ntasks} < # CELLS: {cells} "
                f"(grid fits under QOS_CAP={qos_cap}; {rounds} rounds "
                f"instead of 1 → ~{rounds}x wall time)"
            )
        else:
            warn_msgs.append(
                f"  {sbatch.name}: --ntasks={ntasks} < # CELLS: {cells} "
                f"(grid exceeds QOS_CAP={qos_cap}; {rounds} rounds expected; "
                f"verify --time covers it)"
            )

    if warn_msgs:
        sys.stderr.write(
            "check_pending_sbatches: WARN — sbatch(es) using QOS-clamped "
            "parallelism (informational, not refused):\n"
            + "\n".join(warn_msgs) + "\n\n"
        )
    if missing_header:
        sys.stderr.write(
            "check_pending_sbatches: WARN — sbatch(es) missing `# CELLS: N` "
            "header (required for ntasks-vs-cells check; add near `# ETA: H`):\n"
            + "\n".join(missing_header) + "\n\n"
        )
    if dup_param_msgs:
        sys.stderr.write(
            "check_pending_sbatches: REFUSE — same PARAM_FILE referenced by "
            "multiple pending sbatches (duplicate sweep staged twice):\n"
            + "\n".join(dup_param_msgs)
            + "\n\nSubmitting both runs identical cells on separate GPUs and "
            "creates colliding log groups. Before staging a new sweep, check "
            "slurm_pending/ (and the live queue) for an sbatch already using "
            "this params file. Delete the duplicate sbatch, or — if the second "
            "is intentional (e.g. a different GPU type) — set the override.\n"
            "Override: FORCE_DUP_PARAMS=1\n"
        )
        if os.environ.get("FORCE_DUP_PARAMS") == "1":
            sys.stderr.write("FORCE_DUP_PARAMS=1 set; proceeding anyway.\n")
        else:
            return 1
    if refuse_msgs:
        sys.stderr.write(
            "check_pending_sbatches: REFUSE — ntasks/cells mismatch in "
            "pending sbatch(es):\n"
            + "\n".join(refuse_msgs)
            + "\n\nFor independent disBatch sweeps that fit under the QOS "
            f"cap (={qos_cap}), --ntasks should equal # CELLS so all cells "
            "run in parallel. Inheriting --ntasks from a prior sbatch is the "
            "usual cause; recompute from this grid's cell count.\n"
            "Override: FORCE_NTASKS_MISMATCH=1\n"
        )
        if os.environ.get("FORCE_NTASKS_MISMATCH") == "1":
            sys.stderr.write("FORCE_NTASKS_MISMATCH=1 set; proceeding anyway.\n")
            return 0
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
