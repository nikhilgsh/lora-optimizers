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
import subprocess
import sys
import tempfile
from pathlib import Path

PENDING = Path("slurm_pending")
NTASKS_PATTERN = re.compile(r"^\s*#SBATCH\s+--ntasks=(\d+)\s*$", re.MULTILINE)
CELLS_PATTERN = re.compile(r"^\s*#\s*CELLS:\s*(\d+)\s*$", re.MULTILINE)
# First line that LAUNCHES work (GPU / multi-task). The prep-body dry-run runs
# everything BEFORE this and stops here — so it never allocates a GPU, loads a
# module, or starts disBatch/srun, but DOES execute the inlined task-gen /
# checkpoint / manifest heredocs where runtime bugs (unexported env vars, bad
# paths, python exceptions) hide from bash -n and header checks.
LAUNCHER_PATTERN = re.compile(
    r"^\s*(module\s+load|disBatch|srun|torchrun|accelerate|mpirun|deepspeed|sbatch)\b"
)
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


def _prep_body_dryrun_msgs() -> list[str]:
    """Execute each pending sbatch's PREP BODY (the lines before the first
    launcher) with a stubbed SLURM_JOB_ID, and REFUSE on any runtime error.

    This is the forced equivalent of a dry run: bash -n only checks syntax and
    the header checks only read `#SBATCH`/`# CELLS`, so a runtime bug in an
    inlined heredoc — e.g. `os.environ["RUN_DIR"]` for a var that was never
    exported (real failure 2026-05-31, job 6463166: KeyError 'RUN_DIR' killed
    the job before disBatch started) — passes every static check and only dies
    on the cluster. Running the prep prefix surfaces it here. The prefix stops
    before `module load`/`disBatch`/`srun`, so it allocates nothing and starts
    no training; its only side effect is creating the run_info dir + task file,
    which the real submission recreates anyway."""
    msgs: list[str] = []
    for sbatch in sorted(PENDING.glob("*.sbatch")):
        lines = sbatch.read_text().splitlines()
        prefix: list[str] = []
        found_launcher = False
        for ln in lines:
            if LAUNCHER_PATTERN.match(ln):
                found_launcher = True
                break
            prefix.append(ln)
        # A launcher-less sbatch (no module load / disBatch / srun / torchrun / ...)
        # has no prep-prefix to validate: its BODY is the work (e.g. a
        # `python ...bench.py` microbench, where the python line is the job, not
        # setup). Dry-running it would execute the real job (model load + compile +
        # steps), blow past the 300s cap, and FALSE-refuse. There's nothing to
        # "stop before", so skip the dry-run — bash -n, the orchestration lint, and
        # sbatch --test-only still gate it. The prep-dryrun is a sweep-only check.
        if not found_launcher:
            continue
        body = "\n".join(prefix) + "\n"
        if "python" not in body and "<<" not in body:
            continue  # nothing executable worth dry-running
        env = dict(os.environ, SLURM_JOB_ID="dryrun", DRYRUN="1")
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".dryrun.sh", delete=False
            ) as f:
                f.write(body)
                tmp = f.name
            r = subprocess.run(
                ["bash", tmp], capture_output=True, text=True,
                env=env, timeout=300,
            )
        except subprocess.TimeoutExpired:
            msgs.append(f"  {sbatch.name}: prep body timed out (>300s)")
            continue
        finally:
            if tmp:
                os.unlink(tmp)
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip().splitlines()
            tail = "\n".join("        " + ln for ln in err[-8:])
            msgs.append(
                f"  {sbatch.name}: prep body FAILED (exit {r.returncode}):\n{tail}"
            )
            continue
        # Prep body exited 0 — but generate_task_file.py can exit 0 while emitting
        # a SYNTACtically broken task line (real failure 2026-06-07, jobs
        # 6485808/6485810: a wrapper positional `${5:-${ENV:-default}}` got mangled
        # into an unclosed `${ENV:-default`, swallowing the `> log 2> err` redirect,
        # so every disBatch task was a shell syntax error and the job died in
        # seconds). bash -n each generated task line to catch this whole class
        # (unclosed braces, swallowed redirects) at the gate, not as a dead alloc.
        path_m = re.search(r"Commands written to (.+?), number of lines", r.stdout or "")
        if not path_m:
            continue
        tasks_path = Path(path_m.group(1).strip())
        if not tasks_path.is_file():
            continue
        for i, line in enumerate(tasks_path.read_text().splitlines()):
            if not line.strip():
                continue
            chk = subprocess.run(["bash", "-n", "-c", line],
                                 capture_output=True, text=True)
            if chk.returncode != 0:
                detail = (chk.stderr or "").strip().splitlines()
                tail = "\n".join("        " + ln for ln in detail[-3:])
                msgs.append(
                    f"  {sbatch.name}: generated task line {i} is not valid shell "
                    f"(swallowed redirect / unclosed brace?):\n"
                    f"        {line[:200]}\n{tail}"
                )
                break  # one bad line is enough to refuse this sbatch
    return msgs


def main() -> int:
    if not PENDING.exists():
        return 0
    qos_cap = int(os.environ.get("QOS_CAP", "24"))
    refuse_msgs: list[str] = []
    warn_msgs: list[str] = []
    missing_header: list[str] = []

    dup_param_msgs = _duplicate_param_file_msgs()
    dryrun_msgs = [] if os.environ.get("FORCE_DRYRUN_SKIP") == "1" else _prep_body_dryrun_msgs()

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
    if dryrun_msgs:
        sys.stderr.write(
            "check_pending_sbatches: REFUSE — pending sbatch prep body failed a "
            "dry run (ran the script up to the launcher with SLURM_JOB_ID stubbed):\n"
            + "\n".join(dryrun_msgs)
            + "\n\nThis is a real runtime error that would kill the job on the "
            "cluster (bash -n and the header checks cannot see it). Fix the prep "
            "body — common cause: an inlined `python <<'EOF'` heredoc reads "
            "os.environ[\"VAR\"] for a VAR that was never `export`ed earlier in "
            "the script.\nOverride (NOT recommended): FORCE_DRYRUN_SKIP=1\n"
        )
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
