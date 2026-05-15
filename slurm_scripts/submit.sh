#!/bin/bash
# Usage: ./slurm_scripts/submit.sh <params_json> <group_name> <n_gpus> [sweep_script] [sbatch_script]
# Example: ./slurm_scripts/submit.sh params/lr_sweep.json sweep_lr 4
# Example: ./slurm_scripts/submit.sh params/foo.json my_group 6 scripts/sweep/sweep_2k.sh slurm_scripts/sbatch_h100.sh
#
# Optional env vars (recommended — consumed by analysis tooling):
#   SWEEP_SCOPE="ext_compare,polar_family"   comma-separated tags
#   SWEEP_PURPOSE="E2: AdaMuon-faithful + polar-product geometry"
#
# To exclude an old sweep from analysis, delete its log dir.
set -euo pipefail

# ── Manifest contract refusal ────────────────────────────────────────────────
# Refuse to submit without scope tags. The notebook + tests (lora_playground/
# manifest.py, tests/test_manifests.py) require every populated log group to
# carry a non-empty scope; enforcing it at submission means "untagged sweep"
# becomes impossible by construction rather than by reminder.
if [[ -z "${SWEEP_SCOPE:-}" ]]; then
    echo "ERROR: SWEEP_SCOPE not set. Set scope tags before submitting:" >&2
    echo "  SWEEP_SCOPE=\"ext_compare,polar_family\" \\" >&2
    echo "  SWEEP_PURPOSE=\"E2: AdaMuon-faithful + polar-product geometry\" \\" >&2
    echo "  ./slurm_scripts/submit.sh params/<sweep>.json <group> <n_gpus> [...]" >&2
    echo "" >&2
    echo "Known scopes: ext_compare, muon_family, all_optimizers (r=16 only)," >&2
    echo "              r_extension (r != 16), loraplus_family, svd_oracle," >&2
    echo "              diagnostics, lin_scaled_investigation, polar_family," >&2
    echo "              winner_rerun, pilot, legacy," >&2
    echo "              tight_chord_paper, phase_L, longhorizon_1b," >&2
    echo "              repack_baseline, lr_extension" >&2
    echo "" >&2
    echo "See lora_playground/manifest.py for the full schema." >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Dirty-tree refusal ───────────────────────────────────────────────────────
# Refuse if load-bearing code (lora_playground/*.py, train_lora.py, sweep/
# sbatch shell wrappers) has uncommitted changes. Without this guard, the
# manifest records git_dirty=true and the loader silently excludes the runs.
# Override with FORCE_DIRTY=1.
REPO_DIR="${REPO_DIR}" "${REPO_DIR}/scripts/check_clean_tree.sh"

PARAM_FILE="$1"
GROUP="$2"
N_GPUS="$3"
SWEEP_SCRIPT="${4:-scripts/sweep/sweep.sh}"
SBATCH_SCRIPT="${5:-slurm_scripts/sbatch.sh}"

# Resolve to repo-relative form. If the caller passes an absolute path, strip
# the REPO_DIR prefix; otherwise prepending REPO_DIR below would produce
# `/REPO_DIR//absolute/path` which silently breaks downstream tooling
# (notably audit_sweep_overlap.py's --sweep-script, which reads launcher
# fixed-args; a missing file there causes spurious cross-horizon overlaps).
case "$PARAM_FILE" in /*) PARAM_FILE="${PARAM_FILE#${REPO_DIR}/}";; esac
case "$SWEEP_SCRIPT" in /*) SWEEP_SCRIPT="${SWEEP_SCRIPT#${REPO_DIR}/}";; esac
case "$SBATCH_SCRIPT" in /*) SBATCH_SCRIPT="${SBATCH_SCRIPT#${REPO_DIR}/}";; esac

# ── Reuse-existing-data refusal ──────────────────────────────────────────────
# Refuse to submit if any cell in the cartesian product of PARAM_FILE already
# exists in logs/. Override with FORCE_OVERLAP=1 (and document the reason in
# SWEEP_PURPOSE). The audit covers semantic equivalences (e.g. picard_alpha=0
# is equivalent to picard_iters=1 / uncoupled). See
# scripts/analysis/audit_sweep_overlap.py.
if [[ -z "${FORCE_OVERLAP:-}" ]]; then
    if ! python "${REPO_DIR}/scripts/analysis/audit_sweep_overlap.py" "${PARAM_FILE}" --logs-root "${REPO_DIR}/logs" --sweep-script "${REPO_DIR}/${SWEEP_SCRIPT}"; then
        echo "" >&2
        echo "ERROR: sweep overlaps with existing logs (see ✓ rows above)." >&2
        echo "Drop the duplicate cells from ${PARAM_FILE}, or set FORCE_OVERLAP=1" >&2
        echo "and explain the rerun in SWEEP_PURPOSE." >&2
        exit 1
    fi
fi

if [[ -n "${SWEEP_SUPERSEDES:-}" ]]; then
    echo "WARN: SWEEP_SUPERSEDES is no longer honored." >&2
    echo "      To exclude '${SWEEP_SUPERSEDES}' from analysis, delete its log dir." >&2
fi

RUN_DIR="${REPO_DIR}/logs/${GROUP}/run_info"
mkdir -p "${RUN_DIR}/logs" "${REPO_DIR}/slurm_logs" "${REPO_DIR}/disbatch_logs"

cp "${REPO_DIR}/${SWEEP_SCRIPT}" "${RUN_DIR}/sweep.sh"
cp "${PARAM_FILE}" "${RUN_DIR}/$(basename "$PARAM_FILE")"

python ~/hp_scaling/generate_task_file.py \
    --bash_script="${RUN_DIR}/sweep.sh" \
    --param_file="${RUN_DIR}/$(basename "$PARAM_FILE")" \
    --output_file="${RUN_DIR}/tasks" \
    --full_tasks=True \
    --add_logs=True \
    --log_dir="${RUN_DIR}/logs"

# ── Checkpoint injection ─────────────────────────────────────────────────────
# Each generated task line looks like:
#   sweep.sh <args...> > /…/logs/log_NN.out 2> /…/logs/log_NN.err
# Inject a per-task `CHECKPOINT_DIR=<RUN_DIR>/checkpoints/task_NN` env-var
# prefix so the train.py invocation under the sweep wrapper picks up
# --checkpoint_dir + --resume_from (idempotent: first run finds empty dir,
# resume picks the latest ckpt_step{N}). Opt out with NO_CHECKPOINTS=1.
if [[ -z "${NO_CHECKPOINTS:-}" ]]; then
    python - <<PYEOF
import re
from pathlib import Path
tasks_path = Path("${RUN_DIR}/tasks")
ckpt_root = Path("${RUN_DIR}/checkpoints")
ckpt_root.mkdir(parents=True, exist_ok=True)
out_lines = []
for line in tasks_path.read_text().splitlines():
    if not line.strip():
        out_lines.append(line)
        continue
    m = re.search(r"log_(\d+)\.out", line)
    if not m:
        out_lines.append(line)
        continue
    nn = m.group(1)
    task_ckpt = ckpt_root / f"task_{nn}"
    # Prepend per-task env var so it scopes to this command only. disBatch
    # runs each line via /bin/sh -c, which honors leading KEY=value tokens.
    out_lines.append(f"CHECKPOINT_DIR={task_ckpt} {line}")
tasks_path.write_text("\n".join(out_lines) + "\n")
PYEOF
fi

# ── Snapshot dir injection (opt-in via SNAPSHOTS=1) ──────────────────────────
# Mirrors the CHECKPOINT_DIR block above. Each generated task line gets a
# per-task `SNAPSHOT_DIR=<RUN_DIR>/snapshots/task_NN` env-var prefix so the
# launcher's --snapshot_dir flag picks it up. Snapshots are NOT subject to
# checkpoint pruning or end-of-run cleanup; the dir survives for downstream
# analysis. Opt-in only because most sweeps don't need it.
if [[ "${SNAPSHOTS:-0}" = "1" ]]; then
    python - <<PYEOF
import re
from pathlib import Path
tasks_path = Path("${RUN_DIR}/tasks")
snap_root = Path("${RUN_DIR}/snapshots")
snap_root.mkdir(parents=True, exist_ok=True)
out_lines = []
for line in tasks_path.read_text().splitlines():
    if not line.strip():
        out_lines.append(line)
        continue
    m = re.search(r"log_(\d+)\.out", line)
    if not m:
        out_lines.append(line)
        continue
    nn = m.group(1)
    task_snap = snap_root / f"task_{nn}"
    out_lines.append(f"SNAPSHOT_DIR={task_snap} {line}")
tasks_path.write_text("\n".join(out_lines) + "\n")
PYEOF
fi

# ── Pre-submit log rotation ──────────────────────────────────────────────────
# If a `log_NN.out` already exists from a prior wall-killed run on the same
# group, rotate it to `log_NN.out.resume_K` (K = next available) before
# disBatch creates fresh log_NN.out files. The loader's load_sweep merges
# events across log_NN.out + log_NN.out.resume_K siblings, so the partial
# trajectory survives the resubmit.
python - <<PYEOF
import re
from pathlib import Path
log_dir = Path("${RUN_DIR}/logs")
for src in sorted(log_dir.glob("log_*.out")):
    if not re.fullmatch(r"log_\d+\.out", src.name):
        continue
    if src.stat().st_size == 0:
        continue  # nothing worth keeping
    base = src.name
    k = 0
    while (log_dir / f"{base}.resume_{k}").exists():
        k += 1
    src.rename(log_dir / f"{base}.resume_{k}")
    # Also rotate the paired .err.
    err = src.with_suffix(".err")
    if err.exists() and err.stat().st_size > 0:
        err.rename(log_dir / f"{base.replace('.out', '.err')}.resume_{k}")
PYEOF

echo "Task file (${RUN_DIR}/tasks):"
cat "${RUN_DIR}/tasks"
echo ""
echo "Submitting ${N_GPUS} tasks for group '${GROUP}' ..."

export TASK_FILE="${RUN_DIR}/tasks"
SBATCH_OUT=$(sbatch --ntasks="$N_GPUS" --job-name="$GROUP" "${REPO_DIR}/${SBATCH_SCRIPT}")
echo "${SBATCH_OUT}"
SLURM_JOB_ID=$(echo "${SBATCH_OUT}" | awk '{print $NF}')

# ── Manifest contract ─────────────────────────────────────────────────────────
# Every sweep submission writes meta.json next to the run logs. The notebook
# (and any other downstream analysis) consumes manifests, never raw directory
# listings. Untagged sweeps still produce a manifest — analysis code surfaces
# them as warnings rather than silent dropouts.
GIT_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY="false"
if ! git -C "${REPO_DIR}" diff-index --quiet HEAD 2>/dev/null; then
    GIT_DIRTY="true"
fi
SUBMITTED_AT=$(date -Iseconds)

python - <<PYEOF
import json, os, sys
from pathlib import Path

scope_raw = os.environ.get("SWEEP_SCOPE", "").strip()
scope = [s.strip() for s in scope_raw.split(",") if s.strip()] if scope_raw else []
manifest = {
    "group": "${GROUP}",
    "submitted_at": "${SUBMITTED_AT}",
    "slurm_job_id": "${SLURM_JOB_ID}",
    "n_gpus": int("${N_GPUS}"),
    "params_file": "$(basename "${PARAM_FILE}")",
    "sweep_script": "${SWEEP_SCRIPT}",
    "sbatch_script": "${SBATCH_SCRIPT}",
    "git_commit": "${GIT_COMMIT}",
    "git_dirty": ("${GIT_DIRTY}" == "true"),
    "scope": scope,
    "purpose": os.environ.get("SWEEP_PURPOSE", ""),
    # Default-tag the data pipeline version. Per-run cfg events carry the
    # authoritative value (set by --data_pipeline_version on train.py); this is just
    # a sweep-level hint for analysis filters. Override via env var if a
    # sweep deliberately mixes versions.
    "data_pipeline_version": os.environ.get(
        "SWEEP_DATA_PIPELINE_VERSION", "packed_v1",
    ),
}
out = Path("${RUN_DIR}") / "meta.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote manifest: {out}")
PYEOF
