#!/bin/bash
# Usage: ./slurm_scripts/submit.sh <params_json> <group_name> <n_gpus> [sweep_script] [sbatch_script]
# Example: ./slurm_scripts/submit.sh params/lr_sweep.json sweep_lr 4
# Example: ./slurm_scripts/submit.sh params/foo.json my_group 6 scripts/sweep/sweep_2k.sh slurm_scripts/sbatch_h100.sh
#
# Optional env vars (recommended — consumed by analysis tooling):
#   SWEEP_SCOPE="ext_compare,polar_family"   comma-separated tags
#   SWEEP_PURPOSE="E2: AdaMuon-faithful + polar-product geometry"
#   SWEEP_LOGS_ROOT=/path/to/canonical/logs   shared live catalog (optional)
#
# To exclude an old sweep from analysis, delete its log dir.
set -euo pipefail

# Scope is useful audit metadata, but it is not part of run identity and may
# not block a valid experiment. Missing scope remains visible in catalog audit
# issues and strict manifest checks.
if [[ -z "${SWEEP_SCOPE:-}" ]]; then
    echo "WARN: SWEEP_SCOPE is empty; recording an unscoped audit annotation." >&2
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOGS_ROOT="${SWEEP_LOGS_ROOT:-${REPO_DIR}/logs}"

# --emit-pending: do everything EXCEPT the final `sbatch` — write a self-contained
# pending sbatch to slurm_pending/ instead (for when org policy bars running sbatch).
# The task file is generated + validated in-session here; the pending sbatch just
# disBatches it. Needs TIMING_S_PER_STEP=<wall-inclusive s/step> for the wall buffer.
if [[ "${1:-}" == "--emit-pending" ]]; then
    EMIT_PENDING=1
    shift
fi

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
    if ! python "${REPO_DIR}/scripts/analysis/audit_sweep_overlap.py" "${PARAM_FILE}" --logs-root "${LOGS_ROOT}" --sweep-script "${REPO_DIR}/${SWEEP_SCRIPT}"; then
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

RUN_DIR="${LOGS_ROOT}/${GROUP}/run_info"
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
# Inject per-task checkpoint plumbing plus a fresh explicit attempt ID. The
# checkpoint identity is stable across resubmissions of the same task; the
# attempt ID is new on every invocation. train.py records the actual parent
# only after a checkpoint load succeeds, so an empty first-run directory never
# masquerades as a continuation. Opt out with NO_CHECKPOINTS=1.
if [[ -z "${NO_CHECKPOINTS:-}" ]]; then
    python "${REPO_DIR}/lora_playground/submission.py" inject-task-metadata \
        --tasks "${RUN_DIR}/tasks" \
        --checkpoint-root "${RUN_DIR}/checkpoints" \
        --group "${GROUP}"
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
# physical segments remain separate. New runs carry explicit attempt/checkpoint
# metadata so the lineage layer can validate and merge only a real resume.
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

export TASK_FILE="${RUN_DIR}/tasks"

# ── Optional manifest annotation ─────────────────────────────────────────────
# Write a complete annotation BEFORE submission so a fast-starting task never
# races a half-written/missing meta.json. Physical logs remain discoverable if
# this annotation is later lost; it is audit metadata, not an admission gate.
GIT_COMMIT=$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY="false"
if ! git -C "${REPO_DIR}" diff-index --quiet HEAD 2>/dev/null; then
    GIT_DIRTY="true"
fi
SUBMITTED_AT=$(date -Iseconds)

write_submission_manifest() {
    local job_id="$1"
    local dirty_args=()
    if [[ "${GIT_DIRTY}" == "true" ]]; then
        dirty_args+=(--git-dirty)
    fi
    PYTHONPATH="${REPO_DIR}" python -m lora_playground.manifest write-submission \
        --path "${RUN_DIR}/meta.json" \
        --group "${GROUP}" \
        --submitted-at "${SUBMITTED_AT}" \
        --slurm-job-id "${job_id}" \
        --n-gpus "${N_GPUS}" \
        --params-file "$(basename "${PARAM_FILE}")" \
        --sweep-script "${SWEEP_SCRIPT}" \
        --sbatch-script "${SBATCH_SCRIPT}" \
        --git-commit "${GIT_COMMIT}" \
        "${dirty_args[@]}" \
        --scope "${SWEEP_SCOPE:-}" \
        --purpose "${SWEEP_PURPOSE:-}" \
        --data-pipeline-version "${SWEEP_DATA_PIPELINE_VERSION:-packed_v1}"
}

write_submission_manifest "pending"

if [[ -n "${EMIT_PENDING:-}" ]]; then
    # ── Emit a self-contained pending sbatch instead of calling sbatch ───────
    PENDING_DIR="${PENDING_DIR:-${REPO_DIR}/slurm_pending}"
    CELLS=$(grep -cve '^[[:space:]]*$' "${RUN_DIR}/tasks")
    TASKS_PER_GPU=$(( (CELLS + N_GPUS - 1) / N_GPUS ))
    MAX_STEPS=$(grep -oE 'MAX_STEPS:-[0-9]+' "${RUN_DIR}/sweep.sh" | grep -oE '[0-9]+' | head -1 || true)
    : "${MAX_STEPS:=0}"
    if [[ -z "${TIMING_S_PER_STEP:-}" ]]; then
        echo "EMIT_PENDING: set TIMING_S_PER_STEP=<wall-inclusive s/step> (measure via" >&2
        echo "  'python -m lora_playground.timing record <log> --hardware <hw>' or reuse a" >&2
        echo "  same-class run's value). Needed for the wall-buffer header." >&2
        exit 1
    fi
    # Validate every generated task line is well-formed shell BEFORE it can reach
    # the cluster — catches the generate_task_file nested-default mangle (an
    # unclosed ${...} that swallows the redirect) that otherwise dies silently.
    while IFS= read -r _ln; do
        [[ -z "${_ln// }" ]] && continue
        if ! bash -n -c "$_ln" 2>/dev/null; then
            echo "EMIT_PENDING: a generated task line is not valid shell (swallowed" >&2
            echo "  redirect / unclosed brace — check the wrapper's positional defaults):" >&2
            echo "  ${_ln}" >&2
            exit 1
        fi
    done < "${RUN_DIR}/tasks"
    REQ_TIME_S=$(python3 -c "import math;print(int(math.ceil(${MAX_STEPS}*${TIMING_S_PER_STEP}*${TASKS_PER_GPU}*1.5)))")
    ETA_H=$(python3 -c "print(f'{${MAX_STEPS}*${TIMING_S_PER_STEP}*${TASKS_PER_GPU}/3600:.2f}')")
    TMPL_TIME=$(grep -oE -- '--time[= ][0-9:-]+' "${REPO_DIR}/${SBATCH_SCRIPT}" | grep -oE '[0-9:-]+$' | head -1 || true)
    TMPL_TIME_S=$(python3 -c "
s='${TMPL_TIME:-0}'.strip(); d=0
if '-' in s: d,s=s.split('-',1); d=int(d)
p=[int(x) for x in s.split(':')] if s else [0]
h,m,sec = (p+[0,0,0])[:3] if len(p)==3 else ((p[0],p[1],0) if len(p)==2 else (0,p[0],0))
print(d*86400+h*3600+m*60+sec)")
    if (( MAX_STEPS > 0 && TMPL_TIME_S < REQ_TIME_S )); then
        echo "EMIT_PENDING: template --time (${TMPL_TIME}=${TMPL_TIME_S}s) < required ${REQ_TIME_S}s" >&2
        echo "  (MAX_STEPS ${MAX_STEPS} × ${TIMING_S_PER_STEP} s/step × ${TASKS_PER_GPU} tasks/gpu × 1.5)." >&2
        echo "  Pass a longer sbatch template as arg 5." >&2
        exit 1
    fi
    mkdir -p "${PENDING_DIR}"
    PEND="${PENDING_DIR}/${GROUP}.sbatch"
    {
        grep -E '^#!' "${REPO_DIR}/${SBATCH_SCRIPT}" | head -1
        grep -E '^#SBATCH' "${REPO_DIR}/${SBATCH_SCRIPT}" | grep -vE -- '--output|--error|--ntasks|--job-name'
        echo "#SBATCH --ntasks=${N_GPUS}"
        echo "#SBATCH --job-name=${GROUP}"
        echo "#SBATCH --output=${REPO_DIR}/slurm_logs/slurm_%j.out"
        echo "#SBATCH --error=${REPO_DIR}/slurm_logs/slurm_%j.err"
        echo "# ETA: ${ETA_H}"
        echo "# CELLS: ${CELLS}"
        echo "# MAX_STEPS: ${MAX_STEPS}"
        echo "# TASKS_PER_GPU: ${TASKS_PER_GPU}"
        echo "# TIMING_MEASURED: ${TIMING_S_PER_STEP} s/step (${TIMING_SOURCE:-submit.sh --emit-pending})"
        echo "# Emitted by submit.sh --emit-pending: task file pre-generated in-session."
    } > "$PEND"
    cat >> "$PEND" <<PENDING_EOF

cd ${REPO_DIR}
mkdir -p slurm_logs disbatch_logs
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -eo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_PROJECT=lora-sweeps
export TOKENIZERS_PARALLELISM=false
export TASK_FILE="${RUN_DIR}/tasks"
module load disBatch
disBatch "\$TASK_FILE" --prefix "disbatch_logs"
PENDING_EOF
    echo "EMIT_PENDING: wrote ${PEND}"
    echo "  CELLS=${CELLS} ntasks=${N_GPUS} TASKS_PER_GPU=${TASKS_PER_GPU} ETA=${ETA_H}h (task lines validated)"
    SLURM_JOB_ID="pending"
else
    echo "Submitting ${N_GPUS} tasks for group '${GROUP}' ..."
    SBATCH_OUT=$(sbatch --ntasks="$N_GPUS" --job-name="$GROUP" "${REPO_DIR}/${SBATCH_SCRIPT}")
    echo "${SBATCH_OUT}"
    SLURM_JOB_ID=$(echo "${SBATCH_OUT}" | awk '{print $NF}')
fi

if [[ "${SLURM_JOB_ID}" != "pending" ]]; then
    write_submission_manifest "${SLURM_JOB_ID}"
fi
