#!/bin/bash
# Usage: ./slurm_scripts/submit.sh <params_json> <group_name> <n_gpus> [sweep_script] [sbatch_script]
# Example: ./slurm_scripts/submit.sh params/lr_sweep.json sweep_lr 4
# Example: ./slurm_scripts/submit.sh params/foo.json my_group 6 scripts/sweep_2k.sh slurm_scripts/sbatch_h100.sh
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
    echo "              winner_rerun, pilot, legacy" >&2
    echo "" >&2
    echo "See lora_playground/manifest.py for the full schema." >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARAM_FILE="$1"
GROUP="$2"
N_GPUS="$3"
SWEEP_SCRIPT="${4:-scripts/sweep.sh}"
SBATCH_SCRIPT="${5:-slurm_scripts/sbatch.sh}"

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
}
out = Path("${RUN_DIR}") / "meta.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote manifest: {out}")
PYEOF
