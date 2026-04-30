#!/bin/bash
# Usage: ./slurm_scripts/submit.sh <params_json> <group_name> <n_gpus>
# Example: ./slurm_scripts/submit.sh params/lr_sweep.json sweep_lr 4
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARAM_FILE="$1"
GROUP="$2"
N_GPUS="$3"

RUN_DIR="${REPO_DIR}/logs/${GROUP}/run_info"
mkdir -p "${RUN_DIR}/logs" "${REPO_DIR}/slurm_logs" "${REPO_DIR}/disbatch_logs"

cp "${REPO_DIR}/scripts/sweep.sh" "${RUN_DIR}/sweep.sh"
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
sbatch --ntasks="$N_GPUS" --job-name="$GROUP" "${REPO_DIR}/slurm_scripts/sbatch.sh"
