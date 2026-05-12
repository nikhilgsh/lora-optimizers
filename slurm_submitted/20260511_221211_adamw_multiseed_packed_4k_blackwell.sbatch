#!/bin/bash
# ETA: 1.5
# Multi-seed AdamW noise floor at packed_v1 / 4k canonical horizon.
# Seeds {1,2,3,4} × η=3e-4 × r∈{16,64} = 8 cells, all-parallel on 8 GPUs.
# Scope: diagnostics. Purpose: establish σ-anchor for chord/polar Δ claims.
#
# Self-contained: pre-sbatch setup (audit, task-file generation, file copies)
# already ran. This sbatch only kicks disBatch and writes the manifest.
#
#SBATCH -p gpu
#SBATCH --constraint=rtxblackwell
#SBATCH --reservation=rocky9
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=8
#SBATCH --job-name=adamw_multiseed_packed_4k_blackwell
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.out
#SBATCH --error=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.err

cd /mnt/home/nghosh/lora
mkdir -p slurm_logs disbatch_logs

source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=lora-sweeps
export TOKENIZERS_PARALLELISM=false

GROUP=adamw_multiseed_packed_4k_blackwell
RUN_DIR=/mnt/home/nghosh/lora/logs/${GROUP}/run_info
export TASK_FILE="${RUN_DIR}/tasks"

# Manifest contract (normally written by slurm_scripts/submit.sh after the
# sbatch call; written here at job start since this sbatch was queued via
# submit-pending rather than submit.sh).
GIT_COMMIT=$(git -C /mnt/home/nghosh/lora rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY="false"
if ! git -C /mnt/home/nghosh/lora diff-index --quiet HEAD 2>/dev/null; then
    GIT_DIRTY="true"
fi
SUBMITTED_AT=$(date -Iseconds)

SUBMITTED_AT="$SUBMITTED_AT" GIT_COMMIT="$GIT_COMMIT" GIT_DIRTY="$GIT_DIRTY" \
SLURM_JOB_ID="${SLURM_JOB_ID:-unknown}" GROUP="$GROUP" RUN_DIR="$RUN_DIR" \
python - <<'PYEOF'
import json, os
from pathlib import Path
manifest = {
    "group": os.environ["GROUP"],
    "submitted_at": os.environ["SUBMITTED_AT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "n_gpus": 8,
    "params_file": "adamw_multiseed_packed_4k.json",
    "sweep_script": "scripts/sweep/sweep_4k_packed_diag.sh",
    "sbatch_script": "slurm_pending/adamw_multiseed_packed_4k_blackwell.sbatch",
    "git_commit": os.environ["GIT_COMMIT"],
    "git_dirty": (os.environ["GIT_DIRTY"] == "true"),
    "scope": ["diagnostics"],
    "purpose": ("multi-seed AdamW noise floor under packed_v1 at canonical "
                "4k horizon — σ-anchor for chord/polar single-seed Δ claims"),
    "data_pipeline_version": "packed_v1",
}
out = Path(os.environ["RUN_DIR"]) / "meta.json"
out.write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote manifest: {out}")
PYEOF

module load disBatch
disBatch "$TASK_FILE" --prefix "disbatch_logs"
