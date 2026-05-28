#!/bin/bash
# Blackwell long-wall variant for r>=128 diagnostic reruns.
# TIMING_MEASURED: logs/chord_tight_robustness_llama32_1b_opc_r256_blackwell
#   log_1.out/log_2.out reached 9000 steps in ~3.62 h per task.
# TIMING_BASIS: same model/data/rank/optimizer/diagnostic cadence; chord_slack
#   now uses power iteration in the basic tier.
#SBATCH -p gpu
#SBATCH --constraint=rtxblackwell
#SBATCH --reservation=rocky9
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.out
#SBATCH --error=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.err

cd /mnt/home/nghosh/lora
mkdir -p slurm_logs disbatch_logs

source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_PROJECT=lora-sweeps
export TOKENIZERS_PARALLELISM=false

module load disBatch
disBatch "$TASK_FILE" --prefix "disbatch_logs"
