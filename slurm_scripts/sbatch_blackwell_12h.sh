#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=rtxblackwell
#SBATCH --reservation=rocky9
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.out
#SBATCH --error=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.err

# Blackwell RTX PRO 6000, 12h wall (vs sbatch_blackwell.sh's 8h) for the
# 9000-step full-polar sweeps: 9000 × ~2.5 s/step × 1 task/gpu × 1.5 ≈ 9.4h,
# over the 8h template. Blackwell nodes are under SLURM reservation `rocky9`
# (per ~/.claude/CLAUDE.md); without it jobs sit PD with ReqNodeNotAvail.

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
