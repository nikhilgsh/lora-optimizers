#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.out
#SBATCH --error=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.err

cd /mnt/home/nghosh/lora

mkdir -p slurm_logs disbatch_logs

# Conda FIRST, strict mode after (conda activate references unbound vars)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=lora-sweeps
export TOKENIZERS_PARALLELISM=false

module load disBatch
disBatch "$TASK_FILE" --prefix "disbatch_logs"
