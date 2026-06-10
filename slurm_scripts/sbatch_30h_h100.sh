#!/bin/bash
# Long-wall (30h) variant pinned to h100_pcie, for 8B-scale runs whose
# 9000-step wall-inclusive cost (~7.7 s/step → ~19h) needs the 1.5× contract
# buffer (~29h) and therefore exceeds the 24h tier. Identical to
# sbatch_24h_h100.sh except --time. h100_pcie needs no rocky9 reservation.
#SBATCH -p gpu
#SBATCH --gpus-per-task=h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --output=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.out
#SBATCH --error=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.err

cd /mnt/home/nghosh/lora

mkdir -p slurm_logs disbatch_logs

source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=lora-sweeps
export TOKENIZERS_PARALLELISM=false

module load disBatch
disBatch "$TASK_FILE" --prefix "disbatch_logs"
