#!/bin/bash
# Long-wall (24h) variant pinned to h100_pcie. Use when reproducing a
# failure mode that depends on a specific GPU architecture's float-
# nondeterminism (e.g. the r=256 chord-tight k=3 NaN reproduced only on
# h100_pcie nodes; A100 nodes ran clean with the same code/seed).
#
# Other than the GPU constraint, identical to sbatch_24h.sh.
#SBATCH -p gpu
#SBATCH --gpus-per-task=h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
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
