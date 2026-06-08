#!/bin/bash
#SBATCH -p gpuxl
#SBATCH --constraint=h200
#SBATCH --reservation=rocky9
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.out
#SBATCH --error=/mnt/home/nghosh/lora/slurm_logs/slurm_%j.err

# gpuxl H200 (workergpu[301-324]). Separate QoS from `-p gpu` — gpuxl cap is
# gres/gpu=64 vs gpu's 24, so jobs here don't count against the saturated gpu
# cap (use to dodge QOSMaxGRESPerUser). rocky9 reservation is REQUIRED (it
# covers the gpuxl h100+h200 pools). gpuxl enforces a 4-GPU MINIMUM per job
# (QOSMinGRES) — never submit <4 GPUs here.
#
# NOTE: H200 ≠ the project's Blackwell comparison hardware. Loss / lr-sensitivity
# (curve shape, ranking) transfer fine; per-step TIMING does NOT — do not compare
# tok/s or wall against Blackwell runs.

cd /mnt/home/nghosh/lora
mkdir -p slurm_logs disbatch_logs

source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -eo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_PROJECT=lora-sweeps
export TOKENIZERS_PARALLELISM=false

module load disBatch
disBatch "$TASK_FILE" --prefix "disbatch_logs"
