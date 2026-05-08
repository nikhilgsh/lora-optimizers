#!/bin/bash
# 4-GPU single-node DDP launcher.
#
# Wraps `train_lora.py "$@"` with torchrun so all four CUDA devices on the
# allocated node form a single DistributedDataParallel process group. Use for
# Phase B/C campaign cells where a single 8B/3B run otherwise overflows the
# 24h SLURM wall (per A0.7 in tight_chord_paper_plan.md).
#
# Usage:
#   sbatch slurm_scripts/sbatch_4gpu_ddp.sh \
#       --model_name meta-llama/Meta-Llama-3-8B \
#       --optimizer adam-polar-product-lora-coupled-spectral-chord-tight \
#       --lora_r 256 --lora_alpha 256 \
#       --max_steps 6000 --eval_every 200 \
#       --bf16 --compile \
#       --global_batch_size 32 --batch_size 1 \
#       --data_dir data/magicoder_evol110k_seq2048 \
#       --lr 1e-2 ...
#
# Note: --batch_size is the per-rank micro-batch (sized to fit GPU memory);
# torchrun launches one process per GPU; --global_batch_size is divided
# automatically into per-rank grad_accum.
#
# Pin to A100-80G — Phase A0.1 measured that 8B r=256 + compile OOMs on the
# 40GB SKU. (Generic --constraint=a100 routes to either 40GB or 80GB.)
#
#SBATCH --job-name=lora_ddp4
#SBATCH --partition=gpu
#SBATCH --constraint=a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/home/nghosh/lora/slurm_logs/lora_ddp4_%j.out
#SBATCH --error=/mnt/home/nghosh/lora/slurm_logs/lora_ddp4_%j.err

# Conda first (its activate scripts reference unbound vars), strict mode after.
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/home/nghosh/lora
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"

cd /mnt/home/nghosh/lora
mkdir -p slurm_logs

# torchrun handles RANK / WORLD_SIZE / LOCAL_RANK env vars and process spawning.
# rdzv_endpoint=localhost:0 picks a free port; safe for single-node.
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train_lora.py "$@"
