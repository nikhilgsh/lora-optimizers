#!/bin/bash
# Positional-arg wrapper for generate_task_file.py.
# Arg order must match the JSON param key order in params/*.json.
# cwd is set to repo root by slurm_scripts/sbatch.sh before disBatch runs.
lr=${1:-3e-4}
optimizer=${2:-adamw}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}

python train_lora.py \
    --data_dir data/magicoder_seq512 \
    --device cuda \
    --bf16 \
    --max_steps 500 \
    --eval_every 100 \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed"
