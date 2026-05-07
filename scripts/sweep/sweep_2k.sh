#!/bin/bash
# 2000-step variant of sweep.sh — for testing whether longer training changes optimizer rankings.
# Arg order must match the JSON param key order in params/*.json.
# cwd is set to repo root by slurm_scripts/sbatch.sh before disBatch runs.
lr=${1:-3e-4}
optimizer=${2:-adamw}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_32k \
    --device cuda \
    --bf16 \
    --max_steps "${MAX_STEPS:-2000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    "${wandb_args[@]}"
