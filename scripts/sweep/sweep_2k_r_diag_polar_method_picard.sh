#!/bin/bash
# 2000-step sweep with both --polar_method (6th positional) and
# --picard_iters_override (7th positional). Lets a single sweep config exercise
# the polar approximation method × Picard k-iteration grid.
lr=${1:-3e-4}
optimizer=${2:-adam-polar-product-lora}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}
polar_method=${6:-ns}
picard_iters_override=${7:-3}

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
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --polar_method "$polar_method" \
    --picard_iters_override "$picard_iters_override" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
