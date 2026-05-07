#!/bin/bash
# 2000-step sweep over picard_iters at α=1.
# 6 positional args: lr, lora_plus_multiplier, seed, lora_r, picard_iters_override, optimizer.
# Optimizer fixed to adam-polar-product-lora-coupled; picard_iters_override
# replaces the hardcoded 2 with sweep-controlled value. picard_iters=1 reduces
# to block-diagonal; picard_iters≥2 applies the cross-coupling correction.
lr=${1:-3e-4}
lora_plus_multiplier=${2:-1.0}
seed=${3:-0}
lora_r=${4:-16}
picard_iters_override=${5:-2}
optimizer=${6:-adam-polar-product-lora-coupled}

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
    --picard_iters_override "$picard_iters_override" \
    --log_optim_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
