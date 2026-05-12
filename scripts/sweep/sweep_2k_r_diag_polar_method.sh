#!/bin/bash
# 2000-step sweep with --polar_method as 6th positional. Tests PolarExpress
# (arXiv:2505.16932) and DeepSeek-V4-hybrid against the standard Newton-Schulz
# baseline as the polar approximation method inside the polar pipeline.
lr=${1:-3e-4}
optimizer=${2:-adam-polar-product-lora}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}
polar_method=${6:-ns}

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
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
