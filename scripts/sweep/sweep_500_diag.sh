#!/bin/bash
# 500-step diagnostic-only run for fast cos_pre/cos_post probe.
lr=${1:-1e-3}
optimizer=${2:-adam-lin-lora}
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
    --max_steps 500 \
    --eval_every 500 \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
