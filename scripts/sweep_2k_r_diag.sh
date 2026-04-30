#!/bin/bash
# 2000-step sweep with configurable LoRA rank AND --log_optim_diagnostics.
# 5 positional args: lr, optimizer, lora_plus_multiplier, seed, lora_r.
# lora_alpha is set to lora_r so alpha/r = 1 (matches the r=16 baseline default).
lr=${1:-3e-4}
optimizer=${2:-adam-lin-lora}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}

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
    --log_optim_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
