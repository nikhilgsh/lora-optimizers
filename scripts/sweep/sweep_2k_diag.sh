#!/bin/bash
# 2000-step sweep with --log_optim_diagnostics enabled (H1 probe).
# Same arg order as sweep_2k.sh so it's drop-in compatible with existing param JSON.
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
    --max_steps "${MAX_STEPS:-2000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --log_optim_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
