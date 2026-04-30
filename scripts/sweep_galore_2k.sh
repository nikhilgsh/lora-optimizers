#!/bin/bash
# GaLore 2000-step sweep. Arg order must match params/galore_2k.json key order.
# Uses galore training_mode (unfreezes dense weights, periodic SVD projection).
lr=${1:-3e-4}
seed=${2:-0}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_32k \
    --device cuda \
    --bf16 \
    --max_steps 2000 \
    --eval_every 200 \
    --lr "$lr" \
    --training_mode galore \
    --optimizer galore-adamw \
    --lora_r 16 \
    --galore_update_proj_gap 200 \
    --galore_scale 0.25 \
    --seed "$seed" \
    "${wandb_args[@]}"
