#!/bin/bash
# UCV (orthogonal-core LoRA) η-ranking pilot launcher: 500 steps, eval_every=100.
# Pilots are for ranking-selection only (per project rule); not for measurement.
# 4 positional args: lr, seed, lora_r, muon_ns_steps. lora_alpha = lora_r so α/r = 1.
lr=${1:-3e-4}
seed=${2:-0}
lora_r=${3:-16}
muon_ns_steps=${4:-5}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_32k \
    --device cuda \
    --bf16 \
    --training_mode ucv \
    --optimizer adam-ucv-core-lora \
    --max_steps 500 \
    --eval_every 100 \
    --lr "$lr" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps "$muon_ns_steps" \
    "${wandb_args[@]}"
