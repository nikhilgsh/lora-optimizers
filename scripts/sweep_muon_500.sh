#!/bin/bash
# 500-step pilot variant: fast η-ranking signal (~25 min/run on h100_pcie).
# Arg order matches params/muon_*_2k.json keys: lr, optimizer, lora_plus_multiplier, muon_ns_steps, seed.
lr=${1:-3e-4}
optimizer=${2:-muon-lora}
lora_plus_multiplier=${3:-1.0}
muon_ns_steps=${4:-5}
seed=${5:-0}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_32k \
    --device cuda \
    --bf16 \
    --max_steps 500 \
    --eval_every 100 \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --muon_ns_steps "$muon_ns_steps" \
    --seed "$seed" \
    "${wandb_args[@]}"
