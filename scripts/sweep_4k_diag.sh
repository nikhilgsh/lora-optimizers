#!/bin/bash
# 4000-step sweep with diagnostics, single-pass on the 70k Magicoder subset.
# Replaces sweep_8k_diag.sh, which silently ran ~3.5 epochs on the 32k
# subset (invalidated 2026-05-04). The dataset is sized so 4000 × 16 = 64000
# samples ≤ 70000 train samples (~0.91 epoch, single-pass guard satisfied).
# eval_every fixed at 200 so step granularity matches the 2k notebook.
#
# Positional args (must match params JSON key order):
#   1: lr (default 3e-4)
#   2: optimizer (default adam-polar-product-lora-coupled)
#   3: lora_plus_multiplier (default 1.0)
#   4: seed (default 0)
#   5: lora_r (default 16)
#   6: muon_ns_steps (default 5)
lr=${1:-3e-4}
optimizer=${2:-adam-polar-product-lora-coupled}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}
muon_ns_steps=${6:-5}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_70k \
    --device cuda \
    --bf16 \
    --max_steps "${MAX_STEPS:-4000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps "$muon_ns_steps" \
    --log_optim_diagnostics \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
