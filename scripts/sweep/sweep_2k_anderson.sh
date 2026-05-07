#!/bin/bash
# 2000-step sweep over (picard_iters, anderson_m) at α=1.
# 6 positional args: lr, lora_plus_multiplier, seed, lora_r, picard_iters_override, anderson_m.
# Optimizer fixed to adam-polar-product-lora-coupled. anderson_m=0 = plain Picard.
lr=${1:-3e-4}
lora_plus_multiplier=${2:-1.0}
seed=${3:-0}
lora_r=${4:-16}
picard_iters_override=${5:-16}
anderson_m=${6:-0}

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
    --optimizer adam-polar-product-lora-coupled \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --picard_iters_override "$picard_iters_override" \
    --anderson_m "$anderson_m" \
    --log_optim_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
