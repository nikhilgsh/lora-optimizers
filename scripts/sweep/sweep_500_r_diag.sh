#!/bin/bash
# 500-step diagnostic-probe variant of sweep_2k_r_diag.sh.
# Stage-0 readout for the chord-tight diagnostics (plan
# there-are-a-few-indexed-hickey): 500 steps × probe-every-20 gives 25
# probe samples — enough for stable medians on chord_slack /
# lambda_dir_gain / cos_polar_clip_tight / sat_frac_tight /
# adam_gauge_residual.
# 5 positional args: lr, optimizer, lora_plus_multiplier, seed, lora_r.
# lora_alpha is set to lora_r so alpha/r = 1 (matches the r=16 baseline default).
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_32k_packed \
    --data_pipeline_version packed_v1 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    --max_steps 500 \
    --eval_every 100 \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
