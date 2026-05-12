#!/bin/bash
# 400-step debug smoke for variant 1 bound-violation investigation.
# Same config as the 4k Blackwell head-to-head, but short + probe-every-20
# to catch where chord_slack crosses 1 with the SVD-direct cross-check.
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-direction}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-64}

python train_lora.py \
    --data_dir data/magicoder_seq512_70k_packed \
    --data_pipeline_version packed_v1 \
    --attn_implementation sdpa \
    --device cuda --bf16 \
    --max_steps 400 \
    --eval_every 400 \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps 5 \
    --precond_method higham \
    --log_optim_diagnostics \
    --optim_diagnostics_every 20
