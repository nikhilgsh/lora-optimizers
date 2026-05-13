#!/bin/bash
# Short ranking-selection pilot for the init × damping comparison.
# 600 steps + eval_every=50 + diagnostics on, packed_v1, chord-tight whiten k=1.
# Predictions from docs/notes/polar_product/init_damping_math.md (C2, C3):
#   - Init[AB] should show faster early descent vs Init[A] at r=256
#   - Init[A] + σ_max-relative damping should approximate Init[AB]
# This is a pilot for SELECTION (which cells warrant a 4k follow-up),
# NOT for measurement.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: lora_r
#   6: lora_init_b           (zero/gaussian/symmetric)
#   7: precond_delta_relative (0/1)
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-256}
lora_init_b=${6:-zero}
precond_delta_relative=${7:-0}

compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

damp_args=()
if [ "$precond_delta_relative" = "1" ]; then
    damp_args=(--precond_delta_relative)
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_70k_packed \
    --data_pipeline_version packed_v1 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --max_steps "${MAX_STEPS:-600}" \
    --eval_every "${EVAL_EVERY:-50}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --lora_init_b "$lora_init_b" \
    "${damp_args[@]}" \
    --muon_ns_steps 5 \
    --precond_method higham \
    --log_basic_diagnostics \
    --optim_diagnostics_every 50
