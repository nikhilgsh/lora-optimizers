#!/bin/bash
# Single-cell instrumented blowup capture: r=256 chord-tight k=3 lr=1e-2
# EXACTLY reproduces the failing config (no damping change, no solver
# change). The optimizer is now instrumented with two new event types
# that fire automatically:
#   - `non_finite_detected`  — fires at the START of any step where any
#     pair has non-finite entries in A / B / grad_A / grad_B; identifies
#     pair_name and which tensor, dumps prior-step's per-pair diagnostic.
#   - `non_finite_intermediate` — fires at the END of any step where any
#     intermediate in the chord-tight chain (u_A, SA^{-1/2}, X_A, P_A,
#     geo_A, picard_coeff, ρ, dA, ...) went non-finite. Per-pair global
#     index + pair_name per offending intermediate; pins down the
#     birth-site of the NaN.
# Plus the global `train_norms` event at train_loss_every cadence.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: lora_plus_multiplier  4: seed  5: lora_r
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-256}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_70k_packed \
    --data_pipeline_version packed_v1 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --max_steps "${MAX_STEPS:-500}" \
    --eval_every "${EVAL_EVERY:-100}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps 5 \
    --precond_method higham \
    --picard_iters_override 3 \
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    --train_loss_every 1 \
    "${wandb_args[@]}"
