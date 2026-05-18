#!/bin/bash
# Validation sweep for gram-NS + k=2 default change (commit 766c760).
# Confirms new code doesn't drift from established trajectories.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: lora_r
#   6: ns_form              ← new
#   7: picard_iters_override ← validates k=2
#
# Pass criterion: |Δeval_loss| ≤ 2·σ_AdamW per checkpoint vs rect baseline
# at the same (lr, r, seed). σ_AdamW = 0.0014 at r=64, 0.0017 at r=256.
lr=${1:-3e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight-clean}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-64}
ns_form=${6:-rect}
picard_iters_override=${7:-2}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

diag_args=(--log_basic_diagnostics)
if [ "${LOG_DIAGNOSTICS:-1}" = "0" ]; then
    diag_args=(--no-log_basic_diagnostics)
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_70k_packed \
    --data_pipeline_version packed_v1 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --max_steps "${MAX_STEPS:-4000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps 5 \
    --precond_method higham \
    --precond_delta_relative \
    --precond_delta 1e-2 \
    --picard_iters_override "$picard_iters_override" \
    --ns_form "$ns_form" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
