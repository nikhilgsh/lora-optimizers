#!/bin/bash
# Single-cell diagnostic: r=256 chord-tight k=3 lr=1e-2 with
# --debug_higham_residual ON. Used to convert the strong circumstantial
# case for SB^{-1/2} blowup (cond_SB rising 384 → 2639 over 320 steps) into
# direct evidence: per-call ‖ZHZ − I‖_F + non_finite_Z flag from every
# _spd_inv_half invocation. First terminal NaN in the prior run was step
# ~800; first transient NaN at step 300. 500 steps covers precursor +
# first-NaN window with margin.
#
# Cadence diagnostics every 20 steps (not 80) to tighten the temporal
# alignment between optim_step state and higham_residual events.
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
    --debug_higham_residual \
    "${wandb_args[@]}"
