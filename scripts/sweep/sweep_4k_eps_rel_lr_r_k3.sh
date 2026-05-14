#!/bin/bash
# chord-tight (whiten) k=3, 4k-step, ε_rel × lr × r sweep.
# precond_delta is positional (last arg) so disBatch can vary it.
# `--precond_delta_relative` is always set — this sweep is damped-only
# (no default-δ control; that's already on the leaderboard).
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: lora_r
#   6: precond_delta (ε_rel)
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-64}
precond_delta=${6:-1e-2}

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
    --picard_iters_override 3 \
    --precond_delta_relative \
    --precond_delta "$precond_delta" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
