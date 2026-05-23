#!/bin/bash
# Mirrors sweep_4k_packed_diag_k3.sh but sets --picard_iters_override 2.
# Used for the full-FW Frank-Wolfe variant of chord-tight-clean (§6 of
# docs/notes/polar_product/algorithm_tight_chord.md), which is bit-identical
# to anchored at k=1 and only diverges at k>=2.
#
# Positional args (must match params JSON key order):
#   1: lr (default 3e-3)
#   2: optimizer (default adam-polar-product-lora-coupled-spectral-chord-tight)
#   3: lora_plus_multiplier (default 1.0)
#   4: seed (default 0)
#   5: lora_r (default 16)
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}

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
if [ "${LOG_HEAVY_DIAGNOSTICS:-0}" = "1" ]; then
    diag_args+=(--log_heavy_diagnostics)
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
    --picard_iters_override 2 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
