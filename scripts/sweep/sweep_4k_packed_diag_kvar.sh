#!/bin/bash
# 4000-step single-pass sweep with diagnostics + explicit picard_iters_override.
# Used for the post-sticky-zero-fix rerun at r=64 (chord-tight whiten/no-whiten/
# direction × k ∈ {1, 3}). Mirrors sweep_4k_packed_diag.sh but adds an explicit
# 6th positional arg for picard_iters_override so a single sweep can mix k=1
# and k=3 cells without relying on the build_optimizer alias default.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: lora_r
#   6: picard_iters_override (1 or 3)
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-64}
picard_iters_override=${6:-1}

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
    --picard_iters_override "$picard_iters_override" \
    --muon_ns_steps 5 \
    --precond_method higham \
    "${diag_args[@]}" \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
