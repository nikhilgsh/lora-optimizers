#!/bin/bash
# 4000-step single-pass sweep with diagnostics on the 70k packed_v1 Magicoder
# subset. Mirrors sweep_4k_diag.sh but pinned to packed_v1 (current default)
# + sdpa attention (flash_attention_2 incompatible with packed_v1 varlen).
# 4000 × 16 = 64000 samples ≤ 70000 train samples (single-pass guarded).
# eval_every 200 matches the 2k notebook step granularity.
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

# Diagnostics tiering:
#   - basic probes (norms, sat_frac, cond(S), gauge residual, lambda_dir_gain,
#     cross-coupling magnitudes): ON by default (~2% wall).
#   - heavy probes (chord_slack via SVD, higham accuracy, power-iter ratios,
#     Picard contraction): OFF by default. Set LOG_HEAVY_DIAGNOSTICS=1 to
#     enable for mechanism investigation. ~10x wall overhead at r=64.
# Backward compat: LOG_DIAGNOSTICS=0 also disables basic probes (legacy env
# var from pre-tiered defaults).
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
    "${diag_args[@]}" \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
