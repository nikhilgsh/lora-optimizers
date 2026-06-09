#!/bin/bash
# NESTEROV ABLATION of the r256 opc PE=8 protagonist: identical to
# sweep_curvature_whiten_r256_opc_pe8.sh EXCEPT adds --cw_nesterov (Muon-style
# realizes a TRUE full polar (σ→1) instead of the ns=5 PARTIAL polar the base wrapper
# inherits from train.py default (muon_ns_steps=5). High-rank arm of the SAME-DATASET
# (opc) rank contrast vs sweep_curvature_whiten_r64_opc_pe8.sh; baseline launched
# alongside as diag_shampoo_polar_r256_opc_blackwell (ns=5).
#
# Positional args (must match params JSON key order):
#   1: lr   2: optimizer   3: seed   4: precond_delta
lr=${1:-3e-3}
optimizer=${2:-curvature-whiten-lora}
seed=${3:-0}
precond_delta=${4:-1e-3}

# Full-polar knobs — PE=8 by default; overridable for ablation.
polar_method=${POLAR_METHOD:-polar_express}
muon_ns_steps=${MUON_NS_STEPS:-8}

compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

diag_args=(--log_basic_diagnostics)
if [ "${LOG_DIAGNOSTICS:-1}" = "0" ]; then
    diag_args=(--no-log_basic_diagnostics)
fi

ckpt_args=()
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    ckpt_args=(
        --checkpoint_dir "$CHECKPOINT_DIR"
        --resume_from "$CHECKPOINT_DIR"
        --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST:-2}"
    )
    if [ -n "${CHECKPOINT_EVERY:-}" ]; then
        ckpt_args+=(--checkpoint_every "$CHECKPOINT_EVERY")
    fi
fi

python train_lora.py \
    --data_dir data/opc_sft_stage2_all_packed_seq2048 \
    --data_pipeline_version "${DATA_PIPELINE_VERSION:-packed_v1.1}" \
    --max_seq_length 2048 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --grad_accum_steps "${GRAD_ACCUM:-4}" \
    --max_steps "${MAX_STEPS:-9000}" \
    --eval_every "${EVAL_EVERY:-250}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --seed "$seed" \
    --lora_r 256 \
    --lora_alpha 256 \
    --curvature_beta 0.99 \
    --precond_refresh_every 10 \
    --precond_delta "$precond_delta" \
    --polar_method "$polar_method" \
    --muon_ns_steps "$muon_ns_steps" \
    --cw_nesterov \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
