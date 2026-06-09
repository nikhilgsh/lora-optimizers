#!/bin/bash
# FULL-POLAR (PolarExpress, PE=8) variant of sweep_curvature_whiten_r64_opc.sh.
# Identical config EXCEPT the polar step: --polar_method polar_express --muon_ns_steps 8
# realizes a TRUE full polar (all singular values → 1) instead of the ns=5 PARTIAL
# polar the base wrapper inherits from the train.py default (muon_ns_steps=5). At r64
# ns=5 leaves σ_min≈0.65 (mildly partial) — the low-rank arm of the full-vs-partial
# contrast against the r256 cell. ns=5 baseline: diag_shampoo_polar_r64_opc_blackwell.
#
# Positional args (must match params JSON key order):
#   1: lr   2: optimizer   3: seed   4: precond_delta   5: cw_picard_iters (opt)
#   6: curvature_beta (opt, default 0.99)
lr=${1:-3e-3}
optimizer=${2:-curvature-whiten-lora}
seed=${3:-0}
precond_delta=${4:-1e-3}
cw_picard_iters=${5:-${CW_PICARD_ITERS:-1}}
curvature_beta=${6:-0.99}

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
    --lora_r 64 \
    --lora_alpha 64 \
    --curvature_beta "$curvature_beta" \
    --precond_refresh_every 10 \
    --precond_delta "$precond_delta" \
    --polar_method "$polar_method" \
    --muon_ns_steps "$muon_ns_steps" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    --cw_picard_iters "$cw_picard_iters" \
    "${ckpt_args[@]}"
