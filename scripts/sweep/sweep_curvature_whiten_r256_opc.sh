#!/bin/bash
# Two-sided curvature-whitened momentum (curvature-whiten-lora = no polar;
# curvature-whiten-polar-lora = + Newton-Schulz orthogonalize) at r=256 on
# OLMo-2-1B opc, long-horizon (9000 steps). Matches the phase-L curvature A/B
# data config EXACTLY (pre-packed seq2048 cache, bs4×ga4, eval_every 250) so
# eval_loss is directly comparable to the chord-tight curvature baselines in
# notebooks/olmo2_1b_opc_leaderboard.ipynb — only the optimizer differs.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer (curvature-whiten-lora | curvature-whiten-polar-lora)
#   3: seed
lr=${1:-3e-3}
optimizer=${2:-curvature-whiten-lora}
seed=${3:-0}

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
    --precond_refresh_every 8 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
