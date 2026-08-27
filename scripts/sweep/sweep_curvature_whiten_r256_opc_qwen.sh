#!/bin/bash
# Qwen2.5-1.5B variant of sweep_curvature_whiten_r256_opc.sh — the SOAP-curvature /
# KL-Shampoo "better conditioning" family at r=256 on Qwen opc, long-horizon (9000).
# This is the CONDITIONING TEST: on Qwen opc, chord-tight only ties AdamW (Δ≈-0.001).
# If a better-conditioned algo (curvature-whiten-polar / kl-shampoo-polar) BEATS AdamW
# here, the weak advantage is a conditioning problem; if it also ties, it's
# task/headroom (Qwen barely moves). Matches the Qwen adamw/chord-tight cell config
# EXACTLY (model, qwen25 pre-packed seq2048 cache, bs4xga4, eval_every 250, r256,
# alpha=r) so eval_loss is directly comparable in notebooks/qwen25_1p5b_opc_leaderboard.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer (curvature-whiten[-polar]-lora | kl-shampoo[-polar]-lora)
#   3: seed
#   4: precond_delta  (relative damping delta for the inverse-sqrts; the critical HP.
#       delta=1e-6 amplifies weak curvature directions ~1000x -> NaN; ~1e-3 caps it
#       at ~31x. MUST be set -- the CLI default 1e-6 diverges.)
lr=${1:-3e-3}
optimizer=${2:-curvature-whiten-polar-lora}
seed=${3:-0}
precond_delta=${4:-1e-3}

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
    --no-cw_nesterov \
    --model_name Qwen/Qwen2.5-1.5B \
    --data_dir data/opc_sft_stage2_all_packed_seq2048_qwen25 \
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
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
