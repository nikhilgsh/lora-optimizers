#!/bin/bash
# KL-Shampoo-polar 8B conditioning test: Meta-Llama-3-8B × opc-sft-stage2
# × seq=2048 × global_batch=16 × r=256 × packed_v1.1.
#
# This is the 8B analogue of sweep_curvature_whiten_r256_opc_qwen.sh — the
# "better conditioning" arm. At 8B opc, chord-tight-clean only TIES AdamW
# (best chord-tight 0.5560 vs best AdamW 0.5565, both in-flight). The question:
# does properly-solved KL-Shampoo curvature (coupled two-sided KL fixed point,
# closed-form S^{-1/2} m̂ D^{-1/2}, plus polar) BEAT AdamW where chord-tight ties?
# Win => conditioning problem; tie => task/headroom.
#
# Optimizer: kl-shampoo-polar-lora (kl_coupled, soap_v=False, use_polar=True).
# HPs mirror the qwen kl-shampoo cell EXACTLY: curvature_beta 0.99,
# precond_refresh_every 10, precond_delta 1e-3 (the critical damping HP — the
# CLI default 1e-6 amplifies weak curvature ~1000x and diverges; ~1e-3 caps it).
# ns_steps for the polar step is left at the train.py default (5), matching qwen.
#
# 8B-specific: model/data/batch match the 8B AdamW + chord-tight cells exactly
# (Meta-Llama-3-8B, _llama32 packed corpus, bs2/ga8 = global 16, max_steps 9000,
# eval_every 250, r256/alpha256) so eval_loss is directly comparable in
# notebooks/leaderboard/llama3_8b_opc_leaderboard. bs2 keeps peak memory under
# the 96 GB Blackwell ceiling (chord-tight peaks 61.4 GB at this microbatch;
# kl-shampoo curvature state measured by the pre-submit smoke).
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer (kl-shampoo-polar-lora | kl-shampoo-lora)
#   3: seed
#   4: precond_delta  (relative-off absolute damping floor for the inverse-sqrts;
#       the critical HP — MUST be set, CLI default 1e-6 diverges.)
lr=${1:-3e-3}
optimizer=${2:-kl-shampoo-polar-lora}
seed=${3:-0}
precond_delta=${4:-1e-3}

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

ckpt_args=()
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    ckpt_args=(
        --checkpoint_dir "$CHECKPOINT_DIR"
        --resume_from "$CHECKPOINT_DIR"
        --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST:-2}"
    )
    if [ "${KEEP_CHECKPOINTS:-0}" = "1" ]; then
        ckpt_args+=(--keep_checkpoints)
    fi
fi

python train_lora.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --data_dir data/opc_sft_stage2_all_packed_seq2048_llama32 \
    --data_pipeline_version "${DATA_PIPELINE_VERSION:-packed_v1.1}" \
    --max_seq_length 2048 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --grad_accum_steps "${GRAD_ACCUM:-8}" \
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
    "${ckpt_args[@]}" \
    "${wandb_args[@]}"
