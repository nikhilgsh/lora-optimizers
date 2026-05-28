#!/bin/bash
# Llama-3.2-1B × opc-sft-stage2 (all-4-configs, Llama-tokenized cache) × r=256
# × packed_v1.1. chord-tight-CLEAN, polar_method=ns, muon_ns_steps=8 (full
# whitening), picard ablation: does the 1/eta cross-coupling (picard=2) add
# value over plain polar (picard=1) on Llama at r=256? r=256 rank-extension of
# sweep_robustness_llama32_1b_opc_r64_clean_ns8.sh (only lora_r/alpha 64 -> 256).
# chord-tight-clean uses the pre_norm='none' path, so ns=8 is true full-
# whitening polar (no redundant Frob shrink).
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: picard_iters_override (1 = no cross-coupling / plain polar; 2 = 1/eta coupling)
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight-clean}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
picard=${5:-2}

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
    if [ "${KEEP_CHECKPOINTS:-0}" = "1" ]; then
        ckpt_args+=(--keep_checkpoints)
    fi
fi

python train_lora.py \
    --model_name meta-llama/Llama-3.2-1B \
    --data_dir data/opc_sft_stage2_all_packed_seq2048_llama32 \
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
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r 256 \
    --lora_alpha 256 \
    --muon_ns_steps 8 \
    --polar_method ns \
    --precond_method higham \
    --picard_iters_override "$picard" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${wandb_args[@]}"
