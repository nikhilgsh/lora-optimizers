#!/bin/bash
# Qwen2.5-1.5B × Aya-Bengali × r=256 × packed_v1.1 — AdamW baseline.
# Defines the speed target (fraction-of-steps-to-AdamW-final) so Qwen-Bengali
# becomes a first-class workload in the cross-setting performance_profile aggregate.
# Mirror of sweep_qwen25_1b_opc_r256_adamw.sh with --data_dir swapped to the
# Aya-Bengali (Qwen-tokenized) packed cache.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
lr=${1:-1e-4}
optimizer=${2:-adamw}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}

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

# Checkpoint flags: enabled by the submit path setting CHECKPOINT_DIR per task.
# When the env var is set, --checkpoint_dir AND --resume_from point at the same
# directory. load_checkpoint is idempotent — first submission finds an empty dir
# (no-op), resubmissions auto-pick the latest ckpt_step{N} child and resume in
# place, so a wall-timeout is recoverable instead of a total loss.
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
    "${ckpt_args[@]}" \
    --model_name Qwen/Qwen2.5-1.5B \
    --data_dir data/aya_bengali_packed_seq2048_qwen \
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
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${wandb_args[@]}"
