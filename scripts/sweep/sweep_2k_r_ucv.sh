#!/bin/bash
# UCV (orthogonal-core LoRA, ΔW = U C V^T) sweep launcher.
# Sets --training_mode ucv. 4 positional args: lr, seed, lora_r, muon_ns_steps.
# lora_alpha is set to lora_r so alpha/r = 1.
lr=${1:-3e-4}
seed=${2:-0}
lora_r=${3:-16}
muon_ns_steps=${4:-5}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
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
    --data_dir data/magicoder_seq512_32k \
    --device cuda \
    --bf16 \
    --training_mode ucv \
    --optimizer adam-ucv-core-lora \
    --max_steps "${MAX_STEPS:-2000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps "$muon_ns_steps" \
    "${wandb_args[@]}"
