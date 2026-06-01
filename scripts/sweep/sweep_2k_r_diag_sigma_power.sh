#!/bin/bash
# 2000-step sweep with --polar_sigma_power as 6th positional. HTMuon (arXiv:2603.10067):
# replaces NS polar with SVD σ → σ^p. p=0 = exact polar, p=0.125 = HTMuon paper default,
# p=1 = no orthogonalization.
lr=${1:-3e-4}
optimizer=${2:-adam-polar-product-lora}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}
polar_sigma_power=${6:-0.125}

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
    --max_steps "${MAX_STEPS:-2000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --polar_sigma_power "$polar_sigma_power" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    "${wandb_args[@]}"
