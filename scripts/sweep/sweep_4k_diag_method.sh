#!/bin/bash
# 4000-step sweep with diagnostics + configurable precond_method.
# Mirrors sweep_4k_diag.sh; adds positional 7 = precond_method (eigh|higham).
#
# Positional args (must match params JSON key order):
#   1: lr (default 3e-4)
#   2: optimizer (default adam-polar-product-lora-coupled)
#   3: lora_plus_multiplier (default 1.0)
#   4: seed (default 0)
#   5: lora_r (default 16)
#   6: muon_ns_steps (default 5)
#   7: precond_method (default eigh)
lr=${1:-3e-4}
optimizer=${2:-adam-polar-product-lora-coupled}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}
muon_ns_steps=${6:-5}
precond_method=${7:-eigh}

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
    --data_dir data/magicoder_seq512_70k \
    --device cuda \
    --bf16 \
    --max_steps "${MAX_STEPS:-4000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps "$muon_ns_steps" \
    --precond_method "$precond_method" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
