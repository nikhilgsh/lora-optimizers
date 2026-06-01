#!/bin/bash
# 4000-step single-pass sweep with diagnostics on the 70k packed_v1 Magicoder
# subset. Mirrors sweep_4k_diag.sh but pinned to packed_v1 (current default)
# + sdpa attention (flash_attention_2 incompatible with packed_v1 varlen).
# 4000 × 16 = 64000 samples ≤ 70000 train samples (single-pass guarded).
# eval_every 200 matches the 2k notebook step granularity.
#
# Positional args (must match params JSON key order):
#   1: lr (default 3e-3)
#   2: optimizer (default adam-polar-product-lora-coupled-spectral-chord-tight)
#   3: lora_plus_multiplier (default 1.0)
#   4: seed (default 0)
#   5: lora_r (default 16)
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

# Diagnostics tiering: basic ON by default (~2% wall), heavy OFF by default
# (~10x at r=64). Override with LOG_DIAGNOSTICS=0 (disables basic too, legacy
# compat) and/or LOG_HEAVY_DIAGNOSTICS=1.
diag_args=(--log_basic_diagnostics)
if [ "${LOG_DIAGNOSTICS:-1}" = "0" ]; then
    diag_args=(--no-log_basic_diagnostics)
fi
if [ "${LOG_HEAVY_DIAGNOSTICS:-0}" = "1" ]; then
    diag_args+=(--log_heavy_diagnostics)
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
    --data_dir data/magicoder_seq512_70k_packed \
    --data_pipeline_version packed_v1 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --max_steps "${MAX_STEPS:-4000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps 5 \
    --precond_method higham \
    --picard_iters_override 1 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
