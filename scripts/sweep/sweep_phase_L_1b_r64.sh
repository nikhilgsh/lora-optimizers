#!/bin/bash
# Phase L (long-horizon 1B characterization): OLMo-2-1B × opc-sft-stage2
# all-4-configs × seq=2048 × global_batch=16 × r=64 (FIXED) × packed_v1.
# See docs/notes/polar_product/tight_chord_paper_plan.md.
#
# 1-pass token budget ≈ 225M unique (308M slot-tokens at 150,492 packed slots).
# MAX_STEPS=9000 fits 1-pass at global_batch=16.
#
# Per-microbatch shape: batch=4 × accum=4 = 16 effective (NOT batch=16 × accum=1;
# batch=16 single forward OOMs at seq=2048 on 96 GB Blackwell). Peak memory
# at this shape: 26 GB with compile, 37 GB no-compile.
# Production per-step (Blackwell, compile, warm GPU): 1.72 s/step both opts.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
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

# Diagnostics tiering matches sweep_4k_packed_diag_k1.sh: basic on by default
# (~2% wall), heavy off by default.
diag_args=(--log_basic_diagnostics)
if [ "${LOG_DIAGNOSTICS:-1}" = "0" ]; then
    diag_args=(--no-log_basic_diagnostics)
fi
if [ "${LOG_HEAVY_DIAGNOSTICS:-0}" = "1" ]; then
    diag_args+=(--log_heavy_diagnostics)
fi

# Checkpoint flags: enabled by submit.sh setting CHECKPOINT_DIR per task.
# When the env var is set, --checkpoint_dir AND --resume_from point at the
# same directory. load_checkpoint is idempotent — first submission finds an
# empty dir (no-op), resubmissions auto-pick the latest ckpt_step{N} child
# and resume in place.
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
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r 64 \
    --lora_alpha 64 \
    --muon_ns_steps 5 \
    --precond_method higham \
    --picard_iters_override 1 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${wandb_args[@]}"
