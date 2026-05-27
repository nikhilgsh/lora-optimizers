#!/bin/bash
# Robustness pilot: OLMo-2-1B × tulu3_1b_r256
# × seq=2048 × global_batch=16 × r=256 × packed_v1.1.
# Clone of sweep_phase_L_1b_r256.sh; only --data_dir differs (Tulu-3 cache
# vs opc-sft-stage2). See docs/plans/paper_plan.md and the robustness plan
# at ~/.claude/plans/as-part-of-our-tender-quilt.md.
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

snapshot_args=()
if [ -n "${SNAPSHOT_DIR:-}" ]; then
    snapshot_args=(
        --snapshot_dir "$SNAPSHOT_DIR"
        --snapshot_steps "${SNAPSHOT_STEPS:-0,500,1000,2000,4000,6000,9000}"
    )
fi

python train_lora.py \
    --data_dir data/tulu3_sft_mixture_400k_packed_seq2048 \
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
    --muon_ns_steps 5 \
    --precond_method higham \
    --picard_iters_override 1 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${snapshot_args[@]}" \
    "${wandb_args[@]}"
