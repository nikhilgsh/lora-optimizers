#!/bin/bash
# OLMo-2-0425-1B OPC r=256 damping × NS probe.
#
# Tests whether σ_max-relative preconditioner damping lets the chord-tight
# polar update tolerate more Newton-Schulz steps. Keeps the Phase-L OPC course
# fixed and exposes only the prediction axes needed for the experiment.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: muon_ns_steps
#   6: precond_delta
#   7: precond_delta_relative (true/false)
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
muon_ns_steps=${5:-5}
precond_delta=${6:-1e-6}
precond_delta_relative=${7:-false}

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

precond_args=(--precond_delta "$precond_delta")
case "$precond_delta_relative" in
    true|True|TRUE|1|yes|YES)
        precond_args+=(--precond_delta_relative)
        ;;
    false|False|FALSE|0|no|NO|"")
        ;;
    *)
        echo "precond_delta_relative must be true/false, got: $precond_delta_relative" >&2
        exit 2
        ;;
esac

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
    --lora_r 256 \
    --lora_alpha 256 \
    --muon_ns_steps "$muon_ns_steps" \
    --polar_method ns \
    --precond_method higham \
    --picard_iters_override 1 \
    "${precond_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${snapshot_args[@]}" \
    "${wandb_args[@]}"
