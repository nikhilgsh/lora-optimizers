#!/bin/bash
# kl-diag +polar δ-sensitivity probe: δ=1e-6 (vs the base sweep's canonical
# δ=1e-4) on Qwen3-0.6B-Base × OpenWebMath × r=64. Question: is the kl-diag <
# chord-tight gap a δ-tuning artifact? δ here is the RELATIVE conditioning cap
# in _rdinv ((λ/λ_max + δ)^-1/2); 1e-6 caps conditioning at ~1e6 (aggressive).
# Positional arg 1: task id (lr parsed from trailing _lr<val>).
task=${1:-kl_d1e-6_lr1e-3}

lr="${task##*_lr}"
optimizer="kl-diag-polar-lora"
precond_delta="1e-6"
precond_refresh_every="10"

case "$task" in
    kl_d1e-6_lr*) : ;;
    *) echo "unknown task: $task" >&2; exit 2 ;;
esac
if [ -z "$lr" ] || [ "$lr" = "$task" ]; then
    echo "could not parse lr from task: $task" >&2; exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"

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
    --model_name Qwen/Qwen3-0.6B-Base \
    --data_dir data/openwebmath_qwen3_320m_packed_seq2048 \
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
    --weight_decay 0.0 \
    --lora_plus_multiplier 1.0 \
    --seed "${SEED:-0}" \
    --lora_r 64 \
    --lora_alpha 64 \
    --muon_ns_steps 8 \
    --precond_method higham \
    --precond_refresh_every "$precond_refresh_every" \
    --precond_delta "$precond_delta" \
    --picard_iters_override 1 \
    --cw_picard_iters 1 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${snapshot_args[@]}" \
    "${wandb_args[@]}"
