#!/bin/bash
# OLMo-2-1B × OpenMathInstruct-2 × r=64 Tanya-style polar sweep.
# Positional args:
#   1: task id from params/openmath_tanya_style_polar_r64.json
task=${1:-fw_lr3e-2}

lr=""
optimizer=""
weight_decay="0.0"
picard_iters_override="1"
cw_picard_iters="1"
precond_delta="1e-6"
precond_refresh_every="1"

case "$task" in
    adamw_lr3e-5) lr="3e-5"; optimizer="adamw" ;;
    adamw_lr1e-4) lr="1e-4"; optimizer="adamw" ;;
    adamw_lr3e-4) lr="3e-4"; optimizer="adamw" ;;

    muon_lr1e-3) lr="1e-3"; optimizer="muon-lora" ;;
    muon_lr3e-3) lr="3e-3"; optimizer="muon-lora" ;;
    muon_lr1e-2) lr="1e-2"; optimizer="muon-lora" ;;

    fw_lr1e-2)
        lr="1e-2"
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="1"
        ;;
    fw_lr3e-2)
        lr="3e-2"
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="1"
        ;;
    fw_lr1e-1)
        lr="1e-1"
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="1"
        ;;

    bcd_lr1e-2)
        lr="1e-2"
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="2"
        ;;
    bcd_lr3e-2)
        lr="3e-2"
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="2"
        ;;
    bcd_lr1e-1)
        lr="1e-1"
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="2"
        ;;

    kl_diag_polar_lr1e-2)
        lr="1e-2"
        optimizer="kl-diag-polar-lora"
        precond_delta="1e-4"
        precond_refresh_every="10"
        ;;
    kl_diag_polar_lr3e-2)
        lr="3e-2"
        optimizer="kl-diag-polar-lora"
        precond_delta="1e-4"
        precond_refresh_every="10"
        ;;
    kl_diag_polar_lr1e-1)
        lr="1e-1"
        optimizer="kl-diag-polar-lora"
        precond_delta="1e-4"
        precond_refresh_every="10"
        ;;
    *)
        echo "unknown task: $task" >&2
        exit 2
        ;;
esac

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
    --data_dir data/openmath_instruct_2_2m_packed_seq2048 \
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
    --weight_decay "$weight_decay" \
    --lora_plus_multiplier 1.0 \
    --seed "${SEED:-0}" \
    --lora_r 64 \
    --lora_alpha 64 \
    --muon_ns_steps 5 \
    --precond_method higham \
    --precond_refresh_every "$precond_refresh_every" \
    --precond_delta "$precond_delta" \
    --picard_iters_override "$picard_iters_override" \
    --cw_picard_iters "$cw_picard_iters" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${snapshot_args[@]}" \
    "${wandb_args[@]}"
