#!/bin/bash
# Qwen3-0.6B-Base × OpenWebMath (continued PRETRAINING, all-token loss) × r=64
# polar-family sweep — faithful tanya-style replication
# (tanya_results/owm300m_polar_sweep.md), mapped to repo-canonical optimizers:
#   Frank-Wolfe  -> chord-tight-clean ns=8 k=1
#   BCD-Polar    -> chord-tight-clean ns=8 k=2
#   + AdamW, + KL-diag +polar   (iMuon dropped)
# alpha = r = 64 (repo convention; deviates from tanya's alpha=1.0).
#
# Positional args:
#   1: task id from params/qwen3_openwebmath_polar_r64.json
#      (lr is parsed from the trailing _lr<val>; prefix selects the optimizer)
task=${1:-ct_clean_k1_lr1e-2}

lr="${task##*_lr}"            # everything after the last "_lr"
optimizer=""
weight_decay="0.0"
picard_iters_override="1"
cw_picard_iters="1"
precond_delta="1e-6"
precond_refresh_every="1"

case "$task" in
    adamw_lr*)
        optimizer="adamw"
        ;;
    ct_clean_k1_lr*)
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="1"
        ;;
    ct_clean_k2_lr*)
        optimizer="adam-polar-product-lora-coupled-spectral-chord-tight-clean"
        picard_iters_override="2"
        ;;
    kl_diag_polar_lr*)
        optimizer="kl-diag-polar-lora"
        precond_delta="1e-4"
        precond_refresh_every="10"
        ;;
    *)
        echo "unknown task: $task" >&2
        exit 2
        ;;
esac

if [ -z "$lr" ] || [ "$lr" = "$task" ]; then
    echo "could not parse lr from task: $task" >&2
    exit 2
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
    --no-cw_nesterov \
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
    --weight_decay "$weight_decay" \
    --lora_plus_multiplier 1.0 \
    --seed "${SEED:-0}" \
    --lora_r 64 \
    --lora_alpha 64 \
    --muon_ns_steps 8 \
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
