#!/bin/bash
# ADAM BETA2 sensitivity wrapper: the AdamW baseline with --beta2 as a swept
# positional. Control for scripts/sweep/sweep_protagonist_beta2.sh: if beta2 is
# inert for BOTH, that is a property of this workload's gradient shapes; if it
# matters for Adam but not for the protagonist, the protagonist's max-normalized
# diagonal metric is genuinely insensitive to the EMA window.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: model  5: data_dir  6: lora_r  7: beta2
lr=${1:-3e-4}
optimizer=${2:-adamw}
seed=${3:-0}
model=${4:-allenai/OLMo-2-0425-1B}
data_dir=${5:-data/opc_sft_stage2_all_packed_seq2048}
lora_r=${6:-256}
beta2=${7:-0.999}   # SWEPT AXIS: Adam second-moment decay (default 0.999)

compile_args=()
[ "${COMPILE:-1}" = "1" ] && compile_args=(--compile)

diag_args=(--log_basic_diagnostics)
[ "${LOG_DIAGNOSTICS:-1}" = "0" ] && diag_args=(--no-log_basic_diagnostics)

ckpt_args=()
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    ckpt_args=(
        --checkpoint_dir "$CHECKPOINT_DIR"
        --resume_from "$CHECKPOINT_DIR"
        --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST:-2}"
    )
    [ -n "${CHECKPOINT_EVERY:-}" ] && ckpt_args+=(--checkpoint_every "$CHECKPOINT_EVERY")
fi

python train_lora.py \
    --model_name "$model" \
    --data_dir "$data_dir" \
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
    --seed "$seed" \
    --lora_r "$lora_r" \
    --beta2 "$beta2" \
    --lora_alpha "$lora_r" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
