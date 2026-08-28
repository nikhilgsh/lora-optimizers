#!/bin/bash
# E2 preconditioner x curvature-EMA-decay wrapper.
# Last updated: 2026-08-28T03:10:37-04:00
#
# This is the production protagonist configuration used by the r=16
# preconditioner panel, with curvature_beta promoted to an explicit positional
# so task records and overlap audits can distinguish each beta2 cell.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1
#   6: model  7: data_dir  8: lora_r
#   9: precond  (product | one-sided | factorwise)
#  10: msign    (full | diag)
#  11: curvature_beta
lr=${1:-3e-2}
optimizer=${2:-kl-diag-polar-lora}
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.9}
model=${6:-meta-llama/Llama-3.2-1B}
data_dir=${7:-data/openmath_instruct_2_2m_packed_seq2048_llama32}
lora_r=${8:-16}
precond=${9:-factorwise}
msign=${10:-full}
curvature_beta=${11:-0.9}

case "$precond" in
    product|one-sided|factorwise) ;;
    *) echo "sweep_protagonist_precond_beta2: bad precond '$precond'" >&2; exit 2 ;;
esac
case "$msign" in
    full|diag) ;;
    *) echo "sweep_protagonist_precond_beta2: bad msign '$msign'" >&2; exit 2 ;;
esac

precond_method=${PRECOND_METHOD:-gram_ns}
precond_args=()
[ -n "$precond_method" ] && precond_args=(--precond_method "$precond_method" --higham_iters "${HIGHAM_ITERS:-8}")

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_${USER:-u}_$$}"

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
    --lora_alpha "$lora_r" \
    --beta1 "$beta1" \
    --curvature_beta "$curvature_beta" \
    --precond_refresh_every 10 \
    --precond_delta "$precond_delta" \
    --polar_method polar_express \
    --muon_ns_steps 8 \
    --cw_picard_iters 1 \
    --cw_nesterov \
    --precond "$precond" \
    --msign "$msign" \
    "${precond_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
