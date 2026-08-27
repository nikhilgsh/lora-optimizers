#!/bin/bash
# Continue a factorwise Qwen preconditioner run from its exact step-2000 state
# while keeping the learned small matrices P_A and Q_B fixed. The source and
# destination checkpoint roots are deliberately distinct: this is an ablation
# fork, not a mutation of or merge with the dynamic control trajectory.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1
#   6: model  7: data_dir  8: lora_r  9: precond  10: msign
#  11: source checkpoint root  12: source checkpoint identity prefix
#  13: freeze_factorwise_slots (must be 1)
set -eo pipefail

lr=${1:-1e-2}
optimizer=${2:-kl-diag-polar-lora}
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.9}
model=${6:-Qwen/Qwen2.5-1.5B}
data_dir=${7:-data/openmath_instruct_2_2m_packed_seq2048_qwen25}
lora_r=${8:-16}
precond=${9:-factorwise}
msign=${10:-full}
source_checkpoint_root=${11:-__required_source_checkpoint_root__}
source_identity_prefix=${12:-__required_source_identity_prefix__}
freeze_factorwise_slots=${13:-0}

[[ "$optimizer" == "kl-diag-polar-lora" ]] || {
    echo "freeze launcher requires kl-diag-polar-lora, got '$optimizer'" >&2
    exit 2
}
[[ "$precond" == "factorwise" ]] || {
    echo "freeze launcher requires factorwise preconditioning, got '$precond'" >&2
    exit 2
}
[[ "$msign" == "full" ]] || {
    echo "freeze launcher requires full matrix sign, got '$msign'" >&2
    exit 2
}
[[ "$freeze_factorwise_slots" == "1" ]] || {
    echo "freeze_factorwise_slots must be 1" >&2
    exit 2
}
[[ "$source_checkpoint_root" != "__required_source_checkpoint_root__" ]] || {
    echo "source checkpoint root is required" >&2
    exit 2
}
[[ "$source_identity_prefix" != "__required_source_identity_prefix__" ]] || {
    echo "source checkpoint identity prefix is required" >&2
    exit 2
}
: "${CHECKPOINT_DIR:?submission must inject a unique destination CHECKPOINT_DIR}"
: "${LORA_CHECKPOINT_IDENTITY:?submission must inject a destination checkpoint identity}"

case "$lr" in
    1e-3) source_task=task_00 ;;
    3e-3) source_task=task_03 ;;
    1e-2) source_task=task_06 ;;
    1.7e-2) source_task=task_09 ;;
    3e-2) source_task=task_12 ;;
    *) echo "no verified dynamic source task for lr '$lr'" >&2; exit 2 ;;
esac

final_step=${MAX_STEPS:-9000}
data_pipeline_version=${DATA_PIPELINE_VERSION:-packed_v1.1}
base_checkpoint="${source_checkpoint_root}/${source_task}/ckpt_step2000"
source_identity="${source_identity_prefix}/${source_task}"
resume_from=$(python lora_playground/submission.py \
    resolve-factorwise-freeze-resume \
    --base-checkpoint "$base_checkpoint" \
    --destination-root "$CHECKPOINT_DIR" \
    --source-identity "$source_identity" \
    --destination-identity "$LORA_CHECKPOINT_IDENTITY" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --model-name "$model" \
    --data-dir "$data_dir" \
    --lora-r "$lora_r" \
    --precond-delta "$precond_delta" \
    --beta1 "$beta1" \
    --data-pipeline-version "$data_pipeline_version" \
    --final-step "$final_step")

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_${USER:-u}_$$}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER:-u}_$$}"

compile_args=()
[[ "${COMPILE:-1}" == "1" ]] && compile_args=(--compile)
diag_args=(--log_basic_diagnostics)
[[ "${LOG_DIAGNOSTICS:-1}" == "0" ]] && diag_args=(--no-log_basic_diagnostics)

train_cmd=(
    python train_lora.py
    --model_name "$model"
    --data_dir "$data_dir"
    --data_pipeline_version "$data_pipeline_version"
    --max_seq_length 2048
    --attn_implementation sdpa
    --device cuda
    --bf16
    "${compile_args[@]}"
    --batch_size "${BATCH_SIZE:-4}"
    --grad_accum_steps "${GRAD_ACCUM:-4}"
    --max_steps "$final_step"
    --eval_every "${EVAL_EVERY:-250}"
    --lr "$lr"
    --optimizer "$optimizer"
    --seed "$seed"
    --lora_r "$lora_r"
    --lora_alpha "$lora_r"
    --beta1 "$beta1"
    --curvature_beta 0.99
    --precond_refresh_every 10
    --precond_delta "$precond_delta"
    --polar_method polar_express
    --muon_ns_steps 8
    --cw_picard_iters 1
    --cw_nesterov
    --precond "$precond"
    --freeze_factorwise_slots
    --msign "$msign"
    --precond_method gram_ns
    --higham_iters 8
    "${diag_args[@]}"
    --optim_diagnostics_every "${OPTIM_DIAGNOSTICS_EVERY:-100}"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --resume_from "$resume_from"
    --resume_debug_replay
    --checkpoint_every 1000
    --checkpoint_keep_last 0
    --keep_checkpoints
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${train_cmd[@]}"
    printf '\n'
else
    "${train_cmd[@]}"
fi
