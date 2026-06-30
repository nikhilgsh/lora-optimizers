#!/bin/bash
# rdinv-variant investigation wrapper. Clone of sweep_protagonist_generic.sh
# (KL-diag + full polar PolarExpress + Nesterov, k=1) with one extra trailing
# positional, rdinv_variant, that selects the damping-floor reference scale in
# the large-side diagonal metric:
#   A  = own op-norm   (x/x_max+δ)^{-1/2}   shipped paper protagonist
#   B  = raw/unbiased KL gauge (x+δ·x_max)^{-1/2}   (same op-norm floor)
#   VN = von Neumann / matrix Adafactor (x+δ·Tr(partner))^{-1/2}  trace-scaled
# δ is NOT swept here (precond_delta fixed at the protagonist 1e-4); only lr and
# the variant vary. A is reused from the existing e1 runs, so this wrapper runs B/VN.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1  6: model  7: data_dir
#   8: lora_r  9: precond_method  10: rdinv_variant  11: rdinv_delta (OPTIONAL; empty=coupled to precond_delta)
lr=${1:-3e-2}
optimizer=${2:-kl-diag-polar-lora}
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.9}
model=${6:-meta-llama/Llama-3.2-1B}
data_dir=${7:-data/openmath_instruct_2_2m_packed_seq2048_llama32}
lora_r=${8:-256}
precond_method=${9:-gram_ns}
rdinv_variant=${10:-A}
rdinv_delta=${11:-none}

precond_args=()
[ -n "$precond_method" ] && precond_args=(--precond_method "$precond_method" --higham_iters "${HIGHAM_ITERS:-8}")

# Decoupled diagonal-metric floor. Sentinel "none" (or empty) -> fall through to train.py
# default None -> coupled to precond_delta. ("none" default keeps generate_task_file able to
# register this positional, which needs a non-empty default.)
rdinv_delta_args=()
[ -n "$rdinv_delta" ] && [ "$rdinv_delta" != "none" ] && rdinv_delta_args=(--rdinv_delta "$rdinv_delta")

# Per-task torch.compile cache dir ($$ = task PID) so concurrent disBatch tasks
# don't corrupt a shared inductor/AOTAutograd cache.
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
    --curvature_beta 0.99 \
    --precond_refresh_every 10 \
    --precond_delta "$precond_delta" \
    --polar_method polar_express \
    --muon_ns_steps 8 \
    --cw_picard_iters 1 \
    --cw_nesterov \
    --rdinv_variant "$rdinv_variant" \
    "${rdinv_delta_args[@]}" \
    "${precond_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
