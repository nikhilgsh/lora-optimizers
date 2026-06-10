#!/bin/bash
# Protagonist wrapper + per-factor shape scaling (cw_factor_a, cw_factor_b).
# Identical to sweep_protagonist_generic.sh but exposes the two shape exponents
# as positionals 9-10 so an (a,b) grid can be swept. c=(0,0) is bit-identical to
# the generic protagonist. Investigation: does an asymmetric per-factor radius
# (c_A=(r/d_in)^a, c_B=(d_out/r)^b) beat equal-ρ? See notebooks/factor_scaling_sweep.ipynb.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1  6: model  7: data_dir
#   8: lora_r  9: cw_factor_a  10: cw_factor_b
lr=${1:-3e-2}
optimizer=${2:-diag-shampoo-polar-lora}
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.95}
model=${6:-allenai/OLMo-2-0425-1B}
data_dir=${7:-data/opc_sft_stage2_all_packed_seq2048}
lora_r=${8:-256}
cw_factor_a=${9:-0.0}
cw_factor_b=${10:-0.0}

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
    --cw_factor_a "$cw_factor_a" \
    --cw_factor_b "$cw_factor_b" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
