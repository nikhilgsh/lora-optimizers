#!/bin/bash
# E2 ABLATION wrapper: the `precond` x `msign` arms on the protagonist config.
#
# `precond` picks what fills the two r x r slots (C_B, C_A):
#     product     C_B = B^T P B,  C_A = A Q A^T      (PoLoRA)
#     one-sided   C_B = C_A = I_r
#     factorwise  C_B = P_A,      C_A = Q_B
# `msign` picks how accurately the matrix sign is applied to the whitened momenta
# Z_A = C_B^-1/2 Mhat_A Q^-1/2, Z_B = P^-1/2 Mhat_B C_A^-1/2:
#     full   U = msign(Z)
#     diag   U_A = rownorm(Z_A), U_B = colnorm(Z_B)   (RACS-style, no r x r inverse sqrt)
# The two are ORTHOGONAL and everything else is the protagonist config
# (b1=0.9, delta=1e-4, gram_ns inverse-sqrt, PolarExpress-8, Nesterov, k=1, 9000 steps).
#
# BOTH are passed EXPLICITLY on every cell, never appended-if-set: `--precond`
# defaults to None ("inherit the optimizer spec") and `--msign` to "full", so a
# wrapper that omitted them would silently run the inherited branch and label the
# cell as something it is not.
#
# Parameterized by model / data_dir / lora_r as trailing positionals (encoded
# per-cell in the params JSON) so the cell is captured in the task line — env
# vars do NOT propagate through submit.sh --emit-pending.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1  6: model  7: data_dir  8: lora_r
#   9: precond  (product | one-sided | factorwise)
#  10: msign    (full | diag)
lr=${1:-3e-2}
optimizer=${2:-kl-diag-polar-lora}      # paper protagonist
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.9}                          # locked protagonist β₁
model=${6:-meta-llama/Llama-3.2-1B}
data_dir=${7:-data/openmath_instruct_2_2m_packed_seq2048_llama32}
lora_r=${8:-256}
precond=${9:-product}
msign=${10:-full}

# Fail loudly on a bad cell rather than letting train.py's argparse reject it
# 8 minutes into a model load, or worse, silently accept a typo'd branch.
case "$precond" in
    product|one-sided|factorwise) ;;
    *) echo "sweep_protagonist_precond: bad precond '$precond'" >&2; exit 2 ;;
esac
case "$msign" in
    full|diag) ;;
    *) echo "sweep_protagonist_precond: bad msign '$msign'" >&2; exit 2 ;;
esac

precond_method=${PRECOND_METHOD:-gram_ns}   # protagonist inverse-sqrt: Polar-Express Gram NS
precond_args=()
[ -n "$precond_method" ] && precond_args=(--precond_method "$precond_method" --higham_iters "${HIGHAM_ITERS:-8}")

# Per-task torch.compile cache dir: disBatch co-locates tasks on a node; a SHARED
# inductor/AOTAutograd cache gets corrupted by concurrent compiles (JSONDecodeError
# in AOTAutogradCache.load at startup). $$ (task PID) isolates each task's cache.
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor_${USER:-u}_$$}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER:-u}_$$}"

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
    [ "${KEEP_CHECKPOINTS:-0}" = "1" ] && ckpt_args+=(--keep_checkpoints)
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
    --precond "$precond" \
    --msign "$msign" \
    "${precond_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
