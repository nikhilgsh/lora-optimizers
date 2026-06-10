#!/bin/bash
# 8B protagonist wrapper (Polar-LoRA): identical to sweep_protagonist_generic.sh EXCEPT
# batch_size=2 / grad_accum=8 (global batch 16) baked in — 8B logits (batch×seq×vocab)
# at bs4 OOM an 80GB H100 once the curvature state is added. bs2/ga8 matches the
# validated 8B AdamW + kl-shampoo-polar runs (which fit). Env-default batch does NOT
# propagate through submit.sh --emit-pending, so it must be hardcoded here.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1  6: model  7: data_dir  8: lora_r
lr=${1:-3e-2}
optimizer=${2:-diag-shampoo-polar-lora}
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.95}
model=${6:-meta-llama/Meta-Llama-3-8B}
data_dir=${7:-data/opc_sft_stage2_all_packed_seq2048_llama32}
lora_r=${8:-256}

# Per-task torch.compile cache dir: disBatch co-locates tasks on a node; a SHARED
# inductor/AOTAutograd cache gets corrupted by concurrent compiles (JSONDecodeError
# in AOTAutogradCache.load at startup). $$ (task PID) isolates each task's cache.
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
    --batch_size 2 \
    --grad_accum_steps 8 \
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
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
