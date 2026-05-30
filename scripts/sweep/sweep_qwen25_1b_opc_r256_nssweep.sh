#!/bin/bash
# Qwen2.5-1.5B × opc-sft-stage2 × r=256 × packed_v1.1 — chord-tight ns sweep.
# 3rd-model PLACEMENT on the full-polar (ns=8) ↔ partial-polar (ns=5) axis:
# is OLMo's full-polar lr-aversion an outlier or common? Mirror of the Llama
# r256 chord wrapper with --model_name Qwen/Qwen2.5-1.5B, the Qwen-tokenized
# opc cache, and muon_ns_steps exposed as a positional arg.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: muon_ns_steps          (5 = partial / 8 = full polar)
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
muon_ns_steps=${5:-8}

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

python train_lora.py \
    --model_name Qwen/Qwen2.5-1.5B \
    --data_dir data/opc_sft_stage2_all_packed_seq2048_qwen25 \
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
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r 256 \
    --lora_alpha 256 \
    --muon_ns_steps "$muon_ns_steps" \
    --polar_method ns \
    --precond_method higham \
    --picard_iters_override 1 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${wandb_args[@]}"
