#!/bin/bash
# AdamW-8B baseline (speed target): Meta-Llama-3-8B × OpenMathInstruct-2
# × seq=2048 × global_batch=16 × r=256 × packed_v1.1.
#
# Math-breadth sibling of sweep_8b_opc_r256_adamw.sh: identical except the data
# dir points at the openmath (not opc) _llama32 packed corpus. This is the
# mandatory AdamW speed-target for the 8B math cell (leaderboard metric is
# speed-to-AdamW-target).
#
# 8B-specific: batch_size 2 / grad_accum 8 (global batch 16) — measured peak
# 61.4 GB on a 96 GB Blackwell at this microbatch (bs4 peaks 92 GB, 97%, unsafe).
# Data dir reuses the _llama32 packed corpus — Llama-3/3.1/3.2 share an identical
# token id-mapping (verified: same bos/eos, identical probe encodings).
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer   (adamw here; kept positional so the wrapper mirrors the
#                   robustness wrappers and can be reused for other arms)
#   3: lora_plus_multiplier
#   4: seed
lr=${1:-1e-4}
optimizer=${2:-adamw}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}

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

ckpt_args=()
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    ckpt_args=(
        --checkpoint_dir "$CHECKPOINT_DIR"
        --resume_from "$CHECKPOINT_DIR"
        --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST:-2}"
    )
    if [ "${KEEP_CHECKPOINTS:-0}" = "1" ]; then
        ckpt_args+=(--keep_checkpoints)
    fi
fi

python train_lora.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --data_dir data/openmath_instruct_2_2m_packed_seq2048_llama32 \
    --data_pipeline_version "${DATA_PIPELINE_VERSION:-packed_v1.1}" \
    --max_seq_length 2048 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --batch_size "${BATCH_SIZE:-2}" \
    --grad_accum_steps "${GRAD_ACCUM:-8}" \
    --max_steps "${MAX_STEPS:-9000}" \
    --eval_every "${EVAL_EVERY:-250}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r 256 \
    --lora_alpha 256 \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${wandb_args[@]}"
