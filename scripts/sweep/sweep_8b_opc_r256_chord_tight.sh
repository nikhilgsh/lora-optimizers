#!/bin/bash
# chord-tight(-clean) 8B sweep: Meta-Llama-3-8B × opc-sft-stage2
# × seq=2048 × global_batch=16 × r=256 × packed_v1.1.
#
# Primary arm: chord-tight-clean k=2 gram ns=8 (Algorithm 2', the canonical
# form; assoc-fix landed in optim.py so the Picard cross-term stays (r,r)).
# The optimizer string is positional, and PICARD/NS_STEPS/PRECOND_DELTA are
# env-overridable, so this one wrapper serves both the clean-k2 primary and a
# plain chord-tight k=1 secondary arm:
#   clean k2 (default):  optimizer=...-chord-tight-clean  (PICARD=2)
#   plain  k1 (arm):     optimizer=...-chord-tight  PICARD=1
#
# 8B-specific: batch_size 2 / grad_accum 8 (global 16) — measured peak 61.4 GB
# on a 96 GB Blackwell at this microbatch + clean-k2 (bs4 peaks 92 GB, unsafe).
# Slow-onset NaN is guarded live by --abort_on_nan_eval + nonfinite-grad
# diagnostics + checkpoint-resume (no separate >=1000-step pre-smoke needed).
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight-clean}
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

# Damping mode: abs by default (--precond_delta as an absolute floor); set
# PRECOND_DELTA_RELATIVE=1 to make it relative (ε_rel), e.g. PRECOND_DELTA=1e-3.
rel_args=()
if [ "${PRECOND_DELTA_RELATIVE:-0}" = "1" ]; then
    rel_args=(--precond_delta_relative)
fi

python train_lora.py \
    --model_name meta-llama/Meta-Llama-3-8B \
    --data_dir data/opc_sft_stage2_all_packed_seq2048_llama32 \
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
    --muon_ns_steps "${NS_STEPS:-8}" \
    --polar_method ns \
    --precond_method higham \
    --picard_iters_override "${PICARD:-2}" \
    --precond_delta "${PRECOND_DELTA:-1e-6}" \
    "${rel_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${wandb_args[@]}"
