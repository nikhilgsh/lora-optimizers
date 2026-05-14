#!/bin/bash
# 4k-step single-cell instrumented run: r=256 chord-tight k=3 lr=3e-2,
# NO damping, chain-debug events on. The lr=3e-2 cell historically
# NaN'd at step ~400; canonical 4k-step horizon gives full trajectory
# beyond the failure point.
#
# Captures via the new automatic events:
#   non_finite_detected      — top-of-step per-pair A/B/grad check
#   non_finite_intermediate  — end-of-step chain check naming WHICH
#                              intermediate (u_A, SA^{-1/2}, X_A, P_A,
#                              geo_A, picard_coeff, ρ, dA, ...) and
#                              WHICH pair_name births the NaN
#   train_norms              — global param/grad L2 + n_non_finite_grads
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: lora_plus_multiplier  4: seed  5: lora_r
lr=${1:-3e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-256}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_70k_packed \
    --data_pipeline_version packed_v1 \
    --attn_implementation sdpa \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --max_steps "${MAX_STEPS:-4000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps 5 \
    --precond_method higham \
    --picard_iters_override 3 \
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    --train_loss_every 10 \
    "${wandb_args[@]}"
