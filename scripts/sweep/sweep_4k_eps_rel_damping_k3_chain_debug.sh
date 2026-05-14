#!/bin/bash
# 4k-step single-cell test: r=256 chord-tight k=3 with relative damping
# (ε_rel = 1e-2) AND chain-debug instrumentation enabled. Tests whether
# damping prevents NaN even at the more aggressive lr=3e-2 (which
# historically NaN'd at step 400 on original node). Auto-emits
# `non_finite_detected` + `non_finite_intermediate` events if any
# intermediate goes bad despite the damping.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: lora_plus_multiplier  4: seed  5: lora_r
#   6: precond_delta / ε_rel
lr=${1:-3e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-256}
precond_delta=${6:-1e-2}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

compile_args=()
if [ "${COMPILE:-0}" = "1" ]; then
    compile_args=(--compile)
fi

snapshot_root=${DEBUG_SNAPSHOT_ROOT:-logs/debug_snapshots/chord_tight_k3_eps_rel}
snapshot_dir="${snapshot_root}/lr${lr}_r${lora_r}_eps${precond_delta}_seed${seed}_${SLURM_JOB_ID:-local}_$$"
mkdir -p "$snapshot_dir"

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
    --precond_delta_relative \
    --precond_delta "$precond_delta" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 20 \
    --train_loss_every 10 \
    --log_non_finite \
    --debug_higham_residual \
    --debug_optimizer_state \
    --debug_optimizer_state_every "${DEBUG_OPT_STATE_EVERY:-1}" \
    --debug_snapshot_dir "$snapshot_dir" \
    --debug_snapshot_limit "${DEBUG_SNAPSHOT_LIMIT:-8}" \
    --debug_abort_on_non_finite \
    "${wandb_args[@]}"
