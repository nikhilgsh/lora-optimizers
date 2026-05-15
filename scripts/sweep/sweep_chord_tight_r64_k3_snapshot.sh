#!/bin/bash
# Single-cell snapshot run: chord-tight k=3, r=64, ε_rel=1e-2, lr=3e-2, 4k steps,
# packed_v1. Re-runs the chord_tight_r64_k3_eps_rel_sweep_4k_blackwell task 3
# cell with diagnostic snapshots enabled so a downstream notebook can probe
# (A_pre, B_pre, u_A_pre, u_B_pre, m_A, m_B) at log-spaced step indices.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
lr=${1:-3e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
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

# Checkpoint flags (resume protection) — same convention as Phase L.
ckpt_args=()
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    ckpt_args=(
        --checkpoint_dir "$CHECKPOINT_DIR"
        --resume_from "$CHECKPOINT_DIR"
        --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST:-2}"
    )
    if [ -n "${CHECKPOINT_EVERY:-}" ]; then
        ckpt_args+=(--checkpoint_every "$CHECKPOINT_EVERY")
    fi
    if [ "${KEEP_CHECKPOINTS:-0}" = "1" ]; then
        ckpt_args+=(--keep_checkpoints)
    fi
fi

# Diagnostic snapshot flags. SNAPSHOT_DIR is injected per-task by submit.sh
# when SNAPSHOTS=1 is set on the sbatch. SNAPSHOT_STEPS overridable.
snapshot_args=()
if [ -n "${SNAPSHOT_DIR:-}" ]; then
    snapshot_args=(
        --snapshot_dir "$SNAPSHOT_DIR"
        --snapshot_steps "${SNAPSHOT_STEPS:-0,50,200,500,1000,2000,4000}"
    )
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
    --lora_r 64 \
    --lora_alpha 64 \
    --muon_ns_steps 5 \
    --precond_method higham \
    --picard_iters_override 3 \
    --precond_delta_relative \
    --precond_delta 1e-2 \
    --log_basic_diagnostics \
    --optim_diagnostics_every 80 \
    "${ckpt_args[@]}" \
    "${snapshot_args[@]}" \
    "${wandb_args[@]}"
