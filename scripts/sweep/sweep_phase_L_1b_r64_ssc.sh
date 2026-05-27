#!/bin/bash
# Phase L (long-horizon 1B) with SSC fixed-κ plumbing for chord-tight-clean.
# Clone of sweep_phase_L_1b_r64.sh that adds polar_method / ssc_* / muon_ns_steps
# positional args (mirroring sweep_4k_packed_diag_k3.sh) and sets
# --picard_iters_override 2 — at picard=1 the anchored-FW cross-coupling
# correction is identically zero and chord-tight-clean collapses to polar
# (Lemma 4 / §A2 of docs/notes/polar_product/algorithm_ssc_kappa.md).
#
# Dataset / horizon / batch / r / seq match the canonical Phase L launcher
# (opc-sft-stage2 all-4-configs, seq=2048, global_batch=16, r=64, 9000 steps).
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer
#   3: lora_plus_multiplier
#   4: seed
#   5: polar_method (default ns)
#   6: ssc_c (default none)
#   7: ssc_kappa (default none)
#   8: ssc_kappa_solver (default eigvalsh)
#   9: ssc_nsteps (default 10)
#  10: muon_ns_steps (default 5)
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight-clean}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
polar_method=${5:-ns}
ssc_c=${6:-none}
ssc_kappa=${7:-none}
ssc_kappa_solver=${8:-eigvalsh}
ssc_nsteps=${9:-10}
muon_ns_steps=${10:-5}

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
if [ "${LOG_HEAVY_DIAGNOSTICS:-0}" = "1" ]; then
    diag_args+=(--log_heavy_diagnostics)
fi

polar_args=(--polar_method "$polar_method")
if [ "$polar_method" = "ssc" ]; then
    polar_args+=(--ssc_nsteps "$ssc_nsteps")
    if [ "$ssc_c" != "none" ] && [ -n "$ssc_c" ]; then
        polar_args+=(--ssc_c "$ssc_c")
    fi
    if [ "$ssc_kappa" != "none" ] && [ -n "$ssc_kappa" ]; then
        polar_args+=(--ssc_kappa "$ssc_kappa" --ssc_kappa_solver "$ssc_kappa_solver")
    fi
fi

abort_args=()
if [ -n "${ABORT_ON_EVAL_LOSS_ABOVE:-}" ]; then
    abort_args=(--abort_on_eval_loss_above "$ABORT_ON_EVAL_LOSS_ABOVE")
fi

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

python train_lora.py \
    --data_dir data/opc_sft_stage2_all_packed_seq2048 \
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
    --lora_r 64 \
    --lora_alpha 64 \
    --muon_ns_steps "$muon_ns_steps" \
    --precond_method higham \
    --picard_iters_override 2 \
    "${polar_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${abort_args[@]}" \
    "${wandb_args[@]}"
