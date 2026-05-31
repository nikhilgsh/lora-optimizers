#!/bin/bash
# Phase L (long-horizon 1B) at r=256, chord-tight-clean k=1, with the
# --curvature_whitening flag. polar_method selects plain NS polar (default
# for the curvature A/B) or SSC; at k=1 the Picard cross-coupling is off.
# --curvature_whitening swaps the whitening metric from the geometric Gram
# (BᵀB) to an EMA of the factor-gradient outer products (S_curv =
# EMA(g_A g_Aᵀ) = BᵀHB). Run the false/true arms head-to-head, identical in
# every other respect, to isolate the whitening metric.
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
#  11: curvature_whitening (true/false; default false)
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
curvature_whitening=${11:-false}

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

curv_args=()
case "$curvature_whitening" in
    true|True|TRUE|1|yes|YES) curv_args+=(--curvature_whitening) ;;
    false|False|FALSE|0|no|NO|"") ;;
    *) echo "curvature_whitening must be true/false, got: $curvature_whitening" >&2; exit 2 ;;
esac

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
    --lora_r 256 \
    --lora_alpha 256 \
    --muon_ns_steps "$muon_ns_steps" \
    --precond_method higham \
    --picard_iters_override 1 \
    "${polar_args[@]}" \
    "${curv_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}" \
    "${abort_args[@]}" \
    "${wandb_args[@]}"
