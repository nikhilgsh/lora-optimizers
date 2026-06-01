#!/bin/bash
# Parameterized full-polar (ns=8) damping sweep, r=64, opc — the r=64 mirror of
# scripts/sweep/sweep_epsrel_fullpolar_r256.sh. Used for the eps_rel lr-robustness
# campaign on BOTH OLMo and Llama, and for both the ct-k1 (plain chord-tight) and
# clean-k2 (chord-tight-clean, picard=2) arms. polar_method=ns / muon_ns_steps=8
# matches the r=256 campaign so the eps_rel-effect comparison is method-identical
# across ranks.
#
# Positional args (must match params JSON key order):
#   1: lr
#   2: optimizer                (ct: adam-polar-product-lora-coupled-spectral-chord-tight
#                                clean: ...-chord-tight-clean)
#   3: lora_plus_multiplier
#   4: seed
#   5: muon_ns_steps            (8 = full polar)
#   6: precond_delta            (relative eps; e.g. 1e-3 / 1e-2 / 1e-1)
#   7: precond_delta_relative   (true/false)
#   8: picard_iters             (1 = ct; 2 = clean cross-coupling)
#   9: model_name
#  10: data_dir
lr=${1:-1e-2}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
muon_ns_steps=${5:-8}
precond_delta=${6:-1e-2}
precond_delta_relative=${7:-true}
picard_iters=${8:-1}
model_name=${9:-allenai/OLMo-2-0425-1B}
data_dir=${10:-data/opc_sft_stage2_all_packed_seq2048}

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

precond_args=(--precond_delta "$precond_delta")
case "$precond_delta_relative" in
    true|True|TRUE|1|yes|YES) precond_args+=(--precond_delta_relative) ;;
    false|False|FALSE|0|no|NO|"") ;;
    *) echo "precond_delta_relative must be true/false, got: $precond_delta_relative" >&2; exit 2 ;;
esac

# Checkpoint flags: enabled by submit.sh (or the disbatch skill's self-contained
# sbatch template) setting CHECKPOINT_DIR per task. When the env var is set,
# --checkpoint_dir AND --resume_from point at the same directory. load_checkpoint
# is idempotent — first submission finds an empty dir (no-op), resubmissions
# auto-pick the latest ckpt_step{N} child and resume in place, so a wall-timeout
# is recoverable instead of a total loss.
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
    --model_name "$model_name" \
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
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r 64 \
    --lora_alpha 64 \
    --muon_ns_steps "$muon_ns_steps" \
    --polar_method ns \
    --precond_method higham \
    --picard_iters_override "$picard_iters" \
    "${precond_args[@]}" \
    "${diag_args[@]}" \
    "${ckpt_args[@]}" \
    --optim_diagnostics_every 100 \
    "${wandb_args[@]}"
