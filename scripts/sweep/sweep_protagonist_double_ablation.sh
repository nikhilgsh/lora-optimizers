#!/bin/bash
# DOUBLE-ABLATION wrapper (= the LoRA-Muon step): protagonist + --cw_no_diag_curv
# + --cw_unpinned + --lora_init_b symmetric. Removes BOTH controls — bare partner-Gram
# decoupled sandwich (true-scale roots, no sigma_max(W) pin) = LoRA-Muon Alg 1 / iMuon
# Cor 4.1. The unpinned core is UNSTABLE at B=0, so the symmetric (PiSSA-style, step-0 =
# pretrained) init is REQUIRED — that requirement is itself the magnitude-pillar evidence.
# Parameterized by model / data_dir / lora_r as trailing
# positionals (encoded per-cell in the params JSON) so the cell is captured in the task
# line — env vars do NOT propagate through submit.sh --emit-pending.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1  6: model  7: data_dir  8: lora_r
lr=${1:-3e-2}
optimizer=${2:-kl-diag-polar-lora}      # paper protagonist (was diag-shampoo-polar-lora; pivot 2026-06-11)
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.9}                          # locked protagonist β₁ (was 0.95)
model=${6:-allenai/OLMo-2-0425-1B}
data_dir=${7:-data/opc_sft_stage2_all_packed_seq2048}
lora_r=${8:-256}
precond_method=${9:-gram_ns}            # protagonist inverse-sqrt: Polar-Express Gram NS (was eigh)

precond_args=()
[ -n "$precond_method" ] && precond_args=(--precond_method "$precond_method" --higham_iters "${HIGHAM_ITERS:-8}")

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
    --batch_size "${BATCH_SIZE:-4}" \
    --grad_accum_steps "${GRAD_ACCUM:-4}" \
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
    --cw_no_diag_curv \
    --cw_unpinned \
    --lora_init_b symmetric \
    "${precond_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
