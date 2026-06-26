#!/bin/bash
# GENERIC LoRA-RITE baseline wrapper (Yen et al., ICLR'25, arXiv:2410.20625).
# Added as an adaptive transformation-invariant LoRA baseline alongside Muon/iMuon.
# Same fixed config as the hero cell (matches sweep_adamw_generic.sh) EXCEPT one
# deliberate, documented difference:
#
#   --max_grad_norm 0  -> disables the trainer's GLOBAL grad-norm clip.
#
# Rationale: a raw-gradient-norm clip is NOT transformation-invariant (it scales
# with the arbitrary LoRA factorization), so it would break the invariance
# LoRA-RITE is built to provide. The authors' README instructs disabling the
# trainer clip and using LoRA-RITE's own in-optimizer clip (clip_unmagnified_grad,
# default 1.0, applied in the rotation-invariant coordinates). betas (0.9,0.999),
# eps 1e-6, escape-mass OFF are the reference implementation's defaults (train.py
# --beta1/--beta2 defaults + the LoRARite class defaults); only the lr is swept,
# as for every optimizer.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: model  5: data_dir  6: lora_r
lr=${1:-1e-3}
optimizer=${2:-lora-rite}
seed=${3:-0}
model=${4:-meta-llama/Llama-3.2-1B}
data_dir=${5:-data/openmath_instruct_2_2m_packed_seq2048_llama32}
lora_r=${6:-256}

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
    --max_grad_norm 0 \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
