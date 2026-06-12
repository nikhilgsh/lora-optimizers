#!/bin/bash
# SHORT iMuon lr-BASIN + STABILITY pilot (ranking-only, 1000 steps). Identical to
# sweep_imuon_generic.sh except MAX_STEPS/EVAL_EVERY defaults are short. Used once to
# locate iMuon's (IMuonLoRA = decoupled Cor 4.1) lr basin AND check whether the B=0-init
# A-side transient (S_B^{-1/2} ≈ δ^{-1/2} blowup) destabilizes; then the real E1 grids are
# centered off the result. NOT for measurement — short horizon, ranking selection only.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: model  5: data_dir  6: lora_r
lr=${1:-3e-3}
optimizer=${2:-imuon-lora}
seed=${3:-0}
model=${4:-allenai/OLMo-2-0425-1B}
data_dir=${5:-data/opc_sft_stage2_all_packed_seq2048}
lora_r=${6:-256}

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
    --max_steps "${MAX_STEPS:-1000}" \
    --eval_every "${EVAL_EVERY:-100}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
