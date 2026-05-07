#!/bin/bash
# SVD oracle variant of sweep_2k.sh. Arg order must match params/svd_sweep_2k.json key order.
lr=${1:-3e-4}
training_mode=${2:-svd_step_oracle}
seed=${3:-0}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_32k \
    --device cuda \
    --bf16 \
    --max_steps 2000 \
    --eval_every 200 \
    --lr "$lr" \
    --training_mode "$training_mode" \
    --svd_niter 4 \
    --seed "$seed" \
    "${wandb_args[@]}"
