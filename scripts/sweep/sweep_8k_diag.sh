#!/bin/bash
# 8000-step sweep with diagnostics. Long-horizon variant of
# scripts/sweep/sweep_2k_r_diag_ns_steps.sh — use to test rank-saturation /
# data-bound hypotheses where 2k steps is the canonical comparison horizon
# but a longer trajectory is needed to disambiguate.
#
# Positional args (must match params JSON key order):
#   1: lr (default 3e-4)
#   2: optimizer (default adam-polar-product-lora)
#   3: lora_plus_multiplier (default 1.0)
#   4: seed (default 0)
#   5: lora_r (default 16)
#   6: muon_ns_steps (default 5)
lr=${1:-3e-4}
optimizer=${2:-adam-polar-product-lora}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}
muon_ns_steps=${6:-5}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

# Compile default-on for production sweeps (CLAUDE.md "torch.compile whenever
# it amortizes"). Opt out with COMPILE=0 for debug runs only.
compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

python train_lora.py \
    --data_dir data/magicoder_seq512_32k \
    --device cuda \
    --bf16 \
    "${compile_args[@]}" \
    --max_steps "${MAX_STEPS:-8000}" \
    --eval_every "${EVAL_EVERY:-200}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --lora_plus_multiplier "$lora_plus_multiplier" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --muon_ns_steps "$muon_ns_steps" \
    --log_basic_diagnostics \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
