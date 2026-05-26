#!/bin/bash
# 4000-step single-pass sweep with diagnostics on the 70k packed_v1 Magicoder
# subset. Mirrors sweep_4k_diag.sh but pinned to packed_v1 (current default)
# + sdpa attention (flash_attention_2 incompatible with packed_v1 varlen).
# 4000 × 16 = 64000 samples ≤ 70000 train samples (single-pass guarded).
# eval_every 200 matches the 2k notebook step granularity.
#
# Positional args (must match params JSON key order):
#   1: lr (default 3e-3)
#   2: optimizer (default adam-polar-product-lora-coupled-spectral-chord-tight)
#   3: lora_plus_multiplier (default 1.0)
#   4: seed (default 0)
#   5: lora_r (default 16)
#   6: polar_method (default ns)
#   7: ssc_c (default none)
#   8: ssc_kappa (default none)
#   9: ssc_kappa_solver (default eigvalsh)
#  10: ssc_nsteps (default 10)
#  11: muon_ns_steps (default 5)
lr=${1:-3e-3}
optimizer=${2:-adam-polar-product-lora-coupled-spectral-chord-tight}
lora_plus_multiplier=${3:-1.0}
seed=${4:-0}
lora_r=${5:-16}
polar_method=${6:-ns}
ssc_c=${7:-none}
ssc_kappa=${8:-none}
ssc_kappa_solver=${9:-eigvalsh}
ssc_nsteps=${10:-10}
muon_ns_steps=${11:-5}

wandb_args=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    wandb_args=(--wandb_project "$WANDB_PROJECT")
fi

compile_args=()
if [ "${COMPILE:-1}" = "1" ]; then
    compile_args=(--compile)
fi

# Diagnostics tiering: basic ON by default (~2% wall), heavy OFF by default
# (~10x at r=64). Override with LOG_DIAGNOSTICS=0 (disables basic too, legacy
# compat) and/or LOG_HEAVY_DIAGNOSTICS=1.
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
    --muon_ns_steps "$muon_ns_steps" \
    --precond_method higham \
    --picard_iters_override 3 \
    "${polar_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 80 \
    "${wandb_args[@]}"
