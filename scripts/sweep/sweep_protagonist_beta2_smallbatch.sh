#!/bin/bash
# SMALL-BATCH beta2 contrast. Same swept axis as sweep_protagonist_beta2.sh but at
# 1x1 = 2048 tokens/step instead of 4x4 = 32768, and a short 2000-step horizon.
#
# Why: at 4x4 = 32768 tokens/step the beta2 grid was inert (all four values within
# 0.0001 of each other at step 1750, against sigma = 0.0005 at this cell). Two
# explanations for that: the gradient-energy shape is STRUCTURAL (same every step
# regardless of tokens), or the per-step estimate is already precise at 32768 tokens
# so beta2 has nothing left to average. scripts/grad_shape_splithalf.py separates
# them by correlating the shape from two disjoint halves of the SAME step, as a
# function of tokens per half:
#
#     tokens/estimate   2048    4096    8192    16384   32768
#     split-half cos    0.9594  0.9617  0.9733  0.9658  0.9525
#
# FLAT over a 16x token range, so the residual disagreement is not sampling noise
# that more tokens (or a longer EMA) would average away. And it sits at the same
# value as the across-step autocorrelation (lag 1: 0.9629, lag 40: 0.9509,
# scripts/grad_shape_autocorr.py), so a one-step-stale metric is as good an estimate
# of the current shape as a fresh one computed from the same batch -- staleness has
# nothing to cost.
#
# The prediction is therefore that beta2 stays inert HERE too. This arm is the loss
# ground truth for that prediction: the diagnostic measures the metric shape, the
# sweep measures the objective. If the curves DO separate at small batch, the
# diagnostic is the wrong proxy and staleness mitigation is back on the table.
#
# CONTRAST, not a measurement: 2000 steps at 1/16 the tokens is far off the
# canonical horizon, so read only whether the curves SEPARATE, never the losses.
# Note the grid is swept in beta2, which is a window in STEPS, so the same beta2
# here averages 1/16 as many tokens as in sweep_protagonist_beta2.sh. Marek et al.
# (arXiv:2507.07101, docs/papers/small_batch_2507.07101.pdf) Eq. 2 makes the
# batch-invariant axis the token half-life (B*T)*ln(1/2)/ln(beta2); report the grid
# in that unit when comparing the two sweeps.
# The P,Q metric is accumulated AFTER the step that uses it (optim.py:2020-2024 apply,
# :2068-2069 accumulate), so the metric is stale by one step. At b2=0.99 that is 1% of
# the ~100-step EMA window; at b2=b1^2=0.81 the window is ~5.3 steps and the staleness
# is ~19% of it. This sweep asks whether that matters.
# + Nesterov momentum (β1), k=1. Parameterized by model / data_dir / lora_r as trailing
# positionals (encoded per-cell in the params JSON) so the cell is captured in the task
# line — env vars do NOT propagate through submit.sh --emit-pending.
#
# Positional args (must match params JSON key order):
#   1: lr  2: optimizer  3: seed  4: precond_delta  5: beta1  6: model  7: data_dir  8: lora_r
#   9: precond_method (OPTIONAL — empty=gram_ns (protagonist default); "eigh"/"higham" override)
#  10: cw_metric_init (OPTIONAL — default "1e-12" = εI branch-free init; "zero"/"ones"/"delta" are ablations)
#  11: cw_solved_rho (OPTIONAL — "1" adds --cw_solved_rho, the solved magnitude rule; default off)
lr=${1:-3e-2}
optimizer=${2:-kl-diag-polar-lora}      # paper protagonist (was diag-shampoo-polar-lora; pivot 2026-06-11)
seed=${3:-0}
precond_delta=${4:-1e-4}
beta1=${5:-0.9}                          # locked protagonist β₁ (was 0.95)
model=${6:-allenai/OLMo-2-0425-1B}
data_dir=${7:-data/opc_sft_stage2_all_packed_seq2048}
lora_r=${8:-256}
curvature_beta=${9:-0.99}          # SWEPT AXIS: EMA horizon 1/(1-b2) of the P,Q metric
precond_method=${10:-gram_ns}
cw_metric_init=${11:-1e-12}
cw_solved_rho=${12:-0}

solved_args=()
[ "$cw_solved_rho" = "1" ] && solved_args=(--cw_solved_rho)

# Inverse-sqrt method. Default gram_ns (protagonist). Pass an explicit "eigh"/"higham" at
# positional 9 to override; pass the empty string to fall through to train.py default None
# -> curvature-whiten family default eigh (the legacy 8-positional behavior).
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
    --batch_size "${BATCH_SIZE:-1}" \
    --grad_accum_steps "${GRAD_ACCUM:-1}" \
    --max_steps "${MAX_STEPS:-2000}" \
    --eval_every "${EVAL_EVERY:-100}" \
    --lr "$lr" \
    --optimizer "$optimizer" \
    --seed "$seed" \
    --lora_r "$lora_r" \
    --lora_alpha "$lora_r" \
    --beta1 "$beta1" \
    --curvature_beta "$curvature_beta" \
    --precond_refresh_every 10 \
    --precond_delta "$precond_delta" \
    --polar_method polar_express \
    --muon_ns_steps 8 \
    --cw_picard_iters 1 \
    --cw_nesterov \
    --cw_metric_init "$cw_metric_init" \
    "${solved_args[@]}" \
    "${precond_args[@]}" \
    "${diag_args[@]}" \
    --optim_diagnostics_every 100 \
    "${ckpt_args[@]}"
