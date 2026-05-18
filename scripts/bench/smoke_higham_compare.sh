#!/bin/bash
# Compare fp32 vs fp16 Higham on the PRODUCTION chord-tight-clean k=2 gram
# config, real Magicoder packed_v1 data. Mirrors
# `scripts/sweep/sweep_validation_gram_ns.sh` line-for-line on the algorithm
# flags; only the precision lever differs across the two runs.
set -e
cd /mnt/home/nghosh/lora
source /mnt/home/nghosh/miniforge3/bin/activate ffcv-pl
export WANDB_MODE=offline

N_STEPS=${1:-500}
EVAL_EVERY=${2:-50}
LORA_R=${3:-64}
LR=${4:-3e-2}

COMMON="\
  --data_dir data/magicoder_seq512_70k_packed \
  --data_pipeline_version packed_v1 \
  --attn_implementation sdpa \
  --device cuda --bf16 \
  --optimizer adam-polar-product-lora-coupled-spectral-chord-tight-clean \
  --max_steps $N_STEPS --eval_every $EVAL_EVERY \
  --lr $LR \
  --seed 0 \
  --lora_r $LORA_R --lora_alpha $LORA_R \
  --muon_ns_steps 5 \
  --precond_method higham --higham_iters 10 \
  --precond_delta_relative \
  --precond_delta 1e-2 \
  --picard_iters_override 2 \
  --ns_form gram \
  --no-log_basic_diagnostics \
  --optim_diagnostics_every 1000"

# No --compile in the smoke (compile adds ~60s fixed overhead that hides
# the trajectory comparison; production sweeps DO use compile).

mkdir -p /tmp/smoke_higham_prod

echo "=== fp32 Higham (baseline) — $N_STEPS steps, r=$LORA_R, lr=$LR ==="
python train_lora.py $COMMON --higham_compute_dtype fp32 \
  > /tmp/smoke_higham_prod/fp32.jsonl 2>&1
echo "Last 5 eval events:"
grep '"event": "eval"' /tmp/smoke_higham_prod/fp32.jsonl | tail -5

echo ""
echo "=== fp16 + polish Higham (variant B) — $N_STEPS steps ==="
python train_lora.py $COMMON --higham_compute_dtype fp16 \
  > /tmp/smoke_higham_prod/fp16.jsonl 2>&1
echo "Last 5 eval events:"
grep '"event": "eval"' /tmp/smoke_higham_prod/fp16.jsonl | tail -5
