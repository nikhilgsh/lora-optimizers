#!/bin/bash
# Compare fp32 vs fp16 Higham on a chord-tight-clean k=2 training smoke.
# Emits JSONL events to stdout; redirect to file per variant. Uses tiny
# fixture data so we can run many steps fast — the comparison is "does
# fp16+polish trajectory track fp32?", not "does the model learn".
set -e
cd /mnt/home/nghosh/lora
source /mnt/home/nghosh/miniforge3/bin/activate ffcv-pl
export WANDB_MODE=offline

N_STEPS=${1:-500}
EVAL_EVERY=${2:-50}

COMMON="--device cuda --model_name allenai/OLMo-2-0425-1B \
  --train_file tests/fixtures/tiny_code_train.jsonl \
  --eval_file tests/fixtures/tiny_code_eval.jsonl \
  --optimizer adam-polar-product-lora-coupled-spectral-chord-tight-clean \
  --picard_iters_override 2 \
  --max_steps $N_STEPS --eval_every $EVAL_EVERY \
  --batch_size 1 --grad_accum_steps 1 \
  --max_seq_length 256 --lora_r 64 --lora_alpha 64 --bf16 \
  --precond_method higham --higham_iters 10 \
  --precond_delta 1e-6 \
  --seed 42 \
  --lr 3e-3 \
  --ns_form gram \
  --allow_multi_epoch \
  --no-log_basic_diagnostics"

mkdir -p /tmp/smoke_higham_runs

echo "=== fp32 Higham (baseline) — $N_STEPS steps ==="
python train_lora.py $COMMON --higham_compute_dtype fp32 \
  > /tmp/smoke_higham_runs/fp32.jsonl 2>&1
echo "Last 5 eval events:"
grep '"event": "eval"' /tmp/smoke_higham_runs/fp32.jsonl | tail -5

echo ""
echo "=== fp16 + polish Higham (variant B) — $N_STEPS steps ==="
python train_lora.py $COMMON --higham_compute_dtype fp16 \
  > /tmp/smoke_higham_runs/fp16.jsonl 2>&1
echo "Last 5 eval events:"
grep '"event": "eval"' /tmp/smoke_higham_runs/fp16.jsonl | tail -5
