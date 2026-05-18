#!/bin/bash
# Full-optimizer-step bench: AdamW vs chord-tight-clean k=2 gram-NS,
# with both fp32 and fp16 Higham. Production shapes (1B, packed-style
# batch). Answers the question: "What's the wall-time of the FULL
# chord-tight-clean step under fp16 Higham, vs fp32 Higham, vs AdamW?"
set -e
cd /mnt/home/nghosh/lora
source /mnt/home/nghosh/miniforge3/bin/activate ffcv-pl
export WANDB_MODE=offline

OUT=/tmp/bench_full_step_$(date +%H%M%S).log

COMMON_FLAGS="\
  --model_name allenai/OLMo-2-0425-1B \
  --target_modules all-linear \
  --bf16 \
  --batch_size 2 --seq_len 512 --grad_accum_steps 8 \
  --n_warmup 3 --n_cycles 4 \
  --precond_refresh_every 1 \
  --precond_method higham \
  --higham_iters 10 \
  --ns_form gram \
  --picard_iters_override 2"

OPTIMIZERS="adamw adam-polar-product-lora-coupled-spectral-chord-tight-clean"

run_one() {
  local lora_r="$1"
  local higham_dtype="$2"
  local lora_alpha="$lora_r"
  echo ""
  echo "##### lora_r=$lora_r higham_compute_dtype=$higham_dtype #####" >> $OUT
  python scripts/bench/bench_optimizer_step.py \
    $COMMON_FLAGS \
    --lora_r $lora_r --lora_alpha $lora_alpha \
    --optimizers $OPTIMIZERS \
    --higham_compute_dtype $higham_dtype \
    >> $OUT 2>&1
}

# Serial: 4 runs total — r=64 and r=256, each with fp32 and fp16 Higham.
# AdamW timing is independent of higham_compute_dtype (just included as
# the baseline reference; will be identical across the two dtype rows).
run_one 64 fp32
run_one 64 fp16
run_one 256 fp32
run_one 256 fp16

echo ""
echo "=== FULL OUTPUT ==="
cat $OUT
echo "=== OUTPUT SAVED TO $OUT ==="
