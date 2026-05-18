#!/bin/bash
# Bench chord-tight-clean k=2 gram under three compile configurations,
# to find the smallest-walltime configuration for production.
#
#   B: eager                       (LORA_COMPILE_KERNELS=0)
#   C: torch.compile default       (LORA_COMPILE_KERNELS=1, fullgraph=False)
#   D: torch.compile reduce-overhead (LORA_COMPILE_KERNELS=2, fullgraph=True, CUDA graphs)
#
# AdamW reference timed alongside as ×AdamW comparator.
# Plain fp32 Higham throughout — fp16 Higham showed only 0.5% step
# savings at r=256, dropped from this comparison.
cd /mnt/home/nghosh/lora
source /mnt/home/nghosh/miniforge3/bin/activate ffcv-pl
export WANDB_MODE=offline

OUT=/tmp/bench_compile_modes_$(date +%H%M%S).log

COMMON_FLAGS="\
  --model_name allenai/OLMo-2-0425-1B \
  --target_modules all-linear \
  --bf16 \
  --batch_size 2 --seq_len 512 --grad_accum_steps 8 \
  --n_warmup 5 --n_cycles 4 \
  --precond_refresh_every 1 \
  --precond_method higham \
  --higham_iters 10 \
  --ns_form gram \
  --picard_iters_override 2 \
  --higham_compute_dtype fp32"

OPTIMIZERS="adamw adam-polar-product-lora-coupled-spectral-chord-tight-clean"

run_one() {
  local lora_r="$1"
  local label="$2"
  local lck="$3"
  echo "" >> $OUT
  echo "##### lora_r=$lora_r  $label (LORA_COMPILE_KERNELS=$lck) #####" >> $OUT
  LORA_COMPILE_KERNELS=$lck \
  python scripts/bench/bench_optimizer_step.py \
    $COMMON_FLAGS \
    --lora_r $lora_r --lora_alpha $lora_r \
    --optimizers $OPTIMIZERS \
    >> $OUT 2>&1
}

for r in 64 256; do
  run_one $r "B eager"             0 || true
  run_one $r "C compile default"   1 || true
  run_one $r "D compile reduce-overhead" 2 || true
done

echo "=== OUTPUT SAVED TO $OUT ===" >> $OUT
cat $OUT
