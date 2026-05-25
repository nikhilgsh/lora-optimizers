#!/bin/bash
# End-to-end bench: SSC κ=0.6 picard=2 across (r, refresh-every-N).
# Run on a Blackwell allocation: bash scripts/bench/ssc_kappa_refresh_bench.sh
#
# 60 steps, no compile (per CLAUDE.md ratio-test rule). Eval at step 60.
# Output: logs/bench_ssc_drift/refresh_<r>_<N>.log
set -eo pipefail   # not -u: conda activate scripts reference unbound ADDR2LINE

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT="$REPO/logs/bench_ssc_drift"
mkdir -p "$OUT"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ffcv-pl
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

# Production-faithful config minus --compile and shrunk to 60 steps.
COMMON_ARGS=(
  --data_dir data/magicoder_seq512_70k_packed
  --data_pipeline_version packed_v1
  --attn_implementation sdpa --device cuda --bf16
  --max_steps 100 --eval_every 20
  --optimizer adam-polar-product-lora-coupled-spectral-chord-tight-clean
  --polar_method ssc --ssc_kappa 0.6 --ssc_nsteps 10
  --picard_iters_override 2 --muon_ns_steps 10
  --precond_method higham --precond_delta 1e-6
  --polar_norm_dir frob --ns_form gram
  --lora_init_b zero --seed 0
  --batch_size 2 --grad_accum_steps 8 --max_seq_length 512
  --max_train_samples 8000 --max_eval_samples 512
  --target_modules all-linear
  --beta1 0.9 --beta2 0.999 --weight_decay 0.0 --max_grad_norm 1.0
  --lr_scheduler_type constant --warmup_steps 0
  --lr 1e-2
  --log_basic_diagnostics --optim_diagnostics_every 1
)

for R in 256; do
  for N in 1 5 10; do
    LOG="$OUT/refresh_r${R}_N${N}.log"
    echo "=== r=${R} refresh=${N} ===" | tee -a "$OUT/summary.log"
    if [[ -s "$LOG" ]] && grep -q '"event": "eval"' "$LOG"; then
      echo "  skip (existing log has eval event)" | tee -a "$OUT/summary.log"
      continue
    fi
    python -u train_lora.py "${COMMON_ARGS[@]}" \
      --lora_r "$R" --lora_alpha "$R" \
      --ssc_kappa_refresh_every "$N" \
      > "$LOG" 2>&1
    # Pull per-eval row + total wall.
    grep -E '"event": "eval"|"event": "config"' "$LOG" | tail -3 | tee -a "$OUT/summary.log"
    echo "" | tee -a "$OUT/summary.log"
  done
done

echo "DONE"
