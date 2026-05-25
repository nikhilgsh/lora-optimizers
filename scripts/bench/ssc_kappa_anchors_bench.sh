#!/bin/bash
# Anchor cells for SSC κ-adaptive comparison: fixed-c SSC, NS polar (no SSC),
# AdamW. Same shape as refresh_bench (r=256 picard=2 100 steps no-compile bf16
# packed_v1) so the per-step walls are directly comparable.
set -eo pipefail   # not -u: conda activate scripts reference unbound ADDR2LINE

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

OUT="$REPO/logs/bench_ssc_drift"
mkdir -p "$OUT"

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ffcv-pl
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

COMMON=(
  --data_dir data/magicoder_seq512_70k_packed
  --data_pipeline_version packed_v1
  --attn_implementation sdpa --device cuda --bf16
  --max_steps 100 --eval_every 20
  --lora_r 256 --lora_alpha 256 --lora_init_b zero --seed 0
  --batch_size 2 --grad_accum_steps 8 --max_seq_length 512
  --max_train_samples 8000 --max_eval_samples 512
  --target_modules all-linear
  --beta1 0.9 --beta2 0.999 --weight_decay 0.0 --max_grad_norm 1.0
  --lr_scheduler_type constant --warmup_steps 0
  --lr 1e-2
  --log_basic_diagnostics --optim_diagnostics_every 1
)

CHORD_COMMON=(
  --optimizer adam-polar-product-lora-coupled-spectral-chord-tight-clean
  --picard_iters_override 2 --muon_ns_steps 10
  --precond_method higham --precond_delta 1e-6
  --polar_norm_dir frob --ns_form gram
)

# Cell: fixed-c SSC (cheap-SSC ceiling, no eigvalsh).
echo "=== anchor: fixed-c SSC c=0.3 ===" | tee -a "$OUT/summary.log"
python -u train_lora.py "${COMMON[@]}" "${CHORD_COMMON[@]}" \
  --polar_method ssc --ssc_c 0.3 --ssc_nsteps 10 \
  > "$OUT/anchor_fixedc.log" 2>&1
grep -E '"event": "eval"' "$OUT/anchor_fixedc.log" | tail -1 | tee -a "$OUT/summary.log"
echo "" | tee -a "$OUT/summary.log"

# Cell: NS polar (no SSC). Drops the SSC layer entirely.
echo "=== anchor: NS polar (no SSC) ===" | tee -a "$OUT/summary.log"
python -u train_lora.py "${COMMON[@]}" "${CHORD_COMMON[@]}" \
  --polar_method ns \
  > "$OUT/anchor_ns.log" 2>&1
grep -E '"event": "eval"' "$OUT/anchor_ns.log" | tail -1 | tee -a "$OUT/summary.log"
echo "" | tee -a "$OUT/summary.log"

# Cell: AdamW (cheapest optimizer baseline). No chord-tight, no polar, no SSC.
echo "=== anchor: AdamW ===" | tee -a "$OUT/summary.log"
python -u train_lora.py "${COMMON[@]}" \
  --optimizer adamw \
  > "$OUT/anchor_adamw.log" 2>&1
grep -E '"event": "eval"' "$OUT/anchor_adamw.log" | tail -1 | tee -a "$OUT/summary.log"
echo "" | tee -a "$OUT/summary.log"

echo "DONE_ANCHORS"
