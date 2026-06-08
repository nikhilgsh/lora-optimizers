#!/bin/bash
#SBATCH --job-name=klssc_picard_bw_bench
#SBATCH --partition=gpu
#SBATCH --constraint=rtxblackwell
#SBATCH --reservation=rocky9
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/klssc_picard_bw_bench_%j.out
#SBATCH --error=slurm_logs/klssc_picard_bw_bench_%j.err

# TIMING_BASIS: local A6000 preliminary logs/bench/kl_diag_ssc_history_a6000_prelim.jsonl
# TIMING_ESTIMATE: about 3 minutes for k=1+k=2; 30 minutes is >4x with model-load slack.

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate ffcv-pl
set -eo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export HF_HOME="${HOME}/ceph/hf_cache"

export OUT="logs/bench/kl_diag_ssc_history_blackwell_${SLURM_JOB_ID}.jsonl"

echo "# host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "# out=${OUT}"

python scripts/bench/bench_optimizer_step.py \
  --model_name allenai/OLMo-2-0425-1B \
  --optimizers kl-diag-ssc-history-picard-lora \
  --cw_picard_iters 1 \
  --lora_r 256 \
  --lora_alpha 256 \
  --target_modules all-linear \
  --bf16 \
  --attn_implementation sdpa \
  --batch_size 1 \
  --seq_len 2048 \
  --grad_accum_steps 4 \
  --precond_refresh_every 10 \
  --n_warmup 1 \
  --n_cycles 1 \
  --out "${OUT}"

python scripts/bench/bench_optimizer_step.py \
  --model_name allenai/OLMo-2-0425-1B \
  --optimizers kl-diag-ssc-history-picard-lora \
  --cw_picard_iters 2 \
  --lora_r 256 \
  --lora_alpha 256 \
  --target_modules all-linear \
  --bf16 \
  --attn_implementation sdpa \
  --batch_size 1 \
  --seq_len 2048 \
  --grad_accum_steps 4 \
  --precond_refresh_every 10 \
  --n_warmup 1 \
  --n_cycles 1 \
  --out "${OUT}"

python - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["OUT"])
rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
ks = {
    row["cw_picard_iters"]: row
    for row in rows
    if row["optimizer"] == "kl-diag-ssc-history-picard-lora"
}
k1 = ks[1]
k2 = ks[2]
print("# summary")
print(f"k1_total_sec={k1['mean_sec_per_step']:.6f}")
print(f"k2_total_sec={k2['mean_sec_per_step']:.6f}")
print(f"total_overhead_frac={k2['mean_sec_per_step'] / k1['mean_sec_per_step'] - 1:.6f}")
print(f"k1_opt_sec={k1['opt_sec_per_step']:.6f}")
print(f"k2_opt_sec={k2['opt_sec_per_step']:.6f}")
print(f"opt_overhead_frac={k2['opt_sec_per_step'] / k1['opt_sec_per_step'] - 1:.6f}")
PY
