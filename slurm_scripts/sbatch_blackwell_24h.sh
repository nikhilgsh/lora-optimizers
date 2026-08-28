#!/bin/bash
# Blackwell long-wall variant for r>=128 diagnostic reruns.
# TIMING_MEASURED: logs/chord_tight_phase_L_lrsweep_r256_blackwell
#   log_0/log_1 reached 9000 steps in ~4.5 h train time per task.
# TIMING_BASIS: same model/data/rank/optimizer/diagnostic cadence; chord_slack
#   now uses power iteration in the basic tier.
#SBATCH -p gpu
#SBATCH --constraint=rtxblackwell
#SBATCH --reservation=rocky9
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
repo_root=$(git rev-parse --show-toplevel)
[[ "$repo_root" == "$PWD" ]] || {
    echo "execution root mismatch: pwd=$PWD repo_root=$repo_root" >&2
    exit 2
}
echo "execution_root=$repo_root"
echo "execution_commit=$(git rev-parse HEAD)"
# Scope the dirty check to the LOAD-BEARING closure, not the whole tree. A bare
# `git status --short` aborts on any untracked scratch file -- a preview PNG, a
# half-written note -- none of which the run reads, and this tree is rarely
# empty by that measure (38 entries when this was written).
# `scripts/check_clean_tree.sh` delegates to
# `lora_playground.execution_scope check-clean`, the same import-closure logic
# the loader uses at analysis time, with the contract that anything it accepts
# produces a cfg with execution_source_dirty=False and therefore loads.
# FORCE_DIRTY=1 is its documented override.
scripts/check_clean_tree.sh || {
    echo "execution worktree is dirty in load-bearing paths" >&2
    exit 2
}
mkdir -p slurm_logs disbatch_logs

source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
set -euo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_PROJECT=lora-sweeps
export TOKENIZERS_PARALLELISM=false

module --ignore_cache load disBatch
command -v disBatch >/dev/null
allocated_cpus_per_task=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}
[[ "$allocated_cpus_per_task" =~ ^[1-9][0-9]*$ ]] || {
    echo "invalid SLURM_CPUS_PER_TASK=$allocated_cpus_per_task" >&2
    exit 2
}
export SLURM_CPU_BIND=cores
disBatch --cpusPerTask "$allocated_cpus_per_task" \
    "$TASK_FILE" --prefix "disbatch_logs"
