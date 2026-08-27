#!/bin/bash
# Qwen preconditioner sweeps on H200. The execution root is the submitted
# worktree, so clean isolated worktrees do not fall back to the main checkout.
# TIMING_BASIS: extrapolated from the same Qwen/OpenMath/r=256 optimizer family
# on gpuxl H200 (about 3.25 h through step 9000), with snapshot I/O headroom.
#SBATCH -p gpuxl
#SBATCH --constraint=h200
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
[[ -z "$(git status --short)" ]] || {
    echo "execution worktree is dirty" >&2
    git status --short >&2
    exit 2
}

mkdir -p slurm_logs disbatch_logs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ffcv-pl
set -eo pipefail
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
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
