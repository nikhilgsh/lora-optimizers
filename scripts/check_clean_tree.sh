#!/usr/bin/env bash
# Refuse to submit a sweep if the working tree has uncommitted changes to
# load-bearing code. Called from slurm_scripts/submit.sh and from
# ~/bin/submit-pending.
#
# This script delegates to `python -m lora_playground.execution_scope
# check-clean`, which uses the SAME closure-and-content-hash logic the
# loader uses at analysis time. The contract is: any submission this
# script accepts must produce a cfg event with execution_source_dirty=False
# (or auto-resolve cleanly to a descendant commit), so the loader will
# accept the run.
#
# Load-bearing scope = python import-closure from train_lora.py ∪
# DEFAULT_EXTRA_LOAD_BEARING_GLOBS (sweep / sbatch shell scripts). See
# `lora_playground/execution_scope.py` for the authoritative definition.
#
# Override:
#   FORCE_DIRTY=1 ./check_clean_tree.sh       # honored by the python CLI
#
# Exit:
#   0 → clean (or override). OK to submit.
#   1 → dirty in load-bearing paths. Submission should refuse.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

# The global watcher (slurm-pending-watchd) runs submit-pending -> this script in
# a conda-less systemd environment where bare `python` is not on PATH, so every
# cycle died at "python: command not found" and was misread as a dirty tree.
# Activate the project env when python is absent. No-op when called from an
# already-active env (e.g. slurm_scripts/submit.sh). set +u around activate:
# conda-forge activate.d scripts reference unbound vars and abort under set -u.
if ! command -v python >/dev/null 2>&1; then
    set +u
    source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl
    set -u
fi

# Sanity-check pending sbatches before the load-bearing-cleanliness check:
# refuse if any sbatch declares --ntasks=N while its disBatch task block
# generates a different number of tasks. Catches the "inherited ntasks from
# prior sbatch, didn't recompute for this grid" failure mode — running
# 5 cells through --ntasks=3 doubles wall time and the user pays for
# whatever differential exists.
#
# Override: FORCE_NTASKS_MISMATCH=1 ./check_clean_tree.sh
python scripts/check_pending_sbatches.py

# Orchestration lint (ml_utils.sbatch_lint): the SAME static lint submit-pending
# runs — catches sbatch-BODY failures that bash -n and component smokes miss
# (assignment to a bash special var like GROUPS, a non-executable wrapper →
# Permission denied/exit 126, CELLS-vs-ntasks mismatch, syntax errors). Running
# it HERE means the build-time gate matches submit-pending's gate, so these are
# caught when the sbatch is written, not at submit time. Skips gracefully if
# ml_utils isn't importable; override with SKIP_SBATCH_LINT=1.
if [[ "${SKIP_SBATCH_LINT:-0}" != "1" ]] && python -c "import ml_utils.sbatch_lint" 2>/dev/null; then
    shopt -s nullglob
    PENDING_SBATCHES=(slurm_pending/*.sbatch)
    if [[ ${#PENDING_SBATCHES[@]} -gt 0 ]]; then
        python -m ml_utils.sbatch_lint "${PENDING_SBATCHES[@]}"
    fi
fi

exec python -m lora_playground.execution_scope check-clean
