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

# PICKING AN INTERPRETER. The global watcher (slurm-pending-watchd) runs
# submit-pending -> this script from a systemd environment whose PATH carries
# conda's `condabin` but has NO activated env (CONDA_SHLVL=0), so bare `python`
# resolves to /usr/bin/python = 3.6.8. That parses neither
# `from __future__ import annotations` (3.7+) nor the `dict | None` annotations
# this repo uses (3.10+): it exits with a SyntaxError, submit-pending reads that
# nonzero status as "dirty-tree check failed", and every drain cycle silently
# refuses to submit a perfectly clean tree.
#
# Resolve the env's interpreter by ABSOLUTE PATH rather than `conda activate`.
# Activation runs the env's activate.d hooks, and ffcv-pl's cuda_12.8.sh calls
# `module load cuda/12.8.0`; Lmod's shell function does not exist under systemd,
# so activating there dies with "module: command not found" (exit 127) — trading
# one silent refusal for another. Nothing this script does needs CUDA or any
# other activation side effect: it runs git/hashing and file parsing only.
# When the caller already has a new-enough python (e.g. slurm_scripts/submit.sh
# invoked from an active env), keep it, so this is a no-op on the interactive path.
PY_MIN='import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'
if command -v python >/dev/null 2>&1 && python -c "$PY_MIN" 2>/dev/null; then
    PY=python
elif [[ -x "$HOME/miniforge3/envs/ffcv-pl/bin/python" ]]; then
    PY="$HOME/miniforge3/envs/ffcv-pl/bin/python"
else
    echo "check_clean_tree: no python >= 3.10 found (tried PATH and" >&2
    echo "  ~/miniforge3/envs/ffcv-pl/bin/python). Cannot verify tree state." >&2
    exit 1
fi

# Sanity-check pending sbatches before the load-bearing-cleanliness check:
# refuse if any sbatch declares --ntasks=N while its disBatch task block
# generates a different number of tasks. Catches the "inherited ntasks from
# prior sbatch, didn't recompute for this grid" failure mode — running
# 5 cells through --ntasks=3 doubles wall time and the user pays for
# whatever differential exists.
#
# Override: FORCE_NTASKS_MISMATCH=1 ./check_clean_tree.sh
"$PY" scripts/check_pending_sbatches.py

# Orchestration lint (ml_utils.sbatch_lint): the SAME static lint submit-pending
# runs — catches sbatch-BODY failures that bash -n and component smokes miss
# (assignment to a bash special var like GROUPS, a non-executable wrapper →
# Permission denied/exit 126, CELLS-vs-ntasks mismatch, syntax errors). Running
# it HERE means the build-time gate matches submit-pending's gate, so these are
# caught when the sbatch is written, not at submit time. Skips gracefully if
# ml_utils isn't importable; override with SKIP_SBATCH_LINT=1.
if [[ "${SKIP_SBATCH_LINT:-0}" != "1" ]] && "$PY" -c "import ml_utils.sbatch_lint" 2>/dev/null; then
    shopt -s nullglob
    PENDING_SBATCHES=(slurm_pending/*.sbatch)
    if [[ ${#PENDING_SBATCHES[@]} -gt 0 ]]; then
        "$PY" -m ml_utils.sbatch_lint "${PENDING_SBATCHES[@]}"
    fi
fi

exec "$PY" -m lora_playground.execution_scope check-clean
