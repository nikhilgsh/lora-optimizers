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

exec python -m lora_playground.execution_scope check-clean
