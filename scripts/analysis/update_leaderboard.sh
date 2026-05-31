#!/usr/bin/env bash
# Canonical "regenerate the leaderboard doc now" entry point.
#
# Usage:  ./scripts/analysis/update_leaderboard.sh
#
# Regenerates docs/notes/leaderboard.md from the live logs/ tree. Handles conda
# env activation for you (the underlying python script needs the `lora_playground`
# package importable). The git pre-commit hook reuses this same script, so env
# activation + regen logic lives in exactly one place.
set -eo pipefail  # NOT -u: conda activate scripts abort under set -u

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

# Activate the project env only if lora_playground isn't already importable.
if ! python -c "import lora_playground" 2>/dev/null; then
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate ffcv-pl
fi

python "$ROOT/scripts/analysis/build_leaderboard_doc.py" "$@"
