#!/usr/bin/env bash
# Canonical "regenerate the leaderboard doc now" entry point.
#
# Usage:
#   ./scripts/analysis/update_leaderboard.sh          # regenerate worktree doc
#   ./scripts/analysis/update_leaderboard.sh --stage  # regenerate + stage doc
#
# Regenerates docs/notes/leaderboard.md from the sealed publication archive. Handles conda
# env activation for you (the underlying python script needs the `lora_playground`
# package importable). ``--stage`` renders to a temporary file, protects a
# conflicting unstaged document edit, and stages only the reviewed generated
# output.
set -eo pipefail  # NOT -u: conda activate scripts abort under set -u

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
DOC_REL="docs/notes/leaderboard.md"
DOC="${ROOT}/${DOC_REL}"
GENERATOR_REL="scripts/analysis/build_leaderboard_doc.py"
ARCHIVE_REL="publication/legacy_leaderboard_v1.json"
MODE="write"

case "${1:-}" in
  --stage)
    MODE="stage"
    shift
    ;;
esac

# Activate the project env only if lora_playground isn't already importable.
if ! python -c "import lora_playground" 2>/dev/null; then
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate ffcv-pl
fi

apply_and_stage() {
  local fresh="$1"
  # `git diff` here is worktree-vs-index. A staged manual edit is replaceable:
  # generated files do not accept hand-written content. An UNSTAGED edit may be
  # the user's work, so only stage it directly when it already equals `fresh`.
  if ! git -C "$ROOT" diff --quiet -- "$DOC_REL"; then
    if ! cmp -s "$fresh" "$DOC"; then
      echo "leaderboard update: refusing to overwrite unstaged ${DOC_REL}" >&2
      echo "review or stage that edit, then run: ./scripts/analysis/update_leaderboard.sh --stage" >&2
      return 1
    fi
  elif ! cmp -s "$fresh" "$DOC"; then
    cp "$fresh" "$DOC"
  fi
  git -C "$ROOT" add -- "$DOC_REL"
}

if [[ "$MODE" == "stage" ]]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  FRESH="${TMP_DIR}/leaderboard.md"
  mkdir -p "${TMP_DIR}/mpl" "${TMP_DIR}/cache"
  MPLCONFIGDIR="${TMP_DIR}/mpl" XDG_CACHE_HOME="${TMP_DIR}/cache" \
    python "${ROOT}/${GENERATOR_REL}" \
    --archive "${ROOT}/${ARCHIVE_REL}" \
    --output "$FRESH" --require-archive "$@"
  apply_and_stage "$FRESH"
  echo "regenerated and staged ${DOC_REL}"
  exit 0
fi

python "${ROOT}/${GENERATOR_REL}" "$@"
