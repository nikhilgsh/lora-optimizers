#!/usr/bin/env bash
# Sync paper/manuscript/ <-> the lora-paper GitHub repo that Overleaf mirrors.
#
#   ./paper/sync.sh push   # local commits          -> GitHub -> Overleaf
#   ./paper/sync.sh pull   # Overleaf/GitHub edits   -> local paper/manuscript/
#
# subtree operates on COMMITTED history, not the working tree: commit your
# changes first, then push. If you also edit on Overleaf, always `pull` before
# `push` (a push is rejected if GitHub has commits you don't have).
#
# The default system git lacks git-subtree; git/2.48.1 bundles it. modules.sh
# makes `module` work inside this non-interactive script.
set -eo pipefail

source /etc/profile.d/modules.sh
module load git/2.48.1

cd "$(git rev-parse --show-toplevel)"

PREFIX="paper/manuscript"
REMOTE="paper"
BRANCH="main"

case "${1:-}" in
  push) git subtree push --prefix="$PREFIX" "$REMOTE" "$BRANCH" ;;
  pull) git subtree pull --prefix="$PREFIX" "$REMOTE" "$BRANCH" --squash ;;
  *)    echo "usage: $0 {push|pull}   (commit first; run from anywhere in the lora repo)"; exit 1 ;;
esac
