#!/usr/bin/env bash
# Sync paper/loss-sens/ <-> the "lora-paper (loss-sens)" Overleaf project.
#
#   ./paper/sync-loss-sens.sh pull            # Overleaf -> paper/loss-sens/
#   ./paper/sync-loss-sens.sh push ["msg"]    # paper/loss-sens/ -> Overleaf
#   ./paper/sync-loss-sens.sh status          # what each side is at
#
# Three or more people edit this project at once, so push REFUSES when the
# Overleaf tip has moved past the last pull. Assembling a tree by hand and
# pushing it silently reverts whatever landed in between; that nearly happened
# once already. Push also refuses on a failed build or an undefined reference,
# since both states are cheaper to find here than in the Overleaf log.
set -eo pipefail

cd "$(git rev-parse --show-toplevel)"
PREFIX="paper/loss-sens"
REMOTE="overleaf-review"
BRANCH="main"
SYNC_REF="refs/sync/${REMOTE}-${BRANCH}"

remote_tip() { git rev-parse "$REMOTE/$BRANCH"; }
base_ref()   { git rev-parse -q --verify "$SYNC_REF" 2>/dev/null || true; }

# Tree of the working prefix, honouring its .gitignore, plus main.bbl.
# The .gitignore ignores *.bbl and lives INSIDE the synced tree, so Overleaf
# owns it and a pull can revert any exemption written there. The Overleaf
# project needs main.bbl, so the rule lives here instead, where a pull cannot
# reach it. Never use `add -A -f`: that drags in every build artifact, which
# has been pushed by accident twice.
prefix_tree() {
  local idx; idx=$(mktemp)
  rm -f "$idx"
  GIT_INDEX_FILE="$idx" git --work-tree="$PREFIX" add -A .
  GIT_INDEX_FILE="$idx" git --work-tree="$PREFIX" add -f main.bbl
  GIT_INDEX_FILE="$idx" git write-tree
  rm -f "$idx"
}

build_or_die() {
  source /etc/profile.d/modules.sh
  module load texlive/20240312
  local log; log=$(mktemp)
  ( cd "$PREFIX" && latexmk -C >/dev/null 2>&1
    timeout 600 latexmk -pdf -interaction=nonstopmode main.tex </dev/null >"$log" 2>&1 ) || true
  local errs undef
  errs=$(grep -c '^!' "$log" || true)
  undef=$(grep -ci 'undefined' "$PREFIX/main.log" 2>/dev/null || echo 0)
  if [ "$errs" -ne 0 ]; then
    echo "refusing to push: $errs LaTeX error(s)."; grep -m5 '^!' "$log"; exit 1
  fi
  if [ "$undef" -ne 0 ]; then
    echo "refusing to push: undefined references."
    grep -i 'undefined' "$PREFIX/main.log" | sort -u | head -5
    echo "(override with ALLOW_UNDEF=1 if a coauthor's edit is mid-flight)"
    [ -n "$ALLOW_UNDEF" ] || exit 1
  fi
  echo "build clean: $(grep -o 'main.pdf ([0-9]*' "$PREFIX/main.log" | tail -1 | tr -dc 0-9) pages"
  ( cd "$PREFIX" && latexmk -c >/dev/null 2>&1 )
  git archive "$(remote_tip)" main.bbl 2>/dev/null | tar -x -C "$PREFIX" || true
}

case "${1:-}" in
  status)
    git fetch -q "$REMOTE" "$BRANCH"
    echo "overleaf  $(remote_tip | cut -c1-7)"
    echo "last pull $(base_ref | cut -c1-7)"
    [ "$(base_ref)" = "$(remote_tip)" ] && echo "in sync" || echo "OVERLEAF HAS ADVANCED -- pull before pushing"
    ;;
  pull)
    if [ -n "$(git status --porcelain -- "$PREFIX")" ]; then
      echo "refusing to pull: $PREFIX has uncommitted changes that pull would overwrite."
      git status --porcelain -- "$PREFIX" | sed 's/^/  /'
      echo "Commit them first, then re-run pull and reapply on top."
      exit 1
    fi
    git fetch -q "$REMOTE" "$BRANCH"
    rm -rf "${PREFIX:?}"/*
    git archive "$(remote_tip)" | tar -x -C "$PREFIX"
    git update-ref "$SYNC_REF" "$(remote_tip)"
    echo "pulled $(remote_tip | cut -c1-7) into $PREFIX"
    ;;
  push)
    git fetch -q "$REMOTE" "$BRANCH"
    theirs=$(remote_tip); base=$(base_ref)
    if [ -z "$base" ]; then
      echo "no sync base recorded; run '$0 pull' first."; exit 1
    fi
    if [ "$base" != "$theirs" ]; then
      echo "refusing to push: $REMOTE/$BRANCH advanced since last pull (${base:0:7} -> ${theirs:0:7})."
      echo "Someone else pushed. Run '$0 pull' and reapply your edits on top."
      exit 1
    fi
    build_or_die
    tree=$(prefix_tree)
    msg="${2:-Sync $PREFIX ($(date -u +%Y-%m-%dT%H:%MZ))}"
    new=$(git commit-tree "$tree" -p "$theirs" -m "$msg")
    git push "$REMOTE" "$new:$BRANCH"
    git update-ref "$SYNC_REF" "$new"
    echo "pushed ${theirs:0:7}..${new:0:7} -> $REMOTE/$BRANCH"
    ;;
  *)
    echo "usage: $0 {pull|push [msg]|status}"; exit 1 ;;
esac
