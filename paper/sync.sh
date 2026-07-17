#!/usr/bin/env bash
# Sync paper/manuscript/ <-> the Overleaf project's native git bridge
# (https://git.overleaf.com/<project-id>), no GitHub middleman: a push lands
# directly in the Overleaf project, a pull grabs the editor's edits. There is
# no manual sync step on the Overleaf side.
#
#   ./paper/sync.sh push ["commit message"]    # local paper/manuscript -> Overleaf (direct)
#   ./paper/sync.sh pull                        # Overleaf editor edits   -> local paper/manuscript
#   ./paper/sync.sh publish ["commit message"]  # push HEAD to origin + Overleaf in one call
#
# WHY NOT `git subtree`:
#   lora-paper's history was created by Overleaf's GitHub bridge, not by
#   `git subtree split`, so the synthetic commit SHAs `git subtree push`
#   produces never match the remote -> every push is non-fast-forward. And
#   `git subtree pull --squash` fails ("prefix was never added") because there
#   is no squash-add marker. So we treat the prefix as a plain tree and sync it
#   directly:
#     push -> commit-tree(<prefix tree>, parent=<remote tip>), then a ff push
#     pull -> 3-way merge of (last-synced base, local prefix, remote tip) via
#             `git merge-tree`, extracted into the working prefix
#   The last-synced remote commit is tracked in a local-only ref so push refuses
#   to clobber remote edits it has not pulled. `merge-tree --write-tree` needs
#   git >= 2.38, hence the module load.
#
# Limitation: pull extracts the merged tree over the working prefix (add/
# overwrite); a file deleted on the remote is not removed locally. Rare for this
# manuscript (text + tracked figure PDFs); delete by hand if it ever happens.
set -eo pipefail

source /etc/profile.d/modules.sh
module load git/2.48.1

cd "$(git rev-parse --show-toplevel)"

PREFIX="paper/manuscript"
REMOTE="overleaf"
BRANCH="main"
SYNC_REF="refs/sync/${REMOTE}-${BRANCH}"   # local-only: last-synced remote commit

remote_tip()  { git rev-parse "$REMOTE/$BRANCH"; }
prefix_tree() { git rev-parse "HEAD:$PREFIX"; }
tree_of()     { git rev-parse "$1^{tree}"; }

do_push() {
  local msg="${1:-}"
  git fetch -q "$REMOTE" "$BRANCH"
  local theirs base tree
  theirs=$(remote_tip)
  base=$(git rev-parse -q --verify "$SYNC_REF" 2>/dev/null || true)
  tree=$(prefix_tree)

  # Up to date: local prefix already equals the remote tip's tree.
  if [ "$tree" = "$(tree_of "$theirs")" ]; then
    git update-ref "$SYNC_REF" "$theirs"
    echo "already up to date; nothing to push."
    return 0
  fi

  # Safety: never clobber remote edits we have not merged.
  if [ -z "$base" ]; then
    echo "no sync base recorded and local prefix differs from $REMOTE/$BRANCH."
    echo "Run '$0 pull' first to merge remote state, then push."
    exit 1
  fi
  if [ "$base" != "$theirs" ]; then
    echo "refusing to push: $REMOTE/$BRANCH advanced since last sync ($base -> $theirs)."
    echo "Run '$0 pull' first, then push."
    exit 1
  fi

  [ -n "$msg" ] || msg="Sync paper/manuscript from $(git rev-parse --short HEAD) ($(date -u +%Y-%m-%dT%H:%MZ))"
  local new
  new=$(git commit-tree "$tree" -p "$theirs" -m "$msg")
  git push "$REMOTE" "$new:$BRANCH"
  git update-ref "$SYNC_REF" "$new"
  echo "pushed ${theirs:0:7}..${new:0:7} -> $REMOTE/$BRANCH"
}

do_pull() {
  # The 3-way merge below builds `ours` from HEAD:$PREFIX (the committed tree) and
  # then overwrites the working tree with `git archive | tar -x`. Uncommitted edits
  # under $PREFIX are therefore ignored by the merge AND clobbered by the extract,
  # a silent data loss. Refuse rather than destroy them (mirrors do_publish's guard).
  if [ -n "$(git status --porcelain -- "$PREFIX")" ]; then
    echo "refusing to pull: $PREFIX has uncommitted changes that pull would overwrite and lose."
    echo "Commit them first:  git add $PREFIX && git commit   (then re-run '$0 pull')"
    echo "Uncommitted:"
    git status --porcelain -- "$PREFIX" | sed 's/^/  /'
    exit 1
  fi
  git fetch -q "$REMOTE" "$BRANCH"
  local theirs base
  theirs=$(remote_tip)
  base=$(git rev-parse -q --verify "$SYNC_REF" 2>/dev/null || true)

  # First sync, no recorded base: only safe if local already equals the remote.
  if [ -z "$base" ]; then
    if [ "$(prefix_tree)" = "$(tree_of "$theirs")" ]; then
      git update-ref "$SYNC_REF" "$theirs"
      echo "initialized sync base at ${theirs:0:7}; already up to date."
      return 0
    fi
    echo "no sync base recorded and local prefix != remote tip; cannot 3-way safely."
    echo "Reconcile by hand (diff $REMOTE/$BRANCH:main.tex against HEAD:$PREFIX/main.tex),"
    echo "then: git update-ref $SYNC_REF $theirs"
    exit 1
  fi

  if [ "$base" = "$theirs" ]; then
    echo "already up to date; $REMOTE/$BRANCH unchanged since last sync."
    return 0
  fi

  # 3-way merge: base (last-synced remote) / ours (local prefix) / theirs (remote tip).
  # `ours` is a commit carrying our prefix tree parented on `base`, so base is the
  # genuine merge-base of ours and theirs and merge-tree does a real 3-way.
  local ours out rc merged
  ours=$(git commit-tree "$(prefix_tree)" -p "$base" -m "local prefix")
  set +e
  out=$(git merge-tree --write-tree "$ours" "$theirs"); rc=$?
  set -e
  merged=$(printf '%s\n' "$out" | head -1)

  git archive "$merged" | tar -x -C "$PREFIX"
  git update-ref "$SYNC_REF" "$theirs"

  if [ "$rc" -ne 0 ]; then
    echo "pulled WITH CONFLICTS into $PREFIX (conflict markers written to the files below):"
    printf '%s\n' "$out" | tail -n +2
    echo "Resolve them, then: git add $PREFIX && git commit"
  else
    echo "pulled cleanly into $PREFIX; review 'git status' / 'git diff' and commit."
  fi
}

do_publish() {
  local msg="${1:-}"
  # publish ships HEAD:$PREFIX (like push), so refuse if the prefix has
  # uncommitted edits that would be silently left behind.
  if [ -n "$(git status --porcelain -- "$PREFIX")" ]; then
    echo "refusing to publish: $PREFIX has uncommitted changes (publish ships HEAD:$PREFIX)."
    echo "Commit them first:  git add $PREFIX && git commit"
    exit 1
  fi
  echo "==> push $BRANCH -> origin (GitHub)"
  git push origin "$BRANCH"
  echo "==> push $PREFIX -> $REMOTE (Overleaf)"
  do_push "$msg"   # refuses with pull-first guidance if Overleaf advanced; re-run publish after pull+commit
}

case "${1:-}" in
  push)    shift; do_push "${1:-}" ;;
  pull)    do_pull ;;
  publish) shift; do_publish "${1:-}" ;;
  *)       echo "usage: $0 {push [\"msg\"]|pull|publish [\"msg\"]}   (run from anywhere in the lora repo)"; exit 1 ;;
esac
