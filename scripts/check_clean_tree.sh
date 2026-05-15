#!/usr/bin/env bash
# Refuse to submit a sweep if the working tree has uncommitted changes to
# load-bearing code. Called from slurm_scripts/submit.sh and from
# ~/bin/submit-pending.
#
# Why this exists: the loader requires either git_dirty=false or an explicit
# attestation in lora_playground/exclusions/dirty_attestations.json. Sweeps
# that run with a dirty tree get silently excluded from analysis until a
# human writes an attestation by hand — discovered hours later when the runs
# don't appear in the notebook. This check kills that failure mode at
# submission time, before GPU is allocated.
#
# Load-bearing paths (touching these can change optimizer/training behavior):
#   lora_playground/*.py
#   train_lora.py
#   scripts/sweep/*.sh
#   slurm_scripts/*.sh
#
# Not load-bearing (changes here do NOT block):
#   docs/  notebooks/  tests/  params/  *.md  README*  *.ipynb
#   lora_playground/exclusions/*.json (data artifacts, not code)
#
# Override (use with explicit intent and a SWEEP_PURPOSE that mentions it):
#   FORCE_DIRTY=1  ./submit.sh ...   # accept dirty tree
#
# Exit:
#   0 → clean tree (or override). OK to submit.
#   1 → dirty in load-bearing paths. Submission should refuse.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$REPO_DIR"

if [[ -n "${FORCE_DIRTY:-}" ]]; then
    echo "check_clean_tree: FORCE_DIRTY=1 set — skipping dirty-tree check." >&2
    exit 0
fi

# Collect porcelain lines for load-bearing paths only.
DIRTY_LINES=$(git status --porcelain 2>/dev/null | awk '
    {
        path = $0; sub(/^...?/, "", path);
        sub(/^"/, "", path); sub(/"$/, "", path);
        if (path ~ /^lora_playground\/exclusions\//) next;
        if (path ~ /^lora_playground\/.*\.py$/) print;
        else if (path == "train_lora.py") print;
        else if (path ~ /^scripts\/sweep\/.*\.sh$/) print;
        else if (path ~ /^slurm_scripts\/.*\.sh$/) print;
    }
')

if [[ -z "$DIRTY_LINES" ]]; then
    exit 0
fi

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

cat <<EOF >&2

╭─ DIRTY-TREE SUBMISSION BLOCKED ─────────────────────────────────────╮
│ Working tree has uncommitted changes to LOAD-BEARING code. Submitting
│ now will record git_dirty=true in the manifest and the loader will
│ EXCLUDE these runs from analysis until you write an attestation.
│
│ Dirty load-bearing files:
EOF

while IFS= read -r line; do
    echo "│   $line" >&2
done <<<"$DIRTY_LINES"

cat <<EOF >&2
│
│ Current HEAD: $GIT_COMMIT
│
│ Resolve by ONE of:
│
│   1. COMMIT the changes (preferred — clean tree, clean manifest):
│        git add -p && git commit -m "..."
│        # then re-submit
│
│   2. STASH if the changes aren't ready to commit:
│        git stash
│        # submit
│        git stash pop
│
│   3. ATTEST that the dirty changes are non-load-bearing AT RUNTIME
│      (e.g. new code paths added but not invoked by the sweep's CLI
│      args). Add an entry to lora_playground/exclusions/dirty_attestations.json
│      AFTER the runs complete, naming git_commit=$GIT_COMMIT and the
│      specific runtime reason. Then set FORCE_DIRTY=1 here and submit.
│
│   4. OVERRIDE if you know what you're doing (rare):
│        FORCE_DIRTY=1 ./slurm_scripts/submit.sh ...
│        # Document the reason in SWEEP_PURPOSE.
╰─────────────────────────────────────────────────────────────────────╯
EOF

exit 1
