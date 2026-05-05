#!/usr/bin/env bash
# Delete junk files at repo root. Dry-run by default; pass --apply to delete.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

PATTERNS=(
  'slurm_job_*_resize.sh'
  'slurm_job_*_resize.csh'
)

shopt -s nullglob
matches=()
for pat in "${PATTERNS[@]}"; do
  for f in "$REPO_ROOT"/$pat; do
    matches+=("$f")
  done
done

if (( ${#matches[@]} == 0 )); then
  echo "no junk files found"
  exit 0
fi

echo "${#matches[@]} files matched:"
printf '  %s\n' "${matches[@]}"

if (( APPLY )); then
  rm -f -- "${matches[@]}"
  echo "deleted."
else
  echo
  echo "(dry-run; pass --apply to delete)"
fi
