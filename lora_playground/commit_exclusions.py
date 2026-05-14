"""Per-commit blanket exclusions — thin JSON loader.

Authoritative data: `lora_playground/exclusions/commit_exclusions.json` and
`lora_playground/exclusions/eps_rel_buggy_commits.json`. Edit the JSON; this
module just parses.

See the JSON files' `_doc` blocks for when to add an entry vs. writing an
invariant. Rule of thumb: invariants generalize, commit-exclusions blanket.
"""
from __future__ import annotations

import json
from pathlib import Path

_EX_DIR = Path(__file__).resolve().parent / "exclusions"


def _load_commit_list(filename: str) -> list[tuple[str, str]]:
    path = _EX_DIR / filename
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entry in raw.get("entries", []):
        prefix = entry["commit_prefix"]
        if prefix in seen:
            raise ValueError(
                f"Duplicate commit_prefix in {filename}: {prefix}"
            )
        seen.add(prefix)
        out.append((prefix, entry["reason"]))
    return out


COMMIT_EXCLUSIONS: list[tuple[str, str]] = _load_commit_list("commit_exclusions.json")
EPS_REL_BUGGY_COMMITS: list[tuple[str, str]] = _load_commit_list("eps_rel_buggy_commits.json")


def is_commit_excluded(commit: str | None) -> tuple[bool, str | None]:
    """True iff `commit` matches any prefix in COMMIT_EXCLUSIONS."""
    if not commit:
        return False, None
    for prefix, reason in COMMIT_EXCLUSIONS:
        if commit.startswith(prefix):
            return True, reason
    return False, None


def is_buggy_eps_rel(cfg: dict) -> tuple[bool, str | None]:
    """True iff `cfg` is an ε_rel run on a commit listed in
    EPS_REL_BUGGY_COMMITS. Useful when a group MIXES ε_rel and non-ε_rel
    cells at a partially-buggy commit; the commit-level exclusion would
    kill the good cells.
    """
    if not cfg.get("precond_delta_relative"):
        return False, None
    commit = cfg.get("git_commit") or ""
    for prefix, reason in EPS_REL_BUGGY_COMMITS:
        if commit.startswith(prefix):
            return True, f"buggy ε_rel ({reason})"
    return False, None
