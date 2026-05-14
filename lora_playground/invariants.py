"""Per-run code-correctness invariants for the loader's exclusion DSL.

Each Invariant says: "any cfg matching `when(cfg)` must have run on a commit
that is a git-ancestor of `requires_ancestor`". Runs that fail the check are
excluded with `reason` surfaced in the load summary.

Why a predicate DSL rather than an EXCLUDED_COMMITS enumeration:
- One rule per fix, not one per commit-that-pre-dates-the-fix. New
  intermediate commits inherit the rule automatically through git ancestry.
- Per-cfg-commit predicate, not per-manifest-commit (manifests disagree with
  per-run cfg commits in ~60 groups; the cfg's `git_commit` is authoritative).
- `when` can reference ANY cfg field, so we can write "ε_rel runs require
  λ_max fix" without mentioning specific groups or commit pairs.

Two kinds of exclusion live elsewhere:
- Run-quality exclusions (truncation, contamination) → `run_exclusions.py`.
  Those are opaque to predicates; keyed by `(group, log_filename)`.
- Dirty-tree attestation → `dirty_attestations.py`. Same idea but the run
  was on a dirty tree, so `cfg.git_commit` alone doesn't pin behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Invariant:
    name: str                          # short slug for the exclusion report
    when: Callable[[dict], bool]       # cfg predicate (selects affected runs)
    requires_ancestor: str             # commit fix; merge-base --is-ancestor
    reason: str                        # one-line explanation


# ── Active invariants ─────────────────────────────────────────────────────
#
# Each entry replaces one or more entries in the legacy
# `EXCLUDED_COMMITS` / `BUGGY_EPS_REL_COMMITS` / `EXCLUDED_GROUPS` registries.
# The dual-output mode in loader.py reports disagreements between old and new
# verdicts; when the disagreement set is empty (or every disagreement has a
# recorded decision), the legacy registries are deleted.
INVARIANTS: list[Invariant] = [
    Invariant(
        name="lambda_max_fix_for_eps_rel",
        # ε_rel (precond_delta_relative=True) routes through the Higham
        # inv-sqrt damping path, which used a buggy H@ones λ_max estimate
        # until commit 9677abe (2026-05-13). ε_rel runs before that commit
        # are scaled against the wrong λ_max → drop them.
        when=lambda c: bool(c.get("precond_delta_relative", False)),
        requires_ancestor="9677abe",
        reason="ε_rel runs require λ_max fix in Higham inv-sqrt damping path "
               "(commit 9677abe, 2026-05-13)",
    ),
    # Per-commit blanket exclusions (the legacy EXCLUDED_COMMITS analog) live
    # in commit_exclusions.json instead — invariants are for cases where the
    # bug can be characterized by a cfg predicate, not for "this commit just
    # produced wrong numbers for reasons we couldn't isolate."
]


def evaluate_invariants(
    cfg: dict,
    commit: str | None,
    is_ancestor: Callable[[str, str], bool],
) -> tuple[bool, str | None, str | None]:
    """Run every invariant against `cfg`. Returns (excluded, name, reason).

    `is_ancestor(anc, desc)` is the cached git-ancestor predicate exported
    from loader._is_ancestor. Passing it in keeps this module side-effect-free
    and test-friendly (the test substitutes a fake is_ancestor).

    Stops at the first failing invariant (no need to enumerate all reasons).
    Returns (False, None, None) when no invariant fires.
    """
    if not commit:
        return False, None, None
    for inv in INVARIANTS:
        try:
            if not inv.when(cfg):
                continue
        except Exception:
            # Defensive: a predicate that raises (e.g. missing key on
            # malformed cfg) should not break loading. Treat as not-matched.
            continue
        if not is_ancestor(inv.requires_ancestor, commit):
            return True, inv.name, inv.reason
    return False, None, None
