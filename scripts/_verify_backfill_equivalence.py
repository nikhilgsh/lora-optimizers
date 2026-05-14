"""Side-tool for Phase 2 backfill verification.

Run BEFORE backfill --apply with `snapshot` to save a pickle of all currently-
loaded cfg state. Run AFTER backfill with `diff` to confirm the loader still
produces identical values for every pre-existing key on every run. The only
expected differences are the new schema keys (optimizer_effective, diagnostics,
git_dirty, git_diff_sha, git_untracked_files, _backfill_uncertain) — those
appear ONLY after backfill and don't trip the diff.

Usage:
  python scripts/_verify_backfill_equivalence.py snapshot --out /tmp/pre.pkl
  python scripts/backfill_cfg_events.py --apply
  python scripts/_verify_backfill_equivalence.py diff --pre /tmp/pre.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.loader import load_runs  # noqa: E402


# Keys we expect backfill to ADD; their presence/absence is fine to differ.
ADDED_BY_BACKFILL = {
    "optimizer_effective",
    "diagnostics",
    "git_dirty",
    "git_diff_sha",
    "git_untracked_files",
    "_backfill_uncertain",
}


def snapshot(out_path: Path) -> None:
    runs = load_runs(warn_cross_commit=False)
    snap = {}
    for cfg, evs in runs:
        run_id = tuple(cfg.get("run_id", ("?", "?")))
        # Hash-relevant view: every cfg key the backfill is NOT expected to add.
        view = {
            k: cfg[k] for k in sorted(cfg.keys())
            if k not in ADDED_BY_BACKFILL
        }
        snap[run_id] = view
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(snap, f)
    print(f"Snapshot: {len(snap)} runs → {out_path}")


def diff(pre_path: Path) -> int:
    with pre_path.open("rb") as f:
        pre = pickle.load(f)
    runs = load_runs(warn_cross_commit=False)
    post = {}
    for cfg, evs in runs:
        run_id = tuple(cfg.get("run_id", ("?", "?")))
        view = {
            k: cfg[k] for k in sorted(cfg.keys())
            if k not in ADDED_BY_BACKFILL
        }
        post[run_id] = view

    only_pre = set(pre) - set(post)
    only_post = set(post) - set(pre)
    common = set(pre) & set(post)
    print(f"runs only in pre:  {len(only_pre)}")
    print(f"runs only in post: {len(only_post)}")
    print(f"runs in both:      {len(common)}")

    import json
    import math

    def _normalize(o):
        """JSON-roundtrip + NaN normalization. NaN != NaN under standard
        equality, but it's content-equivalent for our purposes; same for any
        sub-structure containing NaN."""
        if isinstance(o, float) and math.isnan(o):
            return "<nan>"
        if isinstance(o, dict):
            return {k: _normalize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_normalize(x) for x in o]
        return o

    mismatches = 0
    examples = []
    for rid in common:
        pre_view = _normalize(pre[rid])
        post_view = _normalize(post[rid])
        if pre_view == post_view:
            continue
        mismatches += 1
        if len(examples) < 5:
            keys = sorted(set(pre_view.keys()) | set(post_view.keys()))
            for k in keys:
                if pre_view.get(k) != post_view.get(k):
                    examples.append((rid, k, pre_view.get(k), post_view.get(k)))
                    break
    print(f"runs with mismatch on pre-existing keys: {mismatches}")
    if examples:
        print("\nExamples (first 5):")
        for rid, k, pv, pov in examples:
            print(f"  {rid}: key={k!r}")
            print(f"    pre:  {pv!r}")
            print(f"    post: {pov!r}")
    return 0 if mismatches == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--out", required=True, type=Path)
    d = sub.add_parser("diff")
    d.add_argument("--pre", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "snapshot":
        snapshot(args.out)
    elif args.cmd == "diff":
        sys.exit(diff(args.pre))


if __name__ == "__main__":
    main()
