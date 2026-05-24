"""Audit: find (algorithm, seed, lr, lora_r) tuples with multiple surviving
runs after `load_runs` — i.e. cases where the loader's dedup didn't collapse
runs that algorithmically should be one.

Categorizes findings by what fields differ:
  - "schema-growth"  — only raw-vs-derived override or known-runtime drift
  - "real-config"    — data_dir / compile / world_size / model_name differ
                       (genuinely distinct runs; analysis should filter)
  - "loss-outlier"   — losses span >2× the workload σ floor (~0.001);
                       likely a buggy log group worth deleting

Run with: conda run -n ffcv-pl python scripts/audit_loader_dedup.py
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_playground.loader import load_runs
from lora_playground.plotting import series_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50,
                        help="max number of findings to print")
    parser.add_argument("--logs-root", default=str(ROOT / "logs"))
    args = parser.parse_args()

    runs = load_runs(logs_root=args.logs_root, warn_cross_commit=False)
    print(f"Loaded {len(runs)} runs total.")

    # Bucket by (series_id, seed, lr, lora_r): runs that share these all
    # describe the same algorithm at the same model size + lr + seed, and
    # should have collapsed during merge_runs. Surviving as multiple
    # entries = loader dedup miss.
    buckets: dict = defaultdict(list)
    for cfg, evs in runs:
        if not evs:
            continue
        key = (series_id(cfg), cfg.get("seed"), cfg.get("lr"), cfg.get("lora_r"))
        buckets[key].append((cfg, evs))

    findings = [(sid, group) for sid, group in buckets.items() if len(group) > 1]
    print(f"Found {len(findings)} series_id buckets with >1 surviving run "
          f"(should be 0 after correct dedup).\n")

    if not findings:
        print("Loader dedup is clean — no buckets need investigation.")
        return

    # Categorize by loss spread. Every finding IS a dedup miss (since
    # series_id collapsed them, they're algorithmically the same).
    # - "clean" : spread ≤ 0.002 — multiple logs of the same run; tie-break
    #             on newest group / longest trajectory was correct, just
    #             redundant copies. Inflates seed-σ in plots → real harm.
    # - "outlier": spread > 0.005 — one log group's results disagree with
    #             others at the same series. Usually a stale commit with a
    #             since-fixed bug. User should delete that log group.
    cat_clean: list = []
    cat_outlier: list = []
    for key, group in findings:
        losses = [e[-1].get("eval_loss") for _, e in group if e]
        losses = [x for x in losses if x is not None]
        spread = (max(losses) - min(losses)) if len(losses) > 1 else 0.0
        entry = (key, group, losses, spread)
        if spread > 0.005:
            cat_outlier.append(entry)
        else:
            cat_clean.append(entry)

    def _summary_line(key, group, losses, spread):
        sid_dict = dict(key[0])
        opt = sid_dict.get("optimizer", "?")
        seed, lr, lora_r = key[1], key[2], key[3]
        return (
            f"  ({opt[:42]:42s} r={lora_r} lr={lr:.0e} seed={seed}) "
            f"n={len(group)} spread={spread:.4f}",
            [(e[-1].get("eval_loss"), (c.get("log_group") or "?")[:36],
              (c.get("git_commit") or "?")[:7])
             for c, e in group if e],
        )

    print("=" * 80)
    print(f"CATEGORY A: clean-redundant ({len(cat_clean)} buckets)")
    print("Same algorithm + same seed + same lr — surviving as N>1 copies")
    print("inflates seed-σ in plots. Loader fix or log cleanup.")
    print("=" * 80)
    for (key, group, losses, spread) in cat_clean[:args.limit]:
        line, rows = _summary_line(key, group, losses, spread)
        print(line)
        for loss, grp, sha in sorted(rows):
            print(f"    {loss:.4f}  {grp:36s}  {sha}")

    print()
    print("=" * 80)
    print(f"CATEGORY B: loss-outlier ({len(cat_outlier)} buckets)")
    print("Same algorithm but loss spread >5σ — one log group disagrees.")
    print("Almost always a stale commit; delete the outlier log group.")
    print("=" * 80)
    for (key, group, losses, spread) in cat_outlier[:args.limit]:
        line, rows = _summary_line(key, group, losses, spread)
        print(line)
        for loss, grp, sha in sorted(rows):
            print(f"    {loss:.4f}  {grp:36s}  {sha}")

    print()
    print(f"SUMMARY: {len(cat_clean)} clean-redundant, "
          f"{len(cat_outlier)} loss-outlier — total {len(findings)} dedup misses.")

    # Suggest EXCLUDED_COMMITS entries: any commit whose runs are
    # consistently the "high" side of loss-outlier buckets is a
    # registry candidate. A commit qualifies if (a) it appears in
    # >=3 outlier buckets as the worse side, (b) it never appears as
    # the better side, (c) it's not already in EXCLUDED_COMMITS.
    if cat_outlier:
        from lora_playground.manifest import EXCLUDED_COMMITS
        from collections import Counter
        bad_commit_count: Counter = Counter()
        good_commit_count: Counter = Counter()
        for (key, group, losses, spread) in cat_outlier:
            # Only treat as a stale-commit signal if all runs ran to
            # the same max_steps — otherwise a short-pilot horizon
            # surfaces as a "loss outlier" purely because it stopped
            # earlier, not because the algorithm was wrong.
            steps_in_bucket = {c.get("max_steps") for c, _ in group}
            if len(steps_in_bucket) > 1:
                continue
            rows = sorted(
                [(e[-1].get("eval_loss"), (c.get("git_commit") or "?")[:7])
                 for c, e in group if e]
            )
            if len(rows) < 2:
                continue
            best_loss = rows[0][0]
            for loss, sha in rows:
                if loss - best_loss > 0.005:
                    bad_commit_count[sha] += 1
                else:
                    good_commit_count[sha] += 1
        already_excluded = set(EXCLUDED_COMMITS)
        candidates = [
            (sha, n) for sha, n in bad_commit_count.most_common()
            if n >= 3 and good_commit_count[sha] == 0
            and not any(sha.startswith(p) for p in already_excluded)
        ]
        if candidates:
            print()
            print("=" * 80)
            print(f"REGISTRY CANDIDATES: {len(candidates)} commit(s) appearing")
            print("in >=3 outlier buckets as the worse side, never as the better side.")
            print("Paste into `lora_playground.manifest.EXCLUDED_COMMITS`:")
            print("=" * 80)
            for sha, n in candidates:
                print(f"    {sha!r}: \"<describe regression> (appears in {n} outlier buckets)\",")


if __name__ == "__main__":
    main()
