"""One-shot backfill: mark every existing logs/<group>/run_info/meta.json
with `data_pipeline_version: "unpacked_v0"`.

All runs prior to 2026-05-08 used the legacy
DataCollatorForLanguageModeling path (dynamic shapes, no prompt-mask,
full-text loss). This script tags those manifests so analysis can filter
by version uniformly with `load_runs(where={"data_pipeline_version":
"unpacked_v0" or "packed_v1"})`.

Idempotent: if a manifest already has `data_pipeline_version`, leaves it
alone. Only touches manifests that lack the field. Prints a summary at
the end (touched / skipped / missing-meta).

Usage:
    python scripts/data/backfill_pipeline_version.py [--logs-root path]
                                                     [--dry-run]
                                                     [--version VALUE]

`--version` defaults to "unpacked_v0" (the boundary value). Override
only if you have a specific reason; in normal use, run once with the
default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs-root",
        default=str(Path(__file__).resolve().parent.parent.parent / "logs"),
    )
    ap.add_argument("--version", default="unpacked_v0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.logs_root)
    if not root.exists():
        print(f"ERROR: logs root not found: {root}", file=sys.stderr)
        return 1

    touched = 0
    skipped = 0
    missing = 0
    for run_info in sorted(root.glob("*/run_info")):
        meta = run_info / "meta.json"
        if not meta.exists():
            missing += 1
            continue
        try:
            data = json.loads(meta.read_text())
        except json.JSONDecodeError as e:
            print(f"  CORRUPT: {meta} ({e})", file=sys.stderr)
            continue
        if "data_pipeline_version" in data:
            skipped += 1
            continue
        data["data_pipeline_version"] = args.version
        if args.dry_run:
            print(f"  WOULD ADD data_pipeline_version={args.version} to {meta}")
        else:
            meta.write_text(json.dumps(data, indent=2) + "\n")
            print(f"  TAGGED {run_info.parent.name} → {args.version}")
        touched += 1

    print(
        f"\nDone. {touched} manifest(s) "
        f"{'would be' if args.dry_run else 'were'} tagged "
        f"as {args.version}; {skipped} already had the field; "
        f"{missing} group(s) had no meta.json."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
