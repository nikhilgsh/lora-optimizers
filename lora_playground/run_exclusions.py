"""Per-run quality exclusions — thin JSON loader.

Authoritative data lives in `lora_playground/exclusions/run_exclusions.json`.
This module just parses it into a `dict[(group, log_filename), reason]` so
the loader can do O(1) lookups. Don't add entries here; edit the JSON.

Code-correctness exclusions belong in `lora_playground/invariants.py`
(predicate + git-ancestor); only run-quality (truncation, contamination,
supersession) lives here.
"""
from __future__ import annotations

import json
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent / "exclusions" / "run_exclusions.json"


def _load() -> dict[tuple[str, str], str]:
    if not _DATA_FILE.exists():
        return {}
    raw = json.loads(_DATA_FILE.read_text())
    out: dict[tuple[str, str], str] = {}
    for entry in raw.get("entries", []):
        key = (entry["group"], entry["log_filename"])
        if key in out:
            raise ValueError(
                f"Duplicate RUN_EXCLUSIONS entry for {key}: "
                f"{out[key]!r} vs {entry['reason']!r}"
            )
        out[key] = entry["reason"]
    return out


RUN_EXCLUSIONS: dict[tuple[str, str], str] = _load()


def is_run_excluded(group: str | None, log_filename: str | None
                    ) -> tuple[bool, str | None]:
    """True iff (group, log_filename) is in RUN_EXCLUSIONS."""
    if not group or not log_filename:
        return False, None
    reason = RUN_EXCLUSIONS.get((group, log_filename))
    return (reason is not None, reason)
