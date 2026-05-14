"""Dirty-tree attestations — thin JSON loader.

Authoritative data lives in `lora_playground/exclusions/dirty_attestations.json`.
This module parses it into a `dict[(group, log_filename, diff_sha), DirtyAttestation]`
so the loader can do O(1) lookups. Don't add entries here; edit the JSON.

See the JSON file's `_doc` for the loader's dirty-tree resolution policy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DATA_FILE = Path(__file__).resolve().parent / "exclusions" / "dirty_attestations.json"


@dataclass(frozen=True)
class DirtyAttestation:
    treat_as_commit: str
    reason: str


def _load() -> dict[tuple[str, str, str], DirtyAttestation]:
    if not _DATA_FILE.exists():
        return {}
    raw = json.loads(_DATA_FILE.read_text())
    out: dict[tuple[str, str, str], DirtyAttestation] = {}
    for entry in raw.get("entries", []):
        key = (entry["group"], entry["log_filename"], entry["git_diff_sha"])
        if key in out:
            raise ValueError(f"Duplicate DIRTY_ATTESTATIONS entry for {key}")
        out[key] = DirtyAttestation(
            treat_as_commit=entry["treat_as_commit"],
            reason=entry["reason"],
        )
    return out


DIRTY_ATTESTATIONS: dict[tuple[str, str, str], DirtyAttestation] = _load()


def lookup_attestation(group: str | None, log_filename: str | None,
                       diff_sha: str | None) -> DirtyAttestation | None:
    if not group or not log_filename or not diff_sha:
        return None
    return DIRTY_ATTESTATIONS.get((group, log_filename, diff_sha))
