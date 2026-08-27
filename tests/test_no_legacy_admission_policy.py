"""Regression guards for neutral physical-run discovery.

Known-bad-run registries used to be imported lazily from the loader and could
silently change the physical population returned by a query. Discovery now
records provenance without interpreting it. Keep the catalog path independent
of those retired policy modules.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_MODULES = (
    ROOT / "lora_playground" / "loader.py",
    ROOT / "lora_playground" / "run_catalog.py",
    ROOT / "lora_playground" / "run_inventory.py",
    ROOT / "lora_playground" / "run_parsing.py",
)
RETIRED_MODULES = frozenset({
    "commit_exclusions",
    "run_exclusions",
    "invariants",
    "dirty_attestations",
})


def _imported_leaf_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.rsplit(".", 1)[-1])
    return names


def test_discovery_modules_do_not_import_retired_admission_policy():
    offenders: dict[str, list[str]] = {}
    for path in DISCOVERY_MODULES:
        if not path.exists():
            continue
        retired = _imported_leaf_names(path) & RETIRED_MODULES
        if retired:
            offenders[path.relative_to(ROOT).as_posix()] = sorted(retired)
    assert not offenders, (
        "Physical discovery must not import policy that silently rejects "
        f"records: {offenders}"
    )


def test_retired_admission_policy_files_stay_deleted():
    policy_paths = [
        ROOT / "lora_playground" / f"{module}.py"
        for module in RETIRED_MODULES
    ]
    policy_paths.extend((ROOT / "lora_playground" / "exclusions").glob("*.json"))
    present = [
        path.relative_to(ROOT).as_posix()
        for path in policy_paths
        if path.exists()
    ]
    assert not present, f"Retired admission-policy artifacts reappeared: {present}"
