"""Persistent parsed-run caches must stay outside the loading closure."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "lora_playground"


def _imports_run_cache(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "lora_playground.run_cache"
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "lora_playground.run_cache":
                return True
            if node.module in {"lora_playground", None}:
                if any(alias.name == "run_cache" for alias in node.names):
                    return True
    return False


def test_persistent_pickle_cache_module_is_removed():
    assert not (PACKAGE / "run_cache.py").exists()


def test_production_package_does_not_import_removed_cache():
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in PACKAGE.rglob("*.py")
        if _imports_run_cache(path)
    ]
    assert offenders == []


def test_obsolete_cache_entry_points_are_removed():
    assert not (ROOT / "scripts" / "build_logs_cache.py").exists()
    assert not (ROOT / "scripts" / "profile_load_runs.py").exists()
