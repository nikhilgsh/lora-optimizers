import hashlib
import importlib.util
import os
from pathlib import Path
import pathlib
import sys
import types


def test_bootstrap_snapshot_continues_when_subtree_disappears(
    tmp_path, monkeypatch
):
    """A disappearing transient directory must not abort process startup."""
    project_root = Path(__file__).resolve().parents[1]
    entry = tmp_path / "train_lora.py"
    entry.write_bytes((project_root / "train_lora.py").read_bytes())

    stable = tmp_path / "stable" / "kept.py"
    stable.parent.mkdir()
    stable.write_bytes(b"KEPT = True\n")

    excluded = tmp_path / "tests" / "excluded.py"
    excluded.parent.mkdir()
    excluded.write_bytes(b"EXCLUDED = True\n")

    transient = tmp_path / ".agents"
    transient.mkdir()
    transient_file = transient / "racy.py"
    transient_file.write_bytes(b"RACY = True\n")

    # Keep execution of the copied entry point stdlib-only: the production
    # import occurs after the snapshot and is irrelevant to this detector.
    fake_package = types.ModuleType("lora_playground")
    fake_package.__path__ = []
    fake_train = types.ModuleType("lora_playground.train")
    fake_train.main = lambda: None
    monkeypatch.setitem(sys.modules, "lora_playground", fake_package)
    monkeypatch.setitem(sys.modules, "lora_playground.train", fake_train)

    real_scandir = os.scandir
    disappeared = False

    def racing_scandir(path):
        nonlocal disappeared
        if Path(path) == transient and not disappeared:
            # The parent scan has already observed this directory. Remove it
            # immediately before the walker tries to descend into it.
            transient_file.unlink()
            transient.rmdir()
            disappeared = True
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", racing_scandir)
    # Python 3.10's pathlib cached ``os.scandir`` on its normal accessor at
    # import time. Patching that cached callable makes this same detector fail
    # against the former ``Path.rglob`` implementation as it did in practice.
    normal_accessor = getattr(pathlib, "_normal_accessor", None)
    if normal_accessor is not None:
        monkeypatch.setattr(normal_accessor, "scandir", racing_scandir)

    spec = importlib.util.spec_from_file_location(
        "_train_lora_snapshot_race_test", entry
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert disappeared, "detector did not exercise the disappearing-subtree race"
    assert module.SOURCE_SNAPSHOT["stable/kept.py"] == b"KEPT = True\n"
    assert module.SOURCE_SNAPSHOT_SHA["stable/kept.py"] == hashlib.sha256(
        b"KEPT = True\n"
    ).hexdigest()
    assert "tests/excluded.py" not in module.SOURCE_SNAPSHOT
    assert ".agents/racy.py" not in module.SOURCE_SNAPSHOT
