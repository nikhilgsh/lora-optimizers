"""Filesystem-led run catalog and immutable record behavior."""
from __future__ import annotations

import json

import pytest

from lora_playground.run_catalog import RunCatalog


_MISSING = object()


def _write_group(
    root,
    group,
    *,
    config=None,
    manifest=_MISSING,
    filename="log_0.out",
):
    run_info = root / group / "run_info"
    log_dir = run_info / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "event": "config",
        "optimizer": "adamw",
        "lr": 1e-3,
        **(config or {}),
    }
    events = [cfg, {"event": "eval", "step": 100, "eval_loss": 0.8,
                    "lr": cfg["lr"]}]
    (log_dir / filename).write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    if manifest is not _MISSING:
        if isinstance(manifest, str):
            (run_info / "meta.json").write_text(manifest)
        else:
            (run_info / "meta.json").write_text(json.dumps(manifest) + "\n")


def _codes(issues):
    return {issue.code for issue in issues}


def test_physical_discovery_ignores_manifest_admission(tmp_path):
    _write_group(tmp_path, "missing")
    _write_group(tmp_path, "corrupt", manifest="{not-json")
    _write_group(
        tmp_path,
        "empty_scope",
        manifest={"group": "empty_scope", "scope": []},
    )
    _write_group(
        tmp_path,
        "valid",
        manifest={"group": "valid", "scope": ["comparison"],
                  "purpose": "annotation only"},
    )

    catalog = RunCatalog(tmp_path)

    assert catalog.groups == ("corrupt", "empty_scope", "missing", "valid")
    assert "manifest_missing" in _codes(catalog.group_issues["missing"])
    assert "manifest_corrupt" in _codes(catalog.group_issues["corrupt"])
    assert "manifest_empty_scope" in _codes(
        catalog.group_issues["empty_scope"]
    )
    # No manifest state gates the physical runs.
    records = catalog.records
    assert {record.group for record in records} == set(catalog.groups)
    by_group = {record.group: record for record in records}
    assert "manifest_missing" in _codes(by_group["missing"].issues)
    assert "manifest_corrupt" in _codes(by_group["corrupt"].issues)
    assert by_group["valid"].audit_provenance.manifest["purpose"] == (
        "annotation only"
    )


def test_catalog_is_lazy_and_stable_ids_are_physical(tmp_path, monkeypatch):
    _write_group(
        tmp_path,
        "explicit",
        config={"run_id": "logged-run-id", "nested": {"values": [1, 2]}},
    )
    _write_group(tmp_path, "fallback")

    from lora_playground.plotting import loading as legacy_loading

    original = legacy_loading.load_run
    calls = []

    def counted(path):
        calls.append(path.parent.parent.parent.name)
        return original(path)

    monkeypatch.setattr(legacy_loading, "load_run", counted)
    catalog = RunCatalog(tmp_path)
    assert calls == []
    assert catalog.groups == ("explicit", "fallback")
    assert calls == []
    only_fallback = catalog.query(equals={"group": "fallback"})
    assert [record.group for record in only_fallback] == ["fallback"]
    assert calls == ["fallback"]
    calls.clear()
    with pytest.raises(AttributeError, match="immutable discovery snapshot"):
        catalog._groups = ()

    by_group = {record.group: record for record in catalog.records}
    assert calls == ["explicit"]  # fallback stayed cached from the query above
    assert by_group["explicit"].physical_id == "logged-run-id"
    assert by_group["fallback"].physical_id == "fallback/log_0.out"
    assert ({record.group: record.physical_id for record in RunCatalog(tmp_path).records}
            == {group: record.physical_id for group, record in by_group.items()})

    with pytest.raises(TypeError):
        by_group["explicit"].raw_config["optimizer"] = "changed"
    with pytest.raises(TypeError):
        by_group["explicit"].raw_config["nested"]["values"] = ()
    with pytest.raises(TypeError):
        by_group["explicit"].history[0]["eval_loss"] = 0.0

    legacy_cfg, legacy_history = catalog.as_legacy_tuples(
        [by_group["fallback"]]
    )[0]
    assert legacy_cfg["log_group"] == "fallback"
    assert legacy_cfg["run_id"] == "fallback/log_0.out"
    legacy_cfg["optimizer"] = "mutable-copy"
    legacy_history[0]["eval_loss"] = 0.0
    assert by_group["fallback"].raw_config["optimizer"] == "adamw"
    assert by_group["fallback"].history[0]["eval_loss"] == 0.8


def test_catalog_does_not_implicitly_stitch_resume_segments(tmp_path):
    _write_group(tmp_path, "resumed", filename="log_0.out")
    _write_group(
        tmp_path,
        "resumed",
        config={"lr": 1e-3},
        filename="log_0.out.resume_1",
    )

    records = RunCatalog(tmp_path).records

    assert [record.log_filename for record in records] == [
        "log_0.out", "log_0.out.resume_1",
    ]
    assert len({record.physical_id for record in records}) == 2


def test_logged_schema_drives_effective_config_and_explicit_queries(
    tmp_path, monkeypatch
):
    _write_group(
        tmp_path,
        "new_schema",
        config={
            "optimizer": "method",
            "lr": 3e-3,
            "beta1": 0.85,
            "_cli_args": {"brand_new_logged_flag": 17, "beta1": 0.7},
            "optimizer_config": {
                "constructor_only": "logged-constructor-value",
                "beta1": 0.8,
            },
            "optimizer_effective": {
                "effective_picard_iters": 9,
                "beta1": 0.9,
            },
        },
        manifest={"group": "new_schema", "scope": ["new"]},
    )
    _write_group(
        tmp_path,
        "other",
        config={"optimizer": "other", "lr": 1e-2},
        manifest={"group": "other", "scope": ["new"]},
    )

    # The additive catalog must not reach through loader.py to reconstruct old
    # runs using today's argparse or optimizer registry.
    import lora_playground.loader as legacy_loader

    def forbidden(*_args, **_kwargs):
        raise AssertionError("current-code default introspection was consulted")

    monkeypatch.setattr(legacy_loader, "_argparse_defaults", forbidden)
    monkeypatch.setattr(legacy_loader, "_precond_by_optimizer", forbidden)

    catalog = RunCatalog(tmp_path)
    record = catalog.query(equals={"optimizer": "method"})[0]
    effective = record.effective_config
    assert effective["brand_new_logged_flag"] == 17
    assert effective["constructor_only"] == "logged-constructor-value"
    assert effective["effective_picard_iters"] == 9
    assert effective["beta1"] == 0.9
    assert "current_only_default" not in effective

    selected = catalog.query(
        equals={"optimizer": "method"},
        one_of={"lr": [1e-3, 3e-3]},
    )
    assert selected == (record,)
    assert catalog.query(one_of={"optimizer": ["method", "other"]}) == (
        catalog.records
    )

    with pytest.raises(TypeError, match="scalar"):
        catalog.query(equals={"lr": lambda value: value > 0})
    with pytest.raises(TypeError, match="explicit collection"):
        catalog.query(one_of={"optimizer": "method"})
    with pytest.raises(ValueError, match="equals and one_of"):
        catalog.query(equals={"lr": 3e-3}, one_of={"lr": [3e-3]})
