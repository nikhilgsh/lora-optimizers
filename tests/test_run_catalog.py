"""Filesystem-led run catalog and immutable record behavior."""
from __future__ import annotations

import json

import pytest

from lora_playground.run_catalog import RunCatalog
from lora_playground.run_parsing import parse_run_file


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

    import lora_playground.run_catalog as run_catalog_module

    original = run_catalog_module.parse_run_file
    calls = []

    def counted(path, **kwargs):
        calls.append(path.parent.parent.parent.name)
        return original(path, **kwargs)

    monkeypatch.setattr(run_catalog_module, "parse_run_file", counted)
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


def test_semantic_query_header_rejects_without_full_history_parse(
    tmp_path, monkeypatch,
):
    _write_group(tmp_path, "group", filename="log_0.out", config={
        "optimizer": "adamw",
        "_cli_args": {"lora_r": 16},
    })
    _write_group(tmp_path, "group", filename="log_1.out", config={
        "optimizer": "muon",
        "_cli_args": {"lora_r": 64},
    })

    import lora_playground.run_catalog as run_catalog_module

    original = run_catalog_module.parse_run_file
    calls = []

    def counted(path, **kwargs):
        calls.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(run_catalog_module, "parse_run_file", counted)
    catalog = RunCatalog(tmp_path)

    records = catalog.query(equals={"optimizer": "adamw", "lora_r": 16})

    assert [record.log_filename for record in records] == ["log_0.out"]
    assert calls == ["log_0.out"]


def test_catalog_fast_parse_omits_diagnostics_but_preserves_config_and_evals(
    tmp_path,
):
    path = tmp_path / "group" / "run_info" / "logs" / "log_0.out"
    path.parent.mkdir(parents=True)
    events = [
        {"event": "config", "optimizer": "adamw", "lr": 1e-3},
        {"event": "optim_step", "step": 1, "large_probe": list(range(100))},
        {"event": "eval", "step": 10, "eval_loss": 0.8, "lr": 1e-3},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    diagnostic_parse = parse_run_file(path)
    record = RunCatalog(tmp_path).records[0]

    assert len(diagnostic_parse.optim_steps) == 1
    assert diagnostic_parse.raw_config()["_optim_steps"][0]["step"] == 1
    assert "_optim_steps" not in record.raw_config
    assert record.semantic_config["optimizer"] == "adamw"
    assert record.history == diagnostic_parse.evals


def test_header_and_full_parser_share_first_config_authority(tmp_path):
    path = tmp_path / "group" / "run_info" / "logs" / "log_0.out"
    path.parent.mkdir(parents=True)
    events = [
        {"event": "config", "optimizer": "adamw", "lr": 1e-3},
        {"event": "config", "optimizer": "muon", "lr": 2e-3},
        {"event": "eval", "step": 10, "eval_loss": 0.8, "lr": 1e-3},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")

    catalog = RunCatalog(tmp_path)

    assert len(catalog.query(equals={"optimizer": "adamw"})) == 1
    assert catalog.query(equals={"optimizer": "muon"}) == ()


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


def test_catalog_preserves_actual_resume_event_as_audit_metadata(tmp_path):
    run_info = tmp_path / "resumed" / "run_info"
    log_dir = run_info / "logs"
    log_dir.mkdir(parents=True)
    events = [
        {"event": "config", "optimizer": "adamw", "lr": 1e-3,
         "attempt_id": "attempt-b"},
        {"event": "resume", "resumed_from_step": 100,
         "resume_parent_attempt_id": "attempt-a",
         "checkpoint_identity": "group/task-0"},
        {"event": "eval", "step": 200, "eval_loss": 0.7, "lr": 1e-3},
    ]
    (log_dir / "log_0.out").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    record = RunCatalog(tmp_path).records[0]

    assert record.raw_config["_resume"]["resume_parent_attempt_id"] == (
        "attempt-a"
    )
    assert record.raw_config["_resume"]["resumed_from_step"] == 100
    # Audit metadata does not leak into the effective comparison config.
    assert "_resume" not in record.effective_config


def test_catalog_resolves_only_explicit_versioned_lineage(tmp_path):
    log_dir = tmp_path / "resumed" / "run_info" / "logs"
    log_dir.mkdir(parents=True)
    common = {
        "event": "config",
        "run_schema_version": 1,
        "checkpoint_identity": "resumed/task_0",
        "semantic_revisions": {
            "optimizer_impl": 1,
            "data_pipeline": "packed_v1.1",
            "measurement": 1,
        },
        "optimizer": "adamw",
        "lr": 1e-3,
        "data_pipeline_version": "packed_v1.1",
    }
    root_events = [
        {**common, "attempt_id": "attempt-a",
         "resume_parent_attempt_id": None},
        {"event": "eval", "step": 100, "eval_loss": 0.8, "lr": 1e-3},
    ]
    child_events = [
        {**common, "attempt_id": "attempt-b",
         "resume_parent_attempt_id": None},
        {"event": "resume", "attempt_id": "attempt-b",
         "resume_parent_attempt_id": "attempt-a",
         "checkpoint_identity": "resumed/task_0"},
        {"event": "eval", "step": 200, "eval_loss": 0.7, "lr": 1e-3},
    ]
    (log_dir / "log_0.out.resume_0").write_text(
        "\n".join(json.dumps(event) for event in root_events) + "\n"
    )
    (log_dir / "log_0.out").write_text(
        "\n".join(json.dumps(event) for event in child_events) + "\n"
    )

    catalog = RunCatalog(tmp_path)
    resolved = catalog.resolve_lineages()
    legacy = catalog.resolved_legacy_tuples()

    assert len(resolved) == 1
    assert resolved[0].attempt_ids == ("attempt-a", "attempt-b")
    assert [event["step"] for event in resolved[0].history] == [100, 200]
    assert [event["step"] for event in legacy[0][1]] == [100, 200]


def test_query_lineage_closure_does_not_parse_unrelated_versioned_logs(
    tmp_path, monkeypatch,
):
    common = {
        "event": "config",
        "run_schema_version": 1,
        "checkpoint_identity": "logical/task_0",
        "semantic_revisions": {
            "optimizer_impl": 1,
            "data_pipeline": "packed_v1.1",
            "measurement": 1,
        },
        "optimizer": "muon",
        "lr": 2e-3,
        "data_pipeline_version": "packed_v1.1",
    }

    def write(group, events):
        path = tmp_path / group / "run_info" / "logs" / "log_0.out"
        path.parent.mkdir(parents=True)
        path.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n"
        )

    write("root", [
        {**common, "attempt_id": "attempt-a"},
        {"event": "eval", "step": 10, "eval_loss": 0.8, "lr": 2e-3},
    ])
    write("child", [
        {**common, "attempt_id": "attempt-b"},
        {
            "event": "resume",
            "resume_parent_attempt_id": "attempt-a",
            "checkpoint_identity": "logical/task_0",
        },
        {"event": "eval", "step": 20, "eval_loss": 0.7, "lr": 2e-3},
    ])
    # This independent root deliberately shares the checkpoint identity. A
    # checkpoint-bucket implementation would parse it even though no resume
    # edge connects it to the selected chain.
    write("unrelated-same-checkpoint", [
        {**common, "attempt_id": "attempt-z"},
        {"event": "eval", "step": 30, "eval_loss": 0.6, "lr": 2e-3},
    ])
    write("unrelated-other-checkpoint", [
        {
            **common,
            "attempt_id": "attempt-y",
            "checkpoint_identity": "logical/task_9",
        },
        {"event": "eval", "step": 40, "eval_loss": 0.5, "lr": 2e-3},
    ])

    import lora_playground.run_catalog as run_catalog_module

    original = run_catalog_module.parse_run_file
    full_parses = []

    def counted(path, **kwargs):
        full_parses.append(path.parent.parent.parent.name)
        return original(path, **kwargs)

    monkeypatch.setattr(run_catalog_module, "parse_run_file", counted)
    catalog = RunCatalog(tmp_path)

    selected = catalog.query(equals={"group": "child"})
    resolved = catalog.resolve_lineages(selected)

    assert len(resolved) == 1
    assert resolved[0].attempt_ids == ("attempt-a", "attempt-b")
    assert [event["step"] for event in resolved[0].history] == [10, 20]
    assert full_parses == ["child", "root"]


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
            "_cli_args": {
                "brand_new_logged_flag": 17,
                "beta1": 0.7,
                "checkpoint_dir": "/runtime/checkpoint",
                "optim_diagnostics_every": 1,
            },
            "optimizer_config": {
                "constructor_only": "logged-constructor-value",
                "beta1": 0.8,
                "device": "cuda:0",
            },
            "optimizer_effective": {
                "effective_picard_iters": 9,
                "beta1": 0.9,
            },
            "optimizer_variant_semantics": {
                "schema_version": 1,
                "optimizer": "method",
                "config": {"beta1": 0.9},
                "effective": {"effective_picard_iters": 9},
                "semantic_revision": 1,
                "implementation": {
                    "class": "package.Method",
                    "revision": "abc123",
                },
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
    assert "optimizer_variant_semantics" not in effective
    assert record.raw_config["optimizer_variant_semantics"]["optimizer"] == (
        "method"
    )
    assert "current_only_default" not in effective
    assert "checkpoint_dir" not in effective
    assert "optim_diagnostics_every" not in effective
    assert "device" not in effective

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
