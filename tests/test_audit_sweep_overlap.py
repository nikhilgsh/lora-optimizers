from __future__ import annotations

import json

from lora_playground.run_catalog import RunCatalog
from scripts.analysis.audit_sweep_overlap import audit_cell, main


def _write_run(
    logs,
    *,
    group="group",
    filename="log_0.out",
    config=None,
    sweep_script="scripts/sweep/sweep.sh",
):
    log_dir = logs / group / "run_info" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "event": "config",
        "run_schema_version": 2,
        "attempt_id": f"{group}:attempt",
        "checkpoint_identity": f"{group}/task_0",
        "semantic_revisions": {
            "optimizer_impl": 1,
            "data_pipeline": "packed_v1.1",
            "measurement": 1,
        },
        "optimizer": "adamw",
        "lr": 1e-3,
        "seed": 0,
        "optimizer_variant_semantics": {
            "schema_version": 1,
            "optimizer": "adamw",
            "config": {"beta1": 0.9, "beta2": 0.999},
            "effective": {},
            "semantic_revision": 1,
            "implementation": {
                "class": "lora_playground.optim.LoRAPlusAdamW",
                "revision": "commit-a",
            },
        },
    }
    base.update(config or {})
    events = [base, {"event": "eval", "step": 1, "eval_loss": 1.0}]
    (log_dir / filename).write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    if sweep_script is not None:
        (logs / group / "run_info" / "meta.json").write_text(json.dumps({
            "group": group,
            "sweep_script": sweep_script,
        }))


def test_exact_match_uses_catalog_identity_and_reports_physical_provenance(tmp_path):
    logs = tmp_path / "logs"
    _write_run(logs)
    result = audit_cell(
        {"optimizer": "adamw", "lr": "1e-3", "seed": "0"},
        RunCatalog.discover(logs),
        expected_launcher="scripts/sweep/sweep.sh",
    )

    assert result.status == "EXISTS"
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.physical_id == "group/log_0.out"
    assert evidence.group == "group"
    assert evidence.log_filename == "log_0.out"
    assert evidence.variant_view_key.startswith("publication.view.v1:adamw:")
    assert evidence.variant_exact_id.startswith("publication.exact.v1:adamw:")


def test_command_and_launcher_contents_do_not_fill_missing_semantics(tmp_path):
    logs = tmp_path / "logs"
    _write_run(logs, config={"command": "train.py --max_steps 9000"})
    result = audit_cell(
        {"optimizer": "adamw", "lr": "1e-3", "max_steps": "9000"},
        RunCatalog.discover(logs),
        expected_launcher="scripts/sweep/sweep.sh",
    )

    assert result.status == "UNKNOWN"
    assert result.evidence[0].physical_id == "group/log_0.out"
    assert result.evidence[0].missing_fields == ("max_steps",)


def test_matching_fields_without_producer_variant_identity_are_unknown(tmp_path):
    logs = tmp_path / "logs"
    _write_run(
        logs,
        config={"optimizer_variant_semantics": None},
    )
    result = audit_cell(
        {"optimizer": "adamw", "lr": "1e-3"},
        RunCatalog.discover(logs),
        expected_launcher="scripts/sweep/sweep.sh",
    )

    assert result.status == "UNKNOWN"
    assert len(result.evidence) == 1
    assert "producer" in result.evidence[0].issue


def test_missing_launcher_manifest_is_unknown_not_a_cross_scope_match(tmp_path):
    logs = tmp_path / "logs"
    _write_run(logs, sweep_script=None)
    result = audit_cell(
        {"optimizer": "adamw", "lr": "1e-3"},
        RunCatalog.discover(logs),
        expected_launcher="scripts/sweep/sweep.sh",
    )

    assert result.status == "UNKNOWN"
    assert "manifest launcher" in result.evidence[0].issue


def test_cli_preserves_zero_for_new_and_nonzero_for_overlap(tmp_path, capsys):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"optimizer": ["adamw"], "lr": ["1e-3"]}))
    launcher = tmp_path / "sweep.sh"
    launcher.write_text("#!/bin/bash\n--max_steps 9000\n")
    empty_logs = tmp_path / "empty-logs"

    assert main([
        str(params), "--logs-root", str(empty_logs),
        "--sweep-script", str(launcher),
    ]) == 0
    assert "contents not parsed" in capsys.readouterr().out

    logs = tmp_path / "logs"
    _write_run(logs, sweep_script=launcher.resolve().as_posix())
    assert main([
        str(params), "--logs-root", str(logs),
        "--sweep-script", str(launcher),
    ]) == 1
    output = capsys.readouterr().out
    assert "OVERLAP: 1/1" in output
    assert "physical_id='group/log_0.out'" in output
