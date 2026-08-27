"""Contract for the deprecated mutable-tuple loader facade."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from lora_playground import loader


def _write_log(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def _config(**fields) -> dict:
    return {"event": "config", **fields}


def _eval(step: int, loss: float, *, lr: float = 1e-3) -> dict:
    return {"event": "eval", "step": step, "eval_loss": loss, "lr": lr}


def test_load_runs_warns_and_delegates_every_argument(monkeypatch):
    sentinel = object()
    calls = []

    def compatibility_spy(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(loader, "_load_runs_compatibility", compatibility_spy)
    where = {"optimizer": "adamw"}
    runtime_fields = frozenset({"runtime"})

    def postprocess(_cfg, _group):
        return None

    with pytest.warns(
        DeprecationWarning,
        match=r"load_runs\(\) is deprecated; use load_records\(\)",
    ):
        result = loader.load_runs(
            where,
            key_axes=("optimizer", "lr"),
            runtime_fields=runtime_fields,
            cfg_postprocess=postprocess,
            logs_root="/tmp/test-load-runs-compatibility",
            warn_cross_commit=False,
            unique_on=("optimizer",),
            allow_axes=("seed",),
            quiet=False,
        )

    assert result is sentinel
    assert calls == [(
        (where,),
        {
            "key_axes": ("optimizer", "lr"),
            "runtime_fields": runtime_fields,
            "cfg_postprocess": postprocess,
            "logs_root": "/tmp/test-load-runs-compatibility",
            "warn_cross_commit": False,
            "unique_on": ("optimizer",),
            "allow_axes": ("seed",),
            "quiet": False,
        },
    )]


def test_load_runs_facade_preserves_signature_and_public_documentation():
    assert inspect.signature(loader.load_runs) == inspect.signature(
        loader._load_runs_compatibility
    )
    assert "Compatibility API returning mutable" in loader.load_runs.__doc__
    assert "New code should use :func:`load_records`" in loader.load_runs.__doc__
    assert "never" in loader.load_runs.__doc__
    assert "reconstructs missing defaults" in loader.load_runs.__doc__


def test_facade_rejects_retired_key_axes_deduplication(tmp_path):
    with pytest.warns(DeprecationWarning):
        with pytest.raises(
            NotImplementedError,
            match="no longer performs implicit physical-run deduplication",
        ):
            loader.load_runs(
                key_axes=("optimizer", "lr"),
                logs_root=str(tmp_path),
                warn_cross_commit=False,
            )


def test_facade_is_filesystem_led_logged_only_and_does_not_call_legacy_path(
    tmp_path, monkeypatch,
):
    logs = tmp_path / "logs"
    _write_log(
        logs / "missing-manifest" / "run_info" / "logs" / "log_0.out",
        [
            _config(
                optimizer="adamw",
                lr=1e-3,
                optimizer_effective={"effective_mode": "logged"},
                execution_source_sha="audit-only",
            ),
            _eval(10, 0.9),
        ],
    )
    corrupt = logs / "corrupt-manifest" / "run_info"
    _write_log(
        corrupt / "logs" / "log_1.out",
        [_config(optimizer="muon", lr=2e-3), _eval(10, 0.8, lr=2e-3)],
    )
    (corrupt / "meta.json").write_text("{not json")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy reconstruction/admission path was called")

    monkeypatch.setattr(loader, "_load_runs_inner", forbidden)
    monkeypatch.setattr(loader, "_argparse_defaults", forbidden)
    monkeypatch.setattr(loader, "_precond_by_optimizer", forbidden)

    with pytest.warns(DeprecationWarning):
        runs = loader.load_runs(
            where={"optimizer": ["adamw", "muon"]},
            logs_root=str(logs),
            warn_cross_commit=False,
        )

    assert [cfg["log_group"] for cfg, _ in runs] == [
        "corrupt-manifest",
        "missing-manifest",
    ]
    adamw = next(cfg for cfg, _ in runs if cfg["optimizer"] == "adamw")
    assert adamw["effective_mode"] == "logged"
    assert adamw["execution_source_sha"] == "audit-only"
    assert "data_pipeline_version" not in adamw
    assert "_derived" not in adamw
    assert not (logs / "_runs_cache").exists()
    assert not (logs / "_argparse_defaults.json").exists()
    assert not (logs / "_precond_by_optimizer.json").exists()


def test_facade_preserves_literal_list_callable_and_postprocess_where(tmp_path):
    logs = tmp_path / "logs"
    for index, (optimizer, lr, rank) in enumerate((
        ("adamw", 1e-3, 16),
        ("muon", 2e-3, 16),
        ("other", 3e-3, 64),
    )):
        _write_log(
            logs / "group" / "run_info" / "logs" / f"log_{index}.out",
            [_config(optimizer=optimizer, lr=lr, lora_r=rank), _eval(10, 1.0, lr=lr)],
        )

    calls = []

    def postprocess(cfg, group):
        calls.append((cfg["optimizer"], group))
        cfg["postprocessed_rank"] = cfg["lora_r"]

    with pytest.warns(DeprecationWarning):
        runs = loader.load_runs(
            where={
                "optimizer": ["adamw", "muon"],
                "lr": lambda value: value > 1e-3,
                "postprocessed_rank": 16,
            },
            cfg_postprocess=postprocess,
            logs_root=str(logs),
            warn_cross_commit=False,
        )

    assert len(runs) == 1
    assert runs[0][0]["optimizer"] == "muon"
    assert runs[0][0]["postprocessed_rank"] == 16
    assert calls == [
        ("adamw", "group"),
        ("muon", "group"),
        ("other", "group"),
    ]


def test_facade_keeps_legacy_siblings_physical_and_resolves_explicit_lineage(
    tmp_path,
):
    logs = tmp_path / "logs"
    legacy_dir = logs / "legacy" / "run_info" / "logs"
    _write_log(
        legacy_dir / "log_0.out",
        [_config(optimizer="adamw", lr=1e-3), _eval(10, 0.9)],
    )
    _write_log(
        legacy_dir / "log_0.out.resume_0",
        [_config(optimizer="adamw", lr=1e-3), _eval(20, 0.8)],
    )

    common = {
        "run_schema_version": 1,
        "checkpoint_identity": "logical/task-0",
        "semantic_revisions": {
            "optimizer_impl": 1,
            "data_pipeline": "packed_v1.1",
            "measurement": 1,
        },
        "optimizer": "muon",
        "lr": 2e-3,
    }
    _write_log(
        logs / "root" / "run_info" / "logs" / "log_2.out",
        [
            _config(**common, attempt_id="attempt-a", resume_parent_attempt_id=None),
            _eval(10, 0.7, lr=2e-3),
        ],
    )
    _write_log(
        logs / "child" / "run_info" / "logs" / "log_9.out",
        [
            _config(**common, attempt_id="attempt-b", resume_parent_attempt_id=None),
            {
                "event": "resume",
                "resume_parent_attempt_id": "attempt-a",
                "checkpoint_identity": "logical/task-0",
            },
            _eval(20, 0.6, lr=2e-3),
        ],
    )

    with pytest.warns(DeprecationWarning):
        legacy = loader.load_runs(
            where={"optimizer": "adamw"},
            logs_root=str(logs),
            warn_cross_commit=False,
        )
    assert len(legacy) == 2
    assert [history[0]["step"] for _, history in legacy] == [10, 20]
    assert [cfg["_log_filename"] for cfg, _ in legacy] == [
        "log_0.out",
        "log_0.out.resume_0",
    ]

    with pytest.warns(DeprecationWarning):
        resumed = loader.load_runs(
            where={"_group": "child"},
            logs_root=str(logs),
            warn_cross_commit=False,
        )
    assert len(resumed) == 1
    cfg, history = resumed[0]
    assert cfg["run_id"] == "attempt-b"
    assert cfg["log_group"] == "child"
    assert [event["step"] for event in history] == [10, 20]


def test_facade_returns_fresh_mutable_copies_and_keeps_unique_guard(tmp_path):
    logs = tmp_path / "logs"
    for index, variant in enumerate(("a", "b")):
        _write_log(
            logs / "group" / "run_info" / "logs" / f"log_{index}.out",
            [
                _config(optimizer="adamw", lr=1e-3, hidden_variant=variant),
                _eval(10, 0.9),
            ],
        )

    with pytest.warns(DeprecationWarning):
        first = loader.load_runs(
            logs_root=str(logs), warn_cross_commit=False,
        )
    first[0][0]["optimizer"] = "mutated"
    first[0][1][0]["eval_loss"] = -1.0

    with pytest.warns(DeprecationWarning):
        second = loader.load_runs(
            logs_root=str(logs), warn_cross_commit=False,
        )
    assert second[0][0]["optimizer"] == "adamw"
    assert second[0][1][0]["eval_loss"] == 0.9

    with pytest.warns(DeprecationWarning):
        with pytest.raises(loader.UncontrolledAxisError, match="hidden_variant"):
            loader.load_runs(
                logs_root=str(logs),
                warn_cross_commit=False,
                unique_on=("optimizer", "lr"),
            )
