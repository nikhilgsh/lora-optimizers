"""Import-boundary tests for the records-native loading API."""
from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_isolated(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_primary_load_records_import_and_call_stay_dependency_neutral():
    _run_isolated(
        """
        import json
        import sys
        import tempfile
        from pathlib import Path

        from lora_playground.run_catalog import load_records

        def forbidden_modules():
            prefixes = (
                "matplotlib",
                "lora_playground.loader",
                "lora_playground.optim",
                "lora_playground.optim_config",
                "lora_playground.plotting",
            )
            return sorted(
                name for name in sys.modules
                if any(name == prefix or name.startswith(prefix + ".")
                       for prefix in prefixes)
            )

        assert forbidden_modules() == []
        with tempfile.TemporaryDirectory() as directory:
            log = (Path(directory) / "group" / "run_info" / "logs"
                   / "log_0.out")
            log.parent.mkdir(parents=True)
            events = (
                {"event": "config", "optimizer": "adamw", "lr": 0.001},
                {"event": "eval", "step": 10, "eval_loss": 0.9},
            )
            log.write_text("".join(json.dumps(event) + "\\n"
                                   for event in events))
            records = load_records(
                equals={"optimizer": "adamw"},
                logs_root=directory,
                resolve_lineages=False,
            )

        assert len(records) == 1
        assert records[0].effective_config["lr"] == 0.001
        assert forbidden_modules() == []
        """
    )


def test_records_native_workload_consumer_avoids_legacy_loader():
    _run_isolated(
        """
        import json
        import sys
        import tempfile
        from pathlib import Path

        from lora_playground.workloads import Workload, workload_records

        with tempfile.TemporaryDirectory() as directory:
            log = (Path(directory) / "group" / "run_info" / "logs"
                   / "log_0.out")
            log.parent.mkdir(parents=True)
            events = (
                {
                    "event": "config",
                    "optimizer": "adamw",
                    "model_name": "model",
                    "lora_r": 64,
                    "lr": 0.001,
                    "max_steps": 10,
                    "data_dir": "/data/openmath_instruct_2_2m_packed_seq2048",
                },
                {"event": "eval", "step": 10, "eval_loss": 0.9},
            )
            log.write_text("".join(json.dumps(event) + "\\n"
                                   for event in events))
            workload = Workload(
                "model", "openmath", 64, "Model", "OpenMath", 10, 0.001,
                True, min_completed_steps=10,
            )
            records = workload_records(workload, logs_root=directory)

        assert len(records) == 1
        forbidden = (
            "matplotlib",
            "lora_playground.loader",
            "lora_playground.optim",
            "lora_playground.optim_config",
            "lora_playground.plotting",
        )
        unexpected = sorted(
            name for name in sys.modules
            if any(name == prefix or name.startswith(prefix + ".")
                   for prefix in forbidden)
        )
        assert unexpected == []
        """
    )


def test_legacy_loader_reexports_exact_primary_function():
    from lora_playground.run_catalog import load_records as primary
    from lora_playground.loader import load_records as compatibility

    assert compatibility is primary
