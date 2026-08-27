"""Data-pipeline identity is explicit in leaderboard workloads."""
from __future__ import annotations

import pytest

import lora_playground.workloads as workloads
from lora_playground.workloads import Workload


_COMMON = (
    "model",
    "openmath",
    64,
    "Model",
    "OpenMath",
    9000,
    0.001,
    True,
)


def test_workload_requires_a_nonempty_pipeline_identity():
    with pytest.raises(TypeError):
        Workload(*_COMMON)
    with pytest.raises(ValueError, match="data_pipeline_version"):
        Workload(*_COMMON, "")


def test_find_workload_requires_pipeline_when_dimension_is_ambiguous(monkeypatch):
    packed_v1 = Workload(*_COMMON, "packed_v1")
    packed_v11 = Workload(*_COMMON, "packed_v1.1")
    monkeypatch.setattr(workloads, "WORKLOADS", [packed_v1, packed_v11])

    with pytest.raises(KeyError, match="specify data_pipeline_version"):
        workloads.find_workload("model", "openmath", 64)
    assert workloads.find_workload(
        "model", "openmath", 64, "packed_v1.1"
    ) is packed_v11


def test_publication_registry_is_explicitly_pipeline_scoped():
    assert workloads.WORKLOADS
    assert {
        workload.data_pipeline_version for workload in workloads.WORKLOADS
    } == {"packed_v1.1"}
