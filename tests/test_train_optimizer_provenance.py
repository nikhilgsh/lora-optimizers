"""Config-event glue for producer-owned optimizer publication semantics."""
from __future__ import annotations

from lora_playground.train import _optimizer_provenance_fields


def test_optimizer_provenance_fields_share_one_config_snapshot(monkeypatch):
    class ToyOptimizer:
        pass

    config = {
        "_optim_class": "ToyOptimizer",
        "lr": 1e-3,
        "betas": (0.8, 0.9),
    }
    effective = {"mode": "resolved"}
    monkeypatch.setattr(
        "lora_playground.train.optimizer_config_dict", lambda _opt: config
    )
    monkeypatch.setattr(
        "lora_playground.train.optimizer_effective_config",
        lambda _opt: effective,
    )

    fields = _optimizer_provenance_fields(
        optimizer_name="toy",
        optimizer=ToyOptimizer(),
        semantic_revision=4,
        implementation_revision="abc123",
    )

    assert fields["optimizer_config"] is config
    assert fields["optimizer_effective"] is effective
    assert fields["optimizer_variant_semantics"] == {
        "schema_version": 1,
        "optimizer": "toy",
        "config": {"beta1": 0.8, "beta2": 0.9},
        "effective": {"mode": "resolved"},
        "semantic_revision": 4,
        "implementation": {
            "class": f"{ToyOptimizer.__module__}.{ToyOptimizer.__qualname__}",
            "revision": "abc123",
        },
    }
