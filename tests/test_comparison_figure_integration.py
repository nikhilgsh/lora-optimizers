"""Focused integration checks for the comparison core's plotting adapter."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from lora_playground.plotting.figures import compare_variants_figure


def _run(label: str, lr: float, loss: float, *, seed: int = 0, step: int = 4000):
    cfg = {
        "label": label,
        "optimizer": label.lower(),
        "lr": lr,
        "lora_r": 16,
        "max_steps": 4000,
        "seed": seed,
    }
    history = [{"step": step, "eval_loss": loss}]
    return cfg, history


def _render_kwargs():
    return {
        "common_where": {},
        "ref_label": "A",
        "colors": {"A": "#1f77b4", "B": "#d62728"},
        "markers": {"A": "o", "B": "s"},
        "auto_ylim": False,
        "allow_custom_labels": True,
    }


def test_prefetched_path_delegates_aggregation_to_comparison_core(monkeypatch):
    import lora_playground.comparison as comparison

    real_build = comparison.build_comparison
    called = {}

    def spy_build(runs, specs, horizon, completion_slack=300):
        called["runs"] = runs
        called["ids"] = tuple(spec.id for spec in specs)
        called["horizon"] = horizon
        called["completion_slack"] = completion_slack
        return real_build(runs, specs, horizon, completion_slack)

    monkeypatch.setattr(comparison, "build_comparison", spy_build)
    runs = [
        _run("A", 1e-3, 0.5, seed=0),
        _run("A", 1e-3, 0.7, seed=1),
        _run("B", 1e-3, 0.55),
    ]
    fig, table, summary = compare_variants_figure(
        variants={"A": {}, "B": {}},
        prefetched_runs=runs,
        variant_key=lambda cfg: cfg["label"],
        max_steps=4000,
        completion_slack=123,
        **_render_kwargs(),
    )

    assert called == {
        "runs": runs,
        "ids": ("A", "B"),
        "horizon": 4000,
        "completion_slack": 123,
    }
    assert table.loc[1e-3, "A"] == pytest.approx(0.6)
    row_a = summary[summary["variant"] == "A"].iloc[0]
    assert row_a["final"] == pytest.approx(0.6)
    plt.close(fig)


def test_nonprefetched_loader_path_remains_a_compatibility_shim(monkeypatch):
    import lora_playground.comparison as comparison
    import lora_playground.loader as loader

    def fail_build(*_args, **_kwargs):
        raise AssertionError("non-prefetched path must not call build_comparison")

    calls = []

    def fake_load_runs(*, where, **_kwargs):
        calls.append(where)
        label = "A" if where["optimizer"] == "a" else "B"
        loss = 0.5 if label == "A" else 0.6
        return [_run(label, 1e-3, loss)]

    monkeypatch.setattr(comparison, "build_comparison", fail_build)
    monkeypatch.setattr(loader, "load_runs", fake_load_runs)
    fig, table, _summary = compare_variants_figure(
        variants={"A": {"optimizer": "a"}, "B": {"optimizer": "b"}},
        **_render_kwargs(),
    )

    assert calls == [{"optimizer": "a"}, {"optimizer": "b"}]
    assert table.loc[1e-3, "A"] == pytest.approx(0.5)
    assert table.loc[1e-3, "B"] == pytest.approx(0.6)
    plt.close(fig)
