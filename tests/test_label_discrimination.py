"""Series-id / display-label contract tests.

The structural rule:
- ``series_id(cfg)`` is mechanical (cfg minus SERIES_AXIS_FIELDS).
- A display-label function may collapse runs only when they share series_id.
- ``assert_label_discriminates`` enforces this at every plot entry point.

These tests lock the contract so that adding a new train.py CLI flag
which affects optimizer math but isn't reflected in the display-label
function fails LOUDLY at plot time — rather than silently averaging
distinct algorithms as if they were seeds.
"""
import pytest

from lora_playground.manifest import SERIES_AXIS_FIELDS
from lora_playground.plotting import (
    LabelCollisionError,
    RUNTIME_FIELDS,
    assert_label_discriminates,
    series_id,
)


def test_runtime_fields_subset_of_series_axis_fields():
    """Loader's dedup whitelist (RUNTIME_FIELDS) must be a subset of the
    plot-layer's per-series axis whitelist. Otherwise the loader would
    dedup across two cfgs that the plot layer would split into series."""
    assert RUNTIME_FIELDS <= SERIES_AXIS_FIELDS


def test_series_id_ignores_lr_and_seed():
    a = {"optimizer": "adamw", "lora_r": 16, "lr": 1e-3, "seed": 0,
         "lora_init_a": "kaiming"}
    b = dict(a, lr=2e-3, seed=42)
    assert series_id(a) == series_id(b)


def test_series_id_splits_on_optimizer_math_field():
    a = {"optimizer": "polar", "lora_r": 16, "lr": 1e-3, "seed": 0,
         "precond_delta": 0.0}
    b = dict(a, precond_delta=0.01)
    assert series_id(a) != series_id(b)


def test_series_id_splits_on_unknown_future_flag():
    """A flag train.py adds tomorrow that the test/registry has never
    seen MUST split series_id by default (series-defining-by-default)."""
    a = {"optimizer": "polar", "lora_r": 16, "lr": 1e-3, "seed": 0,
         "imagined_future_flag": None}
    b = dict(a, imagined_future_flag="enabled")
    assert series_id(a) != series_id(b)


def test_assert_raises_on_collapsed_distinct_series():
    a = {"optimizer": "polar", "lora_r": 16, "lr": 1e-3, "seed": 0,
         "precond_delta_relative": False}
    b = dict(a, precond_delta_relative=True)
    runs = [(a, []), (b, [])]
    with pytest.raises(LabelCollisionError) as exc_info:
        assert_label_discriminates(runs, lambda c: c["optimizer"])
    assert "precond_delta_relative" in str(exc_info.value)


def test_assert_passes_when_label_discriminates():
    a = {"optimizer": "polar", "lora_r": 16, "lr": 1e-3, "seed": 0,
         "precond_delta_relative": False}
    b = dict(a, precond_delta_relative=True)
    runs = [(a, []), (b, [])]
    label_fn = lambda c: f"polar/{c['precond_delta_relative']}"
    assert_label_discriminates(runs, label_fn)


def test_assert_passes_for_pure_seeds():
    a = {"optimizer": "adamw", "lora_r": 16, "lr": 1e-3, "seed": 0}
    runs = [(a, []), (dict(a, seed=1), []), (dict(a, seed=2), [])]
    assert_label_discriminates(runs, lambda c: c["optimizer"])


def test_assert_passes_when_runs_differ_only_on_lr_within_label():
    """lr is a per-series axis, NOT series-defining."""
    a = {"optimizer": "adamw", "lora_r": 16, "lr": 1e-4, "seed": 0}
    runs = [(a, []), (dict(a, lr=3e-4), []), (dict(a, lr=1e-3), [])]
    assert_label_discriminates(runs, lambda c: c["optimizer"])


def test_assert_splits_buckets_by_lora_r():
    """Two runs at same label + same lr but different lora_r are in
    different buckets — no collision (different model size = different
    bucket axis, not a hidden confounder)."""
    a = {"optimizer": "adamw", "lora_r": 16, "lr": 1e-3, "seed": 0,
         "precond_delta": 0.0}
    b = {"optimizer": "adamw", "lora_r": 64, "lr": 1e-3, "seed": 0,
         "precond_delta": 0.01}  # differs on a non-axis field
    runs = [(a, []), (b, [])]
    assert_label_discriminates(runs, lambda c: c["optimizer"])


def test_series_id_splits_on_a_ctor_knob_recorded_in_the_optimizer_block():
    """A real algorithm knob must split series even when only the block records it.

    Most polar variants are declared ``__init__(self, model, **kwargs)`` and
    delegate down the MRO, so a bare ``inspect.signature`` on the class sees
    almost nothing -- 3 parameters for ``adam-soap-polar-product-lora`` against
    57 real ones. Every name it missed fell through `_series_items`' last
    branch, which drops a key recorded only inside ``optimizer_config`` with NO
    value comparison, on the grounds that the constructor no longer accepts it.
    For a knob the constructor DOES accept that premise is false, and
    ``ns_steps=5`` and ``ns_steps=8`` collapsed to one series_id -- two
    different Newton-Schulz iteration counts averaged together.
    """
    optimizer = "adam-soap-polar-product-lora"

    def cfg(ns_steps):
        return {"optimizer": optimizer, "lr": 1e-3,
                "optimizer_config": {"ns_steps": ns_steps},
                "ns_steps": ns_steps}

    assert series_id(cfg(5)) != series_id(cfg(8))


def test_series_id_still_merges_a_block_recorded_knob_at_its_own_default():
    """The split above must not resurrect the schema-growth splitting."""
    from lora_playground.plotting.dedup import _constructor_defaults

    optimizer = "adam-soap-polar-product-lora"
    default = _constructor_defaults(optimizer)["ns_steps"]
    recorded = {"optimizer": optimizer, "lr": 1e-3,
                "optimizer_config": {"ns_steps": default},
                "ns_steps": default}
    absent = {"optimizer": optimizer, "lr": 1e-3}

    assert series_id(recorded) == series_id(absent)
