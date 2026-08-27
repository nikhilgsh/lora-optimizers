"""Contract for the deprecated mutable-tuple loader facade."""
from __future__ import annotations

import inspect

import pytest

from lora_playground import loader


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
