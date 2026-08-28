"""Every notebook panel entrypoint renders against the recorded logs.

This is the guard the notebook itself used to be. Each of these failed today
only when `jupyter nbconvert --execute` reached the offending cell, minutes into
a full build:

  * `precond_panel(16, model="Qwen/...", matched_revision=True)` raised
    "has no recorded reference arm" because an ` impl-rev=2` label suffix meant
    no run matched any arm;
  * `factorwise_freeze_panel(256)` raised SemanticRevisionConflictError because
    its arms grew past what its slot-cohort view tested;
  * the matched panels raised UncertifiedBaselineError for a cell that records
    no AdamW run.

Panels are DISCOVERED, not listed, so a new one is covered the day it is added
-- which is the point: the failure mode is always "the panel I just wrote
raises deep inside comparison", and nothing was calling them but the notebook.

A panel legitimately returns None (the diagnostic panels print and draw rather
than summarising) and legitimately renders an incomplete arm set while a sweep
is in flight, so the assertion is that it does not RAISE and that it draws
something.
"""
from __future__ import annotations

import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import lora_playground.plotting.paper_plots_lib as plots  # noqa: E402
from lora_playground.plotting.style import apply_notebook_style  # noqa: E402


def _discover():
    """Public `*_panel` entrypoints callable with no required argument."""
    found = []
    for name, fn in sorted(vars(plots).items()):
        if not (inspect.isfunction(fn) and name.endswith("panel")):
            continue
        if name.startswith("_") or fn.__module__ != plots.__name__:
            continue
        required = [p for p in inspect.signature(fn).parameters.values()
                    if p.default is inspect.Parameter.empty]
        if not required:
            found.append(name)
    return found


PANELS = _discover()


def test_discovery_found_the_panels():
    """A rename that empties the list must fail loudly, not pass vacuously."""
    assert len(PANELS) >= 12, PANELS


@pytest.mark.parametrize("name", PANELS)
def test_panel_renders(name, monkeypatch):
    monkeypatch.setattr(plots.plt, "show", lambda *a, **k: None)
    apply_notebook_style()
    try:
        plots.__dict__[name]()
        figures = [plt.figure(n) for n in plt.get_fignums()]
        assert figures, f"{name} drew no figure"
        assert any(f.axes for f in figures), f"{name} drew no axes"
    finally:
        plt.close("all")
