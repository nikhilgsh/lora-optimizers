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
is in flight, so the assertion is that it does not RAISE, that it draws
something, and that nothing it draws falls off the figure.
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import lora_playground.plotting.paper_plots_lib as plots  # noqa: E402
from lora_playground.plotting.style import apply_notebook_style  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "paper" / "paper_plots.ipynb"


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


def _notebook_calls():
    """Every ``P.<name>(...)`` the notebook actually makes, as source text.

    Calling each panel with its DEFAULTS is not the same as calling it the way
    the notebook does, and the difference is where the bugs live: the clipped
    legend that prompted this appeared only at
    ``precond_panel(256, model="Qwen/Qwen2.5-1.5B", data_key="opc")``, whose
    labels are longer than the default cell's, so the default-argument sweep
    below rendered it clean. Reading the calls from the notebook means a cell
    added there is covered without being retyped here.
    """
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    calls = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "P"
                    and node.func.attr in vars(plots)):
                calls.append(ast.unparse(node))
    return sorted(set(calls))


PANELS = _discover()
NOTEBOOK_CALLS = _notebook_calls()


def _clipped_edges(figure):
    """Edges of the rasterized figure that carry ink, i.e. something is cut off.

    A PIXEL test, because the obvious measurement does not work:
    ``Legend.get_window_extent`` reports the same too-wide number whether the
    legend renders complete or clipped, so an extent-based guard passes in both
    states. Rasterizing is the only thing that sees what the reader sees.

    Ink in the outermost row or column means an element ran past the canvas and
    was cut -- a legend entry losing its last characters, a suptitle losing a
    word. Curves cannot cause it: data artists are clipped to their axes, and
    the axes are inset from the figure edge by the layout.
    """
    from io import BytesIO

    import numpy as np
    from PIL import Image

    buffer = BytesIO()
    figure.savefig(buffer, format="png", facecolor="white")
    buffer.seek(0)
    pixels = np.asarray(Image.open(buffer).convert("L"))
    # 250 rather than 255: PNG rendering of white is exact, but antialiasing at
    # a glyph's edge is not, so allow the faintest gray as background.
    edges = {
        "left": pixels[:, 0], "right": pixels[:, -1],
        "top": pixels[0, :], "bottom": pixels[-1, :],
    }
    return sorted(name for name, strip in edges.items()
                  if (strip < 250).any())


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
        for index, figure in enumerate(figures):
            clipped = _clipped_edges(figure)
            assert not clipped, (
                f"{name} figure {index}: content is cut off at the "
                f"{', '.join(clipped)} edge(s)")
    finally:
        plt.close("all")


def test_the_notebook_calls_were_found():
    assert len(NOTEBOOK_CALLS) >= 25, NOTEBOOK_CALLS


@pytest.mark.parametrize("call", NOTEBOOK_CALLS)
def test_notebook_call_renders_without_spilling(call, monkeypatch):
    """Render the notebook's own calls, arguments and all."""
    monkeypatch.setattr(plots.plt, "show", lambda *a, **k: None)
    apply_notebook_style()
    try:
        eval(compile(call, "<notebook>", "eval"), {"P": plots})  # noqa: S307
        for index, number in enumerate(plt.get_fignums()):
            clipped = _clipped_edges(plt.figure(number))
            assert not clipped, (
                f"{call} figure {index}: content is cut off at the "
                f"{', '.join(clipped)} edge(s)")
    finally:
        plt.close("all")


def test_the_clipped_edge_guard_can_actually_fail(monkeypatch):
    """The guard is verified against a known positive, not assumed to work.

    A figure must be RASTERIZED under the rcParams it was LAID OUT under. The
    panels lay out inside `rc_context(NOTEBOOK_RCPARAMS)`; drawing the result
    under matplotlib's defaults uses a different font from the one the layout
    measured, and this legend runs off both edges. That is a real failure mode
    for any preview or export script that forgets `apply_notebook_style`, and
    it is what `_clipped_edges` exists to catch -- so it has to be shown
    catching it, or it is a guard that passes forever.
    """
    monkeypatch.setattr(plots.plt, "show", lambda *a, **k: None)
    plt.close("all")
    with plt.rc_context(matplotlib.rcParamsDefault):
        plots.precond_panel(256, model="Qwen/Qwen2.5-1.5B", data_key="opc",
                            model_label="Qwen2.5-1.5B")
        try:
            assert _clipped_edges(plt.gcf()), (
                "the guard reported a clean figure for a render that clips"
            )
        finally:
            plt.close("all")
