"""The notebook's figure numbers stay in document order.

`### Figure N.` headings are how a figure is named in chat, in notes and in
review, so a number that goes stale the moment a cell moves is worse than no
number: "Figure 24" then means two different panels depending on who is looking.
The numbering is maintained by `scripts/number_notebook_figures.py`; this test
fails when the checked-in notebook has drifted from what that script would
write, which is what happens after a cell is inserted, deleted or reordered
without re-running it.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "paper" / "paper_plots.ipynb"
TOOL = ROOT / "scripts" / "number_notebook_figures.py"


def _tool():
    spec = importlib.util.spec_from_file_location("number_notebook_figures", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_figure_is_numbered_in_document_order():
    tool = _tool()
    notebook = json.loads(NB.read_text())
    changes = tool.renumber(notebook)
    stale = [c for c in changes if "->" in c]
    assert not stale, (
        "figure headings are out of order or unnumbered; re-run "
        "`python scripts/number_notebook_figures.py`:\n  " + "\n  ".join(stale)
    )


def test_every_figure_producing_cell_has_a_heading():
    tool = _tool()
    notebook = json.loads(NB.read_text())
    orphans = [i for i in tool.cells_drawing_figures(notebook)
               if tool.heading_cell_for(notebook, i) is None]
    assert not orphans, (
        f"code cells draw a figure with no heading above them: {orphans}. "
        "A figure with no heading cannot be referred to by number."
    )
