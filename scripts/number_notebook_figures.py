"""Order the notebook's figures and stamp `Figure N.` on each one's heading.

Two jobs, in this order, because the second depends on the first:

1. `--reorder` moves the rank-sweep blocks so rank ASCENDS down the document.
   A block is (heading cell, plot cell), moved as a unit.
2. Numbering walks the notebook in document order and stamps `Figure N.` onto
   the heading above each figure.

The heading -> figure map is DERIVED, never hardcoded to cell indices: an index
list goes stale the moment a cell moves, which is exactly what job 1 does. A
code cell counts as drawing a figure if it has a stored `image/png` output, or
(when outputs are cleared) if it calls a `P.*panel*` entry point or `plt.show`.
Its heading is the nearest preceding markdown cell that has a heading line, so
a compute cell sitting between a heading and its plot cell cannot claim a
number of its own.

Idempotent: a heading already carrying `Figure N.` is renumbered in place. Run
with `--check` to report without writing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "paper" / "paper_plots.ipynb"

# Tracked deliberately. The notebook's figure numbers are part of how the
# figures are referred to in chat, notes and review, and a number that
# silently goes stale when a cell moves is worse than none -- so the tool that
# maintains them lives in the repo, and `tests/test_notebook_figure_numbers.py`
# fails when the notebook drifts from what it would write.

HEADING = re.compile(r"^(#{2,6})\s+(?:Figure\s+\d+\.\s+)?(.*)$")
DRAWS = re.compile(r"P\.\w*panel\w*\(|plt\.show\(|fig\.savefig\(")

# Blocks to reorder, as (call that identifies the plot cell, sort key). Each
# group is reordered among ITSELF, in place, leaving every other cell alone.
RANK_GROUPS = [
    # The r x r metric slot on Llama-3.2-1B / openmath: authored 256, 16, 64.
    [('P.precond_panel(16)', 16),
     ('P.precond_panel(64)', 64),
     ('P.precond_panel(256)', 256)],
    # Same comparison on Qwen2.5-1.5B / openmath: authored 256 then 16.
    [('matched_revision=True', 16), ('matched_revision=True', 256)],
]


def cells_drawing_figures(nb: dict) -> list[int]:
    out = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        has_image = any(
            "image/png" in o.get("data", {}) for o in cell.get("outputs", [])
        )
        if has_image or DRAWS.search("".join(cell["source"])):
            out.append(i)
    return out


def heading_cell_for(nb: dict, code_index: int) -> int | None:
    for j in range(code_index - 1, -1, -1):
        cell = nb["cells"][j]
        if cell["cell_type"] != "markdown":
            continue
        if any(HEADING.match(l) for l in "".join(cell["source"]).split("\n")):
            return j
    return None


def set_source(cell: dict, lines: list[str]) -> None:
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def reorder(nb: dict) -> list[str]:
    """Move (heading, plot) blocks so each group's ranks ascend."""
    notes = []
    for group in RANK_GROUPS:
        blocks = []
        for needle, rank in group:
            found = None
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                if needle not in src:
                    continue
                head = heading_cell_for(nb, i)
                if head is None or head != i - 1:
                    continue          # not a clean (heading, plot) pair
                if needle == "matched_revision=True" and f"{rank}," not in src:
                    continue
                found = (head, i, rank)
                break
            if found is None:
                notes.append(f"  SKIP: no clean (heading, plot) pair for {needle!r} r={rank}")
                break
            blocks.append(found)
        if len(blocks) != len(group):
            continue
        positions = sorted(p for b in blocks for p in (b[0], b[1]))
        if positions != list(range(positions[0], positions[0] + len(positions))):
            notes.append(f"  SKIP: blocks for {group} are not contiguous")
            continue
        # Document order, NOT the order `group` lists them in. Reading the ranks
        # straight off `blocks` returns whatever order this file happens to
        # declare, so the "is it already ascending" test compared a list against
        # itself and always passed.
        blocks.sort(key=lambda b: b[0])
        current = [b[2] for b in blocks]
        wanted = sorted(current)
        if current == wanted:
            notes.append(f"  already ascending: ranks {current}")
            continue
        by_rank = {b[2]: (nb["cells"][b[0]], nb["cells"][b[1]]) for b in blocks}
        rebuilt = [c for rank in wanted for c in by_rank[rank]]
        nb["cells"][positions[0]:positions[0] + len(positions)] = rebuilt
        notes.append(f"  reordered ranks {current} -> {wanted} "
                     f"at cells {positions[0]}-{positions[-1]}")
    return notes


def renumber(nb: dict) -> list[str]:
    changes = []
    seen: set[int] = set()
    number = 0
    for code_index in cells_drawing_figures(nb):
        head = heading_cell_for(nb, code_index)
        if head is None:
            changes.append(f"  WARNING: code cell {code_index} has no heading above it")
            continue
        if head in seen:
            # A heading already numbered for an earlier plot cell: two plot
            # cells under one heading. Leave the number on the first.
            continue
        seen.add(head)
        number += 1
        lines = "".join(nb["cells"][head]["source"]).split("\n")
        target = max(i for i, l in enumerate(lines) if HEADING.match(l))
        hashes, text = HEADING.match(lines[target]).groups()
        new = f"{hashes} Figure {number}. {text}"
        if lines[target] != new:
            changes.append(f"  cell {head:3d}: {lines[target][:58]!r} -> {new[:58]!r}")
            lines[target] = new
            set_source(nb["cells"][head], lines)
    changes.append(f"  {number} figures numbered")
    return changes


def main() -> None:
    nb = json.loads(NB.read_text())
    if "--reorder" in sys.argv:
        print("reorder:")
        for n in reorder(nb):
            print(n)
    print("renumber:")
    for c in renumber(nb):
        print(c)
    if "--check" in sys.argv:
        print("\n(--check: nothing written)")
        return
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {NB}")


if __name__ == "__main__":
    main()
