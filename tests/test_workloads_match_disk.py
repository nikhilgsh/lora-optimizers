"""The declared cell set must match what is on disk.

`lora_playground.workloads` describes itself as predicate-based, but that only
ever governed which RUNS join a cell. WHICH CELLS EXIST was 19 hand-typed
`Workload(...)` lines, so a completed campaign at a new (model, rank) simply never
appeared in the leaderboard and nothing raised. Measured when this test was
written: 24 cells on disk against 19 declared -- 29 completed runs missing, all
`allenai/OLMo-2-0425-1B / magicoder`, which `_DATASET_SUBSTRINGS` had annotated
"legacy; never a leaderboard cell" without anything enforcing it.

This is the enforcement. `discover_cells` is deliberately a CHECK rather than the
source of `WORKLOADS`: making `iter_workloads` discover would change what
`scripts/analysis/build_leaderboard_doc.py` publishes the moment a sweep lands,
and what goes in the paper's leaderboard is a decision, not a scan.

Skips when logs/ has no completed runs, so a fresh clone is not red.
"""
import pytest

import lora_playground.plotting.paper_plots_lib as P
from lora_playground.workloads import (
    LEADERBOARD_CORPORA,
    discover_cells,
    iter_workloads,
)


@pytest.fixture(scope="module")
def discovered():
    found = discover_cells()
    if not found:
        pytest.skip("no completed long-horizon runs in logs/")
    return found


def _declared():
    return {(w.model_name, w.dataset, w.rank) for w in iter_workloads()}


def test_every_cell_on_disk_is_declared(discovered):
    """The failure this test exists for: a finished campaign nobody declared.

    The message names the exact `Workload(...)` key to add, so the fix does not
    require re-deriving what ran.
    """
    missing = sorted(set(discovered) - _declared(), key=str)
    assert not missing, (
        f"{len(missing)} cell(s) with completed runs on disk are NOT declared in "
        f"workloads.WORKLOADS, so they are absent from the leaderboard and from "
        f"every panel:\n"
        + "\n".join(f"  {k}  ({discovered[k]} completed runs)" for k in missing)
        + "\nFix: declare a Workload for each, or -- if the corpus is retired -- "
        "add its dataset id to workloads._LEGACY_DATASETS."
    )


def test_every_declared_cell_has_runs(discovered):
    """The other direction: a declared cell with nothing on disk publishes an
    empty row rather than a missing one, which reads as a measured absence."""
    empty = sorted(_declared() - set(discovered), key=str)
    assert not empty, (
        f"{len(empty)} declared cell(s) have no completed runs on disk:\n"
        + "\n".join(f"  {k}" for k in empty)
        + "\nFix: remove the Workload, or leave it and say here why an empty row "
        "is intended."
    )


def test_declared_datasets_are_leaderboard_corpora():
    """A Workload naming a retired corpus would be declared and discovered-empty
    at the same time, which is confusing rather than loud."""
    bad = sorted({w.dataset for w in iter_workloads()} - LEADERBOARD_CORPORA)
    assert not bad, (
        f"declared cell(s) use non-leaderboard corpora {bad}; either drop them or "
        f"remove the dataset from workloads._LEGACY_DATASETS."
    )


def test_legacy_corpora_are_never_discovered(discovered):
    """Known-negative for the discovery filter: if the legacy exclusion stopped
    working, `test_every_cell_on_disk_is_declared` would start demanding
    magicoder cells, and the honest reading of that failure is ambiguous."""
    assert not (LEADERBOARD_CORPORA & {"magicoder"}), \
        "magicoder must stay in _LEGACY_DATASETS"
    assert all(ds in LEADERBOARD_CORPORA for _m, ds, _r in discovered), \
        f"discovery returned a legacy corpus: {sorted({d for _m, d, _r in discovered})}"


def test_discovery_finds_the_cells_the_panels_use(discovered):
    """Cross-check against the panel list: every panelled cell must have runs, or
    the figure renders empty. Catches a panel pointing at a cell that was
    declared but never actually run.

    Takes the module-scoped `discovered` fixture rather than calling
    `discover_cells()` again: the first call in a process is ~12 s (a full
    load_runs scan) and a second is still ~0.37 s, since the loader's caches
    absorb the subprocess calls but every log file is re-read and re-parsed.
    """
    by_key = set(discovered)
    missing = [c for c in P.CELLS if (c[1], c[2], c[3]) not in by_key]
    assert not missing, (
        f"{len(missing)} panelled cell(s) have no completed runs on disk:\n"
        + "\n".join(f"  {c[0]}" for c in missing)
    )
