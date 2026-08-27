"""`paper_plots_lib.CELLS` must be an ORDER over the workload registry, not a
second list of experiments.

`lora_playground.workloads` calls itself the single source of truth for
(model, dataset, rank) cells. `CELLS` used to retype 13 of them and the two had
drifted in BOTH directions: (Llama-3.2-1B, openmath, 16) was in `CELLS` but
absent from the registry, while r32/64/128/256 were all declared -- so
`find_workload(..., 16)` raised for the very cell the `precond` ablation is read
at -- and the registry carried (OLMo-2-1B, openmath, 64) and
(Qwen3-0.6B, openwebmath, 64) that no panel shows. The leaderboard and the paper
panels disagreed about which experiments exist, with nothing to distinguish a
deliberate omission from an oversight.

These tests pin the partition: every registry cell is either panelled or
explicitly excused, and every panelled cell exists in the registry.
"""
import pytest

import lora_playground.plotting.paper_plots_lib as P
from lora_playground.workloads import find_workload, iter_workloads


def _registry_keys():
    return {(wl.model_display, wl.dataset, wl.rank) for wl in iter_workloads()}


def test_every_panelled_cell_exists_in_the_registry():
    """A panel for an undeclared cell must fail at import, not render a figure
    the leaderboard knows nothing about. `_cell_from_registry` raises KeyError,
    so reaching this assertion at all means import succeeded."""
    missing = [k for k in P._CELLS_ORDER if k not in _registry_keys()]
    assert not missing, (
        f"{len(missing)} panelled cell(s) absent from lora_playground.workloads:\n"
        + "\n".join(f"  {k}" for k in missing)
        + "\nFix: declare the Workload, or drop the panel."
    )


def test_every_registry_cell_is_panelled_or_explicitly_excused():
    """The other direction, which is the one that actually drifted unnoticed:
    a declared cell that no panel shows and no entry excuses."""
    accounted = set(P._CELLS_ORDER) | set(P.CELLS_NOT_PANELLED)
    orphans = sorted(_registry_keys() - accounted)
    assert not orphans, (
        f"{len(orphans)} registry cell(s) neither panelled nor excused:\n"
        + "\n".join(f"  {k}" for k in orphans)
        + "\nFix: add to _CELLS_ORDER to show it, or to CELLS_NOT_PANELLED with a reason."
    )


def test_exclusions_name_real_registry_cells_with_reasons():
    """A stale exclusion is as misleading as a missing one: it reads as
    'deliberately not shown' for a cell that no longer exists."""
    stale = sorted(k for k in P.CELLS_NOT_PANELLED if k not in _registry_keys())
    assert not stale, f"CELLS_NOT_PANELLED names cells the registry does not declare: {stale}"
    unreasoned = sorted(k for k, v in P.CELLS_NOT_PANELLED.items() if not str(v).strip())
    assert not unreasoned, f"exclusions without a reason: {unreasoned}"


def test_a_cell_is_never_both_panelled_and_excused():
    both = sorted(set(P._CELLS_ORDER) & set(P.CELLS_NOT_PANELLED))
    assert not both, f"cells both panelled and excused: {both}"


def test_panel_order_and_identity_are_stable():
    """`panel_n(i)` indexes CELLS and notebook cells are written as
    `P.panel_n(3)`, so a reordering silently repoints every figure. Pin the
    (model, data_key, rank) triple per index; captions may change.
    """
    expected = [
        ("allenai/OLMo-2-0425-1B",     "opc",      256),
        ("Qwen/Qwen2.5-1.5B",          "opc",      256),
        ("meta-llama/Llama-3.2-1B",    "opc",      256),
        ("meta-llama/Meta-Llama-3-8B", "opc",      256),
        ("Qwen/Qwen2.5-1.5B",          "bengali",  256),
        ("allenai/OLMo-2-0425-1B",     "openmath", 256),
        ("Qwen/Qwen2.5-1.5B",          "openmath", 256),
        ("meta-llama/Meta-Llama-3-8B", "openmath", 256),
        ("meta-llama/Llama-3.2-1B",    "openmath",  16),
        ("meta-llama/Llama-3.2-1B",    "openmath",  32),
        ("meta-llama/Llama-3.2-1B",    "openmath",  64),
        ("meta-llama/Llama-3.2-1B",    "openmath", 128),
        ("meta-llama/Llama-3.2-1B",    "openmath", 256),
    ]
    assert [c[1:] for c in P.CELLS] == expected


def test_captions_use_the_paper_name_for_the_8b_model():
    """The registry spells it Meta-Llama-3-8B so its `label` gives 'Meta/...'
    and does not collide with Llama-3.2-1B's 'Llama/...'. The manuscript and the
    project CLAUDE.md say 'Llama-3-8B'. Both must hold at once."""
    captions = [c[0] for c in P.CELLS]
    assert "Llama-3-8B opc r256" in captions
    assert not any("Meta-Llama-3-8B" in c for c in captions)
    assert find_workload("Meta-Llama-3-8B", "opc", 256).model_display == "Meta-Llama-3-8B"


def test_the_r16_cell_is_declared():
    """Regression for the specific drift found: r16 completes a ladder whose
    r32/64/128/256 were all declared, and it is where the precond ablation is
    read. Its absence made find_workload raise for a cell CELLS listed."""
    wl = find_workload("Llama-3.2-1B", "openmath", 16)
    assert wl.rank == 16 and wl.horizon == 9000


def test_find_workload_still_raises_for_a_genuinely_absent_cell():
    """Known-negative: the guard above is only meaningful if a miss raises."""
    with pytest.raises(KeyError):
        find_workload("Llama-3.2-1B", "openmath", 12345)
