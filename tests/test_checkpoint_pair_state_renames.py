"""The pair_state key-rename shim, tested against REAL optimizer key sets.

`CurvatureWhitenLoRA`'s two r x r slots were renamed `L_A`/`R_B` -> `P_A`/`Q_B`
and its eigh eigenbasis `Q_A`/`Q_B` -> `U_A`/`U_B`. `Q_B` therefore appears on
BOTH sides of the boundary with different meanings, which is what makes the
"is this optimizer on the old schema" test easy to get wrong: the first version
of the guard asked whether any renamed key was live, which is true of every
post-rename CW state (it has `Q_B`), so the shim silently never fired and an
old-name checkpoint resumed with its curvature EMAs reset to their init value.

The map itself is SCOPED TO THE OPTIMIZER CLASS (`PAIR_STATE_ALIASES`, read by
`_pair_state_aliases`), which is what keeps `AdamSOAPPolarProductLoRA` — still
using `L_A`, `R_B`, `Q_A`, `Q_B` with their original SOAP meanings — out of the
translation entirely rather than relying on a guard to spare it.

The live key sets here are read off freshly constructed optimizers rather than
hand-typed, so a future pair_state change cannot leave this test asserting
against a schema that no longer exists.

The end-to-end counterpart (save, downgrade the keys on disk, load, resume)
lives in `tests/test_checkpoint.py`, where the save/load fixtures are.
"""
import pytest
import torch

from lora_playground.checkpoint import (
    _PAIR_STATE_RENAMES_FALLBACK,
    _apply_pair_state_renames,
    _pair_state_aliases,
)
from lora_playground.optim import AdamSOAPPolarProductLoRA, CurvatureWhitenLoRA

R, D_IN, D_OUT = 8, 64, 32
# The retired schema, as an old checkpoint's pair_state entry would carry it.
OLD_ENTRY = {"L_A": torch.ones(R, R), "R_B": torch.full((R, R), 2.0),
             "Q_A": torch.eye(R) * 3, "Q_B": torch.eye(R) * 4,
             "m_A": torch.zeros(R, D_IN)}


class _FakeLoRALinear(torch.nn.Module):
    """Minimal PEFT-shaped module: collect_lora_pairs keys off these names."""

    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(D_IN, R, bias=False)})
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(R, D_OUT, bias=False)})
        torch.nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x):
        return self.lora_B["default"](self.lora_A["default"](x))


def _live_keys(opt):
    return set(opt.pair_state[0].keys())


def test_curvature_whiten_live_keys_are_the_new_schema():
    """Guards the premise of the two tests below."""
    live = _live_keys(CurvatureWhitenLoRA(_FakeLoRALinear(), precond="factorwise"))
    assert {"P_A", "Q_B", "U_A", "U_B"} <= live
    assert not ({"L_A", "R_B", "Q_A"} & live)
    # The trap: `Q_B` is live on the NEW schema too, so it cannot discriminate.
    assert "Q_B" in live


def test_old_checkpoint_renames_onto_curvature_whiten():
    live = _live_keys(CurvatureWhitenLoRA(_FakeLoRALinear(), precond="factorwise"))
    out = _apply_pair_state_renames(dict(OLD_ENTRY), live)
    assert set(out) == {"P_A", "Q_B", "U_A", "U_B", "m_A"}
    # The permutation is simultaneous: old L_A -> P_A and old Q_B -> U_B, so the
    # value that was under `Q_B` must land under `U_B`, not stay under `Q_B`.
    assert torch.equal(out["P_A"], OLD_ENTRY["L_A"])
    assert torch.equal(out["Q_B"], OLD_ENTRY["R_B"])
    assert torch.equal(out["U_A"], OLD_ENTRY["Q_A"])
    assert torch.equal(out["U_B"], OLD_ENTRY["Q_B"])


def test_soap_optimizer_keeps_its_own_old_names():
    """AdamSOAPPolarProductLoRA still uses L_A/R_B/Q_A/Q_B with SOAP meanings."""
    live = _live_keys(AdamSOAPPolarProductLoRA(_FakeLoRALinear()))
    assert {"L_A", "R_B", "Q_A", "Q_B"} <= live
    out = _apply_pair_state_renames(dict(OLD_ENTRY), live)
    assert set(out) == set(OLD_ENTRY), "SOAP state must pass through untouched"


def test_new_schema_entry_is_left_alone():
    """A current-schema checkpoint carries no old-only key, so nothing happens."""
    live = _live_keys(CurvatureWhitenLoRA(_FakeLoRALinear(), precond="factorwise"))
    entry = {"P_A": torch.ones(R, R), "Q_B": torch.eye(R), "U_A": torch.eye(R)}
    assert _apply_pair_state_renames(dict(entry), live) == entry


# ── the map is scoped to the optimizer class ────────────────────────────────


class _DeclaresAliases:
    PAIR_STATE_ALIASES = {"old": "new"}


class _DeclaresNoRenames:
    PAIR_STATE_ALIASES = {}


def test_a_declared_alias_map_wins_over_the_module_fallback():
    """The class attribute is the source of truth, empty included — that is
    what makes an identically-spelled key in another optimizer unable to
    participate, rather than something a second guard table has to exclude."""
    assert _pair_state_aliases(_DeclaresAliases()) == {"old": "new"}
    assert _pair_state_aliases(_DeclaresNoRenames()) == {}
    # A copy, so a caller cannot mutate the class attribute through it.
    got = _pair_state_aliases(_DeclaresAliases())
    got["old"] = "tampered"
    assert _DeclaresAliases.PAIR_STATE_ALIASES == {"old": "new"}


def test_an_empty_alias_map_disables_the_rename():
    live = _live_keys(CurvatureWhitenLoRA(_FakeLoRALinear(), precond="factorwise"))
    assert _apply_pair_state_renames(dict(OLD_ENTRY), live, {}) == OLD_ENTRY


def test_temporary_fallback_matches_the_class_attribute():
    """Tripwire for removing the temporary fallback in `checkpoint.py`.

    While `CurvatureWhitenLoRA` has no `PAIR_STATE_ALIASES` this skips. The
    moment it gains one, this checks the declared map agrees with the table
    that stood in for it — and `_PAIR_STATE_RENAMES_FALLBACK`, the fallback
    branch of `_pair_state_aliases`, and this test can all be deleted.
    """
    declared = getattr(CurvatureWhitenLoRA, "PAIR_STATE_ALIASES", None)
    if declared is None:
        pytest.skip("CurvatureWhitenLoRA.PAIR_STATE_ALIASES not declared yet; "
                    "checkpoint._PAIR_STATE_RENAMES_FALLBACK still stands in")
    assert dict(declared) == _PAIR_STATE_RENAMES_FALLBACK, (
        "the declared alias map disagrees with the fallback it replaces — one "
        "of the two is wrong, and an old checkpoint would load into the wrong "
        "slots")


def test_soap_is_untouched_once_curvature_whiten_declares_its_own_map(monkeypatch):
    """Scoping, stated as the property that matters: giving CurvatureWhitenLoRA
    its own alias map must not change how an AdamSOAPPolarProductLoRA
    checkpoint is read, because the map is looked up per instance."""
    monkeypatch.setattr(CurvatureWhitenLoRA, "PAIR_STATE_ALIASES",
                        dict(_PAIR_STATE_RENAMES_FALLBACK), raising=False)
    soap = AdamSOAPPolarProductLoRA(_FakeLoRALinear())
    out = _apply_pair_state_renames(
        dict(OLD_ENTRY), _live_keys(soap), _pair_state_aliases(soap))
    assert set(out) == set(OLD_ENTRY), "SOAP state must pass through untouched"
    for k, v in OLD_ENTRY.items():
        assert torch.equal(out[k], v)
