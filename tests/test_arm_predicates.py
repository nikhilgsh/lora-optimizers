"""Regression guard for the fail-closed arm predicates in
``lora_playground.plotting.arms``.

Why this test exists
--------------------
A figure's arm predicate is only as good as the fields it pins. When a new
optimizer flag is added and a sweep sets it, a hand-typed predicate that never
mentioned the field keeps matching BOTH values: two arms merge into one display
label and the figure silently keeps whichever run has the lowest loss.
``arms.arm`` closes that by pinning every ``OptimizerConfig`` field to its
default, so a new flag is pinned automatically. These tests assert the two
things that has to be true for that to hold:

  1. the derivation stays complete — no config field quietly escapes the pin
     set, and the dataclass default an arm pins is the same value the loader
     backfills (``test_no_config_field_escapes_the_pin_set``,
     ``test_pinned_defaults_agree_with_the_cli``); and
  2. the arms actually discriminate against the runs on disk
     (``test_arm_dict_discriminates``) — the end-to-end check that fails the
     day a new sweep lands that an existing arm cannot separate.

(2) reads the live ``logs/`` tree, so it is the test that turns "the user's
notebook raises ``LabelCollisionError`` mid-analysis" into "CI fails".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_playground.plotting import arms as A
from lora_playground.plotting.dedup import assert_label_discriminates

_LOGS_ROOT = ROOT / "logs"

# One bucket per (workload, lr): runs inside a bucket that carry one display
# label must be the same algorithm. Model / dataset / rank / pipeline version
# are workload identity, not a per-series axis, so they open separate buckets
# rather than counting as collisions.
BUCKET_KEYS = ("model_name", "data_dir", "lora_r", "data_pipeline_version", "lr")


# ── the derivation stays complete ───────────────────────────────────────────

def test_pinned_defaults_agree_with_the_cli():
    """`arm` pins the dataclass default; the loader backfills the argparse
    default onto runs that did not set the flag. If those two ever name
    different values, the arm stops matching its own runs."""
    assert A.check_pinned_defaults_agree_with_cli() == {}


def test_no_config_field_escapes_the_pin_set():
    """A field with no train.py CLI flag never appears in a run cfg, so no arm
    can discriminate on it. Exactly two are in that state today, both
    constructor-only knobs. A NEW one means a flag was added to
    ``OptimizerConfig`` without a CLI flag — add the flag, or the arms cannot
    separate sweeps that set it."""
    assert A.check_config_fields_pinnable()["no_cli_flag"] == ["muon_alpha", "muon_rank"]


def test_arm_pins_every_pinnable_field():
    a = A.arm("adamw")
    assert set(a) == {"optimizer"} | A.PINNED_FIELDS()
    assert a["optimizer"] == "adamw"


def test_arm_override_wins_over_the_default():
    assert A.arm("kl-diag-polar-lora")["msign"] == "full"
    assert A.arm("kl-diag-polar-lora", msign="diag")["msign"] == "diag"


def test_arm_rejects_an_unknown_override():
    """A predicate on a misspelled field pins nothing and fails open — the
    exact failure mode this module exists to prevent, so it must raise."""
    with pytest.raises(ValueError, match="preconditioner"):
        A.arm("kl-diag-polar-lora", preconditioner="one-sided")


def test_every_exported_arm_pins_the_optimizer():
    for name, arms in A.ALL_ARM_DICTS.items():
        for label, pred in arms.items():
            assert "optimizer" in pred, f"{name}[{label}] does not pin `optimizer`"


# ── the arms discriminate against the runs on disk ──────────────────────────

@pytest.fixture(scope="module")
def all_runs():
    """Every packed-pipeline run in the tree, loaded once for all the arm dicts.

    ``packed_v1`` and ``packed_v1.1`` both count (the paper's opc cells are the
    former, its openmath cells the latter); the bucket keys keep them apart, so
    including both widens coverage without merging them. The legacy
    ``unpacked_v0`` runs are excluded: a packed and an unpacked number are not
    comparable in the first place (prompt-mask changes the objective, packing
    changes per-step token density), and those runs carry collisions on fields
    no optimizer predicate can pin — the loader-derived
    ``effective_inner_polar`` and the workload-level ``train_samples``. Those
    are properties of that old data, not holes in the arms. Every sweep
    submitted since 2026-05-08 is packed, so new sweeps — the ones this test is
    here to catch — are all in scope.
    """
    from lora_playground.loader import load_runs
    if not _LOGS_ROOT.is_dir():
        pytest.skip("no logs/ tree")
    runs = load_runs(
        where={"data_pipeline_version": lambda v: str(v).startswith("packed_")},
        logs_root=str(_LOGS_ROOT), warn_cross_commit=False, quiet=True)
    if not runs:
        pytest.skip("no packed-pipeline runs in logs/")
    return runs


def _variant_key(arms):
    """First arm whose predicate matches, else None — mirrors the notebook's
    ``_variant_key_fn`` and ``compare_variants_figure``'s labeling."""
    def key(cfg):
        for label, pred in arms.items():
            for k, v in pred.items():
                if k not in cfg:
                    break
                c = cfg[k]
                if callable(v):
                    if not v(c):
                        break
                elif isinstance(v, (list, set, tuple, frozenset)):
                    if c not in v:
                        break
                elif c != v:
                    break
            else:
                return label
        return None
    return key


@pytest.mark.parametrize("name", sorted(A.ALL_ARM_DICTS))
def test_arm_dict_discriminates(name, all_runs):
    """No (label, workload, lr) bucket may hold two distinct series_ids.

    A failure names the differing cfg field: pin it on the existing arm and
    give the new value its own arm.
    """
    arms = A.ALL_ARM_DICTS[name]
    key = _variant_key(arms)
    guarded = [(c, h) for c, h in all_runs if key(c) is not None]
    # A dict that matches nothing is a dead predicate, not a pass. An
    # over-pinned arm silently selects zero runs and would "discriminate"
    # perfectly, so the emptiness has to be the failure. (A single arm may
    # legitimately be empty while its sweep is in flight — the figures render
    # those with allow_partial — so the assertion is at dict level.)
    assert guarded, (
        f"{name} matches no run in logs/: every arm's predicate is dead. "
        f"Most likely a field was pinned to a value no run carries."
    )
    assert_label_discriminates(guarded, key, bucket_keys=BUCKET_KEYS)
