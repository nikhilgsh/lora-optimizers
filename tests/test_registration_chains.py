"""Guards for the hand-maintained registration chains, so forgetting a step
fails instead of going quiet.

Adding one optimizer requires four separate edits (project CLAUDE.md): the class
in optim.py, an OPTIMIZER_CHOICES entry, a build_optimizer branch, and
registration in OPTIM_COLORS plus at least one OPTIM_FAMILIES set. Two of those
were already enforced -- `test_manifests.test_optim_choices_have_color_entries`
covers OPTIM_COLORS, and `build_optimizer` raises on an unknown name -- while
OPTIM_FAMILIES membership was enforced only by a `warnings.warn` at module
import, which nothing reads and no test asserts.

The wrapper checks here are regression guards: both currently pass with zero
violations across the ~120 scripts/sweep/*.sh, and they exist so the next
train.py rename or the next hand-copied block cannot break that silently.

Flags are extracted STATICALLY from train.py's add_argument calls rather than by
running `train_lora.py --help`, which would import torch and transformers.
"""
from __future__ import annotations

import glob
import re
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "lora_playground" / "train.py"
WRAPPERS = sorted(glob.glob(str(ROOT / "scripts" / "sweep" / "*.sh")))

_ADD_ARG = re.compile(r'add_argument\(\s*"--([a-z0-9][a-z0-9_\-]*)"(.*?)\)\n', re.S)
_FLAG_IN_SH = re.compile(r"--([a-z0-9][a-z0-9_\-]*)")


@lru_cache(maxsize=1)
def _train_flags() -> dict[str, str]:
    """``{flag: the add_argument(...) body}`` for every train.py CLI flag.

    Cached: called 6 times across this module (3 directly and 3 via
    `_default_true_booleans`), each re-reading and re-regexing train.py, which
    cannot change mid-run.
    """
    return {m.group(1): m.group(2) for m in _ADD_ARG.finditer(TRAIN_PY.read_text())}


def _default_true_booleans() -> list[str]:
    """Flags whose OMISSION leaves the feature ENABLED.

    With `BooleanOptionalAction` and `default=True`, or `store_false`, not passing
    the flag keeps it on — so a caller that appends `--x` only in the true branch
    gets the opposite of what it intended in the false branch, silently.
    """
    return sorted(
        f for f, body in _train_flags().items()
        if "default=True" in body
        and ("BooleanOptionalAction" in body or "store_false" in body)
    )


def _all_family_members() -> frozenset:
    """Every optimizer named by any OPTIM_FAMILIES set.

    One idiom: the two tests below computed this union two different ways a
    few lines apart (`set().union(*...)` and a nested comprehension).
    """
    from lora_playground.plotting import OPTIM_FAMILIES
    return frozenset(o for members in OPTIM_FAMILIES.values() for o in members)


@lru_cache(maxsize=1)
def _wrapper_lines() -> tuple[tuple[str, int, str], ...]:
    """``(wrapper name, line number, line)`` for every non-comment line.

    Cached and materialized rather than a generator: two tests iterate it, and
    re-reading all ~120 scripts/sweep/*.sh costs ~135 ms per pass over files
    that are static for the duration of the run.
    """
    return tuple(
        (Path(path).name, i, line)
        for path in WRAPPERS
        for i, line in enumerate(Path(path).read_text().split("\n"), 1)
        if not line.strip().startswith("#")
    )


# ── the OPTIM_FAMILIES link, previously a warning only ──────────────────────


def test_no_optim_colors_entry_is_outside_every_family():
    """`plotting.colors._validate_family_membership` warns about this at import.
    A warning is invisible in a notebook that has already imported the module
    once, and there is no CI here, so assert it too. The failure it prevents:
    a family-filtered plot silently omits the new optimizer.
    """
    from lora_playground.plotting import OPTIM_COLORS

    orphans = sorted(set(OPTIM_COLORS) - _all_family_members())
    assert not orphans, (
        f"{len(orphans)} optimizer(s) have an OPTIM_COLORS entry but belong to no "
        f"OPTIM_FAMILIES set, so every family-filtered plot drops them:\n"
        + "\n".join(f"  {o}" for o in orphans)
        + "\nFix: add each to a family in lora_playground/plotting/colors.py."
    )


def test_every_family_member_has_a_colour():
    """The reverse link: a family naming an optimizer with no colour entry makes
    the notebook's `c["optimizer"] in OPTIM_COLORS` filter drop it."""
    from lora_playground.plotting import OPTIM_COLORS

    missing = sorted(_all_family_members() - set(OPTIM_COLORS))
    assert not missing, (
        f"OPTIM_FAMILIES names optimizer(s) with no OPTIM_COLORS entry: {missing}"
    )


def test_families_only_name_registered_optimizers():
    """A family entry for a retired optimizer reads as coverage that does not
    exist, and a typo there is invisible — nothing dereferences it."""
    from lora_playground.optim import OPTIMIZER_CHOICES

    unknown = sorted(_all_family_members() - set(OPTIMIZER_CHOICES))
    assert not unknown, (
        f"OPTIM_FAMILIES names optimizer(s) absent from OPTIMIZER_CHOICES "
        f"(retired or misspelled): {unknown}"
    )


# ── the wrapper -> train.py flag contract ───────────────────────────────────


def test_every_wrapper_flag_is_accepted_by_train_py():
    """Renaming a train.py flag leaves ~120 wrappers referencing the old name.

    Currently zero violations across 83 distinct flags used by the wrappers, so
    this is the regression guard for the next rename. A wrapper flag train.py
    does not accept either aborts the run at argparse or, worse, is silently
    absorbed by a wrapper that never forwards it.
    """
    accepted = set(_train_flags())
    # BooleanOptionalAction also accepts the --no- form of each boolean.
    accepted |= {f"no-{f}" for f in _train_flags()}
    accepted |= {f"no_{f}" for f in _train_flags()}
    bad: list[str] = []
    for name, i, line in _wrapper_lines():
        for flag in _FLAG_IN_SH.findall(line):
            if flag not in accepted:
                bad.append(f"  {name}:{i}  --{flag}")
    assert not bad, (
        f"{len(bad)} wrapper reference(s) to a flag train.py does not accept:\n"
        + "\n".join(sorted(set(bad))[:20])
        + "\nFix: rename in the wrapper, or add the flag to train.py's parser."
    )


def test_default_true_booleans_are_never_appended_conditionally():
    """`--x` appended only in the true branch of a conditional is a silent
    inversion when train.py's default is True: the false branch leaves the
    feature ON. Render both directions instead (`--x` / `--no-x`).

    Currently zero violations — every one of the ~47 sites passing
    log_basic_diagnostics, cw_nesterov or abort_on_nan_eval does so
    unconditionally, which is redundant but correct. This guards the next one.
    """
    risky = _default_true_booleans()
    assert risky, "expected train.py to have default-True booleans; parser changed?"
    cond = re.compile(r"\bif\b|&&|\|\||\bcase\b|\[\[|\[ ")
    bad: list[str] = []
    for name, i, line in _wrapper_lines():
        if not cond.search(line):
            continue
        for flag in risky:
            # the positive form, not the --no- form
            if re.search(rf"--{re.escape(flag)}\b(?!-)", line):
                bad.append(f"  {name}:{i}  --{flag} appended under a condition")
    assert not bad, (
        f"{len(bad)} conditional append(s) of a default-True boolean, where the "
        f"false branch silently leaves the feature ENABLED:\n" + "\n".join(bad)
        + "\nFix: render both directions — `--x` when true, `--no-x` when false."
    )


@pytest.mark.parametrize("flag", ["log_basic_diagnostics", "cw_nesterov"])
def test_known_default_true_booleans_are_still_default_true(flag):
    """Pins the premise of the test above. If a default flips to False, appending
    the positive form conditionally becomes correct and the guard would be
    enforcing a rule that no longer applies.
    """
    assert flag in _default_true_booleans(), (
        f"--{flag} is no longer a default-True boolean in train.py; revisit "
        f"test_default_true_booleans_are_never_appended_conditionally."
    )


# ── the manifest schema, written in two places ──────────────────────────────


def test_submit_sh_writes_every_manifest_field():
    """`slurm_scripts/submit.sh` builds the manifest dict in inline python, and
    `train.py`'s stub writer builds it via `manifest.build_manifest`. The field
    set therefore lives in two places, and a field added to `MANIFEST_FIELDS`
    without touching submit.sh would make every real sweep's manifest silently
    lack it while the stub writer has it.

    Asserted rather than refactored: submit.sh runs before the conda env is
    necessarily active, and rewiring the production submission path to import
    lora_playground is a bigger risk than a drift guard. (Verified importable
    under system python3 with no torch, so the refactor is possible later.)
    """
    from lora_playground.manifest import MANIFEST_FIELDS

    submit = (ROOT / "slurm_scripts" / "submit.sh").read_text()
    # The manifest literal, bounded so an unrelated dict elsewhere cannot match.
    start = submit.index("manifest = {")
    body = submit[start:submit.index("out = Path(", start)]
    written = set(re.findall(r'^\s*"([a-z_]+)":', body, re.M))
    missing = sorted(set(MANIFEST_FIELDS) - written)
    assert not missing, (
        f"slurm_scripts/submit.sh does not write manifest field(s) {missing}; "
        f"every real sweep's meta.json would lack them while train.py's stub has "
        f"them. Add them to the dict in submit.sh's manifest block."
    )
    extra = sorted(written - set(MANIFEST_FIELDS))
    assert not extra, (
        f"slurm_scripts/submit.sh writes field(s) {extra} that "
        f"lora_playground.manifest.MANIFEST_FIELDS does not declare, so nothing "
        f"reading manifests knows about them. Add to MANIFEST_FIELDS or drop them."
    )
