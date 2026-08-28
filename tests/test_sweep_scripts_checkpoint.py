"""CPU-only enforcement: every sweep wrapper under ``scripts/sweep/`` must wire
SLURM-timeout-resumable checkpointing.

Motivation: a full-polar r64 sweep timed out at step ~8880/9000 (98.7% done) and
was unrecoverable because its wrapper never passed ``--checkpoint_dir`` /
``--resume_from`` — so nothing was written to disk to resume from. The submit
path injects a per-task ``CHECKPOINT_DIR`` env var, but a wrapper that ignores it
silently drops checkpointing. This test makes that regression impossible: any new
wrapper that forgets the ``ckpt_args`` block fails CI.

The canonical block lives in any current wrapper (e.g.
``scripts/sweep/sweep_phase_L_1b_r64.sh``):

    ckpt_args=()
    if [ -n "${CHECKPOINT_DIR:-}" ]; then
        ckpt_args=(
            --checkpoint_dir "$CHECKPOINT_DIR"
            --resume_from "$CHECKPOINT_DIR"
            --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST:-2}"
        )
        ...
    fi

and ``"${ckpt_args[@]}"`` is spliced into the ``python train_lora.py`` invocation.

Fast (~0.1 s), no GPU, no model loads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SWEEP_DIR = ROOT / "scripts" / "sweep"

# Both flags must be present AND wired to the injected env var. Checking the
# literal ``"$CHECKPOINT_DIR"`` (not just ``--checkpoint_dir``) ensures the
# wrapper consumes the per-task dir the submit path injects, rather than
# hardcoding a path that would collide across concurrent tasks.
REQUIRED = (
    '--checkpoint_dir "$CHECKPOINT_DIR"',
    '--resume_from "$CHECKPOINT_DIR"',
    '"${ckpt_args[@]}"',
)


import re

# A wrapper that only dispatches: it sets env vars and hands off to another
# wrapper, which carries the canonical block. Following the handoff is what
# keeps the contract honest without exempting the file.
_EXEC_DELEGATE = re.compile(r"^\s*exec\s+scripts/sweep/(\S+\.sh)", re.M)


def _sweep_scripts() -> list[Path]:
    return sorted(SWEEP_DIR.glob("*.sh"))


def _resumable(path: Path, seen: frozenset[str] = frozenset()) -> list[str]:
    """What ``path`` is missing for a wall-timeout to be resumable, if anything.

    Three structures satisfy the contract, and requiring one spelling of it
    rejected two of them:

    - The canonical ``ckpt_args`` block, which is what almost every wrapper
      uses.
    - A DISPATCHER that sets env vars and ``exec``s another wrapper
      (``sweep_precond_r256_postfix_state.sh`` -> ``sweep_protagonist_precond.sh``).
      The checkpointing is wired one level down, so follow the handoff.
    - A wrapper that resumes from a FORK rather than from its own checkpoint
      dir (``sweep_factorwise_slot_freeze.sh``, whose whole purpose is to
      continue from a checkpoint another run wrote). It still has to pass the
      injected ``--checkpoint_dir "$CHECKPOINT_DIR"`` so a timeout has
      somewhere to resume FROM, but ``--resume_from`` is necessarily its own
      resolved fork path.
    """
    text = path.read_text()
    if not [tok for tok in REQUIRED if tok not in text]:
        return []
    delegate = _EXEC_DELEGATE.search(text)
    if delegate and delegate.group(1) not in seen:
        target = SWEEP_DIR / delegate.group(1)
        if target.is_file():
            return _resumable(target, seen | {delegate.group(1)})
        return [f'exec target {delegate.group(1)} does not exist']
    missing = []
    if '--checkpoint_dir "$CHECKPOINT_DIR"' not in text:
        missing.append('--checkpoint_dir "$CHECKPOINT_DIR"')
    if "--resume_from" not in text:
        missing.append("--resume_from <the checkpoint to continue>")
    return missing


def test_every_sweep_script_wires_checkpointing() -> None:
    scripts = _sweep_scripts()
    if not scripts:
        pytest.skip("no sweep scripts under scripts/sweep/")
    offending = []
    for p in scripts:
        missing = _resumable(p)
        if missing:
            offending.append((p.name, missing))
    assert not offending, (
        f"{len(offending)} sweep script(s) do not wire SLURM-timeout-resumable "
        "checkpointing:\n"
        + "\n".join(f"  {name}: missing {miss}" for name, miss in offending)
        + "\n\nFix: copy the canonical `ckpt_args` block from any current wrapper "
        "(e.g. scripts/sweep/sweep_phase_L_1b_r64.sh) in before the "
        "`python train_lora.py \\` line, and splice `\"${ckpt_args[@]}\"` into the "
        "invocation. This makes a wall-timeout resumable instead of a total loss."
    )
