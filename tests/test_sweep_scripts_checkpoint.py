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


def _sweep_scripts() -> list[Path]:
    return sorted(SWEEP_DIR.glob("*.sh"))


def test_every_sweep_script_wires_checkpointing() -> None:
    scripts = _sweep_scripts()
    if not scripts:
        pytest.skip("no sweep scripts under scripts/sweep/")
    offending = []
    for p in scripts:
        text = p.read_text()
        missing = [tok for tok in REQUIRED if tok not in text]
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
