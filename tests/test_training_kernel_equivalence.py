"""Loss-trajectory equivalence test for the train.py → training_kernel
refactor.

Baseline JSONL files in tests/fixtures/kernel_equivalence/ were captured
from the PRE-refactor train.py at commit acf2853 with seed=1234. After
the refactor, re-running the same command must produce bit-identical
per-step train_loss values — any drift means the refactor changed the
order of operations and confounds future optimizer comparisons.

Marked GPU-only (skipped under CPU). Each command takes ~1 minute on
an A6000. Skipped automatically by `pytest -q` when no GPU is present;
run explicitly via:

    pytest tests/test_training_kernel_equivalence.py -q
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "kernel_equivalence"

BASE_CMD = [
    sys.executable, "-u", "train_lora.py",
    "--device", "cuda",
    "--model_name", "allenai/OLMo-2-0425-1B",
    "--train_file", "tests/fixtures/tiny_code_train.jsonl",
    "--eval_file", "tests/fixtures/tiny_code_eval.jsonl",
    "--training_mode", "lora",
    "--max_steps", "3", "--eval_every", "3", "--train_loss_every", "1",
    "--batch_size", "1", "--grad_accum_steps", "1",
    "--max_seq_length", "128", "--lora_r", "4", "--lora_alpha", "4",
    "--bf16", "--seed", "1234",
    "--data_pipeline_version", "unpacked_v0",
]


def _read_train_step_losses(path: Path) -> list[float]:
    losses = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            rec = json.loads(line)
            if rec.get("event") == "train_step":
                losses.append(rec["train_loss"])
    return losses


def _run_and_collect(optimizer: str, tmp_path: Path) -> list[float]:
    out = tmp_path / f"out_{optimizer}.jsonl"
    cmd = BASE_CMD + ["--optimizer", optimizer]
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    with out.open("w") as f:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, stdout=f,
            stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode != 0:
        pytest.fail(f"train_lora.py failed (rc={result.returncode}); see {out}")
    return _read_train_step_losses(out)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="GPU equivalence test")
@pytest.mark.parametrize("optimizer,fixture", [
    ("adamw", "baseline_lora_adamw.jsonl"),
    ("adam-lin-lora", "baseline_lora_adamlinlora.jsonl"),
])
def test_loss_trajectory_matches_pre_refactor_baseline(
    optimizer, fixture, tmp_path,
):
    fixture_path = FIXTURE_DIR / fixture
    if not fixture_path.exists():
        pytest.skip(f"missing baseline fixture {fixture_path}")
    baseline = _read_train_step_losses(fixture_path)
    assert len(baseline) == 3, f"expected 3 baseline steps, got {len(baseline)}"

    actual = _run_and_collect(optimizer, tmp_path)
    assert len(actual) == 3, f"expected 3 post-refactor steps, got {len(actual)}"

    # bf16 is bit-deterministic on the same hardware with set_seed; the
    # refactor preserves operation ordering so we expect EXACT equality.
    # Allow a tiny rtol just to absorb any harmless float repr round-trip
    # in JSON serialization.
    for i, (b, a) in enumerate(zip(baseline, actual)):
        assert abs(a - b) <= max(1e-12, 1e-7 * abs(b)), (
            f"step {i+1} drift: baseline={b!r} actual={a!r} "
            f"diff={a - b!r}"
        )
