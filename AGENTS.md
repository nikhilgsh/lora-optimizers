# Repository Guidelines

## Project Structure & Module Organization

This is a lean LoRA optimizer playground in the style of `modded-nanogpt`. The main CLI is `train_lora.py`, backed by `lora_playground/train.py`. Optimizers live in `lora_playground/optim.py`; LoRA tensor utilities live in `lora_playground/utils.py`. Use `tests/` for focused unit tests, `docs/` for protocols and notes, and `notebooks/` only for exploration. Treat `wandb/`, caches, and checkpoints as generated artifacts.

The default course adapts a general base model, `allenai/OLMo-2-0425-1B`, to code using `ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response`. Do not default to code-specialized base models; the point is to compare LoRA optimizers on adaptation.

Read `docs/experimental_protocol.md` and `docs/model_dataset_selection.md` before changing model, data, metrics, or smoke-test settings.

## Build, Test, and Development Commands

Run unit tests with:

```bash
python -m pytest tests/test_lora_utils.py -q
```

Inspect the training CLI with:

```bash
python train_lora.py --help
```

The package supports editable installs via `setup.py`; the default local environment is `ffcv-pl`. Use `WANDB_MODE=offline` for W&B-enabled runs. Do not install or mutate Python/conda environments from automation; report missing dependencies.

SLURM controller commands may fail inside the sandbox with `slurm_load_jobs error: Unable to contact slurm controller`. For SLURM/A100 profiling or allocation work, rerun Slurm control commands with escalated permissions rather than treating this as an account permission issue.

## Coding Style & Naming Conventions

Use Python with 4-space indentation, descriptive snake_case functions, and PascalCase classes for optimizers. New optimizers should expose their update rule clearly in `lora_playground/optim.py` and be registered in `OPTIMIZER_CHOICES`. Prefer small helpers only when they remove real duplication.

## Testing Guidelines

Use `pytest`. Add tests next to related coverage in `tests/`, with names like `test_<behavior>.py` and `test_<specific_case>`. Optimizer math tests should check shapes, dtype/device behavior, numerical residuals, and deterministic behavior on tiny tensors before any GPU run. Run only relevant tests, not a broad suite, unless a change touches shared training infrastructure.

## Experiment Rules

The goal is LoRA optimizer comparison, not best code-model fine-tuning. Keep model, dataset, LoRA rank, train/eval split, sequence length, seed, dtype, compile mode, and evaluation cadence fixed across runs. Select settings using held-out validation loss, never training loss. Report validation loss, tokens/sec, wall time, peak memory, and exact command line. Functional smokes should run on GPU, not CPU.

## Commit & Pull Request Guidelines

This repo has just been initialized, so use concise imperative commits such as `Add LoRA optimizer training loop` or `Register AdamLinLoRA option`. PRs should state the optimizer question, list changed files, include smoke commands, and attach before/after metrics when behavior changes.
