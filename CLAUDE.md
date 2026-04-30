# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

This is a lean LoRA optimizer comparison playground in the style of `modded-nanogpt`. The goal is **optimizer comparison**, not best-model production: hold everything else fixed and measure how LoRA optimizer choices affect held-out loss, throughput, and memory when adapting a general LLM to code.

Default course: base model `allenai/OLMo-2-0425-1B`, dataset `ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response`, entry point `train_lora.py`. Use general base models only; do not default to code-specialized bases.

Read `docs/experimental_protocol.md` and `docs/model_dataset_selection.md` before changing model, data, metrics, or smoke-test settings.

## Commands

```bash
# Run unit tests (CPU-only, no GPU needed)
python -m pytest tests/test_lora_utils.py -q
python -m pytest tests/test_svd_oracle.py -q
python -m pytest tests/test_train_helpers.py -q

# Inspect CLI
python train_lora.py --help

# Functional GPU smoke test (use local fixtures to skip dataset download)
python train_lora.py \
  --device cuda \
  --model_name allenai/OLMo-2-0425-1B \
  --train_file tests/fixtures/tiny_code_train.jsonl \
  --eval_file tests/fixtures/tiny_code_eval.jsonl \
  --optimizer adam-lin-lora \
  --max_steps 1 --eval_every 1 \
  --batch_size 1 --grad_accum_steps 1 \
  --max_seq_length 128 --lora_r 4 --lora_alpha 4 --bf16
```

Default local conda environment: `ffcv-pl`. Always set `WANDB_MODE=offline` for W&B runs. Do not install or mutate environments — report missing dependencies.

## Architecture

```
train_lora.py           → thin entry point, calls lora_playground.train.main()
lora_playground/
  train.py              → full training loop, CLI arg parsing, dataset/tokenizer setup
  optim.py              → all optimizer implementations + build_optimizer() factory
  utils.py              → LoRA tensor utilities and SVD helpers
tests/
  fixtures/             → tiny JSONL files for GPU smoke tests
docs/
  experimental_protocol.md
  low_rank_peft_convention.md
  plans/full_finetune_svd_low_rank_oracle.md
```

### Training Modes and Optimizers

`train_lora.py` supports three `--training_mode` values:

- **`lora`** (default): wraps model with PEFT `LoraConfig`; LoRA-aware optimizers operate on `(A, B)` pairs; SVD optimizers are disallowed.
- **`svd_step_oracle`**: unfreezes dense target weights; each Adam step is projected to rank r via truncated SVD before application. Uses `SVDStepAdamW`.
- **`svd_cumulative_oracle`**: same, but accumulated displacement from initialization is projected to rank r, not each step individually. Uses `SVDCumulativeAdamW`.

`OPTIMIZER_CHOICES` in `optim.py` is the registry. `build_optimizer()` is the sole factory; add new optimizers there and register in `OPTIMIZER_CHOICES`.

### LoRA Factor Convention

PEFT convention throughout — A: (r, d_in), B: (d_out, r), adapter output = `(alpha/r) * B @ A`. All optimizer math uses this orientation. See `docs/low_rank_peft_convention.md`.

Custom optimizers collect pairs via `collect_lora_pairs()` in `utils.py` and operate directly on `A.grad`/`B.grad` without going through PyTorch's standard parameter-group mechanics. They store per-pair state in `self.pair_state` (a plain dict) rather than `self.state` to avoid conflicts with `Optimizer.state`.

### Key Math Utilities (`utils.py`)

- `spdify(M, eps)` — symmetrizes and adds δI, outputs float32
- `solve_spd(A, B)` — Cholesky solve AX=B
- `solve_sylvester(SB, SA, RHS)` — solves K·SA + SB·K = RHS via eigendecomposition; used by LinLoRA and AdamLinLoRA
- `truncated_svd(matrix, rank)` — Frobenius-optimal rank-r approximation
- `collect_dense_target_weights(model, target_modules)` — collects `TargetWeight` dataclass instances for SVD oracle modes; `all-linear` excludes `lm_head`

### Logging

Every training run emits JSON lines to stdout via `log_event()`: one `config` event at startup (includes full command line and git commit), and one `eval` event per evaluation step with `eval_loss`, `tokens_per_sec`, `peak_memory_mb`, etc.

## Coding Conventions

- PascalCase for optimizer classes, snake_case for functions
- New optimizers: add class to `optim.py`, add entry to `OPTIMIZER_CHOICES`, add branch in `build_optimizer()`
- Optimizer math operates in float32 (cast inputs, cast updates back to param dtype/device before applying)
- Tests: shapes, dtype/device behavior, numerical residuals, determinism on tiny tensors; CPU-only for unit tests; GPU required for functional smokes

## Experiment Rules

Fix across all optimizer comparisons: model name, dataset + split seed, sample counts, LoRA rank/alpha/dropout, target modules, sequence length, batch size, grad accumulation, dtype, compile mode, eval cadence. Use held-out eval loss for hyperparameter selection (never training loss). Hardware comparison baseline is A100; local RTX A6000 is acceptable only for functional smokes, not for timing or optimizer comparisons.

**Canonical comparison horizon: 2000 steps.** All optimizer-vs-optimizer eval-loss comparisons in this project run to `--max_steps 2000` with `--eval_every 200`. Mechanism probes (cosine trajectories, conditioning, etc.) run to the same 2k horizon — short pilots (500-step etc.) are reserved for η-ranking-selection only, never for measurement (per the global "match canonical horizon" rule). The `lr_sweep_2k`, `optim_compare_high_eta_2k`, and `h*_*_2k` log groups all use this horizon; baseline numbers (AdamW 0.7579 at η=3e-4, adam-lin-lora 0.7564 at η=1e-3, etc.) are at step 2000.
