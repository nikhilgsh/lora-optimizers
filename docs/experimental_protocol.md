# Experimental Protocol

## Objective

This repo compares LoRA optimizers on a fixed code-adaptation course. The goal is not to produce the best code model; it is to measure how optimizer choices affect held-out loss, speed, memory, and stability when adapting a general pretrained language model to code.

See `docs/model_dataset_selection.md` for the model and dataset rationale.

## Default Course

- Base model: `allenai/OLMo-2-0425-1B`
- Dataset: `ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response`
- Entry point: `train_lora.py`
- Default optimizer baseline: `adamw`

Use a general base model, not a code-specialized base. `OLMo-2-0425-1B` is a modern Apache-2.0 general LM with open training details, large enough to produce meaningful LoRA behavior but still small enough for rapid single-GPU iteration. The Magicoder instruction-response dataset provides code-adaptation examples with clean `instruction` and `response` fields. It has only a train split, so the training loop creates a deterministic held-out split when no explicit eval split is present.

## Standard Course Settings

| Parameter | Value | Rationale |
|---|---|---|
| `max_steps` | 2000 | 1 epoch on 32k samples (effective batch 16); enough for Adam+X variants to show characteristic behavior |
| `eval_every` | 200 | 10 eval points per run |
| `max_train_samples` | 32000 | 1-epoch invariant: 2000 × 16 = 32k |
| `max_eval_samples` | 512 | SE ≈ 0.003 nats; detection threshold ~0.01 nats |
| `max_seq_length` | 512 | Magicoder pairs are 300–1500 tokens |
| `batch_size` | 2 | per-device |
| `grad_accum_steps` | 8 | effective batch 16 |
| `lora_r / lora_alpha` | 16 / 16 | |
| `target_modules` | `all-linear` | excludes `lm_head` |
| `lr_scheduler_type` | `constant` | clean, no scheduler interactions |
| `data_dir` | `data/magicoder_seq512_32k` | pre-tokenized Arrow dataset |

Pre-tokenized cache: run `python scripts/data/prepare_data.py --out_dir data/magicoder_seq512_32k --max_train_samples 32000 --max_eval_samples 512` once before sweeping.

**1-epoch invariant:** `max_train_samples` must equal `max_steps × effective_batch_size`. Violating this causes multi-epoch training under constant LR, which diverges. Check before submitting any new sweep.

**Hardware:** Use the canonical hardware for the target campaign. For current polar-product and leaderboard experiments, Blackwell RTX PRO 6000 is canonical. Older H100/A100 runs can be useful for functional checks or loss-only comparisons when the campaign allows it, but do not use their timings as evidence for Blackwell leaderboard claims. Do not compare timing metrics across GPU types.

## Fixed Comparison Rules

Hold these fixed across optimizer comparisons unless the experiment explicitly studies one of them:

- model name and revision
- dataset, split seed, train/eval sample counts
- LoRA rank, alpha, dropout, and target modules
- max sequence length, per-device batch size, gradient accumulation
- dtype, compile mode, gradient checkpointing, and device type
- learning-rate schedule and evaluation cadence

Select hyperparameters using held-out eval loss, never training loss. Report the exact command line for every run.

**LR selection:** Always find each optimizer's best LR via a held-out sweep before comparing optimizers head-to-head. Using AdamW's optimal LR for all methods is not a valid comparison — effective step sizes differ across optimizers.

## Metrics

Primary quality metric:

- `eval_loss`: token-weighted held-out causal LM loss

Required supporting metrics:

- `tokens_per_sec`
- `train_elapsed_sec`
- `peak_memory_mb`
- `train_loss`
- final JSON config event, including command line and git commit when available

Do not compare throughput across different hardware, compile settings, dtype, sequence length, or batch size.

## Hardware Policy

Use the target campaign's canonical hardware for profiling and runtime estimates. For current polar-product and leaderboard experiments, use Blackwell RTX PRO 6000. The local RTX A6000 is acceptable for functional GPU smokes, model-download checks, and quick debugging, but A6000 timings should not be used to estimate production runtime or compare optimizers.

Before every GPU command, check live GPU utilization and memory with `nvidia-smi`. If the local GPU is busy, do not use it. For production profiling, use the interactive GPU or SLURM workflow for the campaign hardware and record the GPU type in the JSON config or run notes.

## Smoke Tests

Unit tests may run on CPU because they only check optimizer math utilities. Functional training smokes must run on GPU.

Use local JSONL fixtures for GPU smokes so dataset downloading is not part of the smoke:

```bash
python train_lora.py \
  --device cuda \
  --model_name allenai/OLMo-2-0425-1B \
  --train_file tests/fixtures/tiny_code_train.jsonl \
  --eval_file tests/fixtures/tiny_code_eval.jsonl \
  --optimizer adam-lin-lora \
  --max_steps 1 \
  --eval_every 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --max_seq_length 128 \
  --target_modules all-linear \
  --lora_r 4 \
  --lora_alpha 4 \
  --bf16
```

This validates model loading, LoRA injection, custom optimizer stepping, eval, JSON logging, and GPU memory. It is not a performance result.

## Profiling Runs

Use `--profile_steps N --profile_dir runs/profiles` for PyTorch profiler traces. Keep profiling runs short and run them on the same GPU type as the comparison run. Treat `--compile` as its own experimental condition; do not compare compiled and uncompiled throughput directly.

## Escalation Path

Start with one-step GPU smoke tests. Then run short course runs on the real dataset with 2-3 candidate learning rates. Only after loss decreases and memory is stable should longer optimizer comparisons be launched.
