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

## Fixed Comparison Rules

Hold these fixed across optimizer comparisons unless the experiment explicitly studies one of them:

- model name and revision
- dataset, split seed, train/eval sample counts
- LoRA rank, alpha, dropout, and target modules
- max sequence length, per-device batch size, gradient accumulation
- dtype, compile mode, gradient checkpointing, and device type
- learning-rate schedule and evaluation cadence

Select hyperparameters using held-out eval loss, never training loss. Report the exact command line for every run.

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

Use A100 runs for profiling and runtime estimates because A100 is the intended comparison hardware. The local RTX A6000 is acceptable for functional GPU smokes, model-download checks, and quick debugging, but A6000 timings should not be used to estimate production runtime or compare optimizers.

Before every GPU command, check live GPU utilization and memory with `nvidia-smi`. If the local GPU is busy, do not use it. For A100 profiling, use the interactive GPU workflow and record the GPU type in the JSON config or run notes.

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
