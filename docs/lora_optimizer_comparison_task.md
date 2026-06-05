# LoRA optimizer comparison — task and settings

This document specifies the experiment so it can be reproduced independently. It
is self-contained: everything you need is the public model and dataset IDs, the
data-preparation recipe, and the fixed training configuration below.

## What the experiment is

Adapt a **general** (not code-specialized) ~1B pretrained base model to an
instruction-tuning corpus using **LoRA**, and compare optimizers. Everything
except the optimizer is held fixed; for each optimizer the learning rate is
swept and the best (by held-out loss) is selected. The quantity of interest is
held-out eval loss as a function of training steps.

A run is one `(model, dataset, rank, optimizer, learning rate, seed)` point.

## Models

| Display | HF model id |
|---------|-------------|
| OLMo-2-1B | `allenai/OLMo-2-0425-1B` |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` |

## Datasets

Three instruction corpora. Each is tokenized once with the **target model's own
tokenizer** (a cache built for one model is not reusable for another) and packed
(see recipe below).

| Display | HF dataset id | configs / fields | raw-row cap |
|---|---|---|---|
| opc-sft-stage2 (Magicoder) | `OpenCoder-LLM/opc-sft-stage2` | concat 4 configs: `educational_instruct,evol_instruct,mceval_instruct,package_instruct`; `{instruction, output}` | 1,000,000 |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | rename `problem → instruction`, `generated_solution → output` | 2,000,000 |
| Tulu-3 SFT mixture | `allenai/tulu-3-sft-mixture` | `messages` chat list, flattened with the Tulu chat template | 400,000 |

The matrix of runs is 3 models × 3 datasets × 2 ranks, minus the cells not run.
The cells that are run:

| Model | Dataset | ranks |
|-------|---------|-------|
| OLMo-2-1B | opc-sft-stage2 | 64, 256 |
| OLMo-2-1B | OpenMathInstruct-2 | 64, 256 |
| OLMo-2-1B | Tulu-3 | 64, 256 |
| Llama-3.2-1B | opc-sft-stage2 | 64, 256 |
| Llama-3.2-1B | OpenMathInstruct-2 | 64, 256 |
| Qwen2.5-1.5B | opc-sft-stage2 | 256 |

## Data-preparation recipe

The same pipeline for every (dataset, model), differing only in dataset id and
tokenizer:

1. **Load** the HF dataset. For opc, load all four configs and concatenate their
   train splits. For Tulu-3, flatten the `messages` chat list into a single
   `{instruction, output}` pair using the Tulu chat template (system/user/assistant
   turns joined, final assistant turn is the response; drop rows with no trailing
   assistant turn). For OpenMathInstruct-2, rename columns as in the table above.
2. **Carve eval split**: `train_test_split` with `test_size = 0.01`, `seed = 0`,
   shuffled. Cap eval at 1024 examples → ~1000 held-out examples per cell.
3. **Cap train** at the raw-row cap in the table (shuffle with `seed = 0` first).
4. **Tokenize** with the target model's tokenizer, tracking the prompt/response
   boundary. Prompt and response are rendered as:
   ```
   Instruction:
   {instruction}

   Response:
   {output}
   ```
   (Tulu data uses the chat template instead; the boundary is still tracked.)
5. **Pack** the train side greedily into fixed **2048-token slots**, dropping
   slots with zero supervised (response) tokens. Eval is left per-document and
   padded to 2048 at batch time.
6. **Prompt-masking**: the loss is computed on **response tokens only**; prompt
   tokens are masked out. (Eval loss is therefore not comparable to a
   full-sequence LM loss.)

Resulting packed-slot counts (≈ the real "number of training samples"):

| Cell | packed 2048-token slots | epochs in 9000 steps |
|---|---|---|
| OLMo × opc | 150,492 | 0.96 (≈1 pass; ~225M unique tokens) |
| OLMo × OpenMathInstruct-2 | 513,877 | 0.28 |
| OLMo × Tulu-3 | 143,558 | 1.00 (exhausts ≈ step 8970) |
| Llama × opc | 149,977 | 0.96 |
| Qwen × opc | 151,502 | 0.96 |

(Slot counts shift slightly across tokenizers; the OpenMath/Llama and Tulu/r256
caches follow the same rule.)

## Fixed training configuration

Identical across every run. These are the answers to "dataset, batch size, seq
len, num tokens, model":

| Setting | Value |
|---|---|
| **Sequence length** | **2048** |
| **Per-device batch size** | **4** |
| **Gradient accumulation** | **4** |
| **Effective (global) batch** | **16** (single GPU) |
| **Training steps (horizon)** | **9000** |
| **Tokens per step** | 32,768 (16 × 2048) |
| **Tokens processed per run** | ~295M (9000 × 32,768 = 294,912,000) |
| **Eval cadence** | every 250 steps |
| **Eval set** | ~1000 held-out examples (1% split), padded to 2048 |
| **Eval loss denominator** | response (supervised) tokens only — ~380k–430k tokens/cell |
| **LoRA rank r** | 64 or 256 |
| **LoRA alpha** | = r (so the `alpha/r` scale is 1.0) |
| **LoRA dropout** | 0.0 |
| **LoRA B init** | zeros (A is the standard PEFT init) |
| **LoRA target modules** | all linear layers **except** `lm_head` |
| **Precision** | bf16 (TF32 matmuls enabled) |
| **torch.compile** | on (default mode) |
| **Attention** | SDPA |
| **Gradient checkpointing** | off |
| **Max grad norm** | 1.0 |
| **LR schedule** | **constant, no warmup, no decay** |

Per-device batch 4 × accum 4 (not a single batch-16 forward, which OOMs at
seq 2048). Peak memory at this shape is ~26 GB with compile, so a single 80–96 GB
GPU is ample.

## Optimizer and learning-rate sweep

- **Baseline**: standard **AdamW** on the LoRA A/B parameters — betas
  `(0.9, 0.999)`, eps `1e-8`, weight decay `0.0`.
- **Learning rate is swept per optimizer** over the grid
  `{3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}` and the best is selected on
  **held-out eval loss** (never training loss). Using one optimizer's best LR for
  another is not a valid comparison — effective step sizes differ across optimizers.
- The best AdamW LR depends on the cell (e.g. ~`3e-4` for OLMo×opc×r64,
  ~`1e-4` for OLMo×opc×r256).

## How methods are compared

Primary metric is **held-out eval loss**: a token-weighted mean over the
**response (supervised) tokens** of the eval set (prompt and padding tokens are
masked). The eval set is ~1000 held-out documents and the loss is averaged over
roughly 380k–430k response tokens per cell. Measured supervised-token counts:

| Cell | eval docs | response tokens (loss denominator) |
|---|---|---|
| OLMo × opc | 1023 | 377,708 |
| OLMo × OpenMathInstruct-2 | 1023 | 394,924 |
| OLMo × Tulu-3 | 994 | 427,936 |
| Llama × opc | 1023 | 377,567 |
| Qwen × opc (r256) | 1022 | 381,253 |

(r64 and r256 share the same eval set within a (model, dataset) pair. The
Llama×OpenMath and Tulu-3×r256 caches follow the same recipe and land in the
same range.)

A convenient summary is *speed-to-baseline*:
the fraction of the 9000-step horizon a method needs to first reach the best
AdamW run's *final* eval loss in the same cell — `1.0` means it needs the whole
run, `0.5` means it gets there in half the steps, lower is better.

Runs are single-seed (seed 0). If you want an error bar on the baseline, the
AdamW spread at this horizon is small (on the order of ~0.002 nats); a multi-seed
σ at the 9000-step horizon has not been measured.

Also record per run, for completeness: tokens/sec, peak memory, wall time, and
the full hyperparameter set + git commit of the training code.

## Gotchas

- **Horizon is 9000 steps at seq 2048.** ~1 pass over opc and Tulu-3, ~0.28 pass
  over OpenMathInstruct-2.
- **Loss is response-only (prompt-masked)** — not comparable to full-sequence LM loss.
- **Tokenizer-specific caches** — rebuild the packed dataset per model.
- **Constant LR**, no warmup/cosine — the per-step LR equals the configured LR at
  every step, so trajectories at the same step index are directly comparable
  across runs of different lengths.
- **Sweep LR per optimizer**, select on held-out loss.
