# AGENTS.md

This file provides guidance to agentic coding tools (Codex, Claude Code, etc.) when working with this repository.

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

Terminology discipline: use **gauge** only for the exact LoRA reparameterization invariance / product-map kernel, e.g. transformations that leave `B @ A` unchanged or first-order factor updates in `ker(d(B @ A))`. Do not use "gauge" as a loose synonym for low-support, low-singular-value, weakly conditioned, or hard-to-interpret factor directions; name the measured quantity instead.

Custom optimizers collect pairs via `collect_lora_pairs()` in `utils.py` and operate directly on `A.grad`/`B.grad` without going through PyTorch's standard parameter-group mechanics. They store per-pair state in `self.pair_state` (a plain dict) rather than `self.state` to avoid conflicts with `Optimizer.state`.

### Key Math Utilities (`utils.py`)

- `spdify(M, eps)` — symmetrizes and adds δI, outputs float32
- `solve_spd(A, B)` — Cholesky solve AX=B
- `solve_sylvester(SB, SA, RHS)` — solves K·SA + SB·K = RHS via eigendecomposition; used by LinLoRA and AdamLinLoRA
- `truncated_svd(matrix, rank)` — Frobenius-optimal rank-r approximation
- `collect_dense_target_weights(model, target_modules)` — collects `TargetWeight` dataclass instances for SVD oracle modes; `all-linear` excludes `lm_head`

### Logging

Every training run emits JSON lines to stdout via `log_event()`: one `config` event at startup (includes full command line and git commit), and one `eval` event per evaluation step with `eval_loss`, `tokens_per_sec`, `peak_memory_mb`, etc.

## Cluster / sbatch conventions

These are the project-specific facts that global skills (`slurm-submit`, `disbatch`, `run-env`) refer back to here.

- **Conda env**: `ffcv-pl` (activate with `source ~/miniforge3/etc/profile.d/conda.sh && conda activate ffcv-pl`). Do not install or mutate; report missing deps.
- **Wall-tier ladder** (`slurm_scripts/`):
  - `sbatch_4h.sh` — A100 4h
  - `sbatch_blackwell.sh` — Blackwell 8h (small Blackwell jobs only)
  - `sbatch_12h.sh` — A100 12h
  - `sbatch_24h.sh` — A100 24h
  - `sbatch_24h_h100.sh` — H100 24h
  - For Blackwell sweeps needing >8h, write a custom pending sbatch with a longer `--time` and the Blackwell directives copied from `sbatch_blackwell.sh`.
- **Default to `sbatch_24h.sh` (or 24h-equivalent) for**: any sweep at r ≥ 128, any sweep at ≥6000 training steps, any sweep using a new/un-measured optimizer family at production scale, any sweep where prior runs on similar workloads finished in >70% of wall.
- **`rocky9` reservation (Flatiron cluster).** Gates **multiple GPU classes**, not just Blackwell. Every sbatch hitting one of these MUST include `#SBATCH --reservation=rocky9` or the job sits PD with `ReqNodeNotAvail,_UnavailableNodes:workergpuXXX` indefinitely (node state shows `IDLE+RESERVED` or `MIXED+RESERVED`):
  - **Blackwell RTX PRO 6000** (`-p gpu --constraint=rtxblackwell`, workergpu[171-193])
  - **gpuxl H200** (`-p gpuxl --constraint=h200`, workergpu[301-324])
  - **gpuxl H100** (`-p gpuxl --constraint=h100`, workergpu[201-226]) — some nodes
  - **Always check** `scontrol show reservation rocky9` for the live node list before assuming a (partition, constraint) is reservation-free.

  If a job is already PD without the reservation: `scontrol update jobid=<id> reservation=rocky9` — typically starts within seconds. The `-p gpu --constraint=a100` / `-p gpu --constraint=h100_pcie` paths do NOT need this reservation.
- **Per-task `--cpus-per-task`**: 8 for the standard sbatch templates here.

## Coding Conventions

- PascalCase for optimizer classes, snake_case for functions
- New optimizers: add class to `optim.py`, add entry to `OPTIMIZER_CHOICES`, add branch in `build_optimizer()`, register in `OPTIM_COLORS` and at least one `OPTIM_FAMILIES` set in `lora_playground/plot_utils.py` (the orphan-warning fires at notebook startup if you forget the family).
- Optimizer math operates in float32 (cast inputs, cast updates back to param dtype/device before applying)
- Tests: shapes, dtype/device behavior, numerical residuals, determinism on tiny tensors; CPU-only for unit tests; GPU required for functional smokes
- **Spectral-norm rescaling is load-bearing.** Any optimizer that divides by an
  estimated `sigma_max` must defend against underestimated denominators.
  Single-vector power iteration can miss the top singular direction when the
  start vector is cold, stale, zero, or nearly in the current Gram nullspace;
  that silently overscales updates and can explode parameters while all tensor
  shapes look correct. Prefer a guarded batched/block estimator with
  deterministic starts and lower-bound floors. When changing a `sigma_max`
  estimator or call site, add a known-positive regression for a bad start vector
  and run a high-rank GPU smoke that checks eval loss, `param_l2`, and nonfinite
  gradient counts.
- **Notebook analysis cells: check `lora_playground/plotting/` first.** Before writing a custom aggregation+plot function for a new comparison cell, grep the plotting package for a primitive that already does it: `compare_variants_figure` (label→extra_where dict, final-vs-lr + best-lr trajectory + summary table with Δσ), `standard_sweep_figure`, `sweep_figure_with_auto_ylim`, `distinct_palette`, `filter_baseline`, `filter_variants`. New comparisons are usually one call into the library plus a small variants dict — not a 100+-line cell that duplicates loader/aggregation/plotting bookkeeping. Add to the library before forking that pattern across multiple cells.

## Experiment Rules

Fix across all optimizer comparisons: model name, dataset + split seed, sample counts, LoRA rank/alpha/dropout, target modules, sequence length, batch size, grad accumulation, dtype, compile mode, eval cadence. Use held-out eval loss for hyperparameter selection (never training loss). Hardware comparison baseline is A100; local RTX A6000 is acceptable only for functional smokes, not for timing or optimizer comparisons.

**LR schedule: constant, no warmup, no cosine.** All sweeps in this project use `train.py` defaults `--lr_scheduler_type constant --warmup_steps 0`. The LR shown in the `config` event and per-eval `lr` field is the LR used at every step — `max_steps` does NOT alter the per-step LR. Loss trajectories at the same `(lr, init, optimizer, seed, lora_r, data_pipeline_version, git_commit)` and same step index are directly comparable across runs regardless of each run's `max_steps`. **Do NOT invoke "different cosine schedule" or "warmup phase" to explain pilot-vs-full-horizon trajectory discrepancies — neither exists here.** If two runs at the same step and same params differ in loss, the cause is elsewhere (code change at the git_commit boundary, sampler/dataloader seeding, diagnostics that mutate state, off-by-one in optimizer init) and must be tracked down rather than rationalized via an imagined schedule.

**Canonical comparison horizon — depends on data pipeline version:**

- **`unpacked_v0` (legacy, pre-2026-05-08): 2000 steps.** All historical `lr_sweep_2k`, `optim_compare_high_eta_2k`, and `h*_*_2k` log groups use this horizon. Baseline numbers (AdamW 0.7579 at η=3e-4, adam-lin-lora 0.7564 at η=1e-3, etc.) are at step 2000 under this version.
- **`packed_v1` (current): 4000 steps** with `--eval_every 200`. All chord-tight / polar-product family runs from 2026-05-08 onward use this horizon (`chord_tight_diag_4k_r16r64_blackwell`, `chord_direction_4k_r16r64_blackwell`, `frob_k{1,3}_4k_r16_blackwell`, `adamw_lr_sweep_packed_4k_r16_blackwell`, etc.). Packing roughly halves per-step token density vs the legacy unpacked path, so the step count was doubled to keep total tokens-seen comparable.

Mechanism probes (cosine trajectories, conditioning, etc.) run to the same horizon as the regime. Short pilots (500-step etc.) are reserved for η-ranking-selection only, never for measurement (per the global "match canonical horizon" rule). Do NOT muscle-memory "2k" when working in the `packed_v1` regime — use 4k.

**Data-pipeline boundary (2026-05-08): `unpacked_v0` → `packed_v1`.** New runs default to `packed_v1` (sequence-packed train side, pad-to-max eval, prompt-masked loss). Numbers across versions are NOT comparable — prompt-mask alone changes the loss objective, and packing changes per-step token density. Filter analyses with `load_runs(where={"data_pipeline_version": ...})`. AdamW noise-floor re-anchor under `packed_v1` is mandatory before any new optimizer Δ claim transfers; until then, optimizer-vs-optimizer comparisons must stay within a single version. See `docs/notes/polar_product/data_pipeline_followups.md` and `lora_playground/data.py`.

**Timing benches: `--optim_diagnostics_every 1` is a trap.** `_emit_basic_diagnostics` does a per-pair `B @ dA` matmul on the OUTER `(d_out, d_in)` shape PLUS ~10 `float(tensor)` syncs per pair. At r=256 all-linear (112 pairs) called every step, this adds ~5+ s/step of pure instrumentation — verified to be the entire "15× chord-tight vs AdamW" wall delta. Production sbatches default to `--optim_diagnostics_every 80`; bench cells measuring per-step wall, tok/s, or fraction-of-step MUST use ≥20 (preferably `--no-log_basic_diagnostics`). If a bench needs per-step c-trajectories or similar, run a SECOND non-timing cell with `--optim_diagnostics_every 1` for the trajectory and the timing cell with diagnostics off — do not conflate the two.

## Sweep manifests — pointer to a mechanically-enforced contract

This section is a **navigation aid**, not the contract itself. The contract is enforced mechanically by three choke points; this doc just tells you where to look when one of them refuses:

1. **`slurm_scripts/submit.sh` refuses to submit without `SWEEP_SCOPE`** — exits non-zero with a list of known scopes. You cannot submit an untagged sweep.
2. **`lora_playground.manifest.load_manifests(strict=True)` raises `UntaggedSweepError`** if any populated log dir lacks a manifest or has empty scope. `strict=False` opts out for ad-hoc exploration.
3. **`tests/test_manifests.py`** walks `logs/` and fails CI on missing/corrupt/empty-scope manifests.

The data flow: `submit.sh` writes `logs/<group>/run_info/meta.json` at submission. Analysis tooling consumes manifests via `lora_playground.loader.load_runs(where=...)` (predicate-based) — never raw directory listings or hand-maintained tuples of group names.

### Loader (current path) — `load_runs(where=…)`

New code uses `lora_playground.loader.load_runs(where=…)`. The `where` dict is one predicate per cfg field; literals match equality, lists/sets/tuples match membership, callables match by predicate. Scope strings are metadata only — they do NOT drive loading. Example:

```python
from lora_playground.loader import load_runs
runs = load_runs(where={"optimizer": ["adam-polar-product-lora", "adamw"], "lora_r": 64})
```

Companion: `inventory_runs(logs_root)` returns a structured audit (orphaned groups, optimizers in logs but missing from `OPTIM_COLORS`, lr-pinning per cell). The audit cell at the top of `sweep_analysis.ipynb` calls it and prints the report — single source of truth for "what could be silently wrong."

To **exclude an old sweep** from analysis, delete its log dir. Newest-wins-on-collision (in `merge_runs`) handles "rerun supersedes old" automatically when the new sweep covers the same configs.

### Updating docs/notes from sweep data — mandatory provenance

Before writing any numerical claim to `docs/notes/*.md` (final losses, Δ vs baseline, "best η", "pinned/not pinned", leaderboard rows), the source MUST be one of:

1. Re-executing the relevant cell in `notebooks/sweep_analysis.ipynb` or `notebooks/lin_scaled_investigation.ipynb` and reading the actual output.
2. A direct call to `lora_playground.loader.load_runs(where=…)` from a fresh script.

**NEVER** a hand-typed list of group names. Hand-typed lists drift from the manifest and silently miss data, producing phantom "pinned at boundary" / "missing data" claims. If you find yourself writing `groups = ['foo_2k', 'bar_2k', ...]`, stop — call the loader.

All numbers in `docs/notes/*.md` are single-seed at the canonical 2k-step horizon unless explicitly multi-seed. Do NOT use vague significance qualifiers — no "within jitter", no "≈ noise", no arbitrary thresholds (0.5%/1%/etc).

**Use multi-seed AdamW as the workload's σ.** `logs/adamw_multiseed/` (params: `adamw_multiseed.json`, seeds 1-4 at η=3e-4) is the project's noise floor at the canonical 2k-step horizon: **r=16 std ≈ 0.0006 (2σ ≈ 0.0012), r=64 std ≈ 0.0007 (2σ ≈ 0.0014)**. When characterizing optimizer Δ values, compute Δ / σ_AdamW and state σ-units explicitly ("X is 2.3σ above Adam-polar"). Reserve "within noise" for |Δ| < 1σ. Do NOT claim "near-equivalent", "matches", or "saturated" without checking the σ-units. Multi-seed verification of variant optimizers themselves is still deferred — but AdamW's σ is the right anchor for characterizing single-seed variant Δ values.

Workflow for `docs/notes/*.md` data-derived edits: pull data via canonical loader → propose concrete diff in chat → user confirms → edit. Never write multi-paragraph data-derived sections in a single unsupervised pass.

**Submitting a sweep:**

```bash
SWEEP_SCOPE="ext_compare,polar_family" \
SWEEP_PURPOSE="E2: AdaMuon-faithful + polar-product geometry" \
./slurm_scripts/submit.sh params/<sweep>.json <group> <n_gpus> [scripts/sweep/sweep_2k_r_diag.sh] [slurm_scripts/sbatch.sh]
```

To exclude an old sweep from analysis, delete its log dir.

**Known scope tags** (one or more, comma-separated):

| scope                      | when to use                                       |
|----------------------------|---------------------------------------------------|
| `ext_compare`              | extension-family optimizer comparison (post-, matrix-, polar-product variants) |
| `muon_family`              | Muon / AdaMuon / ProductMuon variants             |
| `all_optimizers`           | comprehensive optimizer comparison at **fixed r=16** (the reference overlay scope; do NOT tag rank-extension runs with this) |
| `r_extension`              | rank-extension sweeps (r ≠ 16, typically r ∈ {64, 128, 256, …}); kept distinct from `all_optimizers` so the r=16 reference set stays clean |
| `loraplus_family`          | AdamW + LoRA+ B-multiplier sweeps                 |
| `svd_oracle`               | SVD step / cumulative oracle modes                |
| `diagnostics`              | runs whose primary purpose is per-step probes (cos, σ(S), conditioning) |
| `lin_scaled_investigation` | H1–H5 lin/scaled-lora investigation               |
| `polar_family`             | spectral-product polar updates                    |
| `winner_rerun`             | re-run of a known-best config (typically with diagnostics enabled) |
| `pilot`                    | short-step (~500) ranking-selection runs (analysis ignores) |
| `legacy`                   | older sweeps kept for reference; usually excluded |

Schema in `lora_playground/manifest.py`; loader in `lora_playground/loader.py`. **Production sweeps should go through `submit.sh` (or the `disbatch` skill) so manifests and scope metadata are preserved.** Direct `sbatch <file>` is acceptable for ad-hoc/debug jobs when the sbatch itself records the needed configuration and provenance.
