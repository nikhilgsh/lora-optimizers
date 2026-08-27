# AGENTS.md

This file provides guidance to agentic coding tools (Codex, Claude Code, etc.) when working with this repository.

## Project Overview

This is a lean LoRA optimizer comparison playground in the style of `modded-nanogpt`. The goal is **optimizer comparison**, not best-model production: hold everything else fixed and measure how LoRA optimizer choices affect held-out loss, throughput, and memory when adapting a general LLM to code.

Default course: base model `allenai/OLMo-2-0425-1B`, dataset `ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response`, entry point `train_lora.py`. Use general base models only; do not default to code-specialized bases.

Read `docs/experimental_protocol.md` and `docs/model_dataset_selection.md` before changing model, data, metrics, or smoke-test settings.

## PoLoRA public-release boundary

`~/polora` is the public release tree, while this repository is private
research infrastructure. The `tests/`, `_faithfulness/`, and `_investigation/`
directories in `~/polora` are internal-only and must remain excluded from the
public release. Do not track, publish, or recommend publishing tests from those
directories. Cross-repository equivalence checks belong only in these excluded
internal directories.

Tracked files in `~/polora` must not contain paths, imports, names, comments,
or documentation that reveal or depend on this repository, including
`~/lora`, absolute paths to it, and the `lora_playground` package. Before
calling `~/polora` release-ready, scan its tracked files for these references.

## Do not propose more seeds

Seed variance on this project is small, measured, and does not change
conclusions. The clean multi-seed cell -- `kl-shampoo-polar-lora`,
Llama-3.2-1B openmath r256, lr=1e-2, four seeds -- has a final-eval-loss spread
of 0.0009 and sd 0.0004, about a quarter of the 0.0017 AdamW noise floor the
sigma-unit deltas are quoted in. An effect worth reporting here is 1-10 sigma;
seed noise is 0.24 sigma.

So: do not end an analysis by offering to run more seeds, do not caveat a
result as "single seed" as though that were a defect, and do not size a sweep
with a seed axis unless the user asks for one. A comparison at one seed is the
project's normal evidentiary standard, not a provisional version of a real one.

What to do instead when a result looks marginal, in order:
1. Read it at MATCHED STEP across the whole trajectory rather than at the final
   eval alone. A gap that holds its sign over every eval of a run is stronger
   evidence than one final number, and costs nothing -- both curves are already
   on disk.
2. Say the effect is smaller than the quantities being compared, and name the
   size. "-0.5 sigma, sign stable across 12 consecutive evals" is a finding;
   "single seed, would need more" is not.
3. If the cell genuinely cannot settle it, propose a DIFFERENT CELL -- another
   rank, another architecture -- which tests whether the conclusion generalizes.
   That buys information a repeat seed does not.

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

Paper prose must not use internal ablation shorthand such as "bare partner-Gram",
"partner-Gram polar", or "partner-Gram-only whitening". State the actual controls
instead: identity metric (`P=Q=I`) vs learned diagonal metric, and whether the
spectral-norm magnitude rescale is present. For the both-controls-removed arm,
write "the decoupled update with identity metric and no magnitude rescale" or
refer directly to `\Cref{prop:decoupled}` when the proposition is in scope.

When reasoning about LoRA factor-step scaling, do not reduce the objective to
the product output alone. The model sees `(B + dB)(A + dA)`, but optimization
happens in the factor coordinates: conditioning, row/column subspaces,
factor norms, preconditioned directions, and reparameterization geometry can
change training even when an output-feature balance metric looks good. Any
recommendation for `c_A`/`c_B` must state which target it optimizes (factor
update geometry, product-output balance, stability, or held-out loss) and must
not treat one diagnostic as decisive unless it has been connected to training
loss or a clearly stated optimizer mechanism.

When reviewing PoLoRA derivations, audit estimator targets at every
approximation boundary. If a sentence says a stored recurrence, observable
moment, or fitted factor estimates/reproduces/tracks a named quantity, verify
that the target has not changed across per-sample vs batch-collapsed moments,
single-batch vs historical averages, current-weight vs stored-gradient
quantities, dense moments vs projected factor-gradient moments, or exact
moments vs surrogate model factors. Matching notation is not enough; cite the
formula that defines the target actually being estimated.
In PoLoRA estimator prose, do not use "closure" or "closures" as shorthand for
the matrices `C_A=B^T P B`, `C_B=A Q A^T`, fitted response matrices, or
per-step scoring matrices. Name the concrete object instead: `C_A`, `C_B`,
"scoring matrix", "block matrix", or "current `C_A(p)`" as appropriate.
Do not justify PoLoRA estimator choices with ungrounded counterfactuals such
as "if the stored `P_s` were scaled by `a_s`." State the scale dependence of
the actual recurrence or fitted objective directly, and tie it to a concrete
stored quantity, equation, or implementation step.
When a user challenges whether a PoLoRA estimator argument is meaningful, do
not keep defending the framing by restating motivation. Identify the exact
derived bound or objective that supports the claimed usefulness. If the text
only proves an identity, minimizer formula, or residual decomposition without
showing the residual is small in a natural regime, say that it is not an
algorithmic justification and downgrade or remove the claim. Repeated user
pushback on the same claim means stop arguing for it and revise the target or
state the blocker.

When writing theorem-style PoLoRA notes, do not bury the load-bearing
smallness/stability requirements as nested "if" clauses inside the theorem
body. Name them as assumptions before the theorem, state whether they are
primitive, derived, or bootstrap conditions, and make the theorem conclusion a
direct consequence with a reader-visible size interpretation.
Do not replace an opaque derived forcing term by another unexplained tolerance
or arbitrary radius fraction. Use only assumptions already approved by the
user or assumptions stated directly on named observed quantities such as the
gradients, factors, EMA moments, and weighted drift terms already in the note.
If a condition is about a fitted map, response floor, contraction, fixed point,
or online trajectory, label it as a nonprimitive regularity assumption and ask
before using it.
Do not repair a failed PoLoRA tracking proof by adding response-floor,
coordinate-excitation, fixed-point well-definedness, contraction, small-gain,
or online-state closeness assumptions unless the user explicitly approves that
assumption first. In particular, do not assume lower bounds such as every
coordinate of a fitted vector being bounded below, and do not assume
`P_s,Q_s` or normalized EMA states are already close to the target being
proved. If the result is not derivable from the current assumptions, stop and
say exactly which implication is missing instead of patching in a stronger
theorem.
Initial online EMA values should not be framed as a substantive tracking
assumption in PoLoRA arguments. They enter only through the exact EMA tail from
unrolling the recurrence; if the theorem is stated after burn-in, either absorb
that decaying tail explicitly or omit discussion of initial iterates.
For PoLoRA tracking arguments, do not introduce wrapper residual notation like
`U_u`; write the explicit EMA-tail and rescoring terms, such as the
`beta_2` tail and `D_A,D_B`, or their expanded primitive bound. Minimize named
constants in theorem statements: reuse existing constants and inline simple
combinations unless a constant is defined once and used repeatedly in a proof.

When the user narrows a PoLoRA estimator discussion to a chosen target such as
the EMA log-det coordinate fit, keep that target fixed. Do not keep returning
to a model-consistency or on-model-collapse caveat unless it changes the
answer to the narrowed question. If a theorem only justifies a surrogate such
as exact diagonal-Kronecker factors, state that it is secondary and answer the
chosen-estimator question first.

When asked why equal factor radii are chosen, do not merely restate the product
operator-norm bound. State the extra design prior explicitly: equal radii are a
no-preference / isotropic-factor-space allocation after whitening and polar
normalization, not a theorem forced by the product map. If arguing for any
non-equal split, name the additional sensitivity model or measurement that
justifies preferring one factor coordinate over the other.

Do not call a LoRA factor-scaling rule "best", "coherent", or "principled"
unless the optimized objective is stated first. Separate algebraic facts
from design axioms: whitening/polar identities are derivations; choosing an
unweighted or weighted factor trust region is a regularizer choice that needs
its own stated premise.

When proposing static dimension-ratio shape factors for LoRA, define the
ratio convention before naming exponents. Prefer `R_in = r / d_in` and
`R_out = d_out / r` unless the user chooses otherwise. Write the rule as
`c_A = R_in^a`, `c_B = R_out^b` and settle both `a` and `b`; `c_B = 1`
means `b = 0`, not that the B-side question was answered implicitly.
If the stated principle is MuA-style output feature learning, the
load-bearing diagnostic is the direct branch decomposition
`delta1 = B dA x`, `delta2 = dB A x`; isolated A-rowspace or B-expansion
diagnostics are supporting evidence only and must not be promoted to the
optimizer recommendation when they would worsen the measured `delta2/delta1`
balance.

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
- **Never use a full SVD / `eigh` to get a scalar `sigma_max` (or `lambda_max`).**
  `torch.linalg.matrix_norm(X, ord=2)` is a *full SVD* — it was ~80% of the
  curvature-whiten-polar step (224 SVDs/step, ~30 ms each; killing it gave 4.4×).
  Use the library power-iter — `spectral.sigma_max_power_iter` /
  `sigma_max_power_iter_batched` / `lambda_max_power_iter_psd_batched` (matvec-based,
  warm-startable via a cached `v_init`) — and **do not hand-roll** one (library-first;
  grep `lora_playground/spectral.py` before writing any spectral estimator). Reserve
  SVD/`eigh` for when you genuinely need the full spectrum (e.g. the one-time eigenbasis
  seed in a periodic refresh), never for a single top singular/eigen value.
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

**Benchmark the PRODUCTION config, not library defaults.** Any profiling / timing / speed comparison MUST use the exact flags the optimizer's production sweep wrapper passes (grep `scripts/sweep/` for the optimizer — e.g. `--precond_method higham`, `--ns_steps`, `--picard_iters`), NOT `build_optimizer`/argparse defaults. `build_optimizer` defaults `precond_method="eigh"` — a slow per-pair `_spd_inv_half` eigh fallback — while EVERY production chord-tight/polar sweep passes `higham` (a batched, eigh-free refresh). Benchmarking the default silently measures a code path production never runs (it produced a bogus "chord-tight does a 1.1 s per-pair-eigh refresh / curvature-whiten is cheaper" comparison). Either pass the production flags explicitly, or drive the bench through the actual sweep wrapper. When in doubt, print the resolved config (`precond_method`, `ns_steps`, dtype, compile) at the top of every bench. And any *cadence* claim ("Higham refreshes every step", "amortized 1-in-K") must cite the **consuming gate's** `file:line` (e.g. the `if (step-1) % refresh_every == 0:` at `optim.py:5885`), not just the knob's default value or sweep flag — the default tells you the value, only the gate tells you what that value does.

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
