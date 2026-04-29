# Full-Finetune SVD Low-Rank Performance Oracle Plan

## Motivation

LoRA constrains the trainable weight displacement for a targeted linear layer to
the PEFT factor convention documented in `docs/low_rank_peft_convention.md`:

```text
A: (r, d_in)
B: (d_out, r)
adapter update = scale * B @ A
```

This gives a rank-at-most-`r` adapter state, but the learned LoRA displacement
is not generally the best rank-`r` approximation to the displacement learned by
full fine-tuning. If a dense reference run learns `Delta W_full`, the
Frobenius-optimal rank-`r` approximation is the truncated SVD
`Pi_r(Delta W_full)`. The scientific question is performance-first: at the same
rank, how much validation loss gap is there between trained LoRA and evaluating
the SVD-projected full-finetune displacement?

In this note, "FFT" means full fine-tuning/full-rank fine-tuning, not a Fourier
transform.

## Core Performance Questions

1. **LoRA vs SVD-projected FFT.** Train LoRA normally and compare held-out
   validation loss to a model whose rank-`r` adapter is the truncated SVD of a
   full-finetune displacement.
2. **Rank ceiling.** Compare the SVD-projected FFT model against the unprojected
   dense reference. This estimates the loss due only to the rank constraint.
3. **Optimization/factorization gap.** Compare LoRA against the SVD-projected
   FFT model. This estimates the loss due to LoRA's parameterization and
   optimizer path, beyond the rank constraint.

## Important Distinctions

- The primary result is validation performance, not one-step update similarity.
- The first oracle is post-hoc: train a dense target-module reference, project
  its final displacement to rank `r`, then evaluate that projected model.
- This is an oracle baseline because it uses the dense reference endpoint. It is
  not yet an online optimizer that could be run without first doing the dense
  reference training.
- The LoRA adapter state `scale * B @ A` has rank at most `r`, so the matched
  SVD oracle should also evaluate a rank-`r` displacement.

## Primary Experiment

Use a target-matched dense reference first. This keeps the comparison about
LoRA's low-rank parameterization, not about changing which parts of the model
are allowed to move.

1. **Base model.**
   Evaluate the frozen pretrained model.
2. **LoRA baseline.**
   Train LoRA rank `r` on the chosen target modules using the existing
   `train_lora.py` path.
3. **Target-module full finetune.**
   Starting from the same base model, train the dense weights of exactly the
   modules that LoRA targets, while freezing the rest of the model. Save the
   initial weights `W0` and final dense weights `W_T`.
4. **Post-hoc SVD oracle.**
   For every targeted module, compute:

   ```text
   Delta W_full = W_T - W0
   Delta W_svd_r = Pi_r(Delta W_full)
   W_oracle = W0 + Delta W_svd_r
   ```

   Evaluate the resulting model on the held-out validation set.

The main table should be:

```text
model                         eval_loss   train_loss   wall_time   peak_memory
base frozen                   ...
LoRA rank r                   ...
target-module FFT             ...
SVD(FFT displacement), rank r  ...
```

The headline comparison is `eval_loss(LoRA rank r)` versus
`eval_loss(SVD(FFT displacement), rank r)`.

## Experimental Objects

For every targeted PEFT linear module or target-matched dense module, collect:

- module name;
- base dense weight `W0`;
- final dense reference weight `W_T`;
- LoRA factors `A`, `B`, when evaluating a LoRA checkpoint;
- PEFT adapter scale, usually `lora_alpha / r`;
- the effective LoRA displacement `Delta W_lora = scale * B @ A`;
- the SVD-projected displacement `Delta W_svd_r = Pi_r(W_T - W0)`.

Implementation should extend `collect_lora_pairs` or add a neighboring helper
that returns metadata `(name, module, A, B, scale, base_weight)`. Do not compare
raw `B @ A` against a dense update without the PEFT scale.

## Mapping SVD Back Into LoRA Factors

For evaluation, either inject `W0 + Delta W_svd_r` directly into a dense copy of
the target module, or convert the SVD projection into PEFT factors. Direct dense
injection is simpler for a first implementation because it avoids PEFT adapter
initialization details.

If converting to PEFT factors, for
`Delta W_svd_r = U_r diag(S_r) V_r.T` and PEFT scale `scale`, one valid mapping
is:

```text
A = V_r.T
B = U_r diag(S_r) / scale
scale * B @ A = Delta W_svd_r
```

The factor scaling is not unique; evaluation should be invariant as long as the
effective product equals `Delta W_svd_r`.

## Primary Metrics

The performance metrics are primary:

- held-out validation loss;
- training loss for overfit diagnostics, not model selection;
- wall time;
- tokens/sec;
- peak memory;
- exact command line and git commit.

Matrix metrics are secondary sanity checks. They should help verify that the
oracle was constructed correctly, but they are not the main result.

Per targeted layer, optionally log:

- `full_norm = ||Delta W_full||_F`
- `lora_norm = ||Delta W_lora||_F`
- `svd_r_energy = ||Pi_r(Delta W_full)||_F^2 / ||Delta W_full||_F^2`
- `lora_residual = ||Delta W_lora - Delta W_full||_F / ||Delta W_full||_F`
- `svd_r_residual = ||Pi_r(Delta W_full) - Delta W_full||_F / ||Delta W_full||_F`
- `excess_residual_sq = lora_residual^2 - svd_r_residual^2`
- cosine similarity between `Delta W_lora` and `Delta W_full`
- best scalar-aligned residual
  `min_c ||c * Delta W_lora - Delta W_full||_F / ||Delta W_full||_F`

Aggregate metrics should be weighted by `||Delta W_full||_F^2` so small layers
do not dominate the report.

## Two Online SVD Optimizer Variants

There are two distinct online optimizers worth testing. They should be named
separately in code and logs because they answer different questions.

### 1. Per-Step SVD Update Optimizer

This optimizer computes the dense optimizer proposal for the current step, then
applies only the rank-`r` SVD projection of that step:

```text
W_tilde = dense_optimizer_step(W_t)
Delta W_t = W_tilde - W_t
W_{t+1} = W_t + Pi_r(Delta W_t)
```

This may be useful as a cheap or structured optimizer, but it is not a fair
fixed-rank LoRA competitor. The cumulative displacement is:

```text
W_T - W0 = sum_t Pi_r(Delta W_t)
```

and that sum is not necessarily low-rank.

Suggested code name: `svd_step_oracle`.

### 2. Cumulative Fixed-Rank SVD Optimizer

This optimizer keeps a dense accumulator of proposed full updates, then exposes
only the rank-`r` projection of that cumulative update to the model:

```text
C_0 = 0
W_t = W0 + B_t A_t
W_tilde = dense_optimizer_step(W_t)
Delta W_t = W_tilde - W_t
C_{t+1} = C_t + Delta W_t
B_{t+1} A_{t+1} = Pi_r(C_{t+1})
W_{t+1} = W0 + B_{t+1} A_{t+1}
```

Equivalently, the live adapter should satisfy:

```text
B_t A_t approximately equals Pi_r(sum_{i < t} Delta W_i)
```

With exact SVD it is equality up to numerical precision. This is the fairer
LoRA comparison because the live model displacement from `W0` remains rank at
most `r`. It is still an oracle/diagnostic optimizer because it keeps dense
optimizer state and a dense cumulative update `C_t`.

Suggested code name: `svd_cumulative_oracle`.

For the first implementation, train by directly writing projected dense target
weights. Converting each projection to explicit LoRA `B, A` factors is useful
for export and sanity checks, but it is not required for functional evaluation.

### Model Construction

Add a training mode rather than hiding this inside the existing LoRA optimizer
choices:

```text
--training_mode {lora,svd_step_oracle,svd_cumulative_oracle}
--svd_rank R
```

For `training_mode=lora`, keep the current PEFT path.

For either SVD oracle mode:

1. Load the base `AutoModelForCausalLM` without PEFT.
2. Freeze all parameters.
3. Collect dense target modules using the same `--target_modules` argument.
4. Set only those target module weights to `requires_grad=True`.
5. Store a float32 copy of each initial target weight `W0`.
6. Use an SVD-projected dense optimizer over those target weights.

The first version can support `torch.nn.Linear` target modules only. If we need
GPT-2-style `Conv1D` targets, add that after the Linear path is tested.

### Target Module Matching

Implement target collection in a new helper, likely in `lora_playground/oracle.py`:

```text
collect_dense_target_weights(model, target_modules) -> list[TargetWeight]
```

`TargetWeight` should include:

- module name;
- module object;
- weight parameter;
- frozen base copy `W0`;
- shape;
- dtype/device.

For explicit target names such as `q_proj,k_proj,v_proj`, match PEFT-style
suffixes: a module named `model.layers.0.self_attn.q_proj` matches `q_proj`.

For `all-linear`, include all `torch.nn.Linear` modules except obvious output
heads such as `lm_head` for the first implementation. The exact selection must
be logged because target-set mismatches can dominate the result.

### Optimizer Class

Add two optimizers in `lora_playground/oracle.py` or `lora_playground/optim.py`.
Both can subclass `torch.optim.AdamW` so schedulers keep working.

Per-step variant:

```text
class SVDStepAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        before = {target: target.weight.detach().float().clone()}
        loss = super().step(closure)
        for target in self.targets:
            raw_delta = target.weight.float() - before[target]
            step_delta = truncated_svd(raw_delta, self.rank)
            target.weight.copy_(before[target] + step_delta)
        return loss
```

Cumulative fixed-rank variant:

```text
class SVDCumulativeAdamW(torch.optim.AdamW):
    def __init__(self, targets, rank, ...):
        super().__init__([target.weight for target in targets], ...)
        self.targets = targets
        self.rank = rank
        self.accumulators = {
            target.name: torch.zeros_like(target.weight, dtype=torch.float32)
            for target in targets
        }

    @torch.no_grad()
    def step(self, closure=None):
        before = {target.name: target.weight.detach().float().clone()}
        loss = super().step(closure)
        for target in self.targets:
            raw_delta = target.weight.float() - before[target.name]
            self.accumulators[target.name].add_(raw_delta)
            projected = truncated_svd(self.accumulators[target.name], self.rank)
            target.weight.copy_(target.base_weight + projected)
        return loss
```

Initial support should enforce or strongly warn on `weight_decay != 0.0`.
Decaying the full dense weight is not equivalent to decaying the adapter
displacement, and the current LoRA runs default to zero weight decay.

Add SGD variants only if we want geometry-first comparisons. For the main
performance run, AdamW is the useful first target because it matches the default
LoRA baseline.

### SVD Helper

Use exact SVD first:

```text
def truncated_svd(matrix, rank):
    U, S, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    rank = min(rank, S.numel())
    return (U[:, :rank] * S[:rank]) @ Vh[:rank]
```

This is expensive, but it is the oracle baseline. If it is too slow, add a
separate approximate mode later and report it as approximate.

### Training Loop Integration

Refactor model setup in `train_lora.py` into small helpers:

```text
build_lora_model(args, dtype)
build_svd_oracle_model(args, dtype)
build_training_optimizer(model_or_targets, args)
```

Keep the existing dataloading, gradient accumulation, clipping, scheduler,
evaluation, profiling, and JSON logging paths shared.

The config log should add:

- `training_mode`;
- `svd_rank`;
- `target_module_count`;
- sorted target module names;
- `svd_projection="exact"`;
- for SVD modes, `rank_constraint` equal to either `"per_step_update"` or
  `"cumulative_displacement"`;
- whether output heads were excluded from `all-linear`.

### First Tests

Add `tests/test_svd_oracle.py` with CPU-only tests:

1. `truncated_svd` reconstructs a known rank-`r` matrix.
2. `SVDStepAdamW` leaves each applied step rank at most `r`.
3. `SVDCumulativeAdamW` leaves each displacement from `W0` rank at most `r`.
4. Cumulative projection is relative to `W0`, not relative to zero.
5. After two independent rank-1 steps, `svd_step_oracle` is allowed to have a
   cumulative displacement rank greater than `r`; this distinguishes the two
   modes.
6. `collect_dense_target_weights` matches explicit suffix target names.
7. SVD oracle setup freezes non-target parameters and unfreezes target
   weights only.

### First Smoke

After the unit tests pass, run a one-step GPU smoke with fixture data. This
validates model loading, dense target selection, SVD projection, eval, and JSON
logging without downloading the dataset during the smoke:

```text
WANDB_MODE=offline python train_lora.py \
  --training_mode svd_cumulative_oracle \
  --device cuda \
  --model_name allenai/OLMo-2-0425-1B \
  --train_file tests/fixtures/tiny_code_train.jsonl \
  --eval_file tests/fixtures/tiny_code_eval.jsonl \
  --target_modules all-linear \
  --svd_rank 4 \
  --max_steps 1 \
  --eval_every 1 \
  --max_train_samples 8 \
  --max_eval_samples 4 \
  --max_seq_length 128 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --num_workers 0 \
  --bf16
```

Before running this smoke, write the required failure-mode checklist for the
exact command being tested.

## Validation Protocol

- Use held-out validation loss for any optimizer or rank selection.
- Keep model, dataset, LoRA rank, target modules, sequence length, seed, dtype,
  compile mode, batch size, gradient accumulation, evaluation schedule, and data
  order fixed across comparisons.
- Record exact command line, git commit, all CLI defaults, dataset paths, model
  name, seeds, and target modules in the JSON logs.
- Report validation loss together with tokens/sec, wall time, peak memory, and
  optional matrix sanity metrics when they are logged.

## Implementation Sketch

Add a focused module, likely `lora_playground/oracle.py`, containing:

- dense target module collection and freeze/unfreeze helpers;
- `effective_lora_delta(A, B, scale)`;
- `truncated_svd(matrix, rank)`;
- `SVDStepAdamW`;
- `SVDCumulativeAdamW`;
- helpers to load a dense reference checkpoint and build
  `W0 + Pi_r(W_T - W0)`;
- residual/cosine/energy metric helpers;
- optional SVD-to-LoRA-factor conversion for export/sanity checks.

Then add CLI flags to `train_lora.py`:

```text
--training_mode {lora,svd_step_oracle,svd_cumulative_oracle,target_dense,svd_project_eval}
--svd_rank R
--dense_reference_checkpoint PATH
--base_reference_checkpoint PATH
--save_initial_target_weights PATH
--save_final_target_weights PATH
```

The first online implementation should cover `svd_step_oracle` and
`svd_cumulative_oracle`. The post-hoc `svd_project_eval` mode can come after
those are tested.

## Tests

Focused CPU tests should cover:

- PEFT-style scale is included in `effective_lora_delta`;
- truncated SVD gives the known best rank-`r` reconstruction on a tiny matrix;
- residual metrics are zero when the candidate equals the full update;
- SVD-to-factor conversion reconstructs `Delta W_svd_r` after PEFT scaling;
- metadata collection preserves the existing `A: (r, d_in)`,
  `B: (d_out, r)` convention;
- SVD oracle modes only unfreeze the modules selected by `target_modules`.

Do not start with a functional training run. First validate the linear algebra
on tiny tensors with CPU unit tests, then run the one-step GPU smoke above.

## Open Risks

- Target-module dense training uses more memory than LoRA. The first smoke
  should use a tiny model or a narrow target module list.
- Exact SVD every step can dominate runtime. This is acceptable for the oracle
  baseline; approximate SVD should be a separate named mode if needed.
- `svd_step_oracle` is not a fixed-rank LoRA competitor because cumulative
  displacement rank can grow over time.
- `svd_cumulative_oracle` is fixed-rank in model state, but keeps a dense
  accumulator and dense optimizer moments, so it is not memory-fair to LoRA.
- The dense reference optimizer must be matched carefully to the LoRA baseline
  where possible, but the headline result is validation loss, not one-step
  update similarity.
- PEFT implementation details around scaling and merged adapters must be read
  directly before coding the helper. The code comments for the helper should
  cite this plan, `docs/low_rank_peft_convention.md`, and the canonical local
  collection function.
