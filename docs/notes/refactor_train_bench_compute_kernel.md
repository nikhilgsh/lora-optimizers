# Tech debt — extract the per-step compute kernel from train.py + bench

**Status: deferred follow-up, not for mid-campaign work.**

## The problem

`lora_playground/train.py` and `scripts/bench/bench_optimizer_step.py`
both implement the same logical thing — model+optimizer construction
plus the per-step `for grad_accum: fwd; bwd; opt.step()` block — but as
two independent copies. Every feature that touches model, optimizer, or
forward (packed_v1's 4D mask + position_ids; MFU; DDP wrap order;
compile timing; gradient checkpointing; new training modes) currently
has to land twice and stay in sync.

Concrete recent examples:
- packed_v1 wiring: had to add the same `position_ids` + 4D
  `attention_mask` plumbing in both train.py's main loop and bench's
  `make_batch`.
- MFU 4N/6N fix: had to update both train.py and bench to pass
  `flops_per_token_per_param` correctly.
- DDP support in bench (2026-05-08): had to manually mirror train.py's
  `bare_model = model; DDP-wrap; pass bare_model to build_optimizer`
  pattern, including the rationale comment about `collect_lora_pairs`
  needing the un-prefixed module names.

## Proposed shape

```python
# lora_playground/training_kernel.py (NEW)

@dataclass
class TrainingComponents:
    bare_model: nn.Module          # peft + compile, no DDP
    train_model: nn.Module         # bare or DDP-wrapped
    optimizer: torch.optim.Optimizer
    train_collator: Callable | None
    eval_collator:  Callable | None
    n_total_params: int
    flops_per_token_per_param: float
    # ...

def build_training_components(args, device, world_size, local_rank) -> TrainingComponents:
    """Single source of truth for: model load, attn_implementation
    fallback, peft wrap (or SVD/UCV unfreeze), DDP wrap, compile,
    optimizer construction, collator selection. Used by both train.py
    and any bench that wants to time a real production-shape graph."""

@dataclass
class StepStats:
    fwd_sec: float
    bwd_sec: float
    opt_sec: float
    zero_sec: float
    loss_value: float
    n_signal_tokens: int

def run_one_step(components, batch, *, time_phases: bool = False) -> StepStats:
    """The exact fwd+bwd+opt.step+zero loop. `time_phases=True` adds
    cuda.synchronize() between phases for benchmarking; default False
    for production."""
```

Then:
- **train.py::main** keeps dataset loading, epoch / sampler iteration,
  eval cadence, JSONL logging, wandb, profiler, scheduler — and calls
  `run_one_step(components, batch)` per micro-step.
- **bench_optimizer_step.py** keeps synthetic-batch generation, warmup
  loop, timing aggregation, table printing — and calls
  `run_one_step(components, batch, time_phases=True)`.

Both share the same compute path, so packed_v1, MFU, DDP, and any
future feature that touches the model/optimizer/forward only land
once.

## Why it hasn't happened

The refactor is invasive. `train.py::main` has accumulated many
conditional branches (training_mode, optimizer_mode, profiler, wandb,
DDP, compile order, optimizer-diagnostics gating, scheduler choice,
gradient-checkpointing, multi-epoch guard, samples-consumed math) that
are subtle to preserve. Doing it mid-campaign risks a hard-to-find
behavioral change that confounds optimizer comparisons.

**Right time:** between campaigns, when no run is gating on
correctness. Cover with the existing test suite plus a fresh "old vs
new" loss-trajectory equivalence test on a tiny model.

## Lighter intermediate step

If a full extraction is too risky, the highest-value 80% is just:
1. `build_training_components(args, ...)` — the peft + DDP + compile +
   optimizer construction block.
2. `run_one_step(components, batch)` — the inner for-grad_accum loop.

Dataset loading, eval cadence, logging, wandb, profiler all stay in
train.py-only. This already eliminates the parity bugs that cost the
most: every recent gotcha (packed_v1, MFU, DDP) lives in those two
extracted pieces.

## Cost of NOT doing this

Every new feature pays a duplication tax. Today that tax is small (a
few extra Edit calls per feature); over a campaign of 5–10 features
it accumulates into "did we update the bench? did we update train.py?
do they actually compute the same numbers?" — exactly the kind of
silent drift that's hard to debug.

Flag it in any sweep retrospective so it stays visible until done.
