# DDP refactor scope for `train.py`

Scoped 2026-05-07 against the current `lora_playground/train.py` and
custom optimizers in `lora_playground/optim.py`. Purpose: enable
single-node 4-GPU and 8-GPU `torch.distributed.DistributedDataParallel`
to shrink per-cell wall-time for Phase B/C sweeps (see
`tight_chord_paper_plan.md` Hardware Notes — 4-GPU DDP gives ~3.5×
linear speedup with ~zero loss in throughput per GPU-hour).

## Verdict

**~12–20 person-hours** total (1–2 days) for MVP including testing.
~6–8h for Phase 1 minimal viable DDP. Most of the work is mechanical;
one substantive design call below on optimizer state.

## Phase 1 (MVP, ~6–8h)

In `lora_playground/train.py`:

1. **DistributedSampler + rank-aware DataLoader** (`train.py:416–432`) — replace
   the train DataLoader's shuffling with `DistributedSampler(train_dataset,
   num_replicas=world_size, rank=rank, shuffle=True)`. Eval can stay
   single-rank or replicate across ranks. `num_workers` should be capped
   per process (e.g., `max(1, num_workers // world_size)`).
2. **DDP wrap after PEFT** (`train.py:461–462`) — wrap the PEFT-wrapped model in
   `torch.nn.parallel.DistributedDataParallel(model)` AFTER `.to(device)`
   and BEFORE optimizer construction. PEFT submodules (LoRA `A`, `B`)
   are standard `nn.Parameter`s that DDP handles cleanly.
3. **Rank-0 logging** (`train.py:159–160`) — gate every `log_event(...)` call to
   `rank == 0`. One-line wrapper.
4. **Eval on rank-0 only** (`train.py:710–740`) — eval dataset is small (~512
   samples). Run on rank 0; broadcast loss to other ranks if early-stop
   needs it. Diagnostic probes also rank-0.
5. **Launcher**: SLURM script needs `torchrun --nproc_per_node=N` and SLURM
   env-var passthrough (`SLURM_PROCID`, `SLURM_NTASKS_PER_NODE` →
   `RANK`, `WORLD_SIZE`).

## Custom optimizer state — important nuance

The DDP-scoping agent flagged a risk that custom optimizer state in
`pair_state` (e.g. `m_A`, `v_A`, `m_B`, `v_B` for Adam-family;
preconditioner Gram matrices for tight-chord) would diverge across
ranks unless explicitly all-reduced. **This concern is overstated for
Adam-family optimizers like ours.**

Standard DDP wraps model parameters and inserts gradient hooks that
all-reduce `param.grad` during `loss.backward()`. After backward, every
rank sees the SAME averaged gradient on `A.grad` and `B.grad`. Our
optimizer's EMA update reads `A.grad`, so:

```python
m_A = β * m_A + (1 - β) * A.grad   # ← A.grad is identical across ranks
```

If `m_A` starts identical (initialized to zeros, deterministic seed)
and the gradient is identical (DDP all-reduce), the update is
identical, and `m_A` STAYS identical across ranks step-after-step.
**No explicit buffer all-reduce needed for Adam-family momentum.**

Same logic for tight-chord's preconditioner Gram matrices: they're
computed deterministically from the (synchronized) factor weights, so
they match across ranks by construction.

The **actually-divergent state** is:
- Diagnostic probes computed from local (per-rank) activations or
  gradient-derived intermediate quantities BEFORE the all-reduce. Per the existing
  `_emit_optim_diagnostics` path, these are computed from the post-step
  state (including all-reduced grads), so they should match too — but
  worth a sanity check by comparing rank-0 vs rank-1 diag values on a
  smoke run.
- `_sigma_max_power_iter` warm-start vectors stored in `pair_state` —
  if initialized randomly per process, they diverge. Use deterministic
  init.

So the optimizer-state work reduces to: one validation pass to confirm
buffers stay synchronized in practice (1–2h), and deterministic init
for any random warm-starts (trivial). NOT the 4–6h refactor the scoping
report initially suggested.

## Phase 2 (polish, ~2–4h)

- Update `scripts/sweep/sweep.sh` (and the `_diag` variants) to
  conditionally invoke `torchrun` based on a `--world_size` env or arg.
- Add a multi-GPU smoke test: 4-GPU 50-step run on the tiny test
  fixtures, asserting loss matches single-GPU within float32 noise.

## Critical files
- `lora_playground/train.py:416–432, 461–462, 710–740, 159–160`
- `lora_playground/optim.py` — verification only (no expected change)
- `scripts/sweep/sweep_*_diag.sh` — launcher updates
- `slurm_scripts/sbatch.sh` — torchrun integration
- New tests: `tests/test_ddp_smoke.py` (subprocess-spawn DDP smoke)

## Speedup expectation

Per the original A0 cheap-wins analysis: 4-GPU DDP gives ~3.5× linear
speedup on a 4-GPU node. Stacking with already-applied compile + higham:
- 8B r=256 tight-chord 6k steps: 31h → ~9h (fits 24h wall)
- 8B r=256 tight-chord 8.2k steps (270M tok): 43h → ~12h

8-GPU: ~6.5× scaling typical → 8B r=256 8.2k steps ≈ ~7h. Sweep-friendly.
