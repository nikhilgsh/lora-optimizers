# E1 — Coverage Fill (tracking)

Living tracker for the paper's E1 coverage-fill (see `paper/PLAN.md`). Single source of
truth for what runs, its status, and provenance. **Coverage numbers are loader-derived**
(`load_runs`), never hand-typed — re-run the audit cell before editing the matrix.

## Configs (locked)

Paper baselines are **AdamW + iMuon only** (see `paper/PLAN.md`). No other baseline.

- **Protagonist** (`diag-shampoo-polar-lora`): `--polar_method polar_express --muon_ns_steps 8
  --cw_picard_iters 1 --curvature_beta 0.99 --precond_delta 1e-4 --precond_refresh_every 10
  --cw_nesterov --beta1 0.95`. (Existing OLMo cells ran β1=0.9 — ≤0.2σ ≡ 0.95, admissible.)
- **iMuon baseline** (`imuon-lora`): authors' vendored v5; `momentum=0.95` Nesterov, `wd=0`,
  `ns_steps=5`, `adjust_lr`, `ε=1e-6` (see `paper/PLAN.md` E0).
- **AdamW**: universal reference / speedup denominator.
- Protocol (all): global batch 16, seq 2048, 9000 steps, eval 250, packed_v1.1, bf16, compile.

## Coverage matrix (audit 2026-06-09; values = # distinct lrs)

| # | cell | model | data_dir | proto | iMuon | AdamW |
|---|------|-------|----------|:--:|:--:|:--:|
| 1 | OLMo opc r256 | allenai/OLMo-2-0425-1B | opc_…_seq2048 | β.95 in-flight ‡ | **0** | 5 |
| 2 | Qwen opc r256 | Qwen/Qwen2.5-1.5B | opc_…_qwen25 | **0** | **0** | 5 |
| 3 | Llama3.2 opc r256 | meta-llama/Llama-3.2-1B | opc_…_llama32 | **0** | **0** | 5 |
| 4 | Llama3-8B opc r256 | meta-llama/Meta-Llama-3-8B | opc_…_llama32 † | **0** | **0** | 4 |
| 5 | Qwen bengali r256 | Qwen/Qwen2.5-1.5B | aya_bengali_…_qwen | **0** | **0** | 5 |
| 6 | Llama3.2 openmath r64 | meta-llama/Llama-3.2-1B | openmath_…_llama32 | **0** | **0** | 4 |
| 7 | Llama3.2 openmath r128 | meta-llama/Llama-3.2-1B | openmath_…_llama32 | **0** | **0** | **0** |
| 8 | Llama3.2 openmath r256 | meta-llama/Llama-3.2-1B | openmath_…_llama32 | **0** | **0** | 4 |

† Llama-3-8B reuses the `_llama32` opc data (Llama-3.x tokenizer identity — no rebuild).
‡ Cell 1 protagonist @β1=0.95 is the in-flight β1 sweep (`diag_shampoo_polar_r256_opc_beta1_095`,
  job 6492862) — it IS the protagonist config at 0.95, so no separate run. The old β1=0.9
  runs are superseded.

## To-run (16 sweeps)

- **Protagonist** (7): cells 2,3,4,5,6,7,8. (cell 1 = the in-flight β1=0.95 sweep, job 6492862.)
- **iMuon** (8): all cells 1–8.
- **AdamW** (1): cell 7 only (the empty r128 rung; every other cell already has AdamW).

## Wrappers

Existing protagonist wrappers are OLMo-only and don't pass `--beta1`. For a 16-sweep batch,
build **generic per-optimizer wrappers** parameterized by `MODEL`/`DATA_DIR`/`LORA_R` (env)
rather than bespoke files — DECIDED. Wrappers needed:

- [ ] `sweep_protagonist_generic.sh` (PE8+Nesterov+β1=0.95; MODEL/DATA_DIR/LORA_R env)
- [ ] `sweep_imuon_generic.sh` (`imuon-lora`; MODEL/DATA_DIR/LORA_R env)
- [ ] `sweep_adamw_generic.sh` (cell 7 only; MODEL/DATA_DIR/LORA_R env) — or reuse existing

## Gating step

- [ ] **iMuon production smoke** through `train_lora.py` (argparse → build_optimizer →
  train loop), 2 steps, real model, GPU — REQUIRED before any iMuon SLURM. CPU unit test
  (`tests/test_imuon_lora.py`) passed but does NOT exercise the launcher chain.

## Timing / GPU

- Protagonist OLMo r256 PE8: measured **1.77 s/step** (wall-incl). 8B and Qwen/Llama need
  fresh per-step before their sbatch `--time`.
- 1B cells ≈ 4.4h/run; 8B substantially more — measure first.
- Respect QOS GPU cap; stage submission (cannot launch all 16 at once).

## Status log

- 2026-06-09: audit done; matrix populated; configs locked. Nothing submitted yet.
