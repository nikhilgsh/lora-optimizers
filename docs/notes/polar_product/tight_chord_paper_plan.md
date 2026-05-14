# Scale-up validation of `adam-polar-product-lora-coupled-spectral-chord-tight` — plan

## Context

Tight-chord (`adam-polar-product-lora-coupled-spectral-chord-tight`, exact-root ρ=(-s+√(s²+4η))/2, see `docs/notes/polar_product/algorithm_tight_chord.md`) shows a strong single-seed eval-loss advantage over AdamW-LoRA on the canonical packed_v1 4k-step regime (OLMo-2-1B / Magicoder-OSS-75K / seq=512 / global_batch=16 / α=r / all-linear / constant LR / no warmup / 1-pass / packed_v1).

This plan covers the remaining work needed to argue the advantage is paper-grade: long-horizon characterization at 1B, downstream-task eval, and (deferred) larger-base scaling.

## Current state (packed_v1, 4000-step horizon)

Best-LR-per-cell from `notebooks/packed_v1_leaderboard.ipynb` (loader: `lora_playground.loader.load_runs(where={"data_pipeline_version": "packed_v1", "max_steps": 4000})`):

| r   | tight-chord final | AdamW-LoRA final | Δ raw    | Δ in σ-units |
|----:|------------------:|-----------------:|---------:|-------------:|
|  16 |            0.5192 |           0.5265 |  −0.0073 |         6.6σ |
|  64 |            0.5120 |           0.5202 |  −0.0082 |         4.8σ |
| 256 |            0.5103 |           0.5162 |  −0.0059 |        ~3.5σ |

r=128 has no packed_v1 4k cell.

Δ shrinks with r in σ-units. The simplest reading is shrinking headroom — more parameters, lower achievable floor, all reasonable optimizers crowd toward it. Disambiguating "headroom-shrinks-with-r" from "horizon artifact" via last-window slope extrapolation is too brittle on a noisy eval signal; the right test is a long-horizon run at one rank where both optimizers reach asymptote (Phase L below).

**σ anchor (packed_v1)**: σ_AdamW = 0.0011 (r=16), 0.0017 (r=64), measured from `logs/adamw_multiseed_packed_4k_blackwell/` (4 seeds × η=3e-4 + seed=0 LR sweep). All Δ claims must be quoted in σ-units against the r-matched anchor.

**Hardware canon**: Blackwell RTX-PRO-6000 across all tiers, per `docs/notes/polar_product/walltime_profile.md` (2026-05-08). A100 numbers are historical reference.

## What's done

- Phase A0 wall-time profile and explicit `attn_implementation` plumbing → `docs/notes/polar_product/walltime_profile.md`; `train.py:391,747,906`.
- Phase A0.7 DDP refactor → `lora_playground/distributed.py`, `slurm_scripts/sbatch_4gpu_ddp.sh`, `tests/test_ddp_smoke.py`, `--global_batch_size` flag (`train.py:350,558+`).
- packed_v1 data pipeline (prompt-mask + offline packing + block-diagonal causal mask) → `lora_playground/data.py`.
- σ_AdamW(packed_v1) measurement at r ∈ {16, 64} → `logs/adamw_multiseed_packed_4k_blackwell/`.
- Manifest scope contract → `lora_playground/manifest.py`, `slurm_scripts/submit.sh` (refuses untagged submissions).

## Open prereqs

- **P1 — `opc-sft-stage2` dataset slot (all 4 sub-configs concat)**. `prepare_data.py` defaults to Magicoder-OSS-75K. Add support for loading and concatenating multiple `(dataset, config)` pairs, then produce an Arrow cache `data/opc_sft_stage2_all_packed_seq2048`. Target: ~436k raw docs, ~248M tokens 1-pass (measured per `scripts/data/measure_corpus_tokens.py`; `educational_instruct` 118k × 143 = 17M, `evol_instruct` 111k × 515 = 57M, `mceval_instruct` 36k × 852 = 31M, `package_instruct` 171k × 840 = 144M; total 248M). All 4 sub-configs share `instruction`/`output` columns and route through train.py's existing `{instruction, output}` boundary tokenizer. Required for Phase L.
- **P2 — HumanEval / BigCode pass@1 hook**. Zero matches in `lora_playground/` today. Required to call the eval-loss result "paper-grade" downstream; not required to launch Phase L itself. End-of-training only, single suite.
- **P3 — `strong_setting_survey.md`**. One-page synthesis of the convergent best-practice signal across Biderman (2405.09673), Schulman LoRA-Without-Regret 2025, and Hayou μA (2602.06204). Needed before Phase B/C.

## Phase L — long-horizon 1B characterization

**Question**: does tight-chord's eval-loss advantage over AdamW-LoRA at r=64 persist when both optimizers are run to a ~225M-token horizon (Biderman IFT scale)?

**Cell**: OLMo-2-1B × **`opc-sft-stage2` (all 4 sub-configs concat)** × **seq=2048** × **global_batch=16 (batch=4 × accum=4)** × **r=64** × packed_v1 × constant LR × no warmup × 1-pass × α=r × all-linear × bf16 × `--compile` × diagnostic probes on × **single-GPU Blackwell**. Note: batch=16 single-microbatch OOMs at seq=2048 on 96 GB Blackwell (forward + lm_head exceeds budget); the 4×4 split fits comfortably (peak 26 GB with compile).

**Horizon** (measured from the built cache at `data/opc_sft_stage2_all_packed_seq2048`): 431,983 per-doc rows pack into **150,492 slots @ seq=2048** (2.87 docs/slot, 26% pad-tail fraction). 1-pass guard ⇒ ≤ 9,405 steps at `global_batch=16`. **Phase L uses `max_steps=9000`** (96% of 1-pass; leaves headroom for the epoch guard). `eval_every=250` ⇒ 36 eval points. Token accounting: 9000 × 16 × 2048 = **295M slot-tokens, ≈ 225M unique content tokens** (75% fill rate ≈ 431k docs × 525 avg / 295M).

**Runs** (≈10 total):

| Stage | Optimizers | LRs | Seeds | Steps | Count |
|------|------------|-----|-------|-------|------:|
| L1 — LR mini-sweep | AdamW; tight-chord | AdamW {3e-5, 1e-4, 3e-4}; tight {3e-3, 1e-2, 3e-2} | 0 | 6875 | 6 |
| L2 — headline | best-LR-per-opt from L1 | (from L1) | 0 | 6875 | 2 (subsumed by L1 if best LR is on the L1 grid) |
| L3 — sanity rerun | best-LR-per-opt | (from L1) | 1 | 6875 | 2 |

L1 grid is centered on the packed_v1 4k r=64 best-LR per optimizer (AdamW 1e-4, tight 1e-2 per leaderboard). Run L1 at the full 6875 horizon — the optimal η at 2k vs 6.9k can drift and a same-horizon sweep is the only honest pick.

**Pass criterion (L3)**: |Δ(seed=0) − Δ(seed=1)| ≤ 2 × σ_AdamW(packed_v1, r=64) = 0.0034.

**Kill criterion (L2)**: if the seed=0 Δ at step 6875 collapses to <2σ_AdamW(r=64) = <0.0034, the advantage is horizon-fragile. Pause campaign and investigate tight-chord stability before Phase B/C spend.

**Manifest tag**: `tight_chord_paper,phase_L,longhorizon_1b`. Add `phase_L` to the known-scopes table in `slurm_scripts/submit.sh` before submitting.

**Wallclock estimate** (measured directly on Blackwell workergpu181, 2026-05-13, at exact production config — batch=4 accum=4, seq=2048, r=64, compile, packed_v1, opc-sft-stage2 dataset). Warm-GPU steady state:
- AdamW: 1.716 s/step, eval 42.8 s/1024 samples, MFU 23.4%, peak 25.5 GB
- tight-chord: 1.723 s/step, eval 41.5 s/1024 samples, MFU 23.3%, peak 25.8 GB (overhead 1.004× vs AdamW)

Phase L cell wall = 9000 × 1.72 + 36 × 42 ≈ **4.7h** warm-state; cold-start 60-step bench projected 6.13h (cold transient amortizes over a 9000-step run). Conservative SLURM `--time` = 6.13 × 1.5 buffer = **10h**.

**Submission**: write each sbatch to `slurm_pending/`; user runs `submit-pending`. Claude cannot execute `sbatch` (org LAW).

**Pre-flight smoke**: one 5-step single-GPU run at r=64 seq=2048 on `tests/fixtures/tiny_code_*.jsonl` for each optimizer through the same sbatch entry point as the sweep, verifying packing handles seq=2048 without truncation surprises and the LoRA path produces sane init losses. Must go through the production entry point (argparse → main → factory → optimizer init → train loop) — not an isolated unit-test smoke.

## Phase B (3B) and Phase C (8B) — deferred

Pending Phase L outcome. If L2/L3 hold the σ-unit Δ, the ladder reopens to 3B (Llama-3.2-3B, Magicoder-Evol-110K, seq=2048; expected single-GPU per `walltime_profile.md:212`) and 8B (Llama-3.1-8B, Magicoder-Evol-110K, seq=2048; **DDP-4 required at r=256** per `walltime_profile.md:213`, single-GPU OK at r=64). Specific cell choices, LR re-sweeps, and wall budgets get specified at that time — the current plan does not lock the ladder pending the L result.

## Verification

- Every Δ in this doc carries `Δ / σ_AdamW(packed_v1, r-matched)`.
- All loaded run data goes through `lora_playground.loader.load_runs(where=...)`. No hand-typed group lists.
- Pre-submission for Phase L: `inventory_runs(logs_root)` clean (no orphaned groups, no missing manifests, no optimizer-in-logs missing from `OPTIM_COLORS`).
- Phase L pre-flight smoke through `sbatch_4gpu_ddp.sh` end-to-end before SLURM submission.

## Critical files

- `lora_playground/data.py`, `prepare_data.py` — Magicoder-Evol-110K slot (P1).
- `lora_playground/train.py` — HumanEval hook (P2), at end-of-training.
- `slurm_scripts/submit.sh` — register `phase_L` scope.
- `slurm_scripts/sbatch.sh` (single-GPU Blackwell) — Phase L launcher. (`sbatch_4gpu_ddp.sh` is reserved for Phase C 8B r=256.)
- `params/tight_chord_phase_L_r64_lrsweep.json` (new) — L1 grid.
- `params/tight_chord_phase_L_r64_headline.json` (new) — L2/L3.
- `docs/notes/polar_product/strong_setting_survey.md` (new) — P3.
