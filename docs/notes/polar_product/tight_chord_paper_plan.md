# Scale-up validation of `adam-polar-product-lora` (tight-chord) — survey + plan

## Context

Tight-chord (`adam-polar-product-lora-coupled-spectral-chord-tight`, the "tight" / exact-root variant ρ=(-s+√(s²+4η))/2 of the spectral-chord rule, see `docs/notes/polar_product/algorithm_tight_chord.md`) currently has strong **single-seed** results across the rank ladder. From `notebooks/spectral_chord_diag.ipynb` output (4000-step, best lr per rank):

| r | tight chord | AdamW | Δ | Δ in σ-units |
|---:|---:|---:|---:|---:|
| 16 | 0.7319 | 0.7375 | −0.0056 | ~9σ (using r=16 σ≈6e-4) |
| 32 | 0.7214 | 0.7307 | −0.0093 | ~14σ |
| 64 | 0.7159 | 0.7332 | −0.0173 | ~25σ (using r=64 σ≈7e-4) |
| 128 | **0.7104** | 0.7240 | **−0.0136** | **~19σ** |
| 256 | **0.7062** | 0.7185 | **−0.0123** | **~17σ** |

Tight-chord also beats the non-tight (loose, Spectron's ρ=η/(s+1)) variant at r ≥ 64: at r=128 tight = 0.7104 vs loose = 0.7151 (Δ = −0.0047 ≈ 7σ).

The repo's standard setting (OLMo-2-1B + Magicoder-OSS-Instruct, 2k steps, r=16, seq 512, batch 16) is tuned for **rapid optimizer differentiation**, not for paper-grade external comparison. Before claiming a paper-worthy advantage, the result must hold under conditions that mainstream LoRA papers/blogs use, with downstream-task eval (not just held-out NLL), multi-seed, at higher rank / longer horizon / larger base.

## Survey of the three sources

### 1. Biderman et al. 2024 — "LoRA Learns Less and Forgets Less" (TMLR; Databricks/Mosaic) — arXiv 2405.09673
- **Base:** Llama-2-7B
- **Two regimes:** Continued pretraining (CPT, 0.25–20B tokens) and Instruction finetuning (IFT, 1/2/4/8/16 epochs)
- **Domains:** code (CPT: StarCoder-Python 20B; IFT: Magicoder-Evol-Instruct-110K, 73M tok) AND math (CPT: OpenWebMath 14.7B; IFT: MetaMathQA, 103M tok)
- **Target-domain eval:** HumanEval pass@1, GSM8K accuracy. **Source-domain (forgetting):** HellaSwag, WinoGrande, ARC-Challenge.
- **LoRA cfg:** r ∈ {16, 64, 256}, **α=2r** (acknowledged crucial), target=**All transformer modules**.
- **LR:** sweep [1e-5, 5e-4], pick highest stable. LoRA optimal η ≈ 10× FullFT η.
- **Optimizer:** AdamW with cosine LR cooldown.
- **Best-practice recipe (Sec 4.7):** IFT not CPT, target=All, r=256, α=2r, LR sweep [1e-5,5e-4].

### 2. Schulman / Thinking Machines blog "LoRA Without Regret" (Sep 2025)
- **Bases:** Llama-3 8B, Qwen3 8B, Qwen3-30B-MoE
- **Datasets:** Tulu3, OpenThoughts3, MATH, GSM8K, DeepMath-103K
- **LoRA cfg:** r ∈ {1..512}, α=32 fixed, target=**all-linear** (MLP-only > attn-only)
- **Optimizer/schedule:** Adam (ε=0), constant LR, no warmup. LoRA optimal LR ≈ 10× FullFT.
- **Init:** Init[A] uniform 1/√d_in, B=0.

### 3. Chen, Villar, Hayou 2026 — "Learning Rate Scaling across LoRA Ranks" (μA) — arXiv 2602.06204
- **SFT bases:** Llama-3.2-1B/Tulu3, Qwen2.5-3B-Instruct/OpenThoughts-114k, RoBERTa-large/ANLI, ViT-Huge/14, Qwen3-VL-2B/LLaVA. **RLVR:** Llama-3.1-8B/GSM8k/GRPO. **Diffusion:** SD-1.5/Naruto-BLIP.
- **LoRA cfg:** r ∈ {4, 16, 64, 256, 1024}; three init/α configs.
- **Optimizer:** AdamW, weight decay 0.01, grad clip 1.0, **linear warmup 5% + cosine to 0.1× peak**, log2-spaced LR sweep.
- **Headline (μA scaling rules):** Init[A] α=1 ⇒ η_opt ∝ r^(−1/2); Init[B] α=1 ⇒ η rank-invariant and **transfers LoRA→FullFT** (η ∝ n^−1).

### Convergent best-practice signal
| Knob | Biderman | Schulman | Hayou | Repo today (audited) |
|---|---|---|---|---|
| Base model | Llama-2-7B | Llama-3-8B / Qwen3 family | 1B–8B mix | OLMo-2-1B |
| Target modules | All (attn+MLP) | All-linear (MLP critical) | (per-experiment, broad) | **`all-linear`** ✓ (`train.py:220`) |
| α convention | **2r** | 32 fixed | various | **α=r** (every launcher sets `--lora_alpha "$lora_r"`) |
| Rank coverage | 16/64/**256** | 1..**512** | 4..**1024** | 16/64/128 (recent: 256) |
| LR schedule | cosine + warmup | **constant, no warmup** | warmup 5% + cosine to 0.1× | **constant, no warmup** ✓ matches Schulman (`train.py:211-212`) |
| Pass count | multi-epoch (1–16) | **1-pass on big data** | **1-pass on big data** | **1-pass enforced** (`train.py:393` raises unless `--allow_multi_epoch`) |
| Dataset size | Magicoder-110K | Tulu3 939K / OpenThoughts3 | Tulu3 / OT-114K | Magicoder-OSS-75K (32K subset used for 2k-step invariant) |
| LR sweep | mandatory log-grid | mandatory | mandatory log2-grid | yes |
| Eval | HumanEval / GSM8K + held-out | held-out + reward | held-out + task | held-out NLL only |
| Seeds | mostly single | single (smoothed) | single per cell | AdamW noise-floor multiseed only; σ ≈ 0.001 |
| Horizon | 1–16 epochs (≈103M+ tok) | to convergence | to convergence | 2k / 4k steps × 16 batch × 512 seq ≈ 16M / 32M tok |

## Scope (from user answers)

- **Base ladder:** OLMo-2-1B → Llama-3.2-3B (or Qwen2.5-3B-Instruct) → Llama-3.1-8B.
- **Train domain:** code only (confirmed). Magicoder family throughout. No Tulu3 / OpenThoughts3 anchor — the cost of the extra cell isn't worth what's already a strong code-only story with Biderman as external anchor and HumanEval as paper-grade downstream metric.
- **Schedule + recipe:** constant LR, no warmup, 1-pass enforced everywhere. This is Schulman's *recipe* applied to the code domain — note Schulman himself uses Tulu3 / OpenThoughts3 / MATH / DeepMath (no code dataset), so this is a recipe-borrow not an exact replication. AdamW kept with default ε=1e-8 and project's existing weight decay (Schulman's ε=0 is for theoretical invariance analysis, not a practical optimizer recommendation; empirically ε=1e-8 vs ε=0 is negligible on real-magnitude gradients).
- **Token-horizon scaling strategy:** scale via seq_length × dataset, not via epochs or schedule.
  - 1B: Magicoder-OSS-75K, seq=512, batch_eff=16, ≤4k steps → up to ~32M tokens 1-pass (existing protocol).
  - 3B: Magicoder-Evol-Instruct-110K, seq=1024, batch_eff=16, 6k steps → 96K samples × 1024 ≈ 98M tokens 1-pass.
  - 8B: Magicoder-Evol-Instruct-110K, seq=2048, batch_eff=16 (likely micro=1 × grad-accum=16 to fit on A100-80G), 6k steps → 96K samples × 2048 ≈ 200M tokens 1-pass.
  - At each base: confirm the 1-pass invariant `max_train_samples = max_steps × batch_eff` is respected.
- **Eval (pure performance, no forgetting):** held-out NLL on the train dataset's eval split (existing) + **HumanEval pass@1** (target-domain code). No GSM8K, no HellaSwag/WinoGrande/ARC.
- **Seeds:** single-seed per cell, matching the reference papers. Biderman, Schulman, and Hayou all report headline numbers single-seed; their robustness comes from dense LR sweeps + rank/epoch curves, not seed averaging. Repo's measured σ ≈ 0.001 (CLAUDE.md: r=16 std 6e-4, r=64 std 7e-4) means the current Δ ≈ −0.009 vs AdamW is already ~9σ on a single seed — multi-seed has low informational return per 3× compute.
- **Optional sanity-check re-run** at exactly one cell (the 1B r=128 headline): one extra seed for tight-chord and one for AdamW-LoRA, just to confirm the existing single-seed Δ wasn't a tail event. Two extra runs total, not 3-seeds-everywhere.
- **No FullFT baseline.** Paper frame: "tight-chord vs strong LoRA baselines (AdamW-LoRA, adam-lin-lora, adam-polar-product non-tight) across base scale and rank."

## Hardware notes

**A100 is the canonical baseline** per CLAUDE.md ("Hardware comparison baseline is A100; local RTX A6000 is acceptable only for functional smokes"). All existing references in the project (`profiling_a100_canonical_2026_05_04.md`, AdamW noise-floor σ ≈ 0.001) are A100. Phase A0.1 profile and Phase A on 1B both run on A100.

**Worth revisiting H100 / Blackwell for Phase B/C wall-time.** Once started, H100 typically delivers 2–3× per-step throughput vs A100 for bf16 LLM training (Llama-3.1-8B at r=256 seq=2048 on H100 ≈ 1/2 to 1/3 the wall vs A100). Blackwell (RTX-Pro-6000) is faster still and has 96 GB HBM (vs A100's 80 GB), which would relax the seq=2048 micro-batch constraint at 8B. Across a 6k-step Phase B/C run that's hours of compute saved per cell.

When to switch:
- eval_loss is hardware-independent (bf16 numerics ~identical across A100 / H100 / Blackwell), so the *result* values transfer cleanly.
- timing tables that go in the paper need a single hardware baseline — pick A100 OR H100 for the headline timing figure and document it; do not mix.
- queue-time advantage cuts both ways: when A100 is full, H100 may also be full (today's data: all three near 0 idle, but Blackwell had 1 idle node).

**Flash Attention coverage by hardware (verified against dao-ailab/flash-attention 2026-05-07):**
- Ampere (A100): FA-2 supported; PyTorch SDPA already routes to FA-2 internally → no standalone-package install needed.
- Hopper (H100): FA-2 (via SDPA) or FA-3 beta for FP16/BF16 + FP8 forward.
- Blackwell (RTX-Pro-6000, B200): **FA-4** is production-ready, written in CuTeDSL. Install: `pip install flash-attn-4` (use `flash-attn-4[cu13]` on CUDA 13). HuggingFace integration through `attn_implementation` is not explicitly documented in the FA repo — verify the transformers-side wiring with a smoke before committing.

For Phase B/C at 8B + seq=2048, **Blackwell + FA-4 is the most attractive configuration** because (a) memory savings from flash attention scale with seq length (~10–20× at 2k–4k), (b) Blackwell has 96 GB HBM vs A100's 80 GB, relaxing the seq=2048 micro-batch constraint, (c) FA-4 is production-ready not beta. Worth planning for now rather than after Phase A.

Action items, in order:
1. Finish A0.1 baseline + AFTER on A100 (in flight) — establishes the canonical reference and validates compile on our code path.
2. Before Phase B starts, run a single A0.1-style profile cell on Blackwell at 8B r=256 seq=2048 with FA-4. Compare per-step wall and peak memory to the A100 number from this campaign. Needs `pip install flash-attn-4` (env-mod hook will block — ask user).
3. If Blackwell + FA-4 delivers >1.5× over A100 + sdpa at 8B r=256 and the install + HuggingFace `attn_implementation="flash_attention_4"` (or whatever the wiring turns out to be) works cleanly, switch Phase B/C to Blackwell. Otherwise stay on A100.
4. Either way, paper-headline timing tables get one hardware label and stay there.

## Gap analysis

What the repo already has right (confirmed by audit, lock in across the campaign):
- `target_modules = all-linear` ✓
- `lr_scheduler_type = constant`, `warmup_steps = 0` ✓ (Schulman recipe)
- 1-pass invariant enforced ✓
- α=r convention via `--lora_alpha "$lora_r"` in launchers (project memory)

What's missing for paper-grade:

1. **HumanEval eval hook** — BigCode harness, run at end-of-training (and cheaply at canonical 2k/4k checkpoints). Single eval suite, performance-only.
2. **Larger bases** — model loading already works for HF models, but memory budget at 3B/8B with r=256 + diagnostic probes will need bigger SLURM walls and possibly batch/grad-accum retuning.
3. **Higher rank** — extend to r=512 at 1B/3B; r=256 ceiling at 8B unless memory permits more.
4. **Larger-data dataset slot** — current Magicoder-OSS-75K (32K subset) supports up to ~4.7k 1-pass steps at batch 16. To run 6k–10k steps in 1-pass at 1B/3B/8B, swap the data slot to **`ise-uiuc/Magicoder-Evol-Instruct-110K`** (this is exactly Biderman's IFT training set; 110K examples = up to ~6.9k steps × batch 16 in 1-pass). If still not enough at 8B, add a longer-context code IFT (OpenCoder, Tulu3-coder subset) — but defer until needed.

(Note on α: Biderman's "α=2r is crucial" is empirically driven by one sweep at r=256, partly absorbed into the LR sweep — the LoRA update is `(α/r)·BA`, so α/r is equivalent to LR up to a constant. Hayou's μA framework prescribes α=1 (rank-independent) with Init[B] for LR-transfer; rsLoRA prescribes α∝1/√r. The project's α=r convention gives constant ratio α/r=1; this is well-supported and we keep it. No α-ablation needed — a properly resolved LR sweep absorbs the difference.)

## Recommended campaign — phased

Phase numbering is risk bisection, not implementation steps. Each phase has a kill criterion: if its result doesn't preserve tight-chord's edge, stop and re-think before paying the next phase's cost.

### Phase 0 — paper-survey artifacts in repo

Already done in this planning session:
- `docs/papers/biderman_2405.09673.pdf` ✓
- `docs/papers/mua_2602.06204.pdf` ✓ (pre-existing)
- `docs/papers/schulman_lora_without_regret_2025.md` ✓ (careful-read digest including the canonical vLLM repro details: Qwen3-4B SFT on no_robots, batch_eff=32, 200 steps, seq=2048, α=32 hardcoded, AdamW with default ε)

Still to do:
- Write a one-page synthesis at `docs/notes/polar_product/strong_setting_survey.md` distilling the convergent best-practices table above (Phase A prerequisite — first thing to ship after Phase 0).

### Phase A0 — wall-time profile + low-risk perf wins (Phase A prereq)

The campaign at 8B with r=256 and diagnostic probes will be expensive; we should not pay for inefficiencies. Per CLAUDE.md "measure don't guess": profile FIRST.

A0.1 **Profile across the full base ladder (1B / 3B / 8B).** Use `torch.profiler` or simple `time.perf_counter()` instrumentation around forward, backward, optimizer step, eval, diagnostic probes. Run a short profiling sweep (≈20 steps after warmup, no full training) at:
- 1B (OLMo-2-1B, seq=512, batch_eff=16) at r ∈ {128, 512}
- 3B (Llama-3.2-3B, seq=1024, batch_eff=16) at r ∈ {128, 256}
- 8B (Llama-3.1-8B, seq=2048, batch_eff=16, micro=1×accum=16) at r ∈ {64, 256}

Report per-step ms breakdown AND peak GPU memory at each (base, r) cell. Use the same A100-80G hardware throughout for fair scaling. Output table goes to `docs/notes/polar_product/walltime_profile.md` — this table is also load-bearing for SLURM wall-budget choices in Phase B/C.

Run this profile sweep BEFORE A0.2/A0.3 (it's the baseline) and AGAIN AFTER (to measure speedup). Both columns in the same table.

A0.2 **Explicit flash-attention-2.** Add `attn_implementation="flash_attention_2"` to `AutoModelForCausalLM.from_pretrained` call at `train.py:427`, fall back to "sdpa" if unavailable. Confirm via the model's `config._attn_implementation` after load and log it as a `config` event field. (Audit found: currently relies on HF default — may or may not be flash for our targets.)

A0.3 **Default-on `--compile` and `--bf16` for production sweeps.** Pin both in the canonical `scripts/sweep/sweep_*_diag.sh` launchers; keep them off only in unit-test smokes. Per CLAUDE.md "torch.compile whenever amortizes."

A0.4 **Liger Kernel evaluation.** If profile shows MLP/RMSNorm/cross-entropy is >20% of step wall, attempt Liger. Requires `pip install liger-kernel`; package-env hook will block — ask user before installing. Liger does NOT replace LoRA modules so it composes with `AdamPolarProductLoRA` cleanly. Expected: 1.1–1.3× E2E, ~30% memory savings.

A0.5 **Unsloth spike (parallel side task, NOT campaign-blocking).** One day max. Goal: determine whether `AdamPolarProductLoRA` is compatible with Unsloth's modified LoRA forward/backward. Likely Unsloth's fused backward bypasses or rewrites the standard `param.grad` path that our custom optimizers depend on. If yes — switch campaign over (2–3× wins are paper-grade). If no — stay on stock PEFT and document the incompatibility.

A0.6 **Diagnostic probe overhead.** If probes are >15% of step wall (per CLAUDE.md they're nominally ~10%), batch the per-pair operations or downsample the probe cadence. Don't kill them — they're load-bearing per the "diagnostics on by default" memory.

Kill criterion for A0: profile must end with a measured per-step time at r=128 and r=512 on 1B and a ≥1.3× E2E speedup vs the pre-A0 baseline before paying for Phase B. If A0.2+A0.3 alone don't deliver 1.3× → proceed to A0.4 Liger.

### Phase A0.7 — DDP refactor (Phase B/C wall-budget prereq)

**Why now.** A0 measurements (`docs/notes/polar_product/walltime_profile.md`) showed that at 8B r=256 with all cheap wins applied, a single 270M-token cell takes ~43h on A100 — over the 24h SLURM wall. Liger was tested and rejected. DDP is the highest-ROI remaining lever: 4-GPU DDP gives ~3-3.5× wall speedup with ~zero loss in throughput per GPU-hour (we have QoS budget for many concurrent 4-GPU cells), and 8B r=256 270M drops from 43h → ~13-14h. Sweep-friendly.

**Effort estimate.** ~6-8h MVP (Phase A0.7 below), ~2-4h polish. Most mechanical; the optimizer-state risk that the original DDP scoping report flagged is overstated for our Adam-family optimizers (DDP's automatic `param.grad` all-reduce keeps EMA buffers synchronized by construction — see `docs/notes/polar_product/ddp_refactor_scope.md` for the analysis).

**Effective-batch policy (settled).** Campaign default global effective batch = **32** (Schulman canonical, well within his "LoRA matches FullFT" small-batch regime). Per-rank batch and grad-accum are derived per tier to fit memory:
- 1B seq=2048: per-rank batch=8, accum=1 → global=32 on 4 GPUs
- 3B seq=2048: per-rank batch=4, accum=2 → global=32 on 4 GPUs
- 8B seq=2048: per-rank batch=1, accum=8 → global=32 on 4 GPUs

Implementation: add `--global_batch_size` CLI flag (default 32 under DDP; for single-GPU runs caller can still set per-rank `--batch_size`/`--grad_accum_steps` directly for back-compat).

**A0.7.1 — distributed env setup.** New module `lora_playground/distributed.py`:
- `init_distributed()`: idempotent; reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from env (set by torchrun); calls `dist.init_process_group("nccl")` and `torch.cuda.set_device(local_rank)`. No-op if `WORLD_SIZE` unset (single-GPU mode).
- `is_main()`: `rank == 0`.
- `all_reduce_mean(tensor)`: helper for eval loss aggregation.
- `cleanup()`: `dist.destroy_process_group()` at end.

**A0.7.2 — `train.py` changes:**
- Call `init_distributed()` at top of `main()`.
- Wrap train DataLoader (`train.py:422`) with `DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)`. Eval loader: same pattern with `shuffle=False` (we'll all-reduce eval loss).
- Cap `num_workers` per process: `max(1, args.num_workers // world_size)`.
- After PEFT wrap and `.to(device)` (~`train.py:494`), wrap with `DistributedDataParallel(model, device_ids=[local_rank])` when `world_size > 1`.
- Pass `model.module if isinstance(model, DDP) else model` to `build_optimizer()` and `collect_dense_target_weights()` (so PEFT-internal traversal finds the actual `lora_A`/`lora_B` modules, not DDP's wrapper).
- `evaluate()` (`train.py:131`): runs on every rank (DistributedSampler shards eval data); after the per-rank loss accumulation, all-reduce-mean weighted by per-rank token counts, return scalar.
- `log_event()` (`train.py:159`): gate to `is_main()`. One-line wrapper.
- Wandb logging: gate to `is_main()`.
- `train_iter` epoch boundary: call `train_loader.sampler.set_epoch(epoch)` at the start of each epoch (1-pass campaign so only matters if we ever multi-epoch).
- End of `main()`: call `cleanup()`.

**A0.7.3 — `--global_batch_size` flag.** Add CLI arg defaulting to 32. When set under DDP, derive `per_rank_batch_size = global // (world_size × grad_accum_steps)`; raise a clear error if it doesn't divide cleanly. Log derived per-rank values to the config event for reproducibility.

**A0.7.4 — launcher updates.**
- New `slurm_scripts/sbatch_4gpu_ddp.sh` based on `sbatch.sh` template, with:
  - `#SBATCH --gpus=4 --cpus-per-task=32 --constraint=a100-80gb`
  - Replaces `disBatch "$TASK_FILE"` with `torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 train_lora.py "$@"` style invocation
  - For sweep-style use (multiple cells per node), keep disBatch but each task uses 4 GPUs — disBatch handles that via its own GPU-mapping.
- Update `slurm_scripts/submit.sh` to accept `--ddp` flag; when set, sbatch with `--gpus=4*N_CELLS` and pass torchrun-driven sweep wrapper.
- Or simpler intermediate: keep `submit.sh` single-GPU per cell as default; for DDP campaigns invoke `sbatch_4gpu_ddp.sh` directly per-cell.

**A0.7.5 — tests.** New `tests/test_ddp_smoke.py` that uses `torch.multiprocessing.spawn` to launch a 2-process group on a single GPU (gloo backend OK for tiny-model tests; verify identical loss on 1-rank vs 2-rank for the same fixed batch). Existing tests must still pass under the single-GPU code path (no DDP wrap when `WORLD_SIZE` unset).

**A0.7.6 — verification smoke (post-implementation).** A 4-GPU run on `tests/fixtures/tiny_code_*.jsonl`, 50 steps, tight-chord at r=16. Compare eval_loss to the equivalent single-GPU run (same seed, same global batch). Pass criterion: `|Δ eval_loss| ≤ float32 noise floor (~1e-5)`. Then a 4-GPU 8B r=64 200-step run via the bench script (without optimizer logic, just train loop) to confirm ~3× per-step speedup vs single-GPU.

### Phase A — fix the protocol on existing 1B base

The only new code in Phase A is the HumanEval hook (A2). Everything else is configuration, sweep submission via existing `submit.sh` + `disbatch`, and analysis through the existing loader. No optimizer changes, no training-loop refactors.

A1. **Protocol locks** (audit complete, no train.py changes needed except dataset path):
- Target modules: `all-linear` (default ✓)
- LR schedule: constant, no warmup (default ✓ — Schulman recipe; do not switch to cosine)
- 1-pass: enforced (default ✓ — never set `--allow_multi_epoch`)
- α=r (project convention; well-supported, no ablation)
- For horizons >4.7k steps: switch dataset to `ise-uiuc/Magicoder-Evol-Instruct-110K`. Add a `prepare_data.py` invocation for the new dataset and an Arrow cache at `data/magicoder_evol110k_seq512_<N>k`. Keep the `1-epoch invariant` rule from the protocol (max_train_samples = max_steps × effective_batch_size).
A2. **Add HumanEval hook** in `train.py`: load BigCode evaluation harness, run pass@1 at end-of-training (and cheaply at the canonical 2k/4k checkpoints). One eval suite, not five.
A3. **Optional sanity re-run** at the existing headline (r=128, 4k, η=1e-2): one extra seed for tight-chord, one for AdamW-LoRA. Cheap (~2 runs) and rules out the existing Δ being a tail event before paying for Phase B/C.
   - Kill criterion: if the second seed swings the Δ across the σ ≈ 0.001 noise floor (i.e., collapses to <2σ), pause campaign, investigate tight-chord stability before any larger-base spending.
A4. **Rank sweep at canonical 4k horizon, single-seed.** Cells: r ∈ {16, 64, 128, 256, 512} × {tight-chord, AdamW-LoRA, adam-lin-lora, adam-polar-product (non-tight)}. LR pre-tuned per-rank using existing partial sweeps (the three untracked `params/spectral_chord_tight_*_4k.json` already cover much of this) + small extension to r=512 and to the missing baselines. Diagnostic probes (cos sim, σ(S), conditioning) on by default.

No batch-stability sweep. Mechanistic prior: tight-chord's spectral trust region is most active at small batch (noisy gradients → large raw updates → chord clips them); at large batch the per-step ΔW magnitude shrinks naturally and the chord rarely binds, so tight-chord's behavior collapses toward AdamW. Predicted direction is Δ-shrinks-with-batch, not a publishable batch-robustness finding. Picking batch=32 by Schulman/McCandlish theory (post-training critical batch in the 32-256 range) and moving on.

Kill criterion for Phase A: A3 collapses Δ, OR A4 shows tight-chord losing at every other rank.

### Phase B — generalize to a 3B base (Llama-3.2-3B or Qwen2.5-3B-Instruct)

Pick one base; default Llama-3.2-3B because it appears in Hayou's μA SFT set, lining up the LR-scaling prediction with our measurement.

B1. **Reuse Phase A protocol verbatim.** Single change is base model + adjusted batch/grad-accum to fit memory. No retuning of optimizer code.
B2. **Rank cells** at 3B: r ∈ {64, 128, 256}, single-seed, LR re-swept on tight log2 grid per rank (μA predicts how much η should move when r changes; check empirically against the prediction).
B3. HumanEval at end-of-training for each cell.

Kill criterion: tight-chord must hold a >2σ Δ vs AdamW-LoRA at the best-per-rank cell (σ from existing AdamW multiseed extrapolated, or one-cell sanity-seed at 3B if uncertain). If parity at 3B, the optimizer story needs to shift before paying for 8B.

### Phase C — paper-grade headline at Llama-3.1-8B

Only run after Phase B passes its kill criterion.

C1. Restrict to r ∈ {64, 256} (memory budget at 8B is real; 512 deferred unless GPU RAM allows).
C2. Single-seed at the (best r, best η) tight-chord vs AdamW-LoRA cell. Add adam-lin-lora and adam-polar-product non-tight at r=256 only.
C3. HumanEval at end-of-training.
C4. **One non-headline rank cell single-seed** to verify the r-trend continues at 8B.

Plan for resource: 8B at r=256 with bf16 LoRA + diagnostic probes runs comfortably on A100-80G but not A40/A6000 — must use the cluster's A100s (per project rule: A100 baseline for timing, A6000 functional smoke only). Estimate wall before submitting; pick an SBatch script with longer wall (per "ETA — be conservative" memory).

### Phase D — paper write-up scaffolding (parallel with C)

- Result tables: (base × r × optimizer) → eval_loss, HumanEval pass@1, σ-units (vs the AdamW-multiseed σ ≈ 0.001 anchor), wall-time.
- Plots: rank-vs-Δ-vs-AdamW per base; LR-vs-final-loss per (base, r); cos/σ(S) diagnostic trajectories.
- Writeup at `docs/notes/polar_product/tight_chord_paper_draft.md`.

## Critical files / places to touch

- `lora_playground/train.py` — add HumanEval / LM-Eval-Harness hooks; audit target-module default; pin LR schedule.
- `lora_playground/optim.py` — `AdamPolarProductLoRA` is the system-under-test; no expected change unless Phase B/C reveals scale-dependent issue.
- `lora_playground/loader.py` — verify cfg-field schema covers any new flags (eval-suite enable, base-model id, α=2r flag).
- `params/` — new sweep specs per phase, named `tight_chord_<phase>_<descriptor>.json`.
- `slurm_scripts/sbatch.sh`, `slurm_scripts/submit.sh` — verify wall budgets; 3B/8B at 4k steps will need longer walls than 1B/2k presets. Add an A100-80G long-wall script if not present.
- `scripts/sweep/sweep_*_diag.sh` — likely a new variant per base that bumps batch/grad-accum to fit memory.
- New: `docs/notes/polar_product/strong_setting_survey.md` (Phase 0).
- New: `docs/notes/polar_product/tight_chord_paper_plan.md` (durable copy of this plan after approval).
- New: `docs/papers/biderman_2405.09673.pdf`, `docs/papers/schulman_lora_without_regret_2025.md`.

DDP-refactor specific (Phase A0.7):
- New: `lora_playground/distributed.py` — `init_distributed`, `is_main`, `all_reduce_mean`, `cleanup` helpers.
- `lora_playground/train.py` — DDP wrap after PEFT, DistributedSampler, rank-0 gating for log/wandb, all-reduce eval loss, `--global_batch_size` arg.
- New: `slurm_scripts/sbatch_4gpu_ddp.sh` — torchrun-driven 4-GPU 80GB-A100 launcher.
- New: `tests/test_ddp_smoke.py` — 2-process subprocess-spawn smoke verifying loss equivalence.

## Reused infrastructure (don't rewrite)

- `lora_playground.loader.load_runs(where=…)` — for all post-hoc analysis; never hand-typed group lists (per CLAUDE.md).
- Manifest scope tags: introduce one new tag `tight_chord_paper`; tag every run in this campaign with it.
- `notebooks/sweep_analysis.ipynb` — already plots r-sweeps and LR-sweeps; extend rather than create new.
- Multi-seed σ from `logs/adamw_multiseed/` — anchor for 1B/r=64; will need a fresh σ measurement for 3B and 8B (3 seeds AdamW at each base, headline r only).
- `/disbatch` skill for parallel sweep submission.
- `/slurm-submit` for single big runs (8B headline cells).
- BigCode evaluation harness (HumanEval) and LM Evaluation Harness (GSM8K, HellaSwag, WinoGrande, ARC-Challenge) — match Biderman's tool choice for direct comparability.

## Verification

- Phase A correctness: `pytest tests/test_lora_utils.py tests/test_train_helpers.py -q` must pass; smoke run with new HumanEval hook on tiny fixtures must terminate in <5 min and emit a valid pass@1 JSON.
- Per-phase: run `lora_playground.loader.inventory_runs(LOGS)`; ensure no orphaned groups, no missing manifests, no optimizer in logs missing from `OPTIM_COLORS`.
- σ-unit reporting required for every Δ — use the existing AdamW multiseed σ ≈ 0.001 anchor; explicitly compute and write `(Δ / σ)` next to the raw Δ.
- Pre-submission for every sweep: enumerate (cfg-tuple) overlap with existing logs (per CLAUDE.md "manual reuse-check"), report count of overlapping cells, only submit the diff.
- HumanEval numbers cross-checked against Biderman's reported AdamW-LoRA values (e.g. r=64, IFT 4 epochs ≈ 0.417 on Llama-2-7B) — 8B Llama-3.1 should be in the same ballpark or better; 1B/3B have no Biderman anchor, so cross-check is internal consistency only (rank-monotonic, smoothly varying with η).
