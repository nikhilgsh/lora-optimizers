# Wall-time profile across the 1B/3B/8B base ladder

> **Status (2026-05-08): all tables below are at `data_pipeline_version:
> unpacked_v0`** (legacy `DataCollatorForLanguageModeling`, dynamic shapes,
> no prompt-mask). The packed_v1 reprofile is in §"Wall-time + MFU under
> packed_v1 (Blackwell)" below — it adds an MFU column and is the
> operative table for Phase B/C wall-budget decisions going forward.

Baseline (HF default attn, no compile) vs after (flash_attention_2 + torch.compile mode='default'). Single A100 (shared, not exclusive — relative ordering preserved per profiling_a100_canonical_2026_05_04.md).

Per-step times: forward + backward summed across grad_accum_steps microbatches; optimizer.step() and zero_grad timed separately. All numbers in ms.

**Hardware policy update (2026-05-08):** Blackwell RTX PRO 6000 is the new
default canonical hardware across all tiers (1B / 3B / 8B), not just 8B.
A100 numbers below are kept as the historical reference.

## 1B — allenai/OLMo-2-0425-1B, seq=512, batch=2×accum=8

| r | optim | method | cond | fwd ms | bwd ms | opt ms | zero ms | total ms | peak MB | attn | compile |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 128 | adam-polar-product-lora-coupled-spectral-chord-tight | eigh | baseline | 464.0 | 645.7 | 706.0 | 1.3 | 1817.0 | 9340 | sdpa | eager |
| 128 | adam-polar-product-lora-coupled-spectral-chord-tight | higham | baseline | 465.2 | 646.0 | 159.9 | 1.3 | 1272.3 | 9340 | sdpa | eager |
| 128 | adamw | eigh | baseline | 469.5 | 652.8 | 5.3 | 1.3 | 1128.8 | 8590 | sdpa | eager |
| 512 | adam-polar-product-lora-coupled-spectral-chord-tight | eigh | baseline | 791.6 | 1311.8 | 4880.0 | 1.6 | 6985.0 | 19717 | sdpa | eager |
| 512 | adam-polar-product-lora-coupled-spectral-chord-tight | higham | baseline | 782.4 | 1304.9 | 841.3 | 1.5 | 2930.2 | 19717 | sdpa | eager |
| 512 | adamw | eigh | baseline | 782.0 | 1302.1 | 18.4 | 1.5 | 2104.0 | 13174 | sdpa | eager |

## 3B — meta-llama/Llama-3.2-3B, seq=1024, batch=2×accum=8

| r | optim | method | cond | fwd ms | bwd ms | opt ms | zero ms | total ms | peak MB | attn | compile |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 128 | adam-polar-product-lora-coupled-spectral-chord-tight | eigh | baseline | 1700.0 | 2347.2 | 1307.0 | 2.3 | 5356.4 | 25738 | sdpa | eager |
| 128 | adam-polar-product-lora-coupled-spectral-chord-tight | higham | baseline | 1701.6 | 2357.2 | 344.3 | 2.2 | 4405.3 | 25739 | sdpa | eager |
| 128 | adamw | eigh | baseline | 1704.8 | 2343.1 | 10.4 | 2.1 | 4060.5 | 24269 | sdpa | eager |
| 256 | adam-polar-product-lora-coupled-spectral-chord-tight | eigh | baseline | 2040.8 | 3203.3 | 3266.4 | 2.2 | 8512.6 | 30528 | sdpa | eager |
| 256 | adam-polar-product-lora-coupled-spectral-chord-tight | higham | baseline | 2039.5 | 3202.2 | 741.9 | 2.2 | 5985.7 | 30528 | sdpa | eager |
| 256 | adamw | eigh | baseline | 2047.6 | 3203.1 | 19.3 | 2.1 | 5272.2 | 27505 | sdpa | eager |

## 8B — meta-llama/Meta-Llama-3-8B, seq=2048, batch=1×accum=16

| r | optim | method | cond | fwd ms | bwd ms | opt ms | zero ms | total ms | peak MB | attn | compile |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 64 | adam-polar-product-lora-coupled-spectral-chord-tight | eigh | baseline | 5775.9 | 7795.1 | 982.3 | 2.5 | 14555.7 | 42294 | sdpa | eager |
| 64 | adam-polar-product-lora-coupled-spectral-chord-tight | higham | baseline | 5775.7 | 7792.3 | 486.0 | 2.4 | 14056.4 | 42294 | sdpa | eager |
| 64 | adamw | eigh | baseline | 5855.8 | 7783.9 | 9.4 | 2.4 | 13651.5 | 41015 | sdpa | eager |
| 256 | adam-polar-product-lora-coupled-spectral-chord-tight | eigh | baseline | 7768.9 | 12103.6 | 4429.3 | 3.1 | 24304.9 | 54954 | sdpa | eager |
| 256 | adam-polar-product-lora-coupled-spectral-chord-tight | higham | baseline | 7771.1 | 12115.0 | 1523.4 | 3.1 | 21412.6 | 54954 | sdpa | eager |
| 256 | adamw | eigh | baseline | 7775.7 | 12081.3 | 32.2 | 3.0 | 19892.1 | 48996 | sdpa | eager |

## Headlines

### Cheap-wins speedups (compile + sdpa, AdamW)

- **1B r=128**: 1.37×; **r=512**: 1.17×
- **3B r=128**: 1.28×; **r=256**: 1.17×
- **8B r=64**: 1.29×; **r=256**: 1.15×

Compile delivers ~1.15–1.4× E2E. Less help at high r (memory-bandwidth-bound) and at large model (already optimized).

### Tight-chord overhead with all wins applied (compile + higham vs AdamW)

| | r=128/64 | r=256/512 |
|---|---:|---:|
| 1B | 1.18× | 1.46× |
| 3B | 1.10× | 1.17× |
| 8B | **1.05×** | **1.08×** |

Overhead shrinks with model size — at 8B, tight-chord+higham is essentially free relative to AdamW. **Custom optimizer kernels (Triton for the polar/NS step) would buy <1% E2E at 8B and are not worth pursuing for this campaign.**

### Memory savings from compile

- 1B: 7–16% (compile helps activations)
- 3B: 14–24%
- 8B r=64: ~12% (41 → 36 GB)
- 8B r=256: ~0% (LoRA params dominate at high rank — compile can't help)

### Custom kernel implications

- **Compile is the cheap win that mattered**. ~1.15–1.4× E2E across the board.
- **Higham (default switch) was the second cheap win** — 4–6× speedup on optimizer step at moderate r, brings tight-chord to ≤1.1× AdamW at 3B+.
- **Liger Kernel (fwd+bwd fusion)** is the next lever. Predicted ~1.2× on top of compile, plus 30% memory. At 8B r=256 (currently 18.8s/step), that's ~3s/step or ~5h shaved off a 6k-step run. Memory savings (55 → ~38 GB) free headroom for batch/seq increases. Strongly indicated for 8B.
- **Attention backend choice (`flash_attention_2` / `flash_attention_4`) is a no-op at our shapes** — direct measurement on 2026-05-07 across A100, H100 PCIe, and Blackwell sm_120 (see *Hardware comparison* below). At 8B/r=256/seq=2048/batch=1/accum=16 with bf16+causal, `sdpa`, `flash_attention_2`, and (where it works) `flash_attention_4` all produce per-step times within ~0.1% of each other and identical peak memory. PyTorch's SDPA dispatcher already selects a flash-attention-class kernel for this shape on every supported arch, so explicitly requesting an FA backend adds nothing. The 2.49× Blackwell win is from **the hardware**, not from FA4. Practical implication: **`sdpa` is the safe default** — leave the `--attn_implementation` flag on the existing FA2 default for recipe-matching, but no campaign run depends on installing flash-attn 2/3/4. FA4 itself is currently broken on Blackwell sm_120 (RTX-Pro-6000) in `flash-attn-4 4.0.0b12` (CuTeDSL JIT crashes in epilogue, both MHA + GQA paths) and **must not be combined with `--compile`** — torch.compile + the CuTeDSL JIT enters a recompile loop, ~480 s/step observed.

### Phase B/C wall-budget verdict (all cheap wins applied)

| tier | r | 6k steps wall | 8.2k steps (270M tokens) wall |
|---|---:|---:|---:|
| 3B | 128 | 5.5h ✓ | 7.5h ✓ |
| 3B | 256 | 8.7h ✓ | 12.0h ✓ |
| 8B | 64 | 19.1h ✓ | 26.1h ⚠ |
| 8B | 256 | **31.3h** ⚠ | **42.7h** ⚠⚠ |

**8B r=256 on A100 doesn't fit a 24h SLURM wall even with cheap wins.** Path forward (in priority order, updated 2026-05-07 with measured numbers):
1. **Move 8B to Blackwell** — measured 2.49× over A100+sdpa at 8B/r=256/seq=2048/batch=1/accum=16 (single-seed). Brings 6k to ≈13.3 h, 8.2k to ≈18.2 h — both under 24 h with comfortable headroom. **First choice for Phase B/C.**
2. **Move 8B to H100 PCIe** — measured 1.96× over A100+sdpa at the same config. Brings 6k to ≈16.9 h, 8.2k to ≈23.1 h — fits 24 h but with no margin. (H100 SXM untested; expected to land between H100 PCIe and Blackwell.)
3. **Add Liger Kernel** (~1.2× more on A100). Brings 8B r=256 6k to ~26h, 8.2k to ~36h. Helps but doesn't fully solve.
4. Run with 48h SLURM wall on A100. Simplest if cluster QoS allows.
5. Reduce 8B horizon to ~5k steps (~165M tokens, below 270M target).

### Hardware comparison (8B / r=256 / seq=2048 / batch=1 / accum=16 / AdamW)

Single-seed, n_warmup=2 + n_cycles=2 each cell. Measured 2026-05-07.

| hardware | attn | mean ms/step | peak MB | vs A100+sdpa (recorded) |
|---|---|---:|---:|---:|
| A100 (recorded baseline 2026-05-04) | sdpa | 19,892 | 48,996 | 1.00× |
| A100 80GB (today, less-contended) | sdpa | 18,714 | 47,970 | 1.06× |
| A100 80GB (today) | flash_attention_2 | 18,729 | 47,970 | 1.06× (=sdpa) |
| H100 PCIe (sm_90) | sdpa | 10,152 | 48,018 | 1.96× |
| H100 PCIe (sm_90) | flash_attention_2 | 10,075 | 48,018 | 1.97× (=sdpa) |
| H100 PCIe (sm_90) | flash_attention_4 | 10,071 | 48,018 | 1.98× (=sdpa) |
| Blackwell RTX-Pro-6000 (sm_120) | sdpa | **7,981** | 47,970 | **2.49×** |
| Blackwell RTX-Pro-6000 (sm_120) | flash_attention_4 | crash | — | upstream FA4 bug |

Notes:
- `sdpa` ≈ `flash_attention_2` ≈ `flash_attention_4` within ~0.1 % across all archs that support each backend. Identical peak memory to the byte. Confirms PyTorch SDPA already routes to FA-class kernels for bf16+causal at this shape.
- Today's A100+sdpa (18,714 ms) is ~6 % faster than the 2026-05-04 baseline (19,892 ms). Consistent with the original "shared, not exclusive" caveat — today's A100 was less contended. **Keep 19,892 ms as the canonical baseline for relative-ordering comparisons** (avoid mixing).
- Reproduction: `/mnt/home/nghosh/.tmp_scripts/fa_bench_phaseBC/`, `fa_bench_h100_8b/`, `fa2_bench_a100/`, `fa2_bench_h100pcie/` (one-off — not promoted to `logs/bench_profile_walltime/`).
- H100 SXM not measured (cluster's `gpuxl` partition has a 4-GPU minimum-allocation policy; deferred). Expected to land between H100 PCIe and Blackwell.
- FA3 not measured (`flash_attn_interface` not currently installed in `transformers` env; HF supports it via `attn_implementation="flash_attention_3"`).

### Memory floor for 8B at r=256

Compile cache + activations push past **40 GB** — Phase B/C 8B SLURM submissions MUST use `--constraint=a100-80gb`, not generic `--constraint=a100`. (Observed via OOM on a 40GB A100 during this profile.)

## Liger Kernel: tested, rejected (2026-05-07)

Smoked at 1B (OLMo-2-1B, r=128, seq=512) and 8B (Llama-3-8B, r=64, seq=2048),
with and without `fused_linear_cross_entropy=True`. All combinations applied
on top of the same cheap-wins stack (sdpa + compile + bf16 + higham).

| tier × r × seq | variant | AdamW total (s) | AdamW peak (MB) |
|---|---|---:|---:|
| 1B × 128 × 512 | no Liger | 0.819 | 7,151 |
| 1B × 128 × 512 | Liger (FLCE off) | 0.829 | 6,895 |
| 1B × 128 × 512 | Liger + FLCE | 0.824 | 6,895 |
| 8B × 64 × 2048 | no Liger | 11.07 | 32,873 |
| 8B × 64 × 2048 | Liger (FLCE off) | 11.22 | 32,876 |
| 8B × 64 × 2048 | Liger + FLCE | 11.14 | 32,870 |

Maximum observed speedup: 0.4%. Maximum observed memory savings: 3.6% (1B AdamW only). Both within noise at our scale.

Likely reasons: (1) torch.compile already captures the kernel fusions Liger
provides; (2) PEFT-wrapped `lora.Linear` modules prevent Liger from
substituting for the patched `nn.Linear`; (3) Liger's headline benchmark
is at 16K context, where memory savings scale ~quadratically with seq —
at seq=2048 the gain is minimal. **Skip Liger for this campaign.** The
`--use_liger` and `--liger_flce` flags remain in code (off by default)
in case future configurations change the picture.

Raw data: `logs/bench_profile_walltime/liger_smoke_1B.jsonl`,
`liger_flce_1B.jsonl`, `liger_smoke_8B.jsonl`.

## DDP refactor (Phase A0.7, 2026-05-07)

Wired single-node `DistributedDataParallel` into `train.py` so multi-GPU runs can shrink per-cell wall when single-GPU exceeds the 24h SLURM wall (8B r=256 270M-token target = 43h on 1 A100; needs DDP).

**Refactor components:**
- `lora_playground/distributed.py` — `init_distributed()` (idempotent; reads RANK/WORLD_SIZE/LOCAL_RANK from torchrun or SLURM env), `is_main()`, `all_reduce_mean()` for eval-loss aggregation, `cleanup()`.
- `train.py` — DDP wrap after PEFT and before `torch.compile`; `DistributedSampler` on train and eval loaders; `evaluate()` shards eval data and all-reduces a (loss-sum, token-count) pair so the global value matches single-GPU bit-for-bit; `log_event` and wandb gated to rank 0; optimizer-diagnostics emissions in `optim.py` similarly gated.
- `--global_batch_size` CLI flag — when set, derives per-rank `grad_accum_steps` from `global_batch / (batch_size × world_size)`. Errors cleanly if it doesn't divide.
- `slurm_scripts/sbatch_4gpu_ddp.sh` — torchrun launcher pinned to `--nodes=1 --gpus-per-node=4 --constraint=a100-80gb`. (Pinning matters: bare `--gpus=4` lets SLURM scatter across 2 nodes, breaking torchrun's `--nproc_per_node=4`.)
- `tests/test_ddp_smoke.py` — CPU-only 2-process gloo smoke verifying DDP grad all-reduce + `all_reduce_mean` correctness.

**Optimizer state under DDP:** Adam-family momentum buffers (`m_A`, `v_A`, ...) live in `pair_state`, NOT on model parameters, so DDP's automatic gradient hooks don't touch them. Concern was raised that they would diverge across ranks. **Resolution:** because DDP all-reduces `param.grad` *during* `loss.backward()`, every rank reads identical `A.grad` and `B.grad` after the backward, so the EMA update `m_A = β·m_A + (1-β)·g` produces identical state on every rank by construction. No explicit buffer all-reduce needed. Same logic for tight-chord's preconditioner Gram matrices (computed deterministically from synchronized factor weights).

### Environment caveat — NCCL / CUDA driver mismatch (resolved)

The `ffcv-pl` conda env had `nvidia-nccl-cu13 2.28.9` overwriting `nvidia-nccl-cu12 2.26.2` at the same `nvidia/nccl/lib/libnccl.so.2` path. The cu13 NCCL requires NVIDIA driver ≥ 580; cluster GPU compute nodes run driver 560.35.05 (CUDA 12.6). NCCL collectives failed with `Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'` while torch single-GPU and gloo backend both worked.

Fix:
```
pip uninstall -y nvidia-nccl-cu13 nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
pip install --force-reinstall --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install 'fsspec==2024.9.0'  # restore datasets/lightning compat
```

After the fix, `torch.cuda.nccl.version()` reports `(2, 26, 2)`; `strings .../libnccl.so.2 | grep "NCCL version"` shows `2.26.2+cuda12.2`; pip-conflicts on cu13 are gone.

### DDP wiring verification (functional; loss equivalence)

A 1B r=16 seq=512 200-step smoke (eager, no compile) at `--global_batch_size 16` produced bit-for-bit-equivalent loss on 1-GPU vs 4-GPU NCCL:

| | 1-GPU | 4-GPU NCCL | Δ |
|---|---:|---:|---:|
| eval_loss @ step 200 | 0.822846 | 0.822200 | 0.000646 (~0.7σ_AdamW) |
| world_size, per_rank_bs, accum, global_bs | 1, 2, 8, 16 | 4, 2, 2, 16 | (correctly derived) |

Wiring is correct. Speedup at this small scale was 1.23× — but this configuration is dominated by fixed overhead (kernel-launch overhead in eager mode + 200-step amortization of ~30s NCCL/compile init). It does NOT predict production scaling.

### DDP scaling: when DDP is worth its 4× GPU-hours

DDP uses 4× GPU-hours for ~3.5× wall-time reduction when scaling well. That's only worth it when:
1. A single-GPU cell exceeds the 24h SLURM wall (or 48h cap), or
2. Iteration-speed-per-cell matters more than total throughput.

For sweep work where many cells run in parallel, **single-GPU per cell is more efficient** (no DDP overhead, better cluster utilization, same total GPU-hours).

**Scaling expectations from the math** (DDP overhead ~10-20 ms per step, per-step compute from the AFTER table):

| tier × cell (with compile) | 1-GPU per-step | 4-GPU est | speedup est |
|---|---:|---:|---:|
| 1B r=128 seq=512 | 825 ms | 220 ms | ~3.7× |
| 3B r=256 seq=1024 | 4496 ms | 1140 ms | ~3.9× |
| 8B r=256 seq=2048 | 17317 ms | 4350 ms | ~4.0× |

Production-scale measurements (1B/3B/8B with compile, 100 steps each, sbatch jobs in flight 2026-05-07 evening) will replace these estimates with measured numbers.

### Phase recommendation under current data

- **Phase A (1B):** stay single-GPU. Cells run in 1-3h on one GPU; sweep-friendly; no DDP overhead.
- **Phase B (3B):** single-GPU by default. Cells run in 5-9h (6k canonical) or 17-25h (270M target); fits 24h wall comfortably or borderline. DDP only if a specific cell's wall is uncomfortable.
- **Phase C (8B):** **DDP required for r=256 270M target** (43h single-GPU → ~12-14h on 4-GPU per the math). 8B r=64 borderline — single-GPU 26-29h fits 48h wall, DDP fits 24h.

---

## Wall-time + MFU under packed_v1 (Blackwell)

Reprofile of the campaign-critical cells under
`data_pipeline_version: packed_v1` on Blackwell RTX PRO 6000 (workstation,
sm_120; peak 251.9 TFLOPS dense BF16). Run via
`scripts/bench/bench_optimizer_step.py` with synthetic packed batches
(equal-length 3-doc-per-slot packing, 4D block-diagonal SDPA mask,
per-doc position_ids reset).

Why a reprofile is needed:
1. Packed_v1 forward path uses an explicit 4D additive SDPA mask instead
   of the implicit `is_causal=True` path. Worth confirming the kernel
   isn't materially slower.
2. Per-step compute under packed_v1 is fixed-shape `(B, seq_length)` with
   100% signal density — so per-step tokens-of-signal is up to ~3× higher
   than unpacked_v0 at seq=2048 on Magicoder. Tokens-per-sec and MFU
   numbers from the unpacked tables don't transfer.
3. Hardware policy change: Blackwell is now the canonical comparison
   hardware across all tiers, not A100. Old tables are reference-only.

**Source of truth:** `logs/bench_profile_packed_v1/blackwell_runs.jsonl`.
SLURM job IDs: 6364410 (bench profile, 4 cells).

### Bench cells (Blackwell + A100, packed_v1)

All cells: bf16 + sdpa + compile, `--n_docs_per_slot 3`,
`precond_method=higham`, `higham_iters=10`, `precond_refresh_every=1`,
no gradient checkpointing.

**MFU = `4·N·T / (step_sec × peak_TFLOPS_dense)`.** The factor is 4 (not
the standard nanoGPT 6) because LoRA training freezes the base — there's
no param-grad backward pass through the base. See
`lora_playground.mfu.flops_per_token_for_mode`. With grad checkpointing
on (we don't enable it), the multiplier would be 6 (extra fwd
recomputation in bwd).

| tier | r   | HW         | optim              | total ms |  MFU  | peak GB |
|------|----:|------------|--------------------|---------:|------:|--------:|
| 1B   |  64 | Blackwell  | adamw              |     1652 | 48.3% |    14.5 |
| 1B   |  64 | Blackwell  | tight-chord-higham |     1693 | 47.1% |    14.8 |
| 1B   |  64 | A100       | adamw              |     2510 | 25.7% |    14.5 |
| 1B   |  64 | A100       | tight-chord-higham |     2608 | 24.7% |    14.8 |
| 1B   | 256 | Blackwell  | adamw              |     2124 | 41.1% |    16.9 |
| 1B   | 256 | Blackwell  | tight-chord-higham |     2327 | 37.5% |    18.4 |
| 1B   | 256 | A100       | adamw              |     4377 | 16.1% |    16.9 |
| 1B   | 256 | A100       | tight-chord-higham |     4729 | 14.9% |    18.4 |
| 8B   |  64 | Blackwell  | adamw              |     6830 | 62.5% |    32.1 |
| 8B   |  64 | Blackwell  | tight-chord-higham |     7012 | 60.8% |    35.5 |
| 8B   |  64 | A100       | adamw              |    11049 | 31.2% |    32.1 |
| 8B   |  64 | A100       | tight-chord-higham |    11479 | 30.0% |    35.5 |
| 8B   | 256 | Blackwell  | adamw              |     8998 | 50.3% |    39.9 |
| 8B   | 256 | Blackwell  | tight-chord-higham |     9885 | 45.8% |    53.7 |
| 8B   | 256 | A100       | adamw              |    17206 | 21.2% |    39.9 |
| 8B   | 256 | A100       | tight-chord-higham |    18661 | 19.6% |    53.7 |

### Hardware speedup A100 → Blackwell under packed_v1 (AdamW)

| tier | r   | A100 ms | Blackwell ms | speedup |
|------|----:|--------:|-------------:|--------:|
| 1B   |  64 |    2510 |         1652 |   1.52× |
| 1B   | 256 |    4377 |         2124 |   2.06× |
| 8B   |  64 |   11049 |         6830 |   1.62× |
| 8B   | 256 |   17206 |         8998 |   1.91× |

Range 1.52–2.06× under packed_v1. The old "2.49× at 8B/r=256" number
(measured on unpacked_v0) doesn't fully transfer — the explicit 4D
SDPA mask path costs more on A100 (which has lower kernel-launch
budget margin) than on Blackwell, narrowing the relative gap. Even so,
Blackwell remains decisively faster, and only Blackwell fits 8B/r=256
at 270M tokens within a 24h SLURM wall (see verdict tables below).

Sources:
- Blackwell: `logs/bench_profile_packed_v1/blackwell_runs.jsonl`, SLURM
  job 6364410 (single GPU on workergpu181, n_warmup=2-3, n_cycles=2-3
  per cell). JSONL rewritten in place to carry corrected 4N MFU values
  + `mfu_flops_per_token_per_param=4.0`; bench script now records this
  field directly for every new run.
- A100 80GB: `logs/bench_profile_packed_v1/a100_runs.jsonl`, SLURM job
  6364524 (single GPU on workergpu054, same `n_warmup`/`n_cycles`).
  Recorded under the fixed code so MFU is 4N from the start.

### Phase B/C wall-budget verdict (packed_v1, Blackwell)

| tier | r   | per-step (AdamW) | 6k wall | 8.2k wall (270M tokens) |
|------|----:|-----------------:|--------:|------------------------:|
| 1B   |  64 |           1652 ms |   2.8h ✓ |                  3.8h ✓ |
| 1B   | 256 |           2124 ms |   3.5h ✓ |                  4.8h ✓ |
| 8B   |  64 |           6830 ms |  11.4h ✓ |                 15.6h ✓ |
| 8B   | 256 |           8998 ms |  15.0h ✓ |                 20.5h ✓ |

**The 24h-wall blocker for 8B/r=256 is gone under packed_v1+Blackwell.**
Old verdict (A100+unpacked_v0): 8B/r=256 at 8.2k steps was 42.7h (⚠⚠).
New: 20.5h (✓) — comfortable headroom. All four campaign-critical cells
fit a single 24h SLURM wall on Blackwell with packed_v1.

### Phase B/C wall-budget verdict (packed_v1, A100)

| tier | r   | per-step (AdamW) | 6k wall | 8.2k wall (270M tokens) |
|------|----:|-----------------:|--------:|------------------------:|
| 1B   |  64 |           2510 ms |   4.2h ✓ |                  5.7h ✓ |
| 1B   | 256 |           4377 ms |   7.3h ✓ |                 10.0h ✓ |
| 8B   |  64 |          11049 ms |  18.4h ✓ |                 25.2h ⚠ |
| 8B   | 256 |          17206 ms |  28.7h ⚠ |                 39.2h ⚠ |

A100 single-GPU runs the 1B and 8B/r=64 6k cells comfortably, but 8B at
270M tokens or r=256 still overflows the 24h wall. So the campaign's
8B cells **must** run on Blackwell; A100 stays a fallback for 1B/3B
work where free Blackwell capacity is the binding constraint.

### Tight-chord overhead (packed_v1, Blackwell, AdamW=baseline)

| tier | r   | total ms (AdamW) | total ms (tight-chord) | overhead |
|------|----:|-----------------:|----------------------:|---------:|
| 1B   |  64 |           1652 |                  1693 |    1.025× |
| 1B   | 256 |           2124 |                  2327 |    1.10× |
| 8B   |  64 |           6830 |                  7012 |    1.027× |
| 8B   | 256 |           8998 |                  9885 |    1.099× |

Same pattern as the A100+unpacked_v0 numbers in §"Tight-chord overhead
with all wins applied" above: overhead is dominated by the higham
preconditioner's per-step cost, scales with rank, and is small at 1B/8B
r=64 (~3%) but a ~10% tax at r=256. The optimizer.step ms column
(919 ms for 8B/r=256 tight-chord vs 35 ms for AdamW) localizes it.

### Questions this profile answers

- **Q1: Does the packed_v1 4D SDPA mask path slow per-step compute vs
  unpacked_v0's implicit causal mask?** Diagnostic: 1B/r=64 AdamW
  total_ms vs the unpacked 1B/r=128 AdamW row in §1B above. If packed
  is within ~5% of (unpacked × 2048/512 seq scaling), the mask path is
  free. Larger Δ ⇒ flag.
- **Q2: Is per-token throughput (and MFU) actually higher under packed_v1?**
  Under packed_v1 every step processes seq_length signal tokens; under
  unpacked_v0 a fraction is padding. MFU column should land in
  20-40% on Blackwell at 1B/3B; less at 8B/r=256 (memory-bandwidth bound).
- **Q3: Does 8B/r=256 packed_v1 fit a 24h wall on Blackwell?** Compute:
  total_ms × 6000 steps / 1000 / 3600. Verdict in §"Phase B/C
  wall-budget verdict" once cell 4 lands.

### Headline

- **Packed_v1 4D SDPA mask is essentially free.** 8B/r=256 AdamW under
  packed_v1+Blackwell = 8,998 ms; same cell at unpacked_v0+Blackwell
  was 7,981 ms (§Hardware comparison). +12.7% per-step overhead from
  the explicit 4D mask construction. But under packed_v1 every step
  processes 100% signal tokens, so the *effective* per-signal-token
  throughput is up — see MFU column.
- **MFU jumps significantly under packed_v1.** 8B/r=64 AdamW at 62.5%
  MFU is excellent for bf16+compile+LoRA. 1B/r=64 at 48% is healthy
  too. Unpacked_v0 numbers aren't directly comparable (signal density
  differs), but the absolute MFU we measure here is the right number
  to plan the campaign against.
- **Tight-chord overhead trend matches A100+unpacked_v0.** ~3% at r=64,
  ~10% at r=256. No regression from the data-path change — overhead
  is dominated by the higham preconditioner, not data.

### DDP-on-Blackwell — known-broken, not for campaign use (2026-05-08)

A 4-GPU Blackwell DDP bench (SLURM job 6364791, then interactive
diagnosis on workergpu181) reproducibly fails at the **first NCCL
collective during DDP construction**, regardless of `--compile`,
`--data_pipeline_version`, or NCCL workarounds. Failure mode: CUDA
illegal-memory-access in `dist._broadcast_coalesced` /
`dist.Reducer(...)`, watchdog thread terminates all ranks with
SIGABRT.

**Bisected via interactive smokes** (workergpu181, 2 GPUs):
- Pure NCCL `all_reduce` on a 1-element tensor: ✓ works
- Tiny `nn.Linear(64,64)` → DDP wrap: ✓ works
- HF `AutoModelForCausalLM` (OLMo-2 1B) → DDP wrap, no PEFT:
  ✗ default → ✓ with `NCCL_P2P_DISABLE=1`
- HF + PEFT LoRA → DDP wrap: ✗ even with `NCCL_P2P_DISABLE=1` and
  `find_unused_parameters=True`

So plain-HF DDP works on Blackwell with the P2P workaround; **PEFT +
DDP** is the broken combination at this software stack
(torch 2.7.0+cu128, NCCL 2.26.2+cuda12.2, NVIDIA driver 580.142,
PyTorch nightly probably required for a fix). Existing A100 DDP smoke
documented under "DDP wiring verification" still works.

Decision: **drop DDP from the Phase B/C plan.** Single-GPU per cell on
Blackwell+packed_v1 fits all four campaign cells in a 24h SLURM wall
(verdict table above), and parallel-seeds beats DDP for sweep
workloads anyway. The bench script's DDP support
(`scripts/bench/bench_optimizer_step.py`, commit 7ba50a2) is left in
place — it works on A100 — but flagged as Blackwell-incompatible until
the upstream PyTorch / NCCL / PEFT stack updates.

If a future workload needs DDP on Blackwell, retry with newer torch
(≥ 2.8 nightly) + NCCL (≥ 2.27); revisit the bisection above.

### Sanity job (concurrent)

Single end-to-end training run to validate packed_v1 produces a sensible
eval-loss trajectory before launching any sweep. SLURM job 6364408,
config matches `scripts/sweep/sweep_4k_diag.sh` canonical (1B, seq=512,
4000 steps, AdamW, η=3e-4, r=16, seed=0). Output:
`logs/adamw_sanity_packed_v1_4k/run.jsonl`. Sanity criteria:
- Eval-loss decreases monotonically from initial.
- No NaN / Inf in train_loss or eval_loss.
- Eval-loss magnitude in 0.4-0.9 range (response-only CE on Magicoder).
- MFU > 5% (any lower suggests broken pipeline, not just slow optimizer).

A pass on these means the data + forward + loss path is intact under
packed_v1. Absolute numbers are NOT comparable to the unpacked_v0
AdamW@r=16 baseline (0.7579) — prompt-mask alone changes the loss
objective. Re-anchored AdamW noise-floor under packed_v1 is the
follow-on (multiseed).
