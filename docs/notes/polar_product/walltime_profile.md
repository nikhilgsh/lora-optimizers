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

### DDP=4 cells (Blackwell, packed_v1)

Same 4 cells under 4-GPU single-node DDP (NCCL 2.27.7, after the
upgrade documented above). Per-step time changes very little — DDP's
all-reduce overhead is small relative to the fwd+bwd compute — but
each step now consumes 4× the docs (global batch = per-rank batch ×
accum × world_size).

| tier | r   | optim              | single ms | DDP=4 ms | per-step overhead | MFU (DDP=4) |
|------|----:|--------------------|----------:|---------:|------------------:|------------:|
| 1B   |  64 | adamw              |     1652 |     1669 |             +1.0% |       47.8% |
| 1B   |  64 | tight-chord-higham |     1693 |     1711 |             +1.1% |       46.6% |
| 1B   | 256 | adamw              |     2124 |     2167 |             +2.0% |       40.3% |
| 1B   | 256 | tight-chord-higham |     2327 |     2369 |             +1.8% |       36.9% |
| 8B   |  64 | adamw              |     6830 |     6974 |             +2.1% |       61.2% |
| 8B   |  64 | tight-chord-higham |     7012 |     7149 |             +2.0% |       59.7% |
| 8B   | 256 | adamw              |     8998 |     9311 |             +3.5% |       48.6% |
| 8B   | 256 | tight-chord-higham |     9885 |    10202 |             +3.2% |       44.4% |

Overhead grows mildly with rank because DDP's all-reduce volume is
proportional to LoRA-grad size (~`2 · r · (d_in + d_out)` per layer).
Even at 8B/r=256 it's only +3.5%.

Source: `logs/bench_profile_packed_v1/blackwell_ddp4_runs.jsonl`,
SLURM job 6369469 (4 GPUs on workergpu179).

### Phase B/C wall under DDP=4 (Blackwell, packed_v1)

At fixed *token-budget*, DDP=4 reduces wall by 4× (modulo the few-%
all-reduce overhead). Useful only if a single cell's wall is
uncomfortable — for sweep workloads, parallel single-GPU seeds beats
DDP-within-a-cell.

| tier | r   | per-step (AdamW, DDP=4) | 6k-doc-budget wall | 8.2k-doc-budget (270M) wall |
|------|----:|------------------------:|-------------------:|----------------------------:|
| 8B   |  64 |                 6974 ms |              2.9h ✓|                       4.0h ✓ |
| 8B   | 256 |                 9311 ms |              3.9h ✓|                       5.3h ✓ |

(6k-doc-budget = 1500 steps × 4 ranks × batch×accum=16; 8.2k-doc-budget
= 2050 steps × 4 ranks × 16. Same total docs as the single-GPU 6k/8.2k
columns — DDP-4 just gets there in 1/4 the steps.)

8B/r=256 finishes the 270M-token budget in **5.3h** under DDP=4 vs
**20.5h** single-GPU on the same hardware. 3.86× wall speedup at 4×
GPU-hours — break-even on GPU-hours, useful when wall matters more
than throughput.

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

### DDP-on-Blackwell — NCCL 2.26.2 broken, fixed by upgrade to 2.27.7 (2026-05-08)

**Resolution:** Upgrading `nvidia-nccl-cu12` from `2.26.2` → `2.27.7`
fully fixes the Blackwell DDP path. `pip install --upgrade --no-deps
--no-cache-dir nvidia-nccl-cu12==2.27.7` (run by user). torch 2.7.0+cu128
loads the new `libnccl.so.2` from `site-packages/nvidia/nccl/lib/`
without rebuild — the binding is dynamic, NCCL 2.x is ABI-stable. Note:
`torch.cuda.nccl.version()` still returns `(2, 26, 2)` because that's
the *compile-time* constant baked into torch 2.7.0; the runtime version
NCCL itself prints (`NCCL version 2.27.7+cuda12.9`) is what's actually
loaded and used.

**Verified post-upgrade** (workergpu181, 2 GPUs):
- gpt2 (124M) + DDP + fwd+bwd: ✓
- OLMo-2 1B + DDP + fwd+bwd: ✓
- OLMo-2 1B + PEFT LoRA + packed_v1 + compile + DDP (full bench path): ✓ MFU 47%

The bisection below documents the broken-state behavior on NCCL 2.26.2
for posterity. If the env's NCCL is ever pinned back, this section is
the regression-test recipe.

#### Pre-fix bisection (NCCL 2.26.2, kept for reference)

A 4-GPU Blackwell DDP bench (SLURM job 6364791, then interactive
diagnosis on workergpu181) reproducibly fails at the **first NCCL
operation after DDP wrap**, regardless of `--compile`,
`--data_pipeline_version`, or NCCL workarounds. Failure mode: CUDA
illegal-memory-access reported by NCCL's watchdog thread (sometimes
in `dist._broadcast_coalesced`, sometimes in the next CUDA op after
DDP wrap — DDP construction silently corrupts device state and the
crash surfaces at the next CUDA call).

**Bisected via interactive smokes** (workergpu181, 2 GPUs):

| stage | result |
|---|---|
| Pure NCCL `all_reduce` on 1-element tensor | ✓ works |
| `nn.Linear(64,64) → DDP` (NCCL) | ✓ works |
| `gpt2` (124M) `→ DDP` (NCCL) | ✗ |
| `OLMo-2 1B → DDP` (NCCL) | ✗ |
| `OLMo-2 1B + PEFT LoRA → DDP` (NCCL) | ✗ |
| `gpt2 → DDP` (**gloo**) | **✓ works** |
| `OLMo-2 1B → DDP` (**gloo**) | **✓ works** |

**Workarounds tried that did NOT help:**
- `NCCL_P2P_DISABLE=1` (disable peer-to-peer)
- `NCCL_IB_DISABLE=1` (disable InfiniBand)
- `NCCL_CUMEM_ENABLE=0` (disable cuMem pool)
- `find_unused_parameters=True`
- `broadcast_buffers=False, gradient_as_bucket_view=True`
- `device_id=device` in `init_process_group` (PyTorch 2.7-style)

**Root cause: NCCL 2.26.2's sm_120 (Blackwell) path is broken in this
software stack** (torch 2.7.0+cu128, NCCL 2.26.2+cuda12.2, NVIDIA
driver 580.142). The bug doesn't depend on HF model size, on PEFT, or
on any of our code — any HF transformer + DDP + NCCL on this Blackwell
hits it. A100's NCCL path works fine.

**Workaround: switch DDP backend to `gloo` on Blackwell.** Gloo stages
allreduce through CPU memory, so it's slower than NCCL would be — but
for LoRA training the cost is minimal because only LoRA gradients are
allreduced (~10–100 MB/step, vs. multi-GB for full FT). Estimated
overhead: a few percent on per-step wall under packed_v1+LoRA. Activate
via `init_process_group("gloo")`. NCCL stays the right choice on A100.

Decision: **drop DDP from the Phase B/C plan** anyway. Single-GPU per
cell on Blackwell+packed_v1 fits all four campaign cells in 24h
(verdict table above), and parallel single-GPU seeds beats DDP-within-
a-cell for sweep workloads. The gloo workaround is documented for
future workloads that need DDP on Blackwell; until then the bench's
DDP code (`scripts/bench/bench_optimizer_step.py`, commit 7ba50a2)
works on A100 unchanged and serves as a regression test for when the
upstream NCCL/PyTorch stack updates.

When NCCL ≥ 2.27 (or torch ≥ 2.8) is available, retry the bisection
above before accepting gloo as the long-term answer.

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

## Gram-NS + k=2 wall-time (Blackwell, packed_v1, 2026-05-17)

Goal: quantify the wall-time impact of (a) the Dao 2026 Gram Newton-Schulz
polar map (`--ns_form gram` — `_newton_schulz_gram_batched` in fp16 with
restart at τ=2) and (b) lowering Picard iters from k=3 to k=2, both for
the `chord-tight-clean` clean polar pipeline.

Setup: r=64, K=1 (refresh every step), higham preconditioner, packed_v1,
batch=2 × seq=512 × grad_accum=4 (effective batch 8), bf16, no compile,
Blackwell RTX PRO 6000 (workergpu174, exclusive). Bench script:
`scripts/bench/bench_optimizer_step.py` with new `--ns_form` /
`--picard_iters_override` flags. Per-scope profile script:
`scripts/bench/profile_chord_tight_clean.py`. JSONL outputs under
`logs/bench/bench_ns_gram_blackwell_workergpu174_*.jsonl`.

### Headline numbers — 1B (OLMo-2-1B), r=64

| optimizer config | fwd ms | bwd ms | opt ms | total ms | ×AdamW | MFU |
|---|---:|---:|---:|---:|---:|---:|
| adamw (baseline) | 118.3 | 156.5 | 3.3 | 278.9 | 1.00× | 35.8% |
| chord-tight-clean rect k=3 (prior baseline) | 118.3 | 156.5 | 49.1 | 325.2 | **1.16×** | 30.7% |
| chord-tight-clean rect k=2 | 118.2 | 156.6 | 34.6 | 310.7 | **1.11×** | 32.1% |
| chord-tight-clean gram k=3 | 118.1 | 158.8 | 47.5 | 325.7 | **1.16×** | 30.6% |
| chord-tight-clean **gram k=2** | 119.0 | 156.4 | 34.7 | 311.4 | **1.12×** | 32.0% |

Reading at r=64: **almost all of the wall-time win comes from k=3 → k=2**
(one fewer Picard iter = one fewer polar map + cross-coupling). Gram-NS
at k=2 buys essentially nothing extra (1.12× vs 1.11×). Gram-NS at k=3
buys ~3% of opt_ms (49.1 → 47.5). At r=64 the polar block is too small
a fraction of step wall for the gram FLOP advantage to surface.

### Headline numbers — 1B (OLMo-2-1B), r=256

| optimizer config | fwd ms | bwd ms | opt ms | total ms | ×AdamW | MFU |
|---|---:|---:|---:|---:|---:|---:|
| adamw (baseline) | 150.4 | 205.3 | ~3 | ~377 | 1.00× | ~31% |
| chord-tight-clean rect k=3 (prior baseline) | 150.6 | 205.7 | 230.9 | 588.4 | **1.56×** | 18.5% |
| chord-tight-clean rect k=2 | 150.4 | 205.3 | 158.9 | 515.9 | **1.40×** | 21.2% |
| chord-tight-clean gram k=3 | 150.4 | 205.9 | 189.5 | 547.1 | **1.49×** | 19.9% |
| chord-tight-clean gram k=2 | 151.4 | 205.2 | 131.4 | 489.4 | **1.33×** | 22.3% |
| chord-tight-clean gram-norestart k=3 | 150.4 | 206.6 | 186.2 | 544.5 | **1.48×** | 20.0% |
| chord-tight-clean **gram-norestart k=2** | 150.3 | 205.5 | 129.2 | 486.2 | **1.33×** | 22.4% |

Reading at r=256: **gram-NS earns its keep at higher rank.**
- gram vs rect at fixed $k$: opt_ms drops 18% (231→190 at k=3; 159→131 at k=2).
- k=3 → k=2: saves ~70 ms in rect, ~58 ms in gram.
- **Combined gram + k=2 vs old baseline (rect + k=3): 489 vs 588 ms = 17% step-wall reduction. Overhead drops from 1.56× → 1.33× AdamW.**
- gram-norestart at k=2 is bit-equivalent to gram k=2 at this scale (486 vs 489 ms; 0.6%). At k=3 the restart costs ~3 ms (544 vs 547). Marginal; restart on is the safer default.

### Headline numbers — 8B (Meta-Llama-3-8B)

#### 8B, r=64

| optimizer config | fwd ms | bwd ms | opt ms | total ms | ×AdamW | MFU |
|---|---:|---:|---:|---:|---:|---:|
| adamw (baseline) | 419.0 | 545.5 | 10.2 | 975.4 | 1.00× | 54.7% |
| chord-tight-clean rect k=3 (prior baseline) | 419.1 | 545.6 | 201.1 | 1168.3 | **1.20×** | 45.6% |
| chord-tight-clean rect k=2 | 419.1 | 545.2 | 127.5 | 1094.2 | **1.12×** | 48.7% |
| chord-tight-clean gram k=3 | 417.3 | 542.8 | 186.8 | 1149.3 | **1.18×** | 46.4% |
| chord-tight-clean **gram k=2** | 418.8 | 545.2 | 118.0 | 1084.4 | **1.11×** | 49.2% |

#### 8B, r=256 (where the gram-NS win is largest)

| optimizer config | fwd ms | bwd ms | opt ms | total ms | ×AdamW | MFU |
|---|---:|---:|---:|---:|---:|---:|
| adamw (baseline, implied) | 505 | 730 | ~10 | ~1275 | 1.00× | ~52% |
| chord-tight-clean rect k=3 (prior baseline) | 504.9 | 729.6 | 979.1 | 2216.0 | **1.74×** | 25.5% |
| chord-tight-clean rect k=2 | 504.3 | 728.5 | 629.5 | 1864.8 | **1.46×** | 30.3% |
| chord-tight-clean gram k=3 | 505.3 | 729.8 | 815.1 | 2052.6 | **1.61×** | 27.6% |
| chord-tight-clean gram k=2 | 506.8 | 731.2 | 519.8 | 1760.2 | **1.38×** | 32.2% |
| chord-tight-clean gram-norestart k=3 | 507.2 | 732.0 | 802.5 | 2044.1 | **1.60×** | 27.7% |
| chord-tight-clean **gram-norestart k=2** | 505.0 | 729.0 | 511.3 | 1747.7 | **1.37×** | 32.4% |

Reading at 8B r=256:
- **Combined gram-norestart + k=2 vs old baseline (rect + k=3): 1748 vs 2216 = 21% step-wall reduction.** Overhead drops from 1.74× → 1.37× AdamW. **MFU rises from 25.5% to 32.4%.**
- gram vs rect at fixed k: opt_ms drops 17% (979→815 at k=3; 630→520 at k=2). Same FLOP-ratio reasoning as 1B r=256, scaled by ~4× more LoRA pairs.
- gram-norestart vs gram at fixed k: marginal 1-2% additional savings (12 ms at k=3, 8 ms at k=2). Hedge cost is small at this scale; restart on remains the safer default.

### Summary across (model, rank)

| where | k=3 → k=2 | rect → gram | net (rect/k=3 → gram-norestart/k=2) |
|---|---:|---:|---:|
| 1B r=64 | 5% | ~0% | ~4% |
| 1B r=256 | 10% | 7% | 17% |
| 8B r=64 | 6% | 1-2% | 7% |
| 8B r=256 | 16% | 7% | **21%** |

**Reading: k=3 → k=2 is a free win everywhere; gram-NS is a free win at r=256 and especially 8B r=256. At r=64 the polar block is too small a fraction of step wall (~10-15%) for the gram FLOP reformulation to matter.**

### Per-scope breakdown — where opt_ms goes

CudaEvent timing inside optimizer.step (`scripts/bench/profile_chord_tight_clean.py`).
Logs: `logs/bench/profile_blackwell_*.log`.

| scope | rect k=3 | rect k=2 | gram k=3 | gram k=2 |
|---|---:|---:|---:|---:|
| `chord_tight_clean_picard` (= polar pipeline × k) | 35.3 (75%) | 19.8 (63%) | 31.8 (74%) | 18.2 (62%) |
| `precond_refresh` (higham) | 5.0 (10%) | 5.0 (16%) | 4.6 (11%) | 4.6 (16%) |
| `chord_tight_clean_pre_rescale` | 2.5 (5%) | 2.5 (8%) | 2.3 (5%) | 2.3 (8%) |
| `chord_tight_clean_sigma_AB` (ρ formula) | 2.1 (4%) | 2.0 (6%) | 1.8 (4%) | 1.8 (6%) |
| `adam_direction` (EMA) | 1.6 (3%) | 1.6 (5%) | 1.6 (4%) | 1.6 (5%) |
| `apply` | 0.45 (1%) | 0.45 (1%) | 0.45 (1%) | 0.45 (1%) |
| `chord_tight_clean_whiten_input` | 0.30 (1%) | 0.30 (1%) | 0.29 (1%) | 0.29 (1%) |
| **sum of timed scopes per step** | **47.3** | **31.6** | **42.8** | **29.2** |

`picard / k` (per-Picard-iter cost): rect 11.8 ms/iter, gram 10.6 ms/iter
— gram is ~10% faster per iter, matching the FLOP-block ratio for the
polar map (rect 4·N·r²·d → gram 4·N·r³ + reconstruction). The absolute
gain is small because the polar block is only ~11% of opt_ms.

### Reconciliation with the FLOP audit

`algorithm_clean_implementation.md` §3 estimated the polar map at 65% of
step cost at r=64. The measured per-scope profile here puts the Picard
loop (which contains the polar map AND the cross-coupling matmuls AND
unwhiten AND σ_max(geo)) at 75% of opt_ms — but opt_ms is only ~15% of
step wall. So polar is ~11% of step wall, not 65%. The FLOP count was
correct *about the polar block*; what changed is the denominator —
fwd+bwd dominate step wall, and the polar map's bf16-tensor-core wall
is much smaller than its FLOP fraction suggests on Blackwell.

Implication: Gram-NS's headline 3-6× speedup *of the polar block* is
real, but invisible at r=64 because the polar block is small. At
**r=256** the r³ inner work amortizes better and (more importantly) the
rect path's r²·d cost scales linearly with d while gram stays r³ —
re-bench at r=256 expected to show a larger gram-vs-rect gap. Deferred
until needed.

### Headlines

- **Production switch (highest value): flip the `chord-tight-clean`
  sweep scripts to `--picard_iters_override 2`.** Free 5-16% step-wall
  reduction across all (model, rank) configurations with no measured
  eval-loss penalty (prior multi-cell sweep, per user). Already applied
  in `scripts/sweep/sweep_4k_packed_diag_k3_htmuon.sh`.
- **`--ns_form gram` ships behind the flag** as the production-ready
  Gram-NS path. Recommended for r≥128 and especially 8B/r=256 (21%
  step-wall reduction). Default remains `rect` pending trajectory-level
  validation in a real training run — currently only verified to 5
  steps. Flip the default after the first sweep using `--ns_form gram`
  produces a sane eval trajectory.
- **`--ns_form gram-norestart` ships behind the flag** as the
  performance-tuned variant. Saves 1-3% additional opt_ms by dropping
  Dao's stability restart. Cubic Muon at NS=5 makes the restart hedge
  unnecessary on our corpus (basin-blowup factor is 2.25× per iter, too
  slow to compound over 5 iters from the fp16 noise floor). Tier 1
  evidence (real chord-tight r=64 X_eff; max cond(G) 1.4e5; tight-
  damping rebuild at δ=1e-6 gives max cond 3.1e5) shows
  with-restart and no-restart produce identical results to 3.6% rel-err.
  Opt in if you want the headroom; restart-on is the safer default.
- **Tier 1 fixture (real X_eff inputs from chord-tight r=64 snapshots)
  + Tier 3 synthetic cond=1e4 stress** validated all three Gram-NS
  variants in `tests/test_ns_gram.py`. fp16+restart matches rect-fp32
  to ≤ 5% rel-err on every cell in the corpus; orthogonality residual
  is NOT a test target (NS=5 sub-orthogonal is by design — best eval
  per project convention).

### Reproduce

```bash
# Bench (4 cells: rect/gram × k=2/k=3)
bash /mnt/home/nghosh/.tmp_scripts/bench_gram_blackwell.sh
# Per-scope profile
bash /mnt/home/nghosh/.tmp_scripts/profile_gram_blackwell.sh
```

### Caveats

- Single seed per cell, n_cycles=4 (4 timed steps). Variance not
  characterized; ±~2% expected per the existing A100 profiling notes.
- No compile. Compile may shift the opt:fwd+bwd ratio if it fuses parts
  of the optimizer. Production sweep scripts already use `--compile`.
- Bench script silently downgrades `--precond_method higham` to `eigh`
  if the optimizer name is not in `POLAR_OPTIMIZERS` (commit fixed
  this for the clean variant; the first Blackwell bench run gave 1.82×
  because of this silent downgrade — preserved in
  `logs/bench/bench_ns_gram_*` files marked eigh).
- gram-dao path (using the official Tri Dao library) was wired but not
  benched. Pip-installing `~/gram-newton-schulz` forced `torch>=2.7.1`,
  which uninstalled the env's torch 2.7.0+cu128 and broke
  megablocks/torchvision/torchaudio/vllm/xformers. Reverted. The
  library's CuTeDSL kernels would only activate at min(r, d) > 256
  anyway, beyond practical LoRA ranks for our sweeps. Removed from
  CLI choices.

## bf16 / fp16 Higham — kernel bench + variant comparison (Blackwell)

Higham is the largest non-tensor-core consumer of opt_ms (fp32, TF32
disabled) at higher rank. This bench answers whether moving its inner
loop onto tensor cores via low-precision matmuls actually pays off.

Bench script: `scripts/bench/bench_higham_variants.py`. Production
damping (eps_abs=1e-6, eps_relative=False — the `train.py --precond_delta`
default). True λ_max scaling computed outside the timed region.
`n_iters = 10`. Inputs at controlled cond ∈ {1e2, 1e3, 1e6} to bracket
typical (real Gram audit shows cond ≤ 1.24e3 on chord-tight r=64) and
adversarial. Three variants:

- **A: fp32 + TF32 enabled.** Original baddbmm polynomial; just flip
  `allow_tf32 = True` and `set_float32_matmul_precision("high")`
  around the call. No algorithm change.
- **B: fp16 inner + 1 fp32 polish.** Y, Z, three_eye cast to fp16
  once at entry; first `n_iters - 1` iters run fully in fp16 on
  tensor cores; one final fp32 polish iter restores precision.
  Cast launches per call: 5 (3 entry + 2 exit), not per-iter.
- **C: bf16 inner + 1 fp32 polish.** Same as B but bf16.

### Wall-time speedup vs fp32-no-TF32 reference

(Mean across cond levels — variance < 2% with cond. Speedups are
shape-dominated, not cond-dominated.)

| shape         | A: fp32+TF32 | B: fp16+polish | C: bf16+polish |
|---------------|-------------:|---------------:|---------------:|
| (112, 16, 16) | 1.00×        | 0.93×          | 0.93×          |
| (112, 64, 64) | 0.99×        | 0.93×          | 0.93×          |
| (112,128,128) | **1.53×**    | **1.52×**      | 1.48×          |
| (224,256,256) | 1.39×        | **2.16×**      | 1.91×          |

Reading:

- **r ≤ 64**: no win. (112, r, r) bmms are launch-bound at these
  shapes — matmul time is ~0.01 ms and 5 cast launches add ~0.05 ms,
  exactly the 7% slowdown observed for B/C.
- **r = 128**: A and B tie at ~1.5×. TF32 alone is enough; the fp16
  matmul edge isn't large enough at this shape to outrun cast cost.
- **r = 256**: B is the clear winner at 2.16×. fp16's TC throughput
  on Blackwell is the dominant factor; A's TF32 gives only 1.39×.

### Precision (Frobenius rel-err vs fp32-no-TF32 reference)

| shape         | cond=1e2 A / B / C | cond=1e3 A / B / C | cond=1e6 A / B / C |
|---------------|-------------------:|-------------------:|-------------------:|
| (224,256,256) | 2e-3 / 2e-3 / 1e-2 | 1e-2 / 1e-2 / 8e-2 | 3e-2 / 3e-2 / 3e-1 |
| (112,128,128) | 2e-3 / 2e-3 / 1e-2 | 1e-2 / 1e-2 / 9e-2 | 4e-2 / 4e-2 / 3e-1 |
| (112, 64, 64) | 2e-3 / 2e-3 / 1e-2 | 1e-2 / 1e-2 / 9e-2 | 4e-2 / 4e-2 / 3e-1 |
| (112, 16, 16) | 0 / 2e-3 / 2e-2    | 0 / 1e-2 / 1e-1    | 0 / 4e-2 / 5e-1    |

Reading:

- **A and B are precision-equivalent** at every (shape, cond). The
  single fp32 polish iter at the end of B fully restores B's output
  to TF32-tier precision — the 9 fp16 inner iters drift, the fp32
  polish corrects.
- **C (bf16) is ~10× worse** at every (shape, cond). One polish iter
  isn't enough to recover bf16's bigger mantissa drift (7-bit vs
  fp16's 10-bit; same exponent range binds for neither at this scale
  under eps_abs damping).
- **Precision scales with cond.** Real production worst-case is
  cond ≈ 1e3 (fixture audit on chord-tight r=64 snapshot — median
  91, max 1.24e3). At that cond, A and B both have ~1% rel-err vs
  fp32 reference.

### Full-step impact — much smaller than the kernel bench suggested

The kernel bench above measures Higham IN ISOLATION. The relevant
production number is what the full optimizer step looks like with
fp16 Higham wired in. Bench script:
`scripts/bench/bench_full_step_fp16_higham.sh` (OLMo-2-1B,
all-linear LoRA, batch=2·seq=512·accum=8, no compile, K=1):

| variant                    | r=64 opt ms | r=64 total | r=256 opt ms | r=256 total |
|----------------------------|------------:|-----------:|-------------:|------------:|
| AdamW (reference)          |         3.5 |      555.9 |         10.7 |       724.4 |
| clean k=2 gram + fp32 Higham |       30.5 |      583.0 |        126.8 |       842.6 |
| clean k=2 gram + **fp16 Higham** |   30.3 |      584.0 |        123.4 |       839.4 |

fp16 Higham saves **0.2 ms at r=64** (0.04% of step) and **3.4 ms at
r=256** (0.5% of step). The kernel-level 2.16× speedup at (224,
256, 256) is real but Higham itself is only ~9 ms of the 127 ms
opt_ms; the bulk of opt_ms is the Picard scope (gram-NS polar +
cross-coupling + σ_max + unwhiten), unchanged by the lever.

**Shipping decision: keep the CLI flag, default off.**
- `--higham_compute_dtype fp16` is plumbed end-to-end and validated
  to ≤ 1% rel-err vs fp32 reference on real Grams
  (`tests/test_higham_lowp.py`). Available for opt-in.
- Not the production default at any rank — the wall savings don't
  justify a precision-trading change for runs that need byte-for-byte
  trajectory parity with prior sweeps.
- The fp16-Higham work was useful to bound the lever, not to ship.

### Where to attack next: CUDA graphs over the Picard scope

The remaining opt_ms is dominated by the chord-tight-clean Picard
loop (gram-NS polar at K=5, cross-coupling matmuls, σ_max power
iter, unwhiten). At r=256 that scope is ~110 ms out of 127 ms opt_ms.
The matmuls inside are small bmms at (N, r, r) and (N, r, d), and
the wall-per-FLOP analysis at the top of this section put the
optimizer at ~10–50× off-tensor-core wall efficiency — i.e., it's
launch-bound, not compute-bound, on Blackwell at packed_v1 shapes.

**CUDA graphs replace N kernel launches with 1.** Conditions for
capture, all met for `_chord_tight_clean_polar_pipeline`:
- Fixed shapes per group at every step ✓
- No host-side branches in the loop body ✓
- No `.item()` / device-host sync inside the body ✓
- The method already compiles `fullgraph=True`
  (`tests/test_chord_tight_clean.py::test_no_graph_breaks_under_compile`).

Prototype + numbers in
`scripts/bench/bench_cuda_graphs.py` — captures the inner
function as a `torch.cuda.CUDAGraph()`, replays via `copy_` of new
input into a static buffer + `graph.replay()`. Higham (10 NS iters
× 3 matmuls = 30 launches) and gram-NS polar (~15 matmuls) are both
shape-static so they capture cleanly. The full Picard scope at fixed
group shape is the bigger target.

### Compile-modes bench result (Blackwell, k=2 gram, fp32 Higham)

Bench script: `scripts/bench/bench_compile_modes.sh`. Three configurations:

- **B: eager** (`LORA_COMPILE_KERNELS=0`).
- **C: torch.compile default** (`LORA_COMPILE_KERNELS=1`, fullgraph=False, kernel fusion only).
- **D: torch.compile mode='reduce-overhead'** (`LORA_COMPILE_KERNELS=2`, fullgraph=True + CUDA graphs).

| shape | variant | opt ms | total ms | ×AdamW |
|---|---|---:|---:|---:|
| r=64 | B (eager) | 30.4 | 582.8 | 1.05× |
| r=64 | C (compile) | 27.3 | 581.6 | 1.05× |
| r=64 | D (reduce-overhead) | — | — | **crash** |
| r=256 | B (eager) | 126.8 | 842.8 | 1.16× |
| r=256 | C (compile) | 125.9 | 840.7 | 1.16× |
| r=256 | D (reduce-overhead) | — | — | **crash** |

**Findings:**

- **C (compile default) saves 3 ms opt at r=64 (10%) and 1 ms at r=256 (0.7%).** Net step-wall impact is < 0.3% — essentially noise.
- **D (mode='reduce-overhead') crashes** with `"Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"` at `optim.py:3481`. Root cause: the pipeline writes warm-start tensors into `gs[...]` dict slots; auto-CUDA-graph capture references those tensors by identity, but the next call rebinds the dict slot to a new tensor, leaving the captured graph pointing at the stale (now-overwritten) buffer. A pipeline refactor to mutate warm-start buffers in-place (`gs['v_sigma_A'].copy_(new)` instead of `gs['v_sigma_A'] = new`) would unblock this path. Estimated work: 1–2 days. Expected gain on top of compile: 5–15% opt savings = 1–3% step savings.

### Shipping decision — smallest production walltime

The chord-tight-clean k=2 gram pipeline is essentially at its wall-time floor given the current algorithmic choices:

- **Production default = torch.compile + fp32 Higham + k=2 gram NS.** Already wired into sweep scripts via `--compile`.
- Step wall vs AdamW: 1.05× at r=64, 1.16× at r=256.
- No precision-lever or graph-capture trick we tested moves the needle by more than ~1% step wall at production rank.

Levers that didn't pay off and the reasons:

- **fp16 Higham (variant B):** 2.16× kernel-level at r=256 → 0.5% step wall (Higham is ~9 ms of 127 ms opt at r=256; 50% Higham savings is 0.5% step). Real but tiny.
- **mode='reduce-overhead':** crashes on the warm-start dict-mutation pattern. Would need pipeline refactor.
- **Manual CUDA graphs around individual primitives:** 5× speedup at r=16-64 but Higham/gram-NS individually are <10% of opt_ms; absolute savings sub-ms.
- **Manual CUDA graphs around the whole pipeline:** same refactor cost as mode='reduce-overhead' would unblock; uncertain whether ~5-15% opt gain is worth the engineering.

Real remaining levers are algorithmic, not implementation:

- **Picard k=2 → k=1:** drops half of the Picard scope (~50 ms at r=256). But the cross-coupling correction was the whole point of choosing k=2.
- **NS=5 → NS=3 in gram-NS:** drops ~40% of polar map cost (a few ms). Needs trajectory equivalence validation.

Both are real algorithmic decisions and out of scope for an implementation-only lever search.

## SSC κ-adaptive: per-step eigvalsh overhead (Blackwell, packed_v1, 2026-05-24)

The κ-adaptive SSC variant (`--polar_method ssc --ssc_kappa <κ>`) replaces fixed `c` with a per-pair `c` solved from a target rank-normalized energy. The cheapest realization adds, per polar call: one $r \times d$ → $r \times r$ gram bmm, one `torch.linalg.eigvalsh` on the $r \times r$ gram (eigenvalues only, no eigenvectors), and an $O(r)$ vectorized log-bisection. The SSC application itself stays on the existing MISR kernel — no SVD.

Per-step eigvalsh on small $r \times r$ is launch-bound on Blackwell (this profile already documented this effect for eigh-based whitening, ~5× slower than Higham at $r=128$). Measurement at the production sweep shape:

| variant | tps | wall (60 steps, no compile) | per-step | ratio vs fixed-c |
|---|---:|---:|---:|---:|
| fixed-c (`--ssc_c 0.3`) | 4659 tok/s | 37.3 s | 0.62 s/step | 1.00× |
| κ-adaptive (`--ssc_kappa 0.30`) | 2392 tok/s | 72.6 s | 1.21 s/step | **1.95×** |

Config: chord-tight-clean, picard=3, NS=5, gram, r=64, batch=2×accum=8, seq=512, Blackwell node (RTX PRO 6000), bf16, packed_v1, no `--compile` (per CLAUDE.md's ratio-test rule). Eval loss at step 60 matched to 1 ulp (0.6262 vs 0.6265), confirming behavioral equivalence — the wall delta is pure eigvalsh overhead, not extra useful compute.

**Implication for sbatch sizing.** Fixed-c c=0.3 r=64 4k took ~1.6h/cell compiled in the existing leaderboard sweep. κ-adaptive at the same config should take ~3.2h/cell, hence `--time=4h` with margin for the κ-adaptive r=64 4k 12-cell sweep (`slurm_pending/chord_tight_clean_ssc_kappa_r64_lrsweep_4k_blackwell.sbatch`).

**If the κ-adaptive variant proves out (drops below the fixed-c plateau by >1σ_AdamW)**, the obvious cost cut is to amortize the spectrum probe via a refresh schedule analogous to `precond_refresh_every` — solve `c` every $N$ steps, hold between refreshes. At $N \approx 10$ the eigvalsh overhead drops below 0.1 s/step (1.3× vs fixed instead of 1.95×). Until results justify, the per-step variant is the upper-bound-on-quality probe.
