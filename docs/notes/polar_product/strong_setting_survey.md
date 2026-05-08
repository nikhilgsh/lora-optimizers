# Strong-setting LoRA experimental protocol — survey synthesis

This is the one-page synthesis of three current-best LoRA references and
how the repo's protocol stacks up. It exists to ground the choices made
in the tight-chord scale-up campaign (`tight_chord_paper_plan.md`); when
a future reader asks "why did we do X," this is the first stop.

## Sources

- **Biderman et al. 2024**, "LoRA Learns Less and Forgets Less," TMLR.
  arXiv [2405.09673](../../papers/biderman_2405.09673.pdf).
- **Schulman 2025**, "LoRA Without Regret," Thinking Machines blog.
  Digest: [`schulman_lora_without_regret_2025.md`](../../papers/schulman_lora_without_regret_2025.md).
- **Chen, Villar, Hayou 2026**, "Learning Rate Scaling across LoRA Ranks
  and Transfer to Full Finetuning" (μA framework).
  arXiv [2602.06204](../../papers/mua_2602.06204.pdf).

## Convergent best-practices table

| Knob | Biderman | Schulman | Hayou (μA) | Repo today (audited) |
|---|---|---|---|---|
| Base scale | Llama-2-7B | Llama-3-8B / Qwen3 family | 1B–8B mix | OLMo-2-1B (will scale 1B→3B→8B) |
| Target modules | All transformer modules | All-linear (MLP critical) | per-experiment, broad | **`all-linear`** ✓ (`train.py:220`) |
| α convention | α=2r | α=32 fixed | α∈{1, r⁻¹/², r⁻¹} per study | **α=r** (every launcher: `--lora_alpha "$lora_r"`) |
| Rank coverage | 16 / 64 / 256 | 1 → 512 | 4 → 1024 | 16/64/128/256 today; will extend to 512 at 1B/3B |
| LR schedule | cosine + warmup | **constant, no warmup** | linear-warmup-5% + cosine to 0.1× | **constant, no warmup** ✓ (`train.py:211–212`) |
| Pass count | multi-epoch (1–16) | **1-pass** on big data | **1-pass** on big data | **1-pass enforced** (`train.py:393`) |
| Dataset size | Magicoder-Evol-110K | Tulu3 ~939K / OpenThoughts3 | Tulu3 / OT-114K | Magicoder-OSS-75K (32K subset for 2k-step invariant) |
| LR sweep | mandatory log-grid | mandatory | mandatory log₂ grid | yes |
| Seeds | mostly single | single (smoothed) | single per cell | single + AdamW noise-floor multiseed |
| Eval | HumanEval / GSM8K | held-out + reward | held-out + task | held-out NLL only (HumanEval to be added) |
| Horizon | up to ~1.2B tok | to convergence | to convergence | 32M tok at 4k/seq=512; 200M tok at 8B/seq=2048 |

## What we lock in (from convergence)

1. **`target_modules = all-linear`.** All three sources align on
   "include MLP." Schulman is most emphatic: MLP-only beats attn-only
   even at matched parameter count.
2. **constant LR, no warmup, no cooldown.** Schulman's recipe; the
   simplest one to reason about across optimizers; matches repo today.
3. **1-pass, scale-via-data not via epochs.** Schulman + Hayou both
   use 1-pass on large datasets. Multi-epoch under constant LR diverges
   (repo invariant). Multi-epoch under cosine introduces a schedule HP
   that interacts with optimizer choice — confound we don't need.
4. **HumanEval pass@1** as the downstream metric. Biderman's tool
   choice; performance-only target; cleanly comparable to his published
   numbers when we hit Llama-3.1-8B.

## What we deliberately diverge on

- **α=r, not α=2r or α=32.** Biderman's "α=2r is crucial" is one
  empirical sweep at r=256; the LoRA update is `(α/r)·BA`, so changing
  α/r is mathematically equivalent to changing the LR by the same
  factor up to a constant. A properly-resolved log₂ LR sweep absorbs
  the difference at any single cell. The η_opt(r) shape differs, which
  we will report. The project's α=r convention gives constant ratio
  α/r=1 — well-supported, not a question worth re-litigating.
- **AdamW with default ε=1e-8, not Adam ε=0.** Schulman's ε=0 is for
  theoretical invariance analysis (the LoRA update is scale-invariant
  in g/√v with ε=0); empirically ε=1e-8 vs ε=0 is negligible on
  real-magnitude gradients, and Schulman's own canonical vLLM repro
  uses AdamW.
- **Code-only domain, not Schulman's instruction/math mix.** Schulman
  has no code dataset; what we run is "Schulman recipe applied to code
  IFT," with Biderman as the external anchor for HumanEval at 7B/8B
  scale.

## Token-horizon scaling (1-pass, code-only)

| Base | Dataset | seq | batch_eff | steps | tokens (1-pass) |
|---|---|---:|---:|---:|---:|
| OLMo-2-1B | Magicoder-OSS-75K (32K subset) | 512 | 16 | 4000 | 32M |
| Llama-3.2-3B | Magicoder-Evol-Instruct-110K | 1024 | 16 | 6000 | 98M |
| Llama-3.1-8B | Magicoder-Evol-Instruct-110K | 2048 | 16 (μ=1×accum=16) | 6000 | 200M |

Schulman's SFT-canonical (Qwen3-4B on no_robots, vLLM repro) is
batch_eff=32 × 200 steps × seq=2048 ≈ 13M tokens. Our 1B alone exceeds
that horizon; 8B is ~15× longer.

## Expected η_opt values

- LoRA optimal LR ≈ 10× FullFT LR (Schulman, 14 Llama/Qwen models).
- For α=r: η_opt typically 1e-4 to 1e-3 for AdamW-LoRA at our scale
  (current best η=3e-4 at r=16/64 from existing sweeps).
- tight-chord's natural η scale is 10–30× higher than AdamW-LoRA's
  (current best η=1e-2 at r=128, 4k steps).
- Hayou μA prediction (Init[A], α=1): η ∝ r⁻¹/². Under α=r the
  prediction differs; one of the diagnostic outputs of Phase A is
  whether this scaling holds empirically for our optimizers.

## Anchors for cross-checking

- Biderman r=64 IFT 4 epochs on Llama-2-7B Magicoder: HumanEval ≈ 0.417.
  Phase C (Llama-3.1-8B + Magicoder-Evol-110K, 1-pass, r=64) should
  land in the same neighborhood or above.
- AdamW noise-floor σ ≈ 0.001 from `logs/adamw_multiseed/` — use as the
  σ-units anchor for all Δ reporting until/unless we measure σ at a new
  base.
