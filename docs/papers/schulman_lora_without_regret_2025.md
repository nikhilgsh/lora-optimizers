# LoRA Without Regret — Schulman et al. 2025

**URL:** https://thinkingmachines.ai/blog/lora/
**Date:** 2025-09-29
**Org:** Thinking Machines Lab — Connectionism (blog post)
**DOI:** 10.64434/tml.20250929

The blog is HTML, not PDF — this is a project digest, not a copy. Re-read
the source for any claim that this digest is going to be load-bearing on.

## Thesis

LoRA can match FullFT in sample efficiency and final loss in the
**post-training** regime when five details are right. Outside that
regime (pre-training-like, data ≫ adapter capacity) LoRA still
underperforms.

## The five "key details"

1. **Target modules.** Apply LoRA to all weight matrices, especially
   MLP and MoE layers. Attention-only LoRA at r=256 (0.25B params)
   underperforms MLP-only at r=128 (0.24B params) at *matched* parameter
   count on Llama-3.1-8B (Tulu3, OpenThoughts3). Pattern holds for
   Qwen3-30B-A3B-Base MoE.
2. **Rank vs dataset capacity.** Curves plateau when adapter capacity
   runs out at ~1 bit/token (SFT) or ~1 bit/episode (RL). RL matches
   FullFT at ranks as low as r=1 because the per-episode information
   is small.
3. **LR scaling.** Optimal LoRA LR ≈ **10× FullFT LR** across 14 Llama
   and Qwen models tested, both SFT and RL. ~15× for short runs
   (~100 steps), converging to 10× longer. Empirical, not derived.
4. **Batch-size penalty.** LoRA pays a larger loss penalty than FullFT
   at large batch sizes. The gap is **not mitigated by raising r** —
   property of the BA parametrization.
5. **A/B init asymmetry.** A ~ Uniform(±1/√d_in), B = 0. B's spectral
   norm grows past A's during training, giving an implicit LR schedule.

## Optimizer recipe (as written in the blog)

- **Adam, ε=0.** Reason given: theoretical invariance — with ε=0 the
  Adam update is scale-invariant in the gradient, which the blog uses
  to derive the (α, init, LR) invariance group. Empirically ε=1e-8
  vs ε=0 is negligible on real-magnitude gradients, and the canonical
  vLLM repro uses AdamW with default ε. Treat ε=0 as a theoretical
  device, not a practical recommendation.
- **α = 32 fixed across all ranks.** The LoRA update is `(α/r)·BA`,
  so α/r ratio shrinks with r (32 at r=1, 0.0625 at r=512). Different
  from α=r (constant ratio) or α=2r (Biderman) conventions.
- **Constant LR, no warmup, no cooldown.**
- **One epoch (single pass)** on the full training set.

## Datasets and models

| Domain | Dataset | Models |
|---|---|---|
| SFT (instruction) | Tulu3 (~939K) | Llama-3.1-8B, Qwen3-8B, Qwen3-30B-A3B-Base |
| SFT (reasoning)   | OpenThoughts3 (full + 10K subset for batch sweep) | Llama-3.1-8B |
| RL (math)         | MATH, GSM8K, DeepMath-103K (seq 8192) | Llama-3.1-8B, Qwen3-8B-Base |

**No code dataset in the blog.** HumanEval / MagiCoder / StarCoder
are not used. Code is not Schulman's experimental domain.

Rank range tested: **1–512.** RL: r=1 already matches FullFT.

Batch-size sensitivity sweep: 32, 64, 128, 256 on a 10K-example
OpenThoughts3 subset.

## Caveats (from the post)

- Excludes pre-training-like regimes; matching only holds when
  dataset size is moderate relative to adapter capacity.
- Batch-size penalty is unsolved — property of the BA parametrization,
  not of rank.
- 10× LR ratio is empirical, not derived; no theoretical explanation
  for the constant.
- Memorization-vs-generalization capacity left as open question.
- "Random variation in training dynamics" between datasets observed
  at 1B-model scale.

## Reproduction repos surveyed

Two community repros at `~/lora-without-regret-repros/`:

- **`Lora-Without-Regret/`** (Brokttv): DistilBERT on AG-News.
  Implements Schulman's Init[A] uniform 1/√d_in explicitly with
  `nn.init.uniform_(self.lora_A, -scale, scale)`, scale=1/√d_in,
  B=0. AdamW, weight_decay=0.001, betas=(0.9, 0.999), default ε.
  AG-News (10K train, 2K val) classification + WikiText with DistilGPT2.
  Useful for LR-ratio ablation; not LLM-scale.

- **`lora-without-regret/`** (vLLM-based, canonical): production-grade.
  Entry points `sft_lora.py` and `rl_lora.py`.
  - **SFT:** Qwen3-4B on HF `no_robots` (6.4K train, 100 val).
    batch=2 × grad_accum=16 = batch_eff=32, **200 steps** total (1 epoch),
    seq=2048. α=32 hardcoded (does not scale with r). target_modules
    flag for all/mlp/attn. AdamW with default ε. Eval = held-out NLL.
    LR sweep flag.
  - **RL:** Qwen3-1.7B on `qwedsacf/competition_math`. 50 GRPO steps,
    32 prompts × 8 rollouts/prompt, max_new_tokens=1024. Eval =
    boxed-answer symbolic equivalence (hendrycks/math).
  - **No HumanEval, no code metrics.**

Schulman's actual SFT-canonical horizon = 200 × 32 × 2048 ≈ 13M tokens.

## Implications for the tight-chord campaign in this repo

- **Recipe-level alignment is already high.** Repo defaults already match:
  constant LR, no warmup, 1-pass enforced, target=`all-linear`.
- **α convention diverges** (project α=r vs Schulman α=32 fixed). For
  any single (r, α) cell a properly-resolved log₂ LR sweep absorbs the
  α/r difference at that cell. But the η_opt(r) shape is different
  between the two conventions — report η_opt(r) curves and don't
  pretend the conventions are interchangeable.
- **ε=0 is not a knob to flip.** Keep AdamW ε=1e-8.
- **Schulman has no code experiments.** The right framing for our paper
  is "Schulman's recipe applied to code IFT," not "Schulman replication."
- **Token horizon already exceeds Schulman-canonical.** Repo at 4k steps
  × batch_eff=16 × seq=512 = 32M tokens > Schulman SFT canonical of
  13M. Phase B/C plans (98M, 200M) are well above.
- **Batch-size confound is mild for us.** batch_eff=16 is the small end
  of Schulman's sweep range; the LoRA batch-size penalty is small here.
