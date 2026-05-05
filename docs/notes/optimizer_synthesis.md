# Optimizer investigation — synthesis

WARNING STALE

## TL;DR

Across the LoRA-optimizer search on OLMo-2-1B + Magicoder (2k steps, single seed, canonical horizon), the **headline result** is `adam-polar-product-lora-coupled` with `picard_iters` $k=2$ at $r=64$, $\eta=3\mathrm{e}{-4}$, **eval loss 0.7382** ($\Delta = -0.0168$ vs AdamW $r=64$). At $r=16$ the same family's uncoupled $k=1$ variant gives **0.7546** ($\Delta = -0.0033$ vs AdamW $r=16$). The polar-product family is the only one that strictly beats AdamW at both ranks, but the optimal $k$ is rank-dependent ($k=1$ at $r=16$, $k=2$ at $r=64$); no single config wins at both. See [glossary](glossary.md) for pseudocode of every optimizer named below (LoRA pair, Hybrid Picard, `picard_iters`, polar block solve, spectral preconditioner, Adam covector, RMS-align, AdaMuon-faithful, LoRA+ multiplier).

All numbers single-seed at the canonical 2k-step horizon; multi-seed verification deferred. Δ values reported as raw differences without significance qualifiers.

---

## Standing leaderboard (best $\eta$, seed=0, 2k steps)

| rank | optimizer                                       | r  | m  | best η | eval loss  | Δ vs same-r AdamW    |
|------|-------------------------------------------------|----|----|--------|------------|----------------------|
| 1    | **adam-polar-product-lora-coupled** (k=2)       | 64 | 1  | 3e-4   | **0.7382** | −0.0168              |
| 2    | adam-polar-product-lora (k=1, uncoupled)        | 64 | 1  | 3e-4   | 0.7453     | −0.0097              |
| 3    | adamuon-polar-product-lora                      | 64 | 1  | 3e-4   | 0.7486     | −0.0064              |
| 4    | adam-scaled-lora                                | 64 | 1  | 3e-4   | 0.7506     | −0.0044              |
| 5    | adam-muon-lora                                  | 64 | 1  | 3e-3   | 0.7515     | −0.0035              |
| 5    | adamuon-lora (AdaMuon-faithful)                 | 64 | 1  | 3e-4   | 0.7515     | −0.0035              |
| 7    | adam-lin-lora                                   | 64 | 1  | 3e-4   | 0.7527     | −0.0023              |
| 8    | **adam-polar-product-lora** (k=1, uncoupled)    | 16 | 1  | 3e-4   | **0.7546** | −0.0033              |
| 9    | adamw                                           | 64 | 1  | 3e-4   | 0.7550     | baseline (r=64)      |
| 10   | adam-muon-lora (LoRA+, η pinned high)           | 16 | 4  | 3e-3   | 0.7557     | −0.0022              |
| 11   | adam-scaled-lora-post (RMS-align)               | 16 | 1  | 3e-4   | 0.7570     | −0.0009              |
| 12   | adam-scaled-lora                                | 16 | 1  | 1e-3   | 0.7572     | −0.0007              |
| 13   | adamw                                           | 16 | 1  | 3e-4   | 0.7579     | baseline (r=16)      |
| 14   | adam-lin-lora                                   | 16 | 1  | 1e-3   | 0.7581     | +0.0002              |
| 15   | adamuon-lora (AdaMuon-faithful)                 | 16 | 1  | 3e-4   | 0.7603     | +0.0024              |
| 16   | adam-polar-product-lora-coupled (k=2) ⚠ loses to AdamW | 16 | 1  | 3e-4   | 0.7616     | +0.0037              |
| 17   | adam-muon-lora (m=1)                            | 16 | 1  | 3e-3   | 0.7624     | +0.0045              |
| 18   | adamuon-polar-product-lora                      | 16 | 1  | 3e-4   | 0.7653     | +0.0074              |

**Robustness observation.** At $r=64$, $\eta=1\mathrm{e}{-3}$, plain AdamW diverges to 0.89 while `adam-{lin,scaled}-lora` hold at 0.77 and `adam-polar-product-lora` at $\sim 0.79$ mid-trajectory. Geometric/spectral preconditioning gives lr-headroom at high rank.

---

## Investigation lines

**Polar-product family ([Hybrid Picard](glossary.md#optimizer-algorithms)).** The strongest family in the project. Adam-first composition: Adam EMA on factor gradients, then [polar block solve](glossary.md#optimizer-concepts) of the spectrally-preconditioned [Adam covector](glossary.md#optimizer-concepts) sandwiched between $S^{-1/2}$ factors, followed by [RMS-align](glossary.md#optimizer-concepts). Behavioral equivalence test passes (reduces to Muon NS at orthogonal init). Wins at both ranks vs AdamW, with rank-dependent best $k$.

**Picard $k$-coupling.** The cross-coupling iteration ([`picard_iters`](glossary.md#optimizer-concepts)) is rank-dependent. At $r=64$, $k=2$ beats $k=1$ (0.7382 vs 0.7453). At $r=16$, $k=2$ loses to $k=1$ (0.7616 vs 0.7546) and **loses to AdamW** by 0.0037. The full $r=16$ k-sweep is monotonically worse past $k=1$: $0.7546 \to 0.7616 \to 0.7557(k=3) \to 0.7594(k=4)$ (sources `logs/adam_polar_product_coupled_rsweep_2k/log_4.out`, `logs/picard_iters_sweep_2x2/`). The `picard_alpha` damping sweep at $r=16$ ($\alpha \in \{0.25, 0.5, 0.75\}$) likewise fails to recover $k=1$ performance (`logs/alpha_sweep_2x2/`). Cross-coupling iteration is structurally harmful at small $r$. The Picard-$k$ axis at $r=16$ is closed (see `lin_scaled_lora_investigation:H5`).

**Pre-Adam compositions (geometry → Adam).** `adam-lin-lora` (Sylvester) and `adam-scaled-lora` (Gram solve) at $r=16$ tie AdamW (0.7581 / 0.7572 vs 0.7579). Mechanism: `lin_scaled_lora_investigation:H1` cos diagnostics show Adam's per-coord $\sqrt{\hat v}$ erases the rotation the [spectral preconditioner](glossary.md#optimizer-concepts) installs upstream — they are $\varepsilon$-perturbed AdamW at $r=16$. The gap becomes monotonically more favorable as $r$ grows (`lin_scaled_lora_investigation:H3` r-sweep: $+0.023$ at $r=2$, $+0.022$ at $r=4$, $\approx 0$ at $r=16$, $-0.005$ at $r=64$); crossover near $r=16$. The premise that small $r$ would amplify the geometric correction is falsified — small $r$ is *worse*. Also tested `lin_scaled_lora_investigation:H5` (per-pair scalar matrix-Adam to preserve direction): final $r=16$ best 0.7744 ($\eta=1\mathrm{e}{-3}$, 0.018 worse than `adam-lin-lora`), $r=64$ best 0.7723 (0.022 worse than `adam-scaled-lora`); trading per-coord Adam for direction-preservation costs more than it buys.

**`*-Post` compositions (Adam → geometry).** `adam-{lin,scaled}-lora-post` apply Sylvester / Gram solve *after* Adam, hoping to install geometry downstream of $\sqrt{\hat v}$. Unfixed `*-Post` (`lin_scaled_lora_investigation:H4`) loses to AdamW by $\sim 0.03$ — root cause is a magnitude-drift bug ($\sigma_{\min}(S_B)$ climbs $0.011 \to 1.08$, step magnitude varies $100\times$ over training at fixed $\eta$). After [RMS-align](glossary.md#optimizer-concepts) fix: `adam-scaled-lora-post` at $\eta=3\mathrm{e}{-4} \to$ 0.7570 ($r=16$), matching AdamW within 0.0009. Rule of thumb from this line: post-Adam corrections work iff they are structurally meaningful on a sign-magnitude input — Newton–Schulz (spectral cap) qualifies, $S^{-1}$ (Gram-inverse rescaling of a sign vector) does not.

**Muon family.** Plain `muon-lora` (NS without Adam) → 0.7675; `ns_steps=0` baseline → 0.95+ at every $\eta$, so NS contributes $\geq 0.18$ nat. `adam-muon-lora` (Muon NS applied to the Adam step direction) at $r=64$, $m=1$, $\eta=3\mathrm{e}{-3}$ → 0.7515 ($\Delta = -0.0035$ vs AdamW $r=64$); at $r=16$, $m=1$ → 0.7624 (loses). The earlier 0.7557 headline at $r=16$ was driven by [LoRA+](glossary.md#lora-setup) $m=4$; m=1 alone does not beat AdamW at $r=16$. `adam-muon-lora` is rank-dependent: useful at $r=64$, in-noise at $r=16$ unless paired with LoRA+. The [AdaMuon-faithful](glossary.md#optimizer-composition) port (`adamuon-lora`, $m=1$) gives $r=16 \to 0.7603$ ($\Delta = +0.0024$, $\approx$ tied) and $r=64 \to 0.7515$ ($\Delta = -0.0035$); confirms AdaMuon stabilizers (sign(M) before NS, $V$ on NS-output only, RMS-align) recover AdamW pace at $r=16$ and slightly beat at $r=64$. The polar-first ordering (`adamuon-polar-product-lora`) loses to Adam-first at every tested $(r, \eta)$: $r=64 \to 0.7486$ vs Adam-first 0.7453; $r=16 \to 0.7653$ vs Adam-first 0.7546. Spectral-product geometry is the load-bearing piece across both orderings; Adam-first wins among them. AdaMuon's pretraining-scale design argument ("V on polar output is cleaner than V on raw G") does not transfer to LoRA fine-tune scale on this benchmark.

**Diag K-FAC family.** `diag-scaled-lora` $\to$ 0.8153, `kron-grad-lora` $\to$ 0.8263. Diagonal K-FAC family underperforms by $\sim 0.06$ nat. Closed branch unless we revisit the K-FAC formulation more carefully.

**GaLore.** Rank-$r$ projected full gradient via SVD into a low-rank subspace, Adam runs in subspace. $\sim 3\times$ slower per step than LoRA-mode (full dense backward). Result in `logs/galore_fixed_2k/`; does not beat plain LoRA at matched compute on this benchmark.

---

## Mechanism — `lin_scaled_lora_investigation:H1` cos diagnostics at $r=16$

Setup: `adam-lin-lora` and `adam-scaled-lora` at $\eta=1\mathrm{e}{-3}$, 2k steps, with `--log_optim_diagnostics --optim_diagnostics_every 20`. Each step computes both the geometric-Adam step $\Delta_\text{lin}$ and a side-channel plain-AdamW step $\Delta_\text{raw}$ on the same gradient (independent $m, v$ state), then logs cosines and norm ratios across all 112 LoRA pairs.

Final-step values (step 2000, median across 112 pairs):

| optimizer        | cos_A | cos_B | $\|\mathrm{d}A_\text{lin}\| / \|\mathrm{d}A_\text{raw}\|$ | $\sigma_{\min}(S_B)$ | $\|B\|_F$ |
|------------------|-------|-------|----------------------|------------|--------|
| adam-lin-lora    | 0.84  | 0.94  | 0.25                 | 1.08       | 7.68   |
| adam-scaled-lora | 0.88  | 0.97  | 0.27                 | 1.14       | 6.52   |

Three structural conclusions: (1) cos_B $\geq 0.94$ from step 20 — the geometric correction on $B$ is indistinguishable from a plain-AdamW step direction throughout training; (2) cos_A converges to $0.84$–$0.88$, asymptotically tracking AdamW; (3) the geometric step has $\sim \tfrac{1}{4}$ the magnitude of AdamW. Adam's per-coord $\sqrt{\hat v}$ in the denominator divides $g$ by its own EMA RMS coordinate-wise, producing a sign-like update regardless of upstream scaling — whatever $S_B^{-1}$ did to $\nabla A$ is a per-coordinate rescaling that $\hat v$ then undoes. Pre-precondition compositions are $\varepsilon$-perturbed AdamW by construction at $r=16$.

---

## Mechanism — A/B asymmetry across $r$

Extending the cos probe to $r \in \{4, 64\}$ (logs `h1_rsweep_diag_2k`):

| r  | cos_A_median (early) | cos_A_median (late) | cos_B_median  | $\sigma_{\min}(S_B)$ early |
|----|----------------------|----------------------|----------------|--------------------|
| 4  | 0.62 (step 20)       | 0.66 (step 520)     | **0.999** flat | 0.005              |
| 16 | 0.46 (step 20)       | 0.84 (step 1500)    | 0.94–0.98     | 0.011              |
| 64 | 0.36 (step 20)       | 0.72 (step 400)     | 0.97–0.98     | 0.0006              |

Three claims: (1) **cos_B saturates near 1 at every $r$** (0.97–0.999) — the $B$-direction of the geometric step is approximately AdamW's $B$-direction at every rank measured, so any geometric-correction advantage is happening on $A$; (2) **cos_A is rank-dependent but non-monotonically informative** — at $r=4$ cos_A is *lower* than at $r=16$ or $r=64$, yet $r=4$ loses to AdamW by $+0.02$; "different direction" $\neq$ "better direction"; (3) **conditioning is not the lever** — $\sigma_{\min}(S_B)$ is *worse* at $r=64$ ($0.0006$) than at $r=4$ ($0.005$), yet $r=64$ wins. What helps is the dimension of the subspace the geometric solve has to install structure in, not the conditioning of $S_B$.

`H_weak` vs `H_erase` (early-phase, 500-step probe at $r=16$): cos_pre_B $\approx 0.98$ throughout, cos_pre_A reaches 0.65 at step 20 and settles at 0.85 by step 500. Through step 500, $S_A^{-1}$ does not meaningfully rotate $\nabla B$ — the $B$-side correction has nothing to install early because $A$ is approximately a random projection. Whether $S_A$ stays near-isotropic past step 500 is a known-unverified extrapolation.

---

## Parallel work — relevant arxiv papers

### AdaMuon (arxiv 2507.11005v3, Dec 2025)

The closest existing variant to the polar-first composition. Algorithm 1:

```
M_t = β·M_{t-1} + G_t                        # plain SGD momentum
O_t = NewtonSchulz(sign(M_t), T)             # NS on SIGN of momentum
V_t = β·V_{t-1} + (1−β)·O_t ⊙ O_t            # element-wise v on NS output
Õ_t = O_t ⊘ (√V_t + ε)                       # variance-adapt
γ_t = 0.2·√(mn) / ‖Õ_t‖_F                    # RMS-align to Adam magnitude
W_{t+1} = W_t − η·(γ_t·Õ_t + λ·W_t)
```

Three stabilizers vs naïve "NS → Adam": (1) sign(M) before NS bounds post-NS magnitudes; (2) only $v$ on $O_t$ (no Adam $m$ on the NS output); (3) RMS-aligned step with $\gamma_t$. Paper claim: 40%+ training-efficiency gain over Adam at pretraining scale. In this project, the AdaMuon-faithful port (`adamuon-lora`) ties AdamW at $r=16$ (0.7603) and slightly beats at $r=64$ (0.7515). The polar-first composition with spectral-product geometry (`adamuon-polar-product-lora`) loses to the Adam-first analogue at both ranks.

### NorMuon (arxiv 2510.05491v1, Oct 2025)

Different diagnosis: after NS, *singular values* of the update matrix are equalized, but *per-row L² norms* still have high variance. Fix: per-neuron (row-wise) second-order adaptive learning rates on top of Muon orthogonalization — a granularity intermediate between per-element and per-pair Adam:

| granularity        | example optimizer      | adapts                    |
|--------------------|------------------------|---------------------------|
| per-element        | adam-muon-lora, AdaMuon| each parameter coord      |
| **per-row/neuron** | **NorMuon**            | each output unit          |
| per-pair (matrix)  | adam-lin-lora-matrix   | (A, B) pair (one scalar)  |

Claims: 21.74% over Adam, 11.31% over Muon at 1.1B pretraining. Not yet ported to this project.

### Caveat on transfer

Both papers measure pretraining efficiency on 1.1B-class models with hundreds of billions of tokens. This project measures final eval loss after a 2k-step LoRA fine-tune ($\sim 1\mathrm{M}$ tokens). The *direction* of the gain (Muon-family $\geq$ Adam) should transfer; the *magnitude* almost certainly does not — fine-tuning regimes are dominated by Adam's variance adaptation in ways pretraining is not.

---

## Cross-references

- [Glossary](glossary.md) — pseudocode for every optimizer named here, plus project-specific terms.
- Lin/scaled-lora investigation (H1–H5): `docs/notes/lin_scaled/investigation.md`.
- Muon-LoRA "beat AdamW" campaign: `docs/notes/muon_lora/investigation.md`.
- Coupled polar investigation (joint operator-norm family E1–E7, Picard $k$-sweep): `docs/notes/polar_product/investigations.md`.
- Adam-polar-coupled rank investigation: `docs/notes/polar_product/investigations.md`.
- Clipping-prox proposal: `docs/notes/polar_product/proposal.md`.
- Theory (Sylvester preconditioner, spectral product norm): `docs/theory/main.tex`.
- Sweep analysis notebook: `notebooks/sweep_analysis.ipynb`.
- Optimizer code: `lora_playground/optim.py`.
