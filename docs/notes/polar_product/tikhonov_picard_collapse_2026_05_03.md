# Open problem: a rank-stable calibration of the Picard cross-coupling correction in a LoRA optimizer

*2026-05-03*

## Abstract

We study a family of LoRA optimizers that couples Adam preconditioning, a Picard inner loop for cross-coupling between the two LoRA factors, and a polar / Newton–Schulz orthogonalization. The Picard iteration count $k$ has an optimum that flips with the LoRA rank: $k = 1$ wins at rank 16 (best loss 0.7546), $k = 3$ wins at rank 64 (best loss 0.7364, vs 0.7455 at $k = 1$).

We would like to remove $k$ as a knob: a single algorithm that achieves both per-rank optima with no rank-specific configuration. Whatever its internal form, such an algorithm needs to consume some runtime statistic that distinguishes the two regimes. Every spectral statistic we have logged (singular value spreads, effective ranks, Picard contraction rates, cross-coupling magnitudes) fails a cross-rank collapse test: at matched values of the statistic, the loss penalty for varying $k$ has different magnitude and even opposite sign between the two ranks.

This document defines the algorithm, presents the empirical evidence, and lists research directions.

## 1. Background

### 1.1 LoRA fine-tuning

We fine-tune a pretrained transformer by adding a low-rank correction to each frozen weight matrix. For a frozen $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the correction is
$$
\Delta W = \frac{\alpha}{r}\, B A,
\qquad A \in \mathbb{R}^{r \times d_{\text{in}}},\ B \in \mathbb{R}^{d_{\text{out}} \times r},
$$
with $r \ll \min(d_{\text{in}}, d_{\text{out}})$ the *LoRA rank* and $\alpha$ a fixed scaling constant. Only $A, B$ are trained. Backpropagation produces gradients $g_A, g_B$ at each training step.

### 1.2 The polar-product LoRA optimizer family

Per training step, for each layer pair $(A, B)$, the optimizer maps factor gradients $(g_A, g_B)$ to factor updates $(\mathrm{d}A, \mathrm{d}B)$ via three primitives composed in sequence: Adam preconditioning, a Picard inner loop for cross-coupling between the two factors, and a polar / Newton–Schulz orthogonalization invoked inside the Picard loop.

#### Algorithm 0 — Polar via Newton–Schulz, $\mathrm{polar}_{\text{NS-}j}(M)$

For $M = U \Sigma V^\top$ (singular value decomposition), $\mathrm{polar}(M) = U V^\top$ sets every singular value of $M$ to one while preserving its singular vectors. We compute it approximately by:

1. Initialize $X_0 = M / \lVert M \rVert_F$.
2. For $i = 0, \ldots, j-1$: $\quad X_{i+1} = \tfrac{3}{2} X_i - \tfrac{1}{2} X_i X_i^\top X_i$.
3. Return $X_j$.

Default $j = 5$.

#### Algorithm 1 — One optimizer step on layer pair $(A, B)$

**Hyperparameters:** learning rate $\eta$; Adam $\beta_1, \beta_2, \varepsilon$; Picard count $k$; Newton–Schulz iters $j$; preconditioner regularizer $\delta$; LoRA+ $B$-factor multiplier $m$ (default $m = 1$).

**Persistent state:** Adam moments $(m_A, v_A, m_B, v_B)$; step counter $t$.

1. **Adam preconditioning.** Update first and second moments and form the bias-corrected directions:
    $$
    m_A \gets \beta_1 m_A + (1-\beta_1) g_A, \qquad v_A \gets \beta_2 v_A + (1-\beta_2) g_A \odot g_A,
    $$
    $$
    u_A = \frac{m_A / (1-\beta_1^{t})}{\sqrt{v_A / (1-\beta_2^{t})} + \varepsilon},
    $$
    and symmetrically for $B$.

2. **Spectral preconditioners** (the LoRA Gram matrices' inverse square roots, both $r \times r$; cached and refreshed periodically):
    $$
    S_A^{-1/2} = (A A^\top + \delta I)^{-1/2}, \qquad S_B^{-1/2} = (B^\top B + \delta I)^{-1/2}.
    $$

3. **Picard cross-coupling loop.** Initialize $\mathrm{d}A = 0$, $\mathrm{d}B = 0$. For $n = 1, \ldots, k$:
    - Form Adam directions corrected by the previous joint step (no correction at $n = 1$):
        $$
        u_A^{\text{eff}} = u_A + \frac{1}{\eta} B^\top (\mathrm{d}B) A, \qquad u_B^{\text{eff}} = u_B + \frac{1}{\eta} B (\mathrm{d}A) A^\top.
        $$
    - Spectrally precondition, polar-orthogonalize, undo the preconditioning:
        $$
        P_A = \mathrm{polar}_{\text{NS-}j}\!\big(S_B^{-1/2} u_A^{\text{eff}}\big), \qquad P_B = \mathrm{polar}_{\text{NS-}j}\!\big(u_B^{\text{eff}} S_A^{-1/2}\big),
        $$
        $$
        \widetilde{\mathrm{d}A} = S_B^{-1/2} P_A, \qquad \widetilde{\mathrm{d}B} = P_B S_A^{-1/2}.
        $$
    - Frobenius-rescale to the Adam-direction norm, multiply by $-\eta$ (and the LoRA+ multiplier on $B$):
        $$
        \mathrm{d}A = -\eta \, \frac{\lVert u_A \rVert_F}{\lVert \widetilde{\mathrm{d}A} \rVert_F} \, \widetilde{\mathrm{d}A}, \qquad \mathrm{d}B = -\eta\, m \, \frac{\lVert u_B \rVert_F}{\lVert \widetilde{\mathrm{d}B} \rVert_F} \, \widetilde{\mathrm{d}B}.
        $$

4. **Apply:** $A \gets A + \mathrm{d}A$, $\ B \gets B + \mathrm{d}B$.

#### What the algorithm is approximately solving

The variational problem Algorithm 1 targets is the **per-block operator-norm program**:
$$
\min_{\Delta A,\, \Delta B}\ \langle u_A, \Delta A\rangle + \langle u_B, \Delta B\rangle\ +\ \frac{1}{2\eta}\,\lVert B \Delta A + \Delta B\, A\rVert_F^2 \quad \text{s.t.}\ \lVert B \Delta A\rVert_2 \le \tau,\ \ \lVert \Delta B\, A\rVert_2 \le \tau.
$$
Three pieces:

- **Linear cost** in the Adam-preconditioned directions $u_A, u_B$.
- **Frobenius coupling** on the joint tangent $J = B \Delta A + \Delta B\, A$ — the only term that couples the two factors. A step in $\Delta A$ alone is meaningful only via its image $B \Delta A$; the coupling penalizes the joint image.
- **Per-block spectral constraint** that caps the operator norm of each factor's contribution to $J$ separately. This is the LoRA analogue of Muon's spectral cap on the dense weight update.

**Block-coordinate Picard structure.** Solving jointly in $(\Delta A, \Delta B)$ requires an expensive eigendecomposition. The family iterates over blocks: hold $\Delta B$ at its previous inner-iterate, solve the $A$-subproblem; symmetrically for $B$; repeat $k$ times. The $A$-subproblem reduces to spectrally precondition with $S_B^{-1/2}$, apply a per-block prox operator $\mathcal{P}$, undo the preconditioning. Cross-coupling between Picard inner iterates enters only through the $u_A^{\text{eff}}$ and $u_B^{\text{eff}}$ updates in step 3 of Algorithm 1.

**Polar in place of clip.** The variationally correct prox $\mathcal{P}$ for the program above is a singular-value *clip* at threshold $\tau$. Algorithm 1 substitutes the polar map ($\mathcal{P}_{\text{polar}}(X) = U V^\top$, every $\sigma_i \mapsto 1$) followed by a Frobenius rescale to $\lVert u_A\rVert_F$. Polar is not the variational solution but is used because $\tau$ has no defensible workload-independent default and polar sidesteps it; on this workload polar matches or beats clip.

Two extreme choices of $k$:

- $k = 1$: cross-coupling term dropped. The update is the per-block prox with no Picard refinement.
- $k \to \infty$ (when the iteration contracts): converges to the joint update that solves the program self-consistently in both factors.

§2 shows the optimal $k$ flips with the LoRA rank.

### 1.3 Scope

This document is about how to eliminate manual tuning of the Picard iteration count $k$. We leave open *how* the elimination is achieved (adaptive discrete $k$, continuous relaxation of the variational program, or other); §3 fixes the success criterion.

### 1.4 Conventions for empirical numbers

All loss numbers in this document come from fine-tuning the `allenai/OLMo-2-0425-1B` base model on the Magicoder-OSS-Instruct-75K instruction–response dataset, evaluated on a held-out split, single-seed, at training step 2000 unless otherwise stated, with learning rate $\eta = 3 \times 10^{-4}$.

**Units.** Losses and loss differences (Δ) are reported as raw decimals (held-out eval loss). The workload's noise floor, estimated from a four-seed AdamW sweep, is roughly $0.0006$ at rank 16 and $0.0007$ at rank 64; we call out σ-units only where the comparison to noise is the point.

**Sign convention used in every Δ table.** $\Delta = (\text{variant loss}) - (\text{baseline loss})$. **Negative Δ means the variant beats the baseline; positive Δ means it loses.** The baseline for each table is named in its caption (e.g. "vs $k = 1$").

## 2. The Picard sensitivity

Holding all other hyperparameters fixed and varying only the Picard iteration count $k$:

| LoRA rank | Best $k$ | Final loss | Δ vs $k = 1$ |
|---|---|---|---|
| 16 | **1** | 0.7546 | (baseline) |
| 16 | 3 | 0.7557 | $+0.0011$ (worse) |
| 16 | 16 | 0.7582 | $+0.0036$ (worse) |
| 64 | 1 | 0.7455 | (baseline) |
| 64 | **3** | 0.7364 | $-0.0091$ (**better**) |

At rank 16, every $k > 1$ loses to $k = 1$. At rank 64, $k = 3$ beats $k = 1$ by $0.0091$ — about 13× the workload noise σ. Same algorithm, opposite optima.

Within rank 16, the loss penalty for $k > 1$ is non-monotone in $k$: $k = 3$ has a smaller penalty than $k = 2$, which sits below $k = 4$, etc. This is consistent with successive Picard iterates oscillating around their fixed point — the cosine between successive Picard increments is approximately $-0.85$ at rank 16 and $-0.79$ at rank 64.

The rank-16 ladder over training, with the last column showing the Frobenius norm of the cross-coupling correction at the first Picard iterate divided by the Frobenius norm of the total step — i.e. what fraction of the step magnitude the cross-coupling carries.

| Step | $L(k=1)$ | $\Delta(k=2)$ | $\Delta(k=3)$ | $\Delta(k=4)$ | $\Delta(k=8)$ | $\Delta(k=16)$ | $\lVert K_1 \rVert_F / \lVert \mathrm{d}A \rVert_F$ |
|---|---|---|---|---|---|---|---|
| 200  | 0.8314 | $+0.0016$ | $+0.0011$ | $+0.0012$ | $+0.0012$ | $+0.0013$ | 0.66 |
| 1000 | 0.7767 | $+0.0052$ | $+0.0022$ | $+0.0038$ | $+0.0035$ | $+0.0033$ | 0.84 |
| 1400 | 0.7669 | $+0.0058$ | $+0.0012$ | $+0.0041$ | $+0.0032$ | $+0.0032$ | 0.90 |
| 2000 | 0.7546 | $+0.0069$ | $+0.0011$ | $+0.0048$ | $+0.0037$ | $+0.0036$ | **1.00** |

With rank-16 noise σ ≈ 0.0006, the smallest entries are ~2σ effects and the largest ~12σ.

At step 2000 at rank 16 the cross-coupling correction has displaced the diagonal Adam direction entirely — its magnitude equals the magnitude of the total step. Every $k > 1$ loses.

## 3. Success criterion

A single algorithm with one fixed configuration — no per-rank tuning of $k$ or anything else — that matches the per-rank-best discrete-$k$ losses to within ~0.001 at each rank:

- final loss $\le 0.7547$ at LoRA rank 16 (vs the $k = 1$ best of 0.7546);
- final loss $\le 0.7374$ at LoRA rank 64 (vs the $k = 3$ best of 0.7364).

The slack 0.001 is roughly $1.4\,\sigma_{\text{noise}}$ at each rank — matching within single-seed noise. Stretch goals: same configuration also wins at $r = 32, 128$ (rank-invariant), and on a different base model or dataset (workload-invariant).

## 4. What we measured and what we tried

### 4.1 Available spectral diagnostics

The optimizer logs several scalar spectral diagnostics per layer-pair per step, summarized across pairs (max / median / min). They include:

- Extreme singular value ratios $\sigma_{\max}/\sigma_{\min}$ of the polar-stage inputs $u_A, u_B$, separately on each side. (Logged at rank 64; at rank 16 only the ratios of an auxiliary gauge factor matrix are logged.)
- Effective rank measures: stable rank $\lVert M \rVert_F^2 / \lVert M \rVert_2^2$ and numerical rank thresholds ($\sigma_i > 10^{-2} \sigma_{\max}$, $\sigma_i > 10^{-3} \sigma_{\max}$).
- Picard convergence diagnostics: the contraction ratio $\lVert K_3 - K_2 \rVert_F / \lVert K_2 - K_1 \rVert_F$, and the cosine between successive Picard increments.
- Cross-coupling correction magnitude at each Picard inner iterate: $\lVert K_n \rVert_F$, where $K_n^A := \tfrac{1}{\eta} B^\top \Delta B_{n-1} A$ (shape $r \times d_{\text{in}}$) is the correction added to $u_A$ in step 3 of Algorithm 1, with $K_n^B$ defined symmetrically on the $B$-side (shape $d_{\text{out}} \times r$).
- Total step magnitudes $\lVert \mathrm{d}A \rVert_F$, $\lVert \mathrm{d}B \rVert_F$.

These are all scalar summaries. Full singular spectra of the inner matrices $S_A$, $S_B$ (both $r \times r$) and of the cross-coupling corrections $K_n^A, K_n^B$ (each has rank at most $r$) are *not* logged.

### 4.2 Spectral statistics across ranks and steps

The polar-stage input at rank 64 narrows over training on the $B$ side but not the $A$ side:

| Step | $\sigma_{\max}/\sigma_{\min}$, $A$-side input | $\sigma_{\max}/\sigma_{\min}$, $B$-side input |
|---|---|---|
| 20 | 11.2 | **13.6** |
| 1020 | 9.0 | **6.8** |
| 2000 | 9.1 | **6.8** |

At rank 16 the same ratios are smaller and behave differently:

| Step | $\sigma_{\max}/\sigma_{\min}$, $A$-side input | $\sigma_{\max}/\sigma_{\min}$, $B$-side input |
|---|---|---|
| 20 | 1.4 | 4.9 |
| 1020 | 3.3 | 3.3 |
| 2000 | 4.7 | 4.8 |

The Picard contraction loosens over training at both ranks, and is consistently worse at rank 16:

| Step | Picard contraction, rank 16 | Picard contraction, rank 64 |
|---|---|---|
| 20 | 0.002 | 0.001 |
| 1020 | 0.029 | 0.018 |
| 2000 | **0.051** | **0.025** |

At step 2000 the rank-16 inner Picard problem is half as well-contracted as the rank-64 one.

### 4.3 None of the logged statistics distinguishes the two regimes

For an adaptive rule to choose $k$ correctly at each rank, *some* runtime-measurable quantity has to take systematically different values at "cross-coupling helps" (rank 64) vs "cross-coupling hurts" (rank 16). We tested the four statistics above by pooling $(r, k, t)$ records across $r \in \{16, 64\}$ — with $k \in \{2, 3, 4, 8, 16\}$ at rank 16 and $k \in \{3, 4\}$ at rank 64 — and asking: at matched values of the statistic, does the loss penalty $\Delta(r, k, t) = L(r, k, t) - L(r, 1, t)$ agree across ranks?

| Statistic | r=16 range | r=64 range | Overlap | Mean Δ-gap at matched values |
|---|---|---|---|---|
| $\mathrm{stable\_rank}(A) / r$ | [0.41, 0.74] | [0.17, 0.60] | 0.44 | ~0.0035 |
| Picard contraction | [0.008, 0.051] | [0.005, 0.026] | 0.42 | ~0.007 |
| $\lVert K_n^A \rVert_F / \lVert \mathrm{d}A \rVert_F$ (at $n = 1$) | [0.66, 1.00] | [0.54, 0.74] | 0.24 | ~0.007 |
| $\sigma_{\max}/\sigma_{\min}$ ($S_A$) | [1.7, 5.8] | [2.7, 14.9] | 0.25 | ~0.0035 |

A useful adaptive driver would give a Δ-gap below the noise floor (~0.0007); all four are 5–10× above. Worse, the *sign* of Δ flips: in the overlap region of $\lVert K_1^A \rVert_F / \lVert \mathrm{d}A \rVert_F$ near 0.715, rank-64 Δ is around $-0.005$ (cross-coupling helps) while rank-16 Δ is $+0.0017$ (cross-coupling hurts). At the same input value, the right action is opposite. No monotone rule on any of these statistics can give the right answer at both ranks.

## 5. The open problem

The open problem is to design an algorithm meeting §3. Two broad avenues, neither committed-to nor ruled out:

- **Adapt within the current formulation.** Keep the per-block operator-norm program of §1.2 and the polar-product algorithm; replace the fixed $k$ with a rule that picks per layer per step from a runtime input. This requires a runtime-measurable quantity that distinguishes "cross-coupling helps" (rank 64) from "cross-coupling hurts" (rank 16). §4 shows none of the four statistics currently logged does. §6 lists candidate additions to the diagnostic schema.
- **Change the formulation.** A different variational program (different coupling structure, different constraint type) or a different prox in place of polar may not have a rank-dependent optimum in the first place. We have not surveyed this space here.

### 5.1 Constraints on acceptable proposals

The success criterion in §3 is silent on the *form* of an acceptable solution. We add three meta-constraints that rule out classes of tempting hacks meeting the numerical target without contributing understanding:

- **No new tunable hyperparameters.** A fix that introduces a threshold, mixing weight, decision boundary, or schedule that itself needs to be set per workload (or per rank) just relocates the original tuning problem. The goal is fewer knobs, not more.
- **Faster or more accurate convergence to the same target is not progress.** The $k \to \infty$ Picard fixed point is precisely what the rank-16 evidence says we should *not* approach. Any acceleration of the existing iteration (Anderson, better fixed-point solvers, exact joint linear solves, …) converges to the same target and reproduces the same rank-16 failure. Acceptable proposals change *what* is being solved, not just *how well* the existing target is reached.
- **Mechanism over recipe.** A solution that *explains* why the rank-dependent $k$-flip occurs — from the structure of the variational program, the prox, the parameterization, or the data — is strictly preferred over one that detects and reacts to it after the fact. The strongest contribution would identify a reformulation under which the flip does not arise, so no calibration is needed at all.

In short: we want understanding, not just behavior matching.

### 5.2 Approaches already tested and dismissed

To save responders re-deriving directions we have already explored. All numbers below are 2000-step held-out eval losses on the same workload as §2; the per-rank baselines we are trying to match are 0.7546 ($r{=}16$, $k{=}1$) and 0.7364 ($r{=}64$, $k{=}3$).

#### 5.2.1 Gauge-handling reformulations

LoRA factor coordinates have a gauge symmetry: for any invertible $R \in \mathbb{R}^{r \times r}$,
$$
A \mapsto R A, \qquad B \mapsto B R^{-1} \qquad \Longrightarrow \qquad BA \text{ unchanged}.
$$
Infinitesimally, the gauge directions are $(\Delta A, \Delta B) = (-\Omega A,\ B \Omega)$ for any $\Omega \in \mathbb{R}^{r \times r}$, and produce
$$
B\, \Delta A + \Delta B \, A \ =\ B(-\Omega A) + (B\Omega) A \ =\ 0.
$$
So the dense update $J = B\Delta A + \Delta B \, A$ is gauge-invariant, but the factor-coordinate linear cost $\langle u_A, \Delta A\rangle + \langle u_B, \Delta B\rangle$ is not — it equals $\langle u_B, B\Omega\rangle - \langle u_A, \Omega A\rangle$ on a pure gauge direction, which is generically nonzero. This makes it natural to attribute the rank-dependent $k$-flip to gauge contamination: the Picard iteration may be doing work in non-identifiable directions, and the smaller-$r$ regime might suffer disproportionately.

**Modification.** Run the per-block prox of Algorithm 1 to get $(\Delta A, \Delta B)$, then replace them with the unique pair on the gauge surface (the *min-Frobenius lift*) that produces the same $J$:
$$
(\Delta A^{\text{lift}}, \Delta B^{\text{lift}}) \ =\ \arg\min_{(\Delta A', \Delta B')}\ \lVert \Delta A'\rVert_F^2 + \lVert \Delta B'\rVert_F^2 \quad \text{s.t.}\ B \Delta A' + \Delta B' A = J.
$$
This is computed via a small Sylvester solve (size $r \times r$). Optionally, replace the polar prox with the variationally correct singular-value clip at threshold $\tau$ in the per-block step.

**Configurations tested.** Picard $k = 2$, lr swept over $\{3{\times}10^{-4},\ 10^{-3},\ 3{\times}10^{-3}\}$, both ranks. Best (lr-selected) final losses:

| Variant | $r=16$ | $r=64$ |
|---|---|---|
| (baseline) no gauge, $k = 2$ | 0.7615 | 0.7382 |
| Polar prox + gauge lift | **0.7540** | 0.7431 |
| Clip prox + gauge lift | **0.7532** | 0.7420 |
| (target) per-rank best | 0.7546 ($k{=}1$) | 0.7364 ($k{=}3$) |

The gauge lift *helps* at $r=16$ (eliminates the cross-coupling penalty that hits no-gauge $k=2$, recovering near the rank-16 best) but *hurts* at $r=64$ (loses 0.005–0.007 to the rank-64 best). Neither configuration achieves §3's criterion at both ranks. Gauge handling changes the loss landscape but does not resolve the rank-dependence.

The same conclusion rules out variants that operate directly in the joint tangent $J = B\Delta A + \Delta B \, A$ rather than in the factor pair: the Sylvester lift above is precisely the projection onto a $J$-respecting joint update, so a $J$-space optimizer cannot improve on the gauge-lift result.

#### 5.2.2 Alternative polar / Newton–Schulz approximations

A separate sweep at $r=64,\ k=3,\ \eta = 3{\times}10^{-4}$ replaced the default 5-iteration Newton–Schulz polar (NS-5) with five alternatives:

| Variant | Definition |
|---|---|
| $\sigma^0$ (exact polar) | exact $UV^\top$ via SVD; every $\sigma_i \mapsto 1$ |
| $\sigma^{0.125}$ | $UV^\top \cdot \mathrm{diag}(\sigma_i^{0.125})$ via SVD |
| $\sigma^{0.25}$ | $UV^\top \cdot \mathrm{diag}(\sigma_i^{0.25})$ via SVD |
| DeepSeek-hybrid | two-stage degree-5 NS polynomial; more aggressive per-iteration flattening |
| PolarExpress | degree-5 polynomial with per-iteration optimal coefficients |

(The continuous family $\sigma \mapsto \sigma^p$ interpolates between exact polar at $p=0$ and identity at $p=1$.)

Δ vs NS-5 baseline ($\Delta < 0$ means variant beats NS-5):

| Step | $\sigma^0$ | $\sigma^{0.125}$ | $\sigma^{0.25}$ | DeepSeek-hybrid | PolarExpress |
|---|---|---|---|---|---|
| 200  | $-0.0024$ | $-0.0016$ | $-0.0010$ | $-0.0024$ | $-0.0024$ |
| 1000 | $-0.0003$ | $+0.0004$ | $+0.0005$ | $-0.0003$ | $-0.0002$ |
| 1200 | $+0.0002$ ← crossover | $+0.0010$ | $-0.0011$ | $+0.0004$ | $+0.0003$ |
| 1600 | $+0.0060$ | $+0.0066$ | $+0.0042$ | $+0.0060$ | $+0.0063$ |
| 2000 | (cutoff) | (cutoff) | (cutoff) | $+0.0088$ | $+0.0088$ |

The $\sigma^p$ runs hit a wall-clock cutoff at step 1760 with eval losses 0.7521 ($p=0$), 0.7527 ($p=0.125$), 0.7503 ($p=0.25$); the NS-5 trajectory at step 2000 is 0.7364. Every alternative beats NS-5 early but loses by step ~1200 and trails by ~0.009 at step 2000. Polar-stage strength is therefore not the calibration knob in disguise, and "use a better/different polar" does not address the rank-dependent $k$-flip.

## 6. Candidate measurements

A non-exhaustive list of quantities we have not yet logged or analyzed, included to seed discussion rather than to prescribe a plan. None of these are validated; some may turn out to be uninformative.

- **Combinations of the existing four statistics.** The §4.3 candidates fail individually. Whether a low-dimensional combination resolves the regimes is open — existing data would suffice to check (e.g. a small decision tree fit on the pooled records).
- **Richer functionals of the inner matrices' singular spectra.** Only norms and extreme singular ratios of $K_n, S_A, S_B$ are currently logged. Non-extreme functionals — participation ratio $(\sum \sigma_i^2)^2 / \sum \sigma_i^4$, top-vs-tail energy fraction, fitted power-law exponent of the tail, per-mode contraction — could be added cheaply ($O(r)$ floats per pair per step).
- **Iterate-to-iterate variance of the cross-coupling correction.** The variance of $K_n$ across consecutive Picard inner steps would be a noise estimate at the cross-coupling layer, analogous to the Orvieto–Gower true-variance estimator at the gradient layer (arXiv 2505.21829). Whether it correlates with "cross-coupling is signal" vs "cross-coupling is noise" is open.
- **Geometric quantities outside the spectral schema.** The §4.3 sign-flip is consistent with the regimes differing in something invisible to the inputs of the Picard iteration. One geometric candidate is the fraction of $(\mathrm{d}A, \mathrm{d}B)$ that lies in the rank-$r$-feasible tangent subspace of $(A, B)$, but other geometric quantities (and other framings entirely) may be more relevant.
- **Densify the rank-64 sample.** The §4.3 rank-64 arm rests on three cells ($k \in \{1, 3, 4\}$) vs six at rank 16, so a sample-size artifact cannot be ruled out. Sweeping $k \in \{2, 6, 8, 16\}$ at $r=64$ (~8 GPU-hours) would resolve this.

Suggestions for additional measurements, alternative diagnostics, or critiques of the framing above are welcome.

## 7. Caveats

- All numbers single-seed. The qualitative claim ("$k = 1$ best at rank 16, $k = 3$ best at rank 64") is robust to seed noise because the rank-64 gap (0.0091) is ~13× the workload's noise floor (~0.0007). Δ values smaller than the noise floor within a single rank are not robust at single-seed.
- The rank-64 sample is sparse: only $k \in \{3, 4\}$ carry the diagnostic schema used in §4.3, so the cross-rank conclusions there rest on two rank-64 cells vs five at rank 16. Densifying is one of §6's candidate next steps.
- The Picard contraction logged is a scalar summary of the dominant mode of the linearized inner operator; per-mode contraction would carry more signal and is one of §6's candidate measurements.
- Diagnostic schema availability differs across ranks: extreme singular value ratios of the polar-stage inputs $u_A, u_B$ are logged at rank 64 but not at rank 16, so they could not be tested for cross-rank collapse.
