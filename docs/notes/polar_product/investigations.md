# Polar-product investigations: empirical campaigns

Combines the joint-operator-norm core-solver investigation (E1–E7 timeline) with the r=16-specific sub-investigation (why coupled $k=2$ hurts at $r=16$).


> **Status (2026-05-02).** This doc records the empirical study of the
> joint operator-norm core solver (Case 3 of `theory.md`).
> Outcome: experiments E1–E7 falsified that direction — the best variant
> (E3) lands at 0.7490 at $r=64$, losing to the hybrid Picard baseline
> (0.7382). The project has since pivoted to the **adjacent formulation**
> with singular-value clipping (see `proposal.md`). Read
> this doc as the trail that motivated the pivot, not as a live plan.

## TL;DR

- **Variational target tested.** Joint operator-norm of the coupled
  tangent $\|B \Delta A + \Delta B\, A\|_2 \le \lambda$. We have a
  $\tfrac{1}{2}$-approximation core solver (`theory.md`).
- **Baseline.** [Hybrid Picard](glossary.md#optimizer-concepts), the
  strongest empirical LoRA optimizer family in this repo, runs a
  per-factor polar in spectrally-preconditioned space with optional
  cross-coupling iterations. Per-rank best: $r=16$ uncoupled $k=1$ at
  $0.7546$; $r=64$ coupled $k=2$ at $0.7382$.
- **Result.** Seven variants (E1–E7) of the joint-operator-norm solver
  all lose to Picard. Best (E3) is $0.7490$ at $r=64$ vs Picard's
  $0.7382$. E7 (closest analog of Picard with our solver replacing the
  per-factor polar) diverges catastrophically at $r=64$.
- **Diagnostic puzzle.** A variationally principled
  $\tfrac{1}{2}$-approximation to the joint operator-norm tangent step
  *loses* to a fixed-point recipe that makes no variational claim.
  Section 3.3 records the observations; Section 5 (H1–H6) enumerates
  hypotheses.
- **Pivot.** The clipping-prox proposal (`proposal.md`)
  changes the variational target to the **adjacent formulation**
  (per-channel spectral constraints with Frobenius coupling); its exact
  block solve is singular-value clipping rather than polar.

All numbers single-seed, 2k-step horizon, on
[OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)
fine-tuned on
[Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response).
LoRA on `all-linear` excluding `lm_head`, $\alpha = r$, PEFT init
($A \sim$ Kaiming, $B = 0$).

---

## 1. Notation recap

We follow `theory.md`. For each frozen base weight
$W \in \mathbb{R}^{m \times n}$ ($m = d_\text{out}$, $n = d_\text{in}$),
a LoRA correction $W \to W + \tfrac{\alpha}{r} B A$ with
$A \in \mathbb{R}^{r \times n}$, $B \in \mathbb{R}^{m \times r}$.
Per-factor gradients

$$
G_A := \nabla_A L = (\alpha/r)\, B^\top \nabla_W L, \qquad
G_B := \nabla_B L = (\alpha/r)\, \nabla_W L\, A^\top.
$$

Compatibility (raw autograd): $G_A A^\top = B^\top G_B$.

The joint operator-norm step solves

$$
\min_{\Delta A,\Delta B}\; \langle G_A, \Delta A \rangle +
\langle G_B, \Delta B \rangle
\quad \text{s.t.} \quad
\| B \Delta A + \Delta B\, A \|_2 \le \lambda.
$$

---

## 2. Two algorithms

### 2.1 Our solver — projected-quotient-polar core

This is the §2.1 baseline; experiments E1–E7 below are modifications of
it. Implementation: `lora_playground/optim.py::PolarCoupledCoreLoRA`,
helpers `_polar_coupled_core_step`, `_polar_coupled_core_lift`. Matches
`theory.md` Sections 1–3. Per step:

1. Thin QR: $B = Q_L R_L$, $A = R_R Q_R^\top$.
2. [Active core](glossary.md#joint-core-solver-terminology-archive-of-the-dead-family) construction:

   $$
   C \;=\; \tfrac{1}{2}\bigl(R_L^{-\top} G_A Q_R + Q_L^\top G_B R_R^{-\top}\bigr),
   \qquad
   L_\perp = R_L^{-\top} G_A - C Q_R^\top,
   \quad
   R_\perp = G_B R_R^{-\top} - Q_L C.
   $$

   Thin SVD of residuals: $L_\perp = E V^\top$, $R_\perp = U F$.
3. Form
   $\widehat H = \begin{bmatrix} C & E \\ F & 0 \end{bmatrix}
   \in \mathbb{R}^{(r+t)\times(r+s)}$.
4. Compute the compact polar factor $P = \mathrm{polar}(\widehat H) = U V^\top$
   (from the compact SVD $\widehat H = U \Sigma V^\top$), then project the
   $(2,2)$ block to zero (the [$\Pi$ projection](glossary.md#joint-core-solver-terminology-archive-of-the-dead-family))
   and renormalize:

   $$
   R = \Pi(P), \quad \gamma = \|R\|_2, \quad
   \widehat Z_+ = R / \gamma.
   $$

   $\gamma \in [1, 2]$ certifies a deterministic
   $\tfrac{1}{2}$-approximation to the variational program above.
5. Scale (squared-penalty default):
   $\widehat Z_\text{upd} = -\eta\, \tau\, \widehat Z_+$ where
   $\tau = \|\widehat H\|_* / \gamma$.
6. Lift to factor space via
   [min-Frobenius gauge](glossary.md#optimizer-concepts) (Sylvester solve):
   write $\widehat Z_\text{upd}$ as blocks $X, Y, W$ ($X$ is $(1,1)$, $Y$
   is $(1,2)$, $W$ is $(2,1)$). Solve
   $S_L K + K S_R = R_L^\top X R_R^\top$ for $K$ where
   $S_L = R_L^\top R_L + \delta I$, $S_R = R_R R_R^\top + \delta I$, then

   $$
   \Delta A \;=\; S_L^{-1}\bigl(R_L^\top X - K R_R\bigr) Q_R^\top
              + S_L^{-1} R_L^\top Y\, V^\top,
   \quad
   \Delta B \;=\; Q_L\bigl(X R_R^\top - R_L K\bigr) S_R^{-\top}
              + U\, W\, R_R^\top S_R^{-\top}.
   $$

7. Apply $A \leftarrow A + \Delta A$, $B \leftarrow B + \Delta B$.

Diagnostics logged per step: $\gamma$, $\|\widehat H\|_*$,
$\mathrm{relgap} = 1 - 1/\gamma$, $\mathrm{compat} =
\|C_L - C_R\|_F / (\|C_L\|_F + \|C_R\|_F + \varepsilon)$,
$\|\Delta A\|/\|\Delta B\|$, and the **iLoRA imbalance residual**
$\|A A^\top - \rho B^\top B\|_F / (\cdot)$ with $\rho = r/m$ (a measure
of departure from the iLoRA invariant $A A^\top = (r/m) B^\top B$, which
is the natural balance condition for the LoRA factorization).

### 2.2 Picard — `adam-polar-product-lora-coupled`

Implementation: `lora_playground/optim.py::AdamPolarProductLoRA` with
`picard_iters=2`. This is the strong empirical baseline we cannot beat.
**Picard makes no variational claim.** It is a damped fixed-point recipe
on the joint normal equations
$S_B \Delta A + B^\top \Delta B\, A = -\eta u_A$,
$\Delta B\, S_A + B \Delta A\, A^\top = -\eta u_B$ (with
$S_A = A A^\top + \delta I$, $S_B = B^\top B + \delta I$). Per step:

```text
# Adam EMA on raw factor gradients
m_A, v_A = adam_ema(G_A); u_A = m̂_A / (√v̂_A + ε)
m_B, v_B = adam_ema(G_B); u_B = m̂_B / (√v̂_B + ε)

# Spectral preconditioners (refresh each step)
SA_half_inv = (A Aᵀ + δI)^{-1/2}
SB_half_inv = (Bᵀ B + δI)^{-1/2}

# Picard fixed-point iteration (default picard_iters=2)
dA_prev, dB_prev = 0, 0
for k in range(picard_iters):
    if k == 0:
        u_A_eff, u_B_eff = u_A, u_B
    else:
        u_A_eff = u_A + α (Bᵀ dB_prev A) / lr
        u_B_eff = u_B + α (B dA_prev Aᵀ) / lr

    # Per-factor polar in spectrally-preconditioned space
    P_B = polar(u_B_eff @ SA_half_inv)        # (m, r)
    geo_B = P_B @ SA_half_inv
    P_A = polar(SB_half_inv @ u_A_eff)        # (r, n)
    geo_A = SB_half_inv @ P_A

    # RMS-align: rescale step magnitude back to Adam direction norm
    dA = -lr * (‖u_A‖ / ‖geo_A‖) * geo_A
    dB = -lr * (‖u_B‖ / ‖geo_B‖) * geo_B
    dA_prev, dB_prev = dA, dB

apply (dA, dB)
```

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$,
$\delta = 10^{-6}$, $\alpha=1$ (the Picard damping factor — not LoRA
$\alpha$), `picard_iters=2`.

### 2.3 Side-by-side structural differences

| aspect | our solver | Picard |
|---|---|---|
| polar | one **joint** $(r+t)\times(r+s)$ core | two **separate**: $(m \times r)$ and $(r \times n)$ |
| basis extraction | thin QR of $A,B$ + SVD of residuals | none — direct on factors |
| $(2,2)$ extension mode | explicitly zeroed, $\Pi$ projection | not constrained |
| symmetrization | $C = \tfrac{1}{2}(C_L + C_R)$ | none |
| step magnitude | $\eta \tau = \eta \|\widehat H\|_*/\gamma$ | $\eta \|u_A\|$ via RMS-align |
| coupling | one-shot | inner Picard fixed-point (default 2 passes) |
| variational claim | $\tfrac{1}{2}$-optimal for joint op-norm | none |

---

## 3. Empirical results

### 3.1 Reference baselines (best per rank)

| optimizer | $r=16$ best lr | $r=16$ eval | $r=64$ best lr | $r=64$ eval | source |
|---|---|---|---|---|---|
| Uncoupled spectral-product (`adam-polar-product-lora`, k=1) | 3e-4 | **0.7546** | 3e-4 | 0.7453 | `logs/polar_product_2k/log_2.out`, `logs/polar_product_r64_diag_2k/log_1.out` |
| Hybrid Picard (`adam-polar-product-lora-coupled`, k=2) | 3e-4 | 0.7616 | 3e-4 | **0.7382** | `logs/adam_polar_product_coupled_rsweep_2k/log_4.out`, `logs/adam_polar_product_coupled_r64_2k/log_1.out` |
| AdamW | 3e-4 | 0.7579 | 3e-4 | 0.7550 | `logs/lr_sweep_2k/run_info/logs/log_10.out` (r=16); `logs/h3_rsweep_2k/run_info/logs/log_02.out` (r=64) |
| `adam-lin-lora` (Adam-then-Sylvester: Adam EMA on factors → closed-form $L^2$-balanced Sylvester gauge solve, no clipping/polar) | 1e-3 | 0.7581 | — | — | `logs/h1_pre_probe_2k`; referenced by E8 below as a control for the gauge-lift's standalone behavior |

Within-family rank-dependence: **k=1 wins at r=16, k=2 wins at r=64.**
At r=16 the coupled k=2 variant *loses* to AdamW (0.7616 vs 0.7579) —
the cross-coupling iteration with the polar block solve is structurally
harmful at small r. At r=64 k=2 beats AdamW by 0.0168. No single config
of the family wins at both ranks; the bidirectional goal is unmet.

### 3.2 Our solver and its variants — table

We use **experiment ID E1, E2, ...** for our results. Each row gives one
sentence of what changes from the baseline solver of §2.1.

| ID | what changes from §2.1 baseline | $r=16$ best | $r=64$ best | gap to best |
|---|---|---|---|---|
| E1 | nothing (the §2.1 baseline) | 0.8188 (lr 3e-3) | 0.7821 (lr 3e-3) | +0.064 / +0.044 |
| E2 | post-step state rebalance: rotate $(A,B) \to (R^{-1}A, BR)$ to enforce iLoRA invariant $A A^\top = (r/m) B^\top B$, preserving $BA$ exactly | 0.8104 | 0.7686 | +0.056 / +0.030 |
| E3 | wider lr scan on E1 (extend to 1e-2, 3e-2) | 0.8049 | **0.7490** (lr 3e-2) | +0.050 / +0.011 |
| E4 | **core sign-norm** — elementwise normalization $\widehat H \to \widehat H / (\|\widehat H\| + \varepsilon)$ before polar (per-coord adaptivity in core space, no momentum) | **0.7680** (lr 1e-4) | diverges | +0.013 / — |
| E5 | core-EMA + Nesterov (Muon-style on the rotating $Q_L, Q_R$ basis with overlap-matrix transport: $M_t = \beta\, T_L M_{t-1} T_R^\top + (1-\beta)\widehat H_t$) | 0.9073 | 0.8883 | far worse |
| E6 | compounds: E4 ⊕ E5 ⊕ E2 in 4 combinations | 0.7684 | 0.9440 | tied with E4 / worse |
| E7 | factor-Adam preconditioning before §2.1 (Adam EMA on $G_A, G_B$, feed $u_A, u_B$ into the §2.1 solver) — closest analog of Picard with our solver replacing Picard's per-factor polar | 0.7846 (lr 1e-4) | 1.097 (lr 1e-4, step 1600 of 2000; cancelled — clearly falsified) | +0.030 / +0.36 |
| E8 | cross-check on a different LoRA solver: take a Sylvester-based factor-Adam solver that already works (`adam-lin-lora` / Adam-then-Sylvester entry in §3.1, eval 0.7581 at $r=16$) and move its Adam EMA into core space ($r \times r$ Sylvester RHS matrix) instead of factor space — diverges at step 2 | div | div | — |

**Best so far per rank, both from our solver family:**

- $r=16$: **E4** (core sign normalization), $0.7680$. Still $+0.013$
  behind the $r=16$ best (uncoupled spectral-product, $0.7546$).
- $r=64$: **E3** (E1 + lr=3e-2), $0.7490$. Beats AdamW ($0.7550$); still
  $+0.011$ behind hybrid Picard ($0.7382$).

### 3.3 What each experiment tells us (one mechanism per row)

- **E2** (state rebalance): drives the iLoRA imbalance residual
  $\|A A^\top - (r/m) B^\top B\|_F / (\cdot)$ from $1.0 \to 10^{-3}$ in
  two steps. Mechanism works as designed; eval gain $\le 0.014$.
  Conclusion: factor-state imbalance is real but not the bottleneck.
- **E3** (wider lr scan): the §2.1 baseline's lr ceiling is around
  $3\times$ the canonical Adam lr ($3 \times 10^{-2}$ vs
  $3 \times 10^{-4}$). Beats AdamW at $r=64$. At $r=16$ ceiling
  $\sim 0.80$, so wider lr does not save us at low rank.
- **E4** (core sign-norm): per-coord adaptivity in **core space** (after
  the $Q_L, Q_R$ basis projection). First variant to break the $r=16$
  ceiling; useless at $r=64$.
- **E5** (transported core EMA, Muon-style): the principled
  "Muon-on-LoRA-tangent" answer — should be the right thing
  theoretically, but is the worst variant tested. Diagnostics in §4.2
  show why.
- **E7** (factor-Adam, the closest analog of Picard with our solver):
  replicates Picard's preconditioning step but feeds into the §2.1
  solver instead of Picard's per-factor polar. **Does not help.** At
  $r=16$, best lr ($1\mathrm{e}{-4}$, plain) lands at $0.7846$
  ($+0.030$ over the $r=16$ best). At $r=64$ the same configuration is
  dramatically worse ($1.097$ at step 1600, vs Picard $0.7382$); the
  $r=64$ run was cancelled at step 1600 of 2000 — the gap is $+0.36$
  and shows no closing trend, so the conclusion does not depend on the
  last 400 steps. This is the experiment that most directly tests "is
  the missing piece factor-space adaptivity?" Answer: no.
- **E8** (cross-check on a different working solver — the
  Adam-then-Sylvester baseline from §3.1, which gets $0.7581$ at $r=16$;
  we move its Adam EMA from factor space to its $r \times r$
  core/Sylvester matrix and rerun): diverges at step 2 of OLMo-2-1B
  smoke; Cholesky fails at step 3. Mechanism: $\sqrt{v_M}$ on a small
  $r\times r$ matrix degenerates to $\approx 3\,\mathrm{sign}(M)$ at
  step 1, inflating step magnitude. The core matrix's coordinates all
  live on the same scale (no heterogeneous coordinate scales akin to
  parameters of different layer widths or roles), so Adam's
  $\sqrt{v}$-rescaling — designed to normalize precisely those
  cross-coordinate scale differences — acts as near-pure sign
  saturation rather than as a calibrating preconditioner. This
  independently confirms E5's failure mode generalizes: core-space
  Adam-style momentum is structurally broken.

---

## 4. Diagnostic findings

### 4.1 `compat` — gradient compatibility violation

Defined as $\|C_L - C_R\|_F / (\|C_L\|_F + \|C_R\|_F + \varepsilon)$,
where $C_L = R_L^{-\top} G_A Q_R$, $C_R = Q_L^\top G_B R_R^{-\top}$.

- Variants 1–6 (raw factor gradients): $\mathrm{compat} \approx
  \varepsilon_\text{machine}$. Compatibility holds by construction; the
  averaging step in (2.1.2) is averaging two equal numbers.
- Variant 7 (factor-Adam): $\mathrm{compat} \in [0.65, 0.88]$ in r=4
  smoke at early steps. **Factor-Adam genuinely breaks compatibility**;
  the $\tfrac{1}{2}(C_L + C_R)$ averaging is doing real work and may be
  lossy.

### 4.2 `align_inst` and `align_mom` — alignment of EMA core with chosen direction

For E5 (transported core EMA). Per-pair median across the model:

- $\mathrm{align\_inst}$ (cosine of instantaneous core $\widehat H_t$
  with chosen polar direction $\widehat Z_+$): $\approx 0.45$–$0.50$.
- $\mathrm{align\_mom}$ (cosine of EMA-preconditioned core with the
  same direction): $\approx 0.30$–$0.55$, **frequently below
  align_inst**.

EMA averaging in core space does **not** accumulate constructively.
Successive cores point in different directions in the rotating
$Q_L, Q_R$ frame; averaging dilutes rather than reinforces signal.

### 4.3 `transport_residual` — basis transport error in E5

$\|M_t - M_\text{transported}\|_F / (\|M_\text{transported}\|_F + \varepsilon)$
where $M_\text{transported} = T_L M_{t-1} T_R^\top$ with overlap matrices
$T_L = U_\text{cur}^\top U_\text{prev}$,
$T_R = V_\text{cur}^\top V_\text{prev}$.

Median $\approx 0.04$, max $\approx 0.10$. **Small.** Transport is fine.
Variant 5's failure is the EMA itself, not the transport mechanism.

### 4.4 $\gamma$, $\mathrm{relgap}$ — solver health

$\gamma \in [1, 2]$ on every step. $\mathrm{relgap} = 1 - 1/\gamma$
typically $0.05$–$0.15$. The $\tfrac{1}{2}$-approximation certificate
holds. Polar is computed correctly.

### 4.5 `imbalance_residual`

Drops from $\approx 1.0$ at PEFT init to $\approx 10^{-3}$ in 2 steps
under state rebalance and stays there. The factor-state geometry can be
fully restored to the iLoRA invariant — and it doesn't help.

---

## 5. Open hypotheses

Live items only. Closed/folded items are noted with one sentence.

### H1. We are solving the wrong variational problem (folded)

The joint operator-norm constraint is variationally clean but may not be
what the fine-tuning loss landscape rewards. **Folded into the
clipping-prox proposal as an ablation** (per-factor polar with our gauge
lift); see `proposal.md`.

### H2. The $(2,2)$ zero projection discards real signal (open)

Our active-core construction sets $\widehat H_{22} := 0$ (the "extend
both $A$ and $B$ into new directions simultaneously" mode). The solver
doc justifies this from the $(2,2)$ block of feasible $\widehat Z$ being
zero. Picard's per-factor polar has no such restriction — its step on
$A$ can flow signal into directions $B$ doesn't currently span, and vice
versa, simultaneously.

**Test:** un-zero the $(2,2)$ block, replace $\Pi(P)$ with $P$ itself in
step (2.1.4). One-line change; it would violate the variational story
but tells us if the projection is empirically costly. Residual to the
dead joint-operator-norm branch.

### H3. Step magnitude mismatch (open)

Our step magnitude is $\eta \tau = \eta \|\widehat H\|_* / \gamma$.
Picard's is $\eta \|u_A\|$ (RMS-aligned to Adam direction norm). At
fixed $\eta$ these are different scales. We see this empirically: E3
(E1 + wider lr) needs $\eta = 3 \times 10^{-2}$ at $r=64$ to be
competitive, while Picard's optimum is $\eta = 3 \times 10^{-4}$ — a
$100\times$ ratio.

**Test:** plot $\|\Delta A\|, \|\Delta B\|$ trajectories of E1 vs Picard
at their best lr's. Are the effective per-step magnitudes matched? If
ours is way larger (or smaller), that's a knob we haven't calibrated.

### H4. Symmetrization $\tfrac{1}{2}(C_L + C_R)$ is lossy when compat is high (open)

Variant 7 directly hits this: factor-Adam → compat 0.65–0.88. The
averaging projects two genuinely different views into a single one.
Picard avoids it by never building $C$ — instead, two separate
per-factor polars, each operating on its own preconditioned gradient.

**Test:** version of our solver that never symmetrizes — keep $C_L$ and
$C_R$ as separate inputs, do separate per-factor polars in core space,
lift separately. (Closely related to H1's folded test.)

### H5. Picard's iteration is doing something we lack (closed at $r=16$)

The hypothesis is empirically falsified at $r=16$ by the
`picard_iters` sweep. Evidence:

- iters=1 (cross-coupling disabled, `adam-polar-product-lora`
  uncoupled): eval $0.7546$ at $r=16$.
- iters=2 (default): eval $0.7616$ at $r=16$ (loses to AdamW 0.7579).
- iters=3: eval $0.7557$ at $r=16$.
- iters=4: eval $0.7594$ at $r=16$.

iters=2 is *substantially worse* than iters=1 (Δ=+0.0070), and iters=3,
4 do not recover. The cross-coupling iteration with the polar block
solve is structurally harmful at small $r$ — at $k=2$ it even pushes the
family below AdamW. Extending `picard_iters` further is not informative;
the sweep already covers $k \in \{1, 2, 3, 4\}$ exhaustively at this
rank.

### H6. We have a bug (open)

Possible defects worth re-checking:

- The $\alpha/r$ LoRA scaling. PEFT applies $\alpha/r$ at the model
  layer; our solver does not separately scale. Picard also does not.
  Should match in principle. Worth a 1-pair trace to confirm.
- The Sylvester lift formula in (2.1.6). Test 4 in
  `tests/test_polar_coupled_core.py` verifies that on synthetic random
  $(A, B, G_A, G_B)$ pairs, our solver in `core_norm="frobenius"` mode
  matches a hand-derived Sylvester closed form to $10^{-5}$. This
  passes. Worth a single real-LoRA-pair trace at $r=4$ to verify the
  operator-norm path matches an alternative implementation.
- The $B = 0$ PEFT-init boundary case. Our solver triggers
  `_zero_B_fallback` at step 1; thereafter regular path. Picard's
  $S_B^{-1/2} = (B^\top B + \delta I)^{-1/2} \to \delta^{-1/2} I$
  smoothly handles it. Cheap test.

---

## 6. Reproducibility

All sweeps logged in `logs/<group>/run_info/`. Pull final results via
`lora_playground.loader.load_runs(where={...})`. Diagnostics on every
cell via `--log_optim_diagnostics --optim_diagnostics_every 200`.

Sweep groups referenced in this doc:

- `polar_coupled_core_2k` — variants 1, 5
- `state_rebalanced_2k` — E2
- `polar_core_wide_lr_2k` — E3
- `polar_core_sign_2k` — E4
- `polar_core_sign_followup_2k` — E6
- `polar_factor_adam_2k` — E7

Code: `lora_playground/optim.py`, classes `PolarCoupledCoreLoRA`
(variants 1–4), `MuonCoupledCoreLoRA` (E5),
`PolarCoupledCoreFactorAdamLoRA` (E7), `AdamLinCoreLoRA` (E8
cross-check), `AdamPolarProductLoRA` (Picard).

---

## Appendix: $r=16$ sub-investigation — coupled vs uncoupled


Working doc. Investigates why [`AdamPolarProductLoRACoupled`](glossary.md)
(Picard `picard_iters` $k=2$) underperforms the uncoupled variant ($k=1$)
specifically at $r=16$, while winning at $r=64, 128, 256$. See
[glossary.md](glossary.md) for Hybrid Picard, polar block solve, spectral
preconditioner, Adam covector, RMS-align, `picard_iters`, `picard_alpha`.

### TL;DR

At $r=16$, coupled loses to uncoupled by $\approx 0.007$ eval loss; at
$r \in \{64, 128, 256\}$ coupled wins by $0.001$–$0.011$. The single
variable that flips sign at the same boundary is $\kappa(S_B)$ — the
condition number of the right-side spectral preconditioner $S_B = B^\top B + \delta I$
($\delta = 10^{-6}$): $\kappa \approx 2.5$ at $r=16$ versus $\ge 11$ at
$r \ge 64$. When $S_B$ is near-flat, the polar pipeline's $S_B^{-1/2}$
factor acts like a scalar and the Picard cross-term becomes a destabilizing
perturbation; when $S_B$ is peaky, the cross-term refines a meaningful
preconditioner. Mechanistically, the cross-term in $\Delta B$ has column
space inside $\mathrm{col}(B)$ — call this **bilinear feedback in $B$**
(the cross-coupling correction reinforces directions $B$ already occupies)
— which concentrates rank when $B$ is near-saturated. Cross-coupling is
correlated with stable-rank drift in the predicted direction; causation
is not pinned down. The α-sweep and picard_iters sweep have completed
(see "Sweep results" below).

### The gap to explain

Final eval losses at $\eta=3 \times 10^{-4}$, 2k steps, single seed (pulled
via `load_runs(where={"optimizer": ["adam-polar-product-lora", "adam-polar-product-lora-coupled"], "lr": 3e-4, "lora_r": [16, 64, 128, 256]})` over default-`picard_iters`/`picard_alpha` runs, newest-wins on collision):

| $r$ | uncoupled (k=1) | coupled (k=2) | $\Delta$ (coupled − uncoupled) | source (coupled)                                        |
|-----|-----------------|---------------|---------------------------------|---------------------------------------------------------|
| 16  | 0.7546          | 0.7615        | **+0.0069** (worse)             | `diag_h1234_2x2`                                        |
| 64  | 0.7454          | 0.7382        | $-0.0072$                       | `adam_polar_product_coupled_r64_2k`                     |
| 128 | 0.7458          | 0.7358        | $-0.0100$                       | `diag_h1234_2x2`                                        |
| 256 | 0.7471          | 0.7364        | $-0.0107$                       | `adam_polar_product_coupled_r256_rerun_2k`              |

At every $r$ except 16, the cross-coupled iter-2 update beats
block-diagonal iter-1. At $r=16$ it loses by a comparable magnitude.
The question is what makes $r=16$ special.

### What "coupled" actually does

`adam-polar-product-lora` solves the joint normal equations of the
adjacent variational formulation under the **spectral-product metric**
(separately whitens $\Delta A$ on the right by $S_B^{-1/2}$ and $\Delta B$
on the left by $S_A^{-1/2}$, where $S_A = A A^\top + \delta I$,
$S_B = B^\top B + \delta I$, $\delta = 10^{-6}$) via Picard iteration.
The equations:

$$S_B \cdot \Delta A + B^\top \cdot \Delta B \cdot A = -\eta \cdot u_A$$
$$\Delta B \cdot S_A + B \cdot \Delta A \cdot A^\top = -\eta \cdot u_B$$

`picard_iters=1` (uncoupled) drops the cross-terms $B^\top \Delta B\, A$
and $B\, \Delta A\, A^\top$. `picard_iters=2` (coupled) does one Picard
fixed-point step: feed the iter-1 $(\Delta A, \Delta B)$ back as
`dA_prev`, `dB_prev` and recompute with the cross-terms folded into the
linear cost:

$$u_A^\text{eff} = u_A + \alpha \cdot (B^\top \cdot \mathrm{dB\_prev} \cdot A) / \eta$$
$$u_B^\text{eff} = u_B + \alpha \cdot (B \cdot \mathrm{dA\_prev} \cdot A^\top) / \eta$$

Then through the polar pipeline (Newton-Schulz orthogonalization,
RMS-align). The damping coefficient $\alpha$ (`picard_alpha`) defaults
to 1 and was added as a sweep knob.

### H1–H4′ — what we instrumented and what we found

We instrumented `AdamPolarProductLoRA.step` in `lora_playground/optim.py`
(commit `6ae1ec7`) with per-pair stats; `_emit_optim_diagnostics`
aggregates min/median/max across pairs into one `optim_step` JSONL event.

Definitions (project-specific; not in glossary):

- **$\gamma_A, \gamma_B$** — relative magnitude of the cross-coupling
  correction: $\gamma_A = \|B^\top\, \mathrm{dB}\, A / \eta\|_F / \|u_A\|_F$,
  symmetric for $\gamma_B$. Always-on.
- **stable rank** — $\mathrm{sr}(M) := \|M\|_F^2 / \|M\|_2^2$, a soft
  rank measure ($\le \mathrm{rank}(M)$, equals true rank when all
  nonzero $\sigma_i$ are equal). Reported as $\mathrm{sr}_B / r$ to
  normalize.
- **$\mathrm{nrank}_\tau(S)$** — count of singular values of $S$
  exceeding $\tau \cdot \sigma_\max(S)$ (a hard rank measure). Always-on.
- **picard_contract_A_12, A_23** — successive-iter Picard increment
  ratios with the **iterate-1-vs-2 ratio** convention `_12` =
  $\|\mathrm{dA}^2 - \mathrm{dA}^1\|/\|\mathrm{dA}^1\|$ and `_23` =
  $\|\mathrm{dA}^3 - \mathrm{dA}^2\|/\|\mathrm{dA}^2\|$. Probe-step only
  (every `diagnostics_every` steps).
- **polar_cos_A_12, B_12** — cos between iter-1 and iter-2 polar (NS)
  outputs. Probe-step only.

**Hypotheses (this doc; primed `H4′` distinguishes from H1–H4 in the
glossary, which are gauge/rank-axis labels in `investigations.md`):**

- **H1 (this doc) — cross-term dominance.** $\gamma \gtrsim 1$ and
  r-dependent (smaller $r \to$ larger $\gamma$ because $B$'s top
  $\sigma$ are bigger). Predicts: $\gamma$ explosion at $r=16$.
- **H2 (this doc) — Picard non-contracting.** picard_contract_*_23
  not $\ll$ _12; successive iterates don't shrink.
- **H3 (this doc) — polar amplifies perturbations.** polar_cos_*_12
  noticeably below 1 (Newton-Schulz non-Lipschitz at degenerate spectra).
- **H4′ — bilinear feedback into $B$ at small $r$.** Because the
  cross-term $B \cdot \mathrm{dA} \cdot A^\top$ has column space
  strictly inside $\mathrm{col}(B)$, repeated coupling reinforces
  directions $B$ already occupies; at small $r$ where $B$'s column
  space is near-saturated this concentrates effective rank.
  Predicts: $\mathrm{sr}_B/r$ drops in coupled relative to uncoupled,
  more dramatically at $r=16$.

### What the data says (2×2 sweep at $\eta=3 \times 10^{-4}$, 2k steps)

Group: `diag_h1234_2x2`. Configs: $\{$coupled, uncoupled$\} \times \{r=16, r=128\}$.
Diagnostics every 20 steps.

**stable_rank_B / $r$** (out of $r$):

| step | uncoupled $r=16$ | coupled $r=16$ | uncoupled $r=128$ | coupled $r=128$ |
|------|------------------|-----------------|--------------------|------------------|
| 200  | 10.64            | 10.40           | 30.99              | 30.61            |
| 500  | 11.04            | 10.36           | 33.83              | 32.54            |
| 1000 | 11.24            |  9.96           | 35.18              | 32.69            |
| 1500 | 11.20            |  9.50           | 36.46              | 32.82            |
| 1800 | 11.21            |  9.26           | 36.40              | 33.14            |

- $r=16$: uncoupled drifts up to 11.2/16 ($\approx 70\%$ of available
  rank); coupled drifts down to 9.3/16 ($\approx 58\%$). Gap = 1.95
  (12% of available). The two trajectories diverge over training.
- $r=128$: uncoupled rises to 36.4/128 ($\approx 28\%$); coupled rises
  to 33.1/128 ($\approx 26\%$). Gap = 3.3 (2.5% of available). Same
  direction (coupled has lower stable rank than uncoupled), but the
  relative magnitude is much smaller.

**$\gamma$ trajectories:** $\gamma$ stays small (max $\approx 0.12$ in
any cell), and $\gamma$ at $r=128 \ge \gamma$ at $r=16$ in every cell —
opposite of what H1 predicted. **H1 (cross-term dominance) is refuted.**

**picard_contract:** _A_23 / _A_12 ratio $\approx 0.1$–$0.5$ throughout,
_A_23 substantially smaller than _A_12. Picard contracts cleanly.
**H2 (Picard non-contracting) is refuted.**

**polar_cos:** $\ge 0.992$ everywhere; Newton-Schulz behaves as identity
between iter-1 and iter-2. **H3 (polar amplifies perturbations) is
refuted.**

**Eval losses at step 1800** (current sweep, reproduces prior data):

| cell             | step 1800 |
|------------------|-----------|
| uncoupled $r=16$  | 0.7577    |
| coupled $r=16$    | 0.7643    |
| uncoupled $r=128$ | 0.7496    |
| coupled $r=128$   | 0.7388    |

### Mechanism diagnosis (claim, not proof)

The iter-2 update to $\mathrm{dB}$ is:

$$\mathrm{dB} \propto \mathrm{polar\_pipeline}\!\left((u_B + \alpha \cdot B \cdot \mathrm{dA} \cdot A^\top / \eta) \cdot S_A^{-1/2}\right) \cdot S_A^{-1/2}$$

The cross-term $B \cdot \mathrm{dA} \cdot A^\top / \eta$ has its column
space **strictly inside $\mathrm{col}(B)$** — $B$ appears as a left
factor. The polar (Newton-Schulz) operator preserves column space;
right-multiplication by $S_A^{-1/2}$ acts on rows. So the iter-2
contribution to $\mathrm{dB}$ lives preferentially within $\mathrm{col}(B)$;
the iter-1 contribution does not (its column space comes from $u_B$,
which depends on the loss gradient, not on $B$'s structure).

Repeated application reinforces $\mathrm{col}(B)$: each step preferentially
refreshes already-occupied directions of $B$. At small $r$, where $B$'s
column space is already near-saturated ($\mathrm{sr}_B / r \approx 0.7$
at $r=16$), this concentrates the limited rank into fewer effective
directions. At large $r$, where $\mathrm{sr}_B / r \approx 0.25$, there's
spare column space for the reinforcement to absorb without harm.

This is **inherent to (a) the LoRA factorization $\Delta W = BA$ (so
$\partial \Delta W / \partial \Delta A = B$ brings $B$ into the chain
rule for any joint update) and (b) joint coupling on $(A, B)$ variables**.
Within LoRA + joint coupling, the bilinear cross-term is unavoidable;
the polar / metric / Picard machinery doesn't change its column-space
property because polar preserves column space and right-side whitening
doesn't act on the left factor.

The mechanism is supported correlatively by the data: coupled-$r=16$
$\mathrm{sr}_B$ diverges from uncoupled-$r=16$ $\mathrm{sr}_B$ in the
direction predicted, while the divergence at $r=128$ is small. But the
data does not establish causation: the rank concentration may be a side
effect of some other axis the optimizers differ on, rather than the
cause of the loss gap.

### Dispersion of $S_B$ predicts the sign of $\Delta$ at every existing cell

Pulled all $(r, \eta=3 \times 10^{-4})$ cells where both coupled and
uncoupled were recorded. Computed $\kappa_B^\text{median}$ = median
across pairs of $\sigma_\max(S_B) / \sigma_\min(S_B)$, late-window
(step 1000–2000) of the uncoupled run.

| $r$ | $\Delta$ (cou − unc) | $\kappa_B$ median (uncoupled, late) |
|-----|-----------------------|-------------------------------------|
| 16  | $+0.0069$ (lose)      | 2.45                                |
| 64  | $-0.0072$ (win)       | 10.96                               |
| 128 | $-0.0100$ (win)       | 35.85                               |
| 256 | $-0.0107$ (win)       | 159.87                              |

$\kappa(S_B)$ is monotone in $r$ and the sign of $\Delta$ flips between
$r=16$ ($\kappa \approx 2.5$, near-flat spectrum) and $r=64$
($\kappa \approx 11$). At $r=16$, the polar pipeline's $S_B^{-1/2}$
factor is close to a scalar — it doesn't reshape directions meaningfully;
coupled's iter-2 refinement on top has nothing to refine and the
cross-term destabilizes. At $r \ge 64$, $S_B$ is peaky enough for the
preconditioner to do meaningful directional work; coupled refines
usefully.

Mechanism details (whether the failure mode at low $\kappa$ is best
described as "polar near-identity → cross-term is perturbation on Adam
direction" or as "bilinear cross-term concentrates $\mathrm{col}(B)$")
may be different framings of the same phenomenon. Both predict
$\kappa$-as-predictor.

### Sweep results

Two follow-up sweeps have completed (results in `logs/`).

**`alpha_sweep_2x2`** — $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\} \times
r \in \{16, 128\}$ at $\eta=3 \times 10^{-4}$, `picard_iters=2`. Per the
glossary entry for `picard_alpha`: at $r=16$, interior
$\alpha \in \{0.25, 0.5, 0.75\}$ are all worse than both endpoints
($\alpha = 0$ recovers uncoupled, $\alpha = 1$ is the default coupled).
There is no benign middle setting that splits the difference at $r=16$.

**`picard_iters_sweep_2x2`** — $k \in \{1, 2, 3, 4\} \times r \in \{16, 128\}$
at $\eta=3 \times 10^{-4}$, $\alpha=1$. Per the glossary entry for
`picard_iters` and H5: at $r=16$, $k=1$ is best and $k \ge 2$ is worse;
at $r=64$ (and by extension the larger-$r$ regime), $k=2$ wins.
Converging more deeply to the joint-NE fixed point makes things worse
at $r=16$, consistent with the bilinear-feedback diagnosis: more
iterations apply the cross-term more times, reinforcing $\mathrm{col}(B)$
further.

Both sweep outcomes are consistent with the H4′ prediction
(monotone-worse-with-coupling at low $\kappa$). They do not by themselves
upgrade the claim from correlative to causal.

### What is not pinned down

- **Causation.** Correlation of $\mathrm{sr}_B$ drift with the loss gap
  is consistent with H4′ but does not rule out shared upstream causes.
- **Whether $\alpha$ and `picard_iters` trace the same axis** or are
  separable controls along different paths into the joint-NE solution.
  The α-sweep result (interior $\alpha$ worse than endpoints at $r=16$)
  hints at non-separability — a smaller cross-term is not simply a
  fractional version of a larger one — but does not settle it.

### Logged groups

- `diag_h1234_2x2` — 2×2 (coupled/uncoupled × $r \in \{16, 128\}$) at
  $\eta = 3 \times 10^{-4}$ with H1–H4′ probes.
- `alpha_sweep_2x2` — $\alpha \in \{0, 0.25, 0.5, 0.75, 1\} \times
  r \in \{16, 128\}$ at $\eta = 3 \times 10^{-4}$, `picard_iters=2`.
- `picard_iters_sweep_2x2` — `picard_iters` $\in \{1, 2, 3, 4\}
  \times r \in \{16, 128\}$ at $\eta = 3 \times 10^{-4}$, $\alpha = 1$.
