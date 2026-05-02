# Coupled-polar LoRA optimizer — empirical investigation and open puzzle

Companion to `polar_coupled_problem.md` (problem statement) and
`polar_coupled_core_solver.md` (candidate solution). Those docs state
the variational problem and propose a solver. This doc records the
empirical study of that solver against a strong empirical baseline
(hybrid Picard) and surfaces the open puzzle: **why does a
variationally principled $\tfrac{1}{2}$-approximation to the joint
operator-norm tangent step lose to a damped fixed-point recipe that
makes no variational claim?**

We are looking for outside advice on what to try next.

All numbers single-seed, 2k-step horizon, on
[OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B)
fine-tuned on
[Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K-Instruction-Response).
LoRA on `all-linear` excluding `lm_head`, $\alpha = r$,
PEFT init ($A \sim$ Kaiming, $B = 0$).

---

## 1. Notation recap

We follow `polar_coupled_problem.md`. For each frozen base weight
$W \in \mathbb{R}^{m \times n}$ ($m = d_\text{out}$,
$n = d_\text{in}$), a LoRA correction
$W \to W + \tfrac{\alpha}{r} B A$ with
$A \in \mathbb{R}^{r \times n}$, $B \in \mathbb{R}^{m \times r}$.
Per-factor gradients

$$
G_A := \nabla_A L = (\alpha/r)\, B^\top \nabla_W L, \qquad
G_B := \nabla_B L = (\alpha/r)\, \nabla_W L\, A^\top.
$$

Compatibility (raw autograd):
$G_A A^\top = B^\top G_B$.

The joint operator-norm step solves

$$
\min_{\Delta A,\Delta B}\; \langle G_A, \Delta A \rangle +
\langle G_B, \Delta B \rangle
\quad \text{s.t.} \quad
\| B \Delta A + \Delta B\, A \|_2 \le \lambda.
$$

---

## 2. Two algorithms

### 2.1 Our solver — projected-quotient-polar core (this is the §2.1 baseline; experiments E1–E7 below are modifications of this)

Implementation: `lora_playground/optim.py::PolarCoupledCoreLoRA`,
helpers `_polar_coupled_core_step`, `_polar_coupled_core_lift`. Matches
`polar_coupled_core_solver.md` Sections 1–3. Per step:

1. Thin QR: $B = Q_L R_L$, $A = R_R Q_R^\top$.
2. Active core construction:

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
4. Compact polar $P = \mathrm{polar}(\widehat H)$, then project the
   $(2,2)$ block to zero, renormalize:

   $$
   R = \Pi(P), \quad \gamma = \|R\|_2, \quad
   \widehat Z_+ = R / \gamma.
   $$

   $\gamma \in [1, 2]$ certifies a deterministic
   $\tfrac{1}{2}$-approximation to ($\dagger$).
5. Scale (squared-penalty default):
   $\widehat Z_\text{upd} = -\eta\, \tau\, \widehat Z_+$ where
   $\tau = \|\widehat H\|_* / \gamma$.
6. Lift to factor space via min-Frobenius gauge (Sylvester solve):
   write $\widehat Z_\text{upd}$ as blocks $X, Y, W$
   ($X$ is $(1,1)$, $Y$ is $(1,2)$, $W$ is $(2,1)$). Solve
   $S_L K + K S_R = R_L^\top X R_R^\top$ for $K$ where
   $S_L = R_L^\top R_L + \delta I$,
   $S_R = R_R R_R^\top + \delta I$, then

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
$\|\Delta A\|/\|\Delta B\|$, the iLoRA imbalance residual
$\|A A^\top - \rho B^\top B\|_F / (\cdot)$ with $\rho = r/m$.

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

Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, $\delta
= 10^{-6}$, $\alpha=1$ (the Picard damping factor — not LoRA $\alpha$),
`picard_iters=2`.

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

| optimizer | $r=16$ best lr | $r=16$ eval | $r=64$ best lr | $r=64$ eval |
|---|---|---|---|---|
| Uncoupled spectral-product (Picard with `picard_iters=1`) | 3e-4 | **0.7546** | — | — |
| Hybrid Picard (`picard_iters=2`, the algorithm of §2.2) | 3e-4 | 0.7557 | 3e-4 | **0.7382** |
| AdamW | 3e-4 | 0.7601 | 3e-4 | 0.7550 |
| Adam-then-Sylvester (Adam EMA on factors, then closed-form L²-balanced Sylvester step) | 1e-3 | 0.7581 | — | — |

At $r=16$ the spread among baselines is small (Picard $\approx$ AdamW).
At $r=64$ Picard has a real $\sim 0.017$ lead over AdamW. **The
serious test is $r=64$.**

### 3.2 Our solver and its variants — table

We use **experiment ID E1, E2, ...** for our results. Each row gives
one sentence of what changes from the baseline solver of §2.1.

| ID | what changes from §2.1 baseline | $r=16$ best | $r=64$ best | gap to best |
|---|---|---|---|---|
| E1 | nothing (the §2.1 baseline) | 0.8188 (lr 3e-3) | 0.7821 (lr 3e-3) | +0.064 / +0.044 |
| E2 | post-step state rebalance: rotate $(A,B) \to (R^{-1}A, BR)$ to enforce iLoRA invariant $A A^\top = (r/m) B^\top B$, preserving $BA$ exactly | 0.8104 | 0.7686 | +0.056 / +0.030 |
| E3 | wider lr scan on E1 (extend to 1e-2, 3e-2) | 0.8049 | **0.7490** (lr 3e-2) | +0.050 / +0.011 |
| E4 | core sign-norm, $\widehat H \to \widehat H / (\|\widehat H\| + \varepsilon)$ elementwise before polar (per-coord adaptivity in core space, no momentum) | **0.7680** (lr 1e-4) | diverges | +0.013 / — |
| E5 | core-EMA + Nesterov (Muon-style on the rotating $Q_L, Q_R$ basis with overlap-matrix transport: $M_t = \beta\, T_L M_{t-1} T_R^\top + (1-\beta)\widehat H_t$) | 0.9073 | 0.8883 | far worse |
| E6 | compounds: E4 ⊕ E5 ⊕ E2 in 4 combinations | 0.7684 | 0.9440 | tied with E4 / worse |
| E7 | factor-Adam preconditioning before §2.1 (Adam EMA on $G_A, G_B$, feed $u_A, u_B$ into the §2.1 solver) — closest analog of Picard with our solver replacing Picard's per-factor polar | 0.7846 (lr 1e-4) | extrap $\sim$0.95 | +0.030 / extrap +0.21 |
| E8 | cross-check on a DIFFERENT LoRA solver: take a Sylvester-based factor-Adam solver that already works (the one labeled "Adam-then-Sylvester" in §3.1, eval 0.7581 at $r=16$) and move its Adam EMA into core space ($r \times r$ Sylvester RHS matrix) instead of factor space — DIVERGES at step 2 | div | div | — |

**Best so far per rank, both from our solver family:**

- $r=16$: **E4** (core sign normalization), $0.7680$. Still $+0.013$
  behind the $r=16$ best (uncoupled spectral-product, $0.7546$).
- $r=64$: **E3** (E1 + lr=3e-2), $0.7490$. Beats AdamW ($0.7550$);
  still $+0.011$ behind hybrid Picard ($0.7382$).

### 3.3 What each experiment tells us (one mechanism per row)

- **E2** (state rebalance): drives the iLoRA invariant
  $\|A A^\top - (r/m) B^\top B\|_F / (\cdot)$ from $1.0 \to 10^{-3}$ in
  two steps. Mechanism works as designed; eval gain $\le 0.014$.
  Conclusion: factor-state imbalance is real but not the bottleneck.
- **E3** (wider lr scan): the §2.1 baseline's lr ceiling is around $3\times$
  the canonical Adam lr ($3 \times 10^{-2}$ vs $3 \times 10^{-4}$).
  Beats AdamW at $r=64$. At $r=16$ ceiling $\sim 0.80$, so wider lr
  does not save us at low rank.
- **E4** (core sign): per-coord adaptivity in **core space** (after
  the $Q_L, Q_R$ basis projection). First variant to break the $r=16$
  ceiling; useless at $r=64$.
- **E5** (transported core EMA, Muon-style): the principled
  "Muon-on-LoRA-tangent" answer — should be the right thing
  theoretically, but is the worst variant tested. Diagnostics in §4.2
  show why.
- **E7** (factor-Adam, the closest analog of Picard with our solver):
  replicates Picard's preconditioning step but feeds into the §2.1
  solver instead of Picard's per-factor polar. **Does not help.**
  This is the experiment that most directly tests "is the missing
  piece factor-space adaptivity?" Answer: no.
- **E8** (cross-check on a different working solver — the
  Adam-then-Sylvester baseline from §3.1, which gets $0.7581$ at
  $r=16$; we move its Adam EMA from factor space to its $r \times r$
  core/Sylvester matrix and rerun): **diverges at step 2** of OLMo-2-1B
  smoke; Cholesky fails at step 3. Mechanism: $\sqrt{v_M}$ on a small
  $r\times r$ matrix (homogeneous coordinate scales) degenerates to
  $\approx 3\,\mathrm{sign}(M)$ at step 1, inflating step magnitude.
  This independently confirms E5's failure-mode generalizes — core-
  space Adam-style momentum is structurally broken because the core
  object lacks the heterogeneous coordinate scales that Adam's
  $\sqrt{v}$ rescaling exists to normalize.

---

## 4. Diagnostic findings

### 4.1 `compat` — gradient compatibility violation

Defined as
$\|C_L - C_R\|_F / (\|C_L\|_F + \|C_R\|_F + \varepsilon)$, where
$C_L = R_L^{-\top} G_A Q_R$, $C_R = Q_L^\top G_B R_R^{-\top}$.

- Variants 1–6 (raw factor gradients): $\mathrm{compat} \approx
  \varepsilon_\text{machine}$. Compatibility holds by construction; the
  averaging step in (2.1.2) is averaging two equal numbers.
- Variant 7 (factor-Adam): $\mathrm{compat} \in [0.65, 0.88]$ in r=4
  smoke at early steps. **Factor-Adam genuinely breaks
  compatibility**; the $\tfrac{1}{2}(C_L + C_R)$ averaging is doing
  real work and may be lossy.

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
where $M_\text{transported} = T_L M_{t-1} T_R^\top$ with overlap
matrices $T_L = U_\text{cur}^\top U_\text{prev}$, $T_R =
V_\text{cur}^\top V_\text{prev}$.

Median $\approx 0.04$, max $\approx 0.10$. **Small.** Transport is
fine. Variant 5's failure is the EMA itself, not the transport
mechanism.

### 4.4 $\gamma$, $\mathrm{relgap}$ — solver health

$\gamma \in [1, 2]$ on every step. $\mathrm{relgap} = 1 - 1/\gamma$
typically $0.05$–$0.15$. The $\tfrac{1}{2}$-approximation certificate
holds. Polar is computed correctly.

### 4.5 `imbalance_residual`

Drops from $\approx 1.0$ at PEFT init to $\approx 10^{-3}$ in 2 steps
under state rebalance and stays there. The factor-state geometry can
be fully restored to the iLoRA invariant — and it doesn't help.

---

## 5. The puzzle

Theory says our solver should beat Picard. The argument:

1. Both algorithms are doing "Muon-style" updates on the LoRA tangent.
2. Picard makes no variational claim — it's a damped fixed-point
   iteration.
3. Our solver has a deterministic $\tfrac{1}{2}$-approximation
   guarantee for the principled joint operator-norm objective.
4. Therefore at fixed preconditioning, ours should be no worse and
   plausibly better.

But E7 directly tests this: replace our raw factor gradients
with Picard's exact Adam-preconditioned $u_A, u_B$, run our solver,
compare to Picard at the same lr. Result: ours is $\sim 0.025$ worse
at $r=16$ (extrapolated from step 1200 trajectory) and catastrophically
worse at $r=64$. Even with matched preconditioning, our step
direction loses to Picard's.

**This is the puzzle we are stuck on.** Our principled solver is
producing a worse practical step than Picard's recipe.

---

## 6. Open hypotheses (for external review)

In our order of belief, with a quick test for each.

### H1. We are solving the wrong variational problem

The joint operator-norm constraint
$\|B \Delta A + \Delta B\, A\|_2 \le \lambda$ is variationally clean
but may not be what the fine-tuning loss landscape rewards. Picard's
recipe is "factor-Adam → per-factor polar in spectrally-preconditioned
space → RMS-align" — none of which corresponds to constraining the
joint tangent operator norm. Empirically, that recipe wins.

**Test:** drop the joint constraint and run a per-factor polar in our
gauge framework — i.e.\ apply our Sylvester gauge lift on top of
Picard's per-factor polar steps, post-hoc enforcing
$B^\top \Delta B = \Delta A\, A^\top$. If this beats Picard, our gauge
analysis is the missing piece. If it ties, gauge is irrelevant. If it
loses, our solver is structurally wrong.

### H2. The $(2,2)$ zero projection discards real signal

Our active-core construction sets $\widehat H_{22} := 0$ (the "extend
both $A$ and $B$ into new directions simultaneously" mode). The
solver doc justifies this from the $(2,2)$ block of feasible
$\widehat Z$ being zero. Picard's per-factor polar has no such
restriction — its step on $A$ can flow signal into directions $B$
doesn't currently span, and vice versa, simultaneously.

**Test:** un-zero the $(2,2)$ block, replace $\Pi(P)$ with $P$
itself in step (2.1.4). One-line change; it would violate the
variational story but tells us if the projection is empirically
costly.

### H3. Step magnitude mismatch

Our step magnitude is $\eta \tau = \eta \|\widehat H\|_* / \gamma$.
Picard's is $\eta \|u_A\|$ (RMS-aligned to Adam direction norm). At
fixed $\eta$ these are different scales. We see this empirically:
E3 (E1 + wider lr) needs $\eta = 3 \times 10^{-2}$ at $r=64$ to be
competitive, while Picard's optimum is $\eta = 3 \times 10^{-4}$
— a 100$\times$ ratio.

**Test:** plot $\|\Delta A\|, \|\Delta B\|$ trajectories of E1 vs
Picard at their best lr's. Are the effective per-step magnitudes
matched? If ours is way larger (or smaller), that's a knob we haven't
calibrated. (E7 supposedly fixes this by inheriting Picard's
preconditioning — but we still see catastrophic behavior at $r=64$,
suggesting the magnitude story is more subtle.)

### H4. Symmetrization $\tfrac{1}{2}(C_L + C_R)$ is lossy when compat is high

Variant 7 directly hits this: factor-Adam → compat 0.65–0.88. The
averaging projects two genuinely different views into a single one.
Picard avoids it by never building $C$ — instead, two separate
per-factor polars, each operating on its own preconditioned gradient.

**Test:** version of our solver that never symmetrizes — keep $C_L$
and $C_R$ as separate inputs, do separate per-factor polars in core
space, lift separately. (Closely related to H1's test.)

### H5. Picard's iteration is doing something we lack

Picard's $k=2$ inner step adds
$\alpha (B^\top \Delta B_\text{prev} A)/\eta$ to $u_A$ and
$\alpha (B \Delta A_\text{prev} A^\top)/\eta$ to $u_B$. This is a
fixed-point iteration on the joint normal equations with damping
$\alpha = 1$. Our solver is one-shot. Even if our one-shot direction
is $\tfrac{1}{2}$-optimal for the *single-step* objective, Picard's
iteration may be converging to a better fixed point of the implicit
training dynamics across steps — or to a different objective entirely.

**Test:** sensitivity of Picard to `picard_iters $\in \{1, 2, 3, 5, 10\}$.
- iters=1 disables the cross-coupling entirely — that's
  `adam-polar-product-lora` (uncoupled), eval $0.7546$ at $r=16$.
- iters=2 (default) is $0.7557$ at $r=16$.
- We have not swept iters=3+. **This is a config-only experiment
  with zero new code.**

If iters=2 is significantly better than iters=1, Picard's iteration
is doing real work, and our one-shot solver structurally cannot match.
If iters $\geq 2$ is flat or worse, the iteration isn't the
explanation.

### H6. We have a bug

Possible defects worth re-checking:

- The $\alpha/r$ LoRA scaling. PEFT applies $\alpha/r$ at the model
  layer; our solver does not separately scale. Picard also does not.
  Should match in principle. Worth a 1-pair trace to confirm.
- The Sylvester lift formula in (2.1.6). Our test 4 in
  `tests/test_polar_coupled_core.py` verifies that on synthetic random
  $(A, B, G_A, G_B)$ pairs, our solver in `core_norm="frobenius"` mode
  (i.e.\ no operator constraint, equivalent to plain GD on the joint
  problem) matches a hand-derived Sylvester closed form to $10^{-5}$.
  This passes. Worth a single real-LoRA-pair trace at $r=4$ to verify
  the operator-norm path matches an alternative implementation.
- The $B = 0$ PEFT-init boundary case. Our solver triggers
  `_zero_B_fallback` at step 1; thereafter regular path. Picard's
  $S_B^{-1/2} = (B^\top B + \delta I)^{-1/2}$ → $\delta^{-1/2} I$
  smoothly handles it. Could the fallback step's magnitude differ
  from what the regular path would compute on $B = \varepsilon\, X$
  for small $\varepsilon$? Cheap test.

---

## 7. What we plan to try (subject to advice)

In order of expected information per GPU-hour:

1. **Picard sensitivity to `picard_iters`** (tests H5). Config-only,
   $\sim 50$ min wall on 16 GPUs. Free.
2. **Variant 1 with $(2,2)$ block UN-zeroed** (tests H2). One-line
   change. Tells us if $\Pi$ is empirically costly.
3. **Step-magnitude diagnostic**: trajectory plot of $\|\Delta A\|,
   \|\Delta B\|$ for E1 vs Picard at best lr's (tests H3).
   No new code.
4. **Per-factor polar with our gauge lift** (tests H1, H4). New
   optimizer: do Picard's per-factor polar steps, but post-hoc lift
   into the min-Frobenius gauge. If it beats Picard, we have a
   reason to ship it; if it ties or loses, our gauge analysis is
   irrelevant.
5. **Real-LoRA-pair Sylvester-lift trace** (tests H6).
   $\sim 30$ min, one-pair printout.

---

## 8. What we want from external review

Concretely:

- **Are we mis-formulating the variational problem?** Is the
  operator-norm constraint on
  $\|B \Delta A + \Delta B\, A\|_2$ even the right object for
  fine-tuning? Should it be a different norm, a different combination,
  or constrained on each factor separately?
- **Is the $(2,2)$-zero projection actually justified?** The doc's
  argument is that feasible $\widehat Z$ has zero $(2,2)$ block, so
  $\widehat H_{22}$ is undefined-as-data. But the *polar of $\widehat H$*
  has nonzero $(2,2)$ entries before $\Pi$ is applied; we throw them
  away. Is there a principled reformulation where we don't?
- **Is the symmetrization $C = \tfrac{1}{2}(C_L + C_R)$ throwing away
  signal Picard preserves?** When $\mathrm{compat}$ is high (e.g.,
  factor-Adam), is there a better way to combine $C_L$ and $C_R$ than
  averaging?
- **Why does core-space momentum fail?** Variant 5's `align_mom`
  $<$ `align_inst` data and E8's outright divergence say
  core-space EMA-Adam is structurally broken. Is the rotating
  $Q_L, Q_R$ basis truly incompatible with momentum, or is there a
  variant of basis transport / parallel-transport that would fix it?
- **Is Picard's fixed-point iteration empirically equivalent to
  something we could compute one-shot?** If `picard_iters=10` matches
  the iters=2 result, we know the iteration is essentially
  block-Jacobi. If iters=10 is much better, our one-shot story has
  a structural problem.

---

## 9. Reproducibility

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

Code: `lora_playground/optim.py`,
classes `PolarCoupledCoreLoRA` (variants 1–4),
`MuonCoupledCoreLoRA` (E5),
`PolarCoupledCoreFactorAdamLoRA` (E7),
`AdamLinCoreLoRA` (E8 cross-check),
`AdamPolarProductLoRA` (Picard).
