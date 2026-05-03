# Clipping-prox optimizer — proposal

## 0. Background and vocabulary

This proposal modifies one line inside an existing optimizer family. To see which line, you need the algorithmic skeleton; everything else is a substitution into it.

### 0.1 Setup

A LoRA pair is

$$A \in \mathbb{R}^{r \times n},\qquad B \in \mathbb{R}^{m \times r},$$

contributing $BA$ to the effective weight $W + BA$ ($\alpha/r$ folded in at use-time). One optimizer step on a pair produces an update $(\Delta A, \Delta B)$.

Each factor carries its own Adam state. The Adam updates are

$$u_A := \hat m_A / (\sqrt{\hat v_A} + \varepsilon),\qquad u_B := \hat m_B / (\sqrt{\hat v_B} + \varepsilon).$$

The polar-product family uses $u_A, u_B$ in place of the raw gradients $G_A, G_B$ throughout the per-block solves below — this is the "Adam-covector compromise" discussed in §2.6.

### 0.2 What the optimizer is solving

**The coupled program.** For each LoRA pair, the polar-product family targets the per-step program

$$\min_{\Delta A,\, \Delta B}\ \underbrace{\langle u_A,\, \Delta A\rangle + \langle u_B,\, \Delta B\rangle}_{\text{linear cost (Adam direction)}}\ +\ \underbrace{\frac{1}{2\eta}\,\|B \Delta A + \Delta B\, A\|_F^2}_{\text{Frobenius penalty on the joint tangent}} \quad \text{s.t.}\ \underbrace{\|B \Delta A\|_2 \le \tau,\ \ \|\Delta B\, A\|_2 \le \tau}_{\text{per-block spectral constraint}}.$$

Three pieces, each named for use throughout:

- **Linear cost.** Inner product of the Adam-preconditioned covector with the update — the descent term. The variationally clean version uses raw gradients $G_A, G_B$ here; the substitution to $u_A, u_B$ is a known compromise (§2.6).
- **Frobenius penalty.** Acts on the *joint tangent* $J := B \Delta A + \Delta B\, A$. This is the only term that couples the two blocks: a step on $\Delta A$ alone cannot be evaluated without knowing $\Delta B$, because both feed into $J$. Penalty weight $1/(2\eta)$ — discussion of this choice in §2.3.
- **Per-block spectral constraint.** Caps the operator norm of each block's contribution to $J$. The radius $\tau$ is the only quantity the program leaves free; we set it via a scale-invariant shape parameter $c$ in §0.4.

(`theory.md` derives this program as the formulation that hybrid Picard implicitly targets; §2.1 reviews the derivation.) The program is **coupled** through the Frobenius penalty.

**Why iterate over blocks.** Solving the program jointly in $(\Delta A, \Delta B)$ requires a large eigendecomposition per step — too expensive. The family does **block-coordinate (Picard) iteration** instead:

- Hold $\Delta B$ fixed at its previous inner-iterate; solve the $A$-subproblem.
- Hold $\Delta A$ fixed; solve the $B$-subproblem.
- Repeat for $k$ inner iterations.

**The cross-coupling target.** The off-block enters the on-block only through a single $r \times n$ matrix:

$$T_A\ :=\ -Q_B^\top\, \Delta B_\text{prev}\, A,\qquad T_B\ :=\ -B\, \Delta A_\text{prev}\, Q_A$$

($Q_B, Q_A$ from the QR of $B, A$, see the "Whitening" piece below). Whatever the other block's previous-pass update was, only this matrix matters for the current pass's subproblem.

**Iteration order.** The repo uses **Jacobi**:

- Both blocks read the *previous pass's* $\Delta A, \Delta B$.
- Both update together at the end of the pass.

The alternative is Gauss–Seidel — feed the fresh $\Delta A$ into the $B$-subproblem within the same pass. We do not use it.

Naming:
- $k = 1$ — **uncoupled**. Cross-coupling targets are zero on the first pass.
- $k \ge 2$ — **coupled**. Each pass after the first sees a nonzero target.

**Each block subproblem has three pieces.** The $A$-subproblem (with $\Delta B$ held at $\Delta B_\text{prev}$) is a quadratic in $\Delta A$ with a spectral-norm constraint. It decomposes as:

1. **Whitening.** The Frobenius penalty $\|B \Delta A + \Delta B_\text{prev} A\|_F^2$ has $B$ in front, which mixes the singular directions of $\Delta A$ in an awkward way. Take the thin QR

   $$B = Q_B R_B,\qquad Q_B \in \mathbb{R}^{m \times r}\ \text{column-orthonormal},\quad R_B \in \mathbb{R}^{r \times r}\ \text{upper-triangular},$$

   and change variables to $X := R_B^\top \Delta A \in \mathbb{R}^{r \times n}$. Then $B \Delta A = Q_B X$, and using $T_A = -Q_B^\top \Delta B_\text{prev} A$ from above, the Frobenius penalty collapses to the clean form $\|X - T_A\|_F^2$. (At the first inner pass, $\Delta B_\text{prev} = 0$ so $T_A = 0$.) The linear cost $\eta \langle u_A, \Delta A \rangle$ becomes $\eta \langle R_B^{-\top} u_A, X \rangle$ under the same change of variables.

2. **Unconstrained block prox.** Setting the gradient of (linear cost) + $(1/2\lambda)$(coupling penalty) to zero gives

   $$X_{A,\text{unc}} = T_A - \eta\, R_B^{-\top} u_A.$$

   (We commit to $\lambda = \eta$ in §2.3.) This is the optimum *without* the spectral constraint.

3. **Spectral operator $\mathcal{P}$.** The spectral-norm constraint $\|X\|_2 \le \tau$ is enforced by projecting $X_{A,\text{unc}}$ through a per-block operator $\mathcal{P}(\cdot;\, \tau)$. §2.1 derives the variationally correct $\mathcal{P}$ as **singular-value clip**. Picard currently uses **polar** instead (§0.3 below). *This is the only design freedom in the per-block solve, and it is the one line this proposal changes.*

The $B$-subproblem is symmetric: row QR $A = R_A Q_A^\top$ (with $Q_A \in \mathbb{R}^{n \times r}$ column-orthonormal, $R_A \in \mathbb{R}^{r \times r}$); decision variable $Y := \Delta B\, R_A^\top$; target $T_B := -B \Delta A_\text{prev} Q_A$; unconstrained $Y_{B,\text{unc}} = T_B - \eta\, u_B R_A^{-1}$.

**The lift.** After $\mathcal{P}$ returns $(X_A^\star, Y_B^\star)$, we need a factor pair $(\Delta A, \Delta B)$ whose joint tangent matches the prox solution. Three facts:

- The map $(\Delta A, \Delta B) \mapsto B \Delta A + \Delta B\, A$ is many-to-one (gauge freedom in the bilinear parametrization).
- The family picks the **min-Frobenius representative** — the unique pair minimizing $\|\Delta A\|_F^2 + \|\Delta B\|_F^2$.
- It is computed via a small Sylvester solve (formula in §2.5).

For the skeleton, treat $\mathrm{Lift}(X_A^\star, Y_B^\star) \to (\Delta A, \Delta B)$ as a black box.

**Skeleton.** Putting the pieces together, per step per pair:

$$
\begin{aligned}
&\textbf{inputs: }\ A,\ B,\ u_A,\ u_B,\ \eta,\ k,\ \mathcal{P} \\[2pt]
&\text{QR: }\ B = Q_B R_B,\quad A = R_A Q_A^\top \\[2pt]
&\Delta_A \leftarrow 0,\quad \Delta_B \leftarrow 0 \\[2pt]
&\textbf{for }\ j = 1, \dots, k\ \textbf{ do} \\
&\quad T_A \leftarrow -Q_B^\top\, \Delta_B\, A &&\text{(cross-coupling target;\ }= 0\text{ on }j=1\text{)} \\
&\quad T_B \leftarrow -B\, \Delta_A\, Q_A &&\text{(cross-coupling target;\ }= 0\text{ on }j=1\text{)} \\
&\quad X_{A,\text{unc}} \leftarrow T_A\ -\ \eta\, R_B^{-\top}\, u_A &&\text{(unconstrained }A\text{-block prox)} \\
&\quad Y_{B,\text{unc}} \leftarrow T_B\ -\ \eta\, u_B\, R_A^{-1} &&\text{(unconstrained }B\text{-block prox)} \\
&\quad X_A^\star \leftarrow \mathcal{P}(X_{A,\text{unc}};\, \tau_A) &&\text{(}\leftarrow\text{ polar today; clip in this proposal)} \\
&\quad Y_B^\star \leftarrow \mathcal{P}(Y_{B,\text{unc}};\, \tau_B) &&\text{(}\leftarrow\text{ polar today; clip in this proposal)} \\
&\quad (\Delta_A,\, \Delta_B) \leftarrow \mathrm{Lift}(X_A^\star,\, Y_B^\star) &&\text{(min-Frobenius gauge, §2.5)} \\
&\textbf{end for} \\[2pt]
&A \leftarrow A + \Delta_A,\quad B \leftarrow B + \Delta_B
\end{aligned}
$$

Every line was motivated above. The two $\mathcal{P}$ lines are the only design freedom; $\tau_A, \tau_B$ come from the shape parameter $c$ (§0.4).

### 0.3 The operator $\mathcal{P}$ — polar vs clip

$\mathcal{P}$ acts on a matrix $X$ via its compact SVD $X = U \Sigma V^\top$ with a target spectral radius $\tau \ge 0$:

$$\mathcal{P}_\text{polar}(X; \tau)\ =\ \tau \cdot U V^\top \qquad \text{(every }\sigma_i \to \tau\text{)}$$

$$\mathcal{P}_\text{clip}(X; \tau)\ =\ U\,\mathrm{diag}\bigl(\min(\sigma_i, \tau)\bigr)\,V^\top \qquad \text{(only }\sigma_i > \tau\text{ are capped)}$$

Picard's current implementation uses $\mathcal{P}_\text{polar}$. **This proposal swaps in $\mathcal{P}_\text{clip}$ at exactly those two lines in the skeleton — nothing else changes.**

Where they differ:

- On any direction with $\sigma_i < \tau$: polar **amplifies** it up to $\tau$; clip **leaves it alone**.
- They agree only when every singular value is already at or above $\tau$.

§2.1 derives clip as the variationally correct block-prox solution to the program in §0.2; polar is a different operator that the current code uses heuristically.

### 0.4 The threshold $\tau$ — clip is deferred for this campaign

The variational form (§2.1) requires a numerical value for $\tau$ but does not derive one. Sweeping $\tau$ to find the best value re-introduces a per-problem hyperparameter, which is exactly what we are trying to avoid. We do not have a theoretical rule that picks $\tau$ a priori from $(A, B, u_A, u_B, \eta)$ in a way that survives across ranks and workloads.

**Decision:** the clip operator is deferred. This campaign tests the new **architecture** (QR whitening + Sylvester min-Frob lift, §3) with $\mathcal{P} = \mathcal{P}_\text{polar}$ — i.e., $\tau = \infty$ in the skeleton, no clipping. The headline question becomes: does the QR + Sylvester architecture, with polar as the operator, beat hybrid Picard at both ranks?

Clip remains the variationally-derived correct operator (§2.1 still stands as theory). It moves to a follow-up campaign once a defensible $\tau$-rule is identified — either from cross-workload theory or from a separate calibration experiment that picks one fixed $\tau$ and tests it without further tuning.

## 1. TL;DR

**Goal.** Ship one optimizer config (no problem-tunable hyperparameters — a single $(c, k)$ tuple must work across the workload distribution; rank-stability across $r \in \{16, 64\}$ on this workload is the necessary condition tested here) that simultaneously beats both per-rank winners of the polar-product family: $r=16$ at **0.7546** (uncoupled, $k=1$) and $r=64$ at **0.7382** (coupled, $k=2$). No existing config of the family meets both bars — at $r=16$ the coupled $k=2$ variant *loses* to AdamW (0.7616 vs 0.7579).

**Headline open question.** Within the polar-product family, replace polar with singular-value clip in the per-block operator. **Does clip at $k=2$ widen the lead at $r=16$ without regressing at $r=64$?**

**Mechanism hypothesis (not yet verified).** A guess about *why* clip might widen the lead at $r=16$:

- Polar saturates **every** nonzero singular direction in $X$ to magnitude $\tau$ — including small-$\sigma$ directions.
- We suspect that at small $r$, the small-$\sigma$ directions of the cross-coupling target $T_A = -Q_B^\top \Delta B_\text{prev} A$ are dominated by estimation noise (few rank directions; Adam-preconditioned; stale prior inner-iterate).
- Polar then amplifies that noise up to first-rank magnitude. Clip leaves it small.
- This would explain why polar's $k=2$ helps at $r=64$ (cross-coupling carries signal) but hurts at $r=16$ (cross-coupling is mostly noise).

If the mechanism is correct, clip at $k=2$ wins at both ranks. Q3 (§4) is the ride-along diagnostic that tests it.

**Falsification path.** A single $c$-sweep at $r=16$, $k=2$, fixed best $\eta$. If no $c$ beats 0.7546, clip doesn't help where the family is weakest, and the family is hyperparameter-saturated within this formulation. End the campaign.

## 2. Details deferred from §0

§0 stated the program (§0.2) and the algorithmic skeleton (§0.2 and §0.3). This section fills in the four pieces §0 deferred:

- §2.1 — proof that $\mathcal{P}_\text{clip}$ is the variationally correct operator.
- §2.3 — choice $\lambda = \eta$ in the Frobenius-penalty weight.
- §2.5 — explicit Sylvester formula for $\mathrm{Lift}$.
- §2.6 — the Adam-covector compromise.
- §2.7 — the $B = 0$ init boundary.

### 2.1 Why $\mathcal{P}_\text{clip}$ is the variationally correct operator

The program of §0.2 (with raw gradients $G_A, G_B$ written for the linear cost; the substitution to $u_A, u_B$ is §2.6's compromise) was shown by `theory.md` to be the formulation that hybrid Picard implicitly targets via block-coordinate descent.

Within one block-coordinate step on the $A$-subproblem, fix $\Delta B = \Delta B_\text{prev}$. Apply the change of variables from §0.2 — $X := R_B^\top \Delta A$, $T_A := -Q_B^\top \Delta B_\text{prev} A$, $L_0 := R_B^{-\top} u_A$. The subproblem becomes

$$\min_{X \in \mathbb{R}^{r \times n}}\ \eta\,\langle L_0,\, X\rangle + \tfrac{1}{2\eta}\,\|X - T_A\|_F^2 \quad \text{s.t.}\ \|X\|_2 \le \tau.$$

Steps:

- The unconstrained minimum (drop the spectral constraint) is found by setting the gradient to zero:

  $$X_{A,\text{unc}}\ =\ T_A\ -\ \eta\, L_0\ =\ T_A\ -\ \eta\, R_B^{-\top} u_A.$$

  This is the same expression as the skeleton's $X_{A,\text{unc}}$ line.

- Adding the constraint $\|X\|_2 \le \tau$ turns the problem into a Euclidean projection onto the spectral-norm ball of radius $\tau$, centered at $X_{A,\text{unc}}$ (because the linear term has been absorbed into the squared term — completing the square).
- The Euclidean projection of a matrix onto the spectral-norm ball is exactly **singular-value clipping**: SVD $X_{A,\text{unc}} = U\Sigma V^\top$, return

  $$X^\star\ =\ U\,\mathrm{diag}\bigl(\min(\sigma_i, \tau)\bigr)\,V^\top.$$

  (Standard fact: the closest matrix in Frobenius norm to $X$ subject to $\|\cdot\|_2 \le \tau$ caps each singular value at $\tau$ and leaves the singular vectors fixed.)

So $\mathcal{P}_\text{clip}$ is the closed form. Polar is a different operator (every singular value $\to \tau$, including those already below $\tau$); it has a different fixed point.

The $B$-subproblem is symmetric: row QR $A = R_A Q_A^\top$, decision variable $Y := \Delta B\, R_A^\top$, target $T_B := -B \Delta A_\text{prev} Q_A$, linear cost $\eta\, u_B R_A^{-1}$ on the right.

### 2.3 The $1/(2\eta)$ penalty weight

The Frobenius penalty in §0.2 is written with weight $1/(2\lambda)$ for $\lambda = \eta$. Why $\lambda = \eta$?

- In the limit $T_A = 0$, $c \to \infty$ (no cross-coupling, no clipping), the program reduces to the Frobenius-coupled Sylvester closed form.
- Its step Frobenius norm scales as $\eta \cdot \|R_B^{-\top} u_A\|_F$.
- Picard's update has step Frobenius norm $\eta \|u_A\|_F$, in a different basis ($S_B^{-1/2}$ rather than $R_B^{-\top}$).
- These two bases are inverse-square-roots of $S_B$ that differ by an orthogonal rotation, which preserves Frobenius norm. So the unclipped-limit step magnitudes match.

$\lambda = \eta$ is the choice that makes the unclipped limit step-magnitude-comparable to Picard.

**Natural-prox magnitude (no rescale).** Apply the clipped output directly:

- Picard currently rescales its polar output to Frobenius norm $\eta \|u\|_F$ ("Picard-rescale" / "RMS-align" in legacy code — despite the name, it is a Frobenius rescale, not an element-wise RMS).
- For the clip variant the rescale would violate the clip radius and convert the experiment from a test of the prox formulation into a direction-shaping heuristic.
- Picard-rescale is retained only as one ablation cell in Phase 3 (§5).

**Aside on $\tau$.** A naive Frobenius-scale choice $\tau = \eta \|u_A\|_F / \sqrt{\mathrm{tr}(S_B^{-1})}$ controls the *Frobenius* norm of the step, not its spectral norm — it does not act as a clip threshold. The threshold must be set against $\sigma_\text{max}(X_\text{unc})$ (i.e. the $c$ parametrization of §0.4).

### 2.5 Lift formula

§0.2 introduced $\mathrm{Lift}$ as a black box returning the min-Frobenius representative of the joint tangent. Closed form: solve the small Sylvester equation

$$S_B K + K S_A\ =\ R_B^\top X^\star R_A^\top, \qquad K \in \mathbb{R}^{r \times r}$$

(with $S_B = R_B^\top R_B$, $S_A = R_A R_A^\top$), then

$$\Delta A\ =\ S_B^{-1}\,\bigl(R_B^\top X^\star - K R_A\bigr)\,Q_A^\top \qquad \text{(symmetric for }\Delta B\text{)}.$$

This is the §4 lift formula in `theory.md` with the off-block-diagonal extensions zeroed out. Rationale: a rank-$r$ tangent has no component orthogonal to $\mathrm{col}(B)$ on the left or $\mathrm{row}(A)$ on the right, so those extensions vanish.

Sylvester solve and spectral preconditioners use existing utilities (`solve_sylvester`, `spdify` in `lora_playground/utils.py`); no new math infrastructure.

### 2.6 The Adam-covector compromise

The §2.1 variational program was written with raw factor gradients $G_A, G_B$ as the linear cost. The polar-product family substitutes the Adam-preconditioned covectors $u_A, u_B$ instead. This breaks an algebraic identity that $G_A, G_B$ would have satisfied, so the resulting block solve is not a literal optimum of §2.1 — it is the step-shape compromise Picard already lives with. Spelled out:

**The identity (raw gradients).** Let $W = BA$ and $G_W := \partial L / \partial W$. The chain rule gives

$$G_A\ =\ B^\top G_W,\qquad G_B\ =\ G_W A^\top.$$

Therefore

$$G_A A^\top\ =\ B^\top G_W A^\top\ =\ B^\top G_B.$$

This identity, $G_A A^\top = B^\top G_B$, says the two blocks see consistent linear cost projected through the bilinear parametrization.

**Why it breaks.** Adam preconditioning is applied **independently per factor** — $u_A$ uses $A$'s own first/second moments, $u_B$ uses $B$'s. There is no reason the rescaled quantities should still satisfy $u_A A^\top = B^\top u_B$. So substituting $(u_A, u_B)$ for $(G_A, G_B)$ in §2.1 turns the clean coupled program into a heuristic.

**Why we do it anyway.** Picard already lives with this compromise; AdaMuon and NorMuon do too. Using the same one across clip and polar keeps the experiment a clean A/B test of the $\mathcal{P}$ operator — both branches inherit the same break, so the differential effect is orthogonal to whether the substitution is "clean".

### 2.7 Init boundary: $B = 0$ at PEFT step 1

**Where the problem is.** The skeleton in §0.2 contains the line

$$X_{A,\text{unc}}\ =\ T_A\ -\ \eta \cdot R_B^{-\top}\, u_A.$$

PEFT initializes $B = 0$, so at step 1 the QR of $B$ has $R_B = 0$ and $R_B^{-\top}$ does not exist. Same for $R_A^{-1}$ if $A$ is ever zero (it is not, by Kaiming init). The skeleton is undefined as written.

**The fallback.** Picard's implementation (in `optim.py`) replaces the whitening matrix at step 1 by $\delta^{-1/2} I$ — the limit of $S_B^{-1/2} = (B^\top B + \delta I)^{-1/2}$ as $B \to 0$. Plugged back into the skeleton this gives, at step 1 only:

$$X_{A,\text{unc}}\ =\ -\eta \cdot \delta^{-1/2}\, u_A,\qquad \Delta B = 0,\qquad \Delta A \propto u_A,$$

i.e. a plain (unwhitened) Adam step on $A$ alone, with $B$ held at zero. From step 2 onward, $B \ne 0$ and the standard skeleton runs unchanged.

**Inherited verbatim by the clip variant.** Step 1 has no information to clip — the operator $\mathcal{P}$ is bypassed along with the whitening. Steps $\ge 2$ run the full skeleton with $\mathcal{P} = \mathcal{P}_\text{clip}$.

**Smoke check.** For early steps where $\sigma_\text{min}(R_B)$ is positive but small, $R_B^{-\top}$ can have large singular values and amplify $u_A$ before clipping sees it. Log $\|R_B^{-\top} u_A\|_F$ for the first ~50 steps of a smoke run; if it stays within an order of magnitude of $\|u_A\|_F$ the whitening is well-conditioned in practice.

## 3. Current state

All numbers single-seed, 2k-step horizon, sourced from logs.

| rank | current best | $\eta$ | eval @ 2k | source |
|---|---|---|---|---|
| 16 | uncoupled `adam-polar-product-lora` ($k=1$) | 3e-4 | **0.7546** | `logs/polar_product_2k/run_info/logs/log_2.out` |
| 64 | coupled `adam-polar-product-lora-coupled` ($k=2$) | 3e-4 | **0.7382** | `logs/adam_polar_product_coupled_r64_2k/run_info/logs/log_1.out` |
| 16 | AdamW (baseline) | 3e-4 | 0.7579 | `logs/lr_sweep_2k/run_info/logs/log_10.out` |
| 64 | AdamW (baseline) | 3e-4 | 0.7550 | `logs/h3_rsweep_2k/run_info/logs/log_02.out` |

Gap to close vs AdamW: 0.0033 at $r=16$, 0.0168 at $r=64$.

**Within-family rank-dependence.** Same family wins both ranks but with different optimal $k$:

| $r$ | $k=1$ | $k=2$ | $k=3$ | $k=4$ |
|---|---|---|---|---|
| 16 | **0.7546** | 0.7616 | 0.7557 | 0.7594 |
| 64 | 0.7453 | **0.7382** | — | — |

Sources: `logs/polar_product_2k/log_2.out` ($k=1, r=16$); `adam_polar_product_coupled_rsweep_2k/log_4.out` ($k=2, r=16$); `picard_iters_sweep_2x2/log_{0,1}.out` ($k=3,4$ at $r=16$); `polar_product_r64_diag_2k/log_1.out` ($k=1, r=64$); `adam_polar_product_coupled_r64_2k/log_1.out` ($k=2, r=64$).

**At $r=16$, $k=2$ coupled loses to AdamW** (0.7616 vs 0.7579). No existing config wins at both ranks — the bidirectional goal is unmet.

### Closed dead-ends

To prevent re-proposing approaches the project has ruled out:

| approach | result | source |
|---|---|---|
| Picard $k \in \{1,2,3,4\}$ at $r=16$ | $k=1$ best (0.7546); $k \ge 2$ worse, $k=2$ loses to AdamW | logs above |
| `picard_alpha` damping at $k=2$, $r=16$, $\alpha \in \{0.25, 0.5, 0.75\}$ | 0.7562 / 0.7582 / 0.7600 — interior worse than $\alpha \in \{0,1\}$ | `logs/alpha_sweep_2x2/` |
| Joint operator-norm core solver (E1–E7 in `investigations.md`) | best E3 = 0.7490 at $r=64$, lost to Picard 0.7382; E7 diverges at $r=64$ | `investigations.md` §3 |
| Polar-first composition (`adamuon-polar-product-lora`) | $r=16$ 0.7653, $r=64$ 0.7486 — worse than Adam-first at both | `optimizer_synthesis.md` leaderboard |
| Core-space momentum / Adam (E5, E8) | structurally broken: divergence or `align_mom < align_inst` | `investigations.md` §3 |

### How this proposal differs from existing variants

Each closely-related variant produces $(\Delta A, \Delta B)$ from $(u_A, u_B, A, B)$ by a different sequence of operations. Written as math:

#### Hybrid Picard, uncoupled — `AdamPolarProductLoRA`, $k=1$ (current $r=16$ winner)

With $S_A := AA^\top + \delta I$, $S_B := B^\top B + \delta I$, $\mathrm{polar}(M) = UV^\top$ for SVD $M = U\Sigma V^\top$ (computed via Newton–Schulz):

$$
\begin{aligned}
\widetilde{u}_A &= S_B^{-1/2}\, u_A,
&\widetilde{u}_B &= u_B\, S_A^{-1/2}, \\
g_A &= S_B^{-1/2}\,\mathrm{polar}(\widetilde{u}_A),
&g_B &= \mathrm{polar}(\widetilde{u}_B)\, S_A^{-1/2}, \\
\Delta A &= -\eta \cdot \tfrac{\|u_A\|_F}{\|g_A\|_F}\, g_A,
&\Delta B &= -\eta \cdot \tfrac{\|u_B\|_F}{\|g_B\|_F}\, g_B.
\end{aligned}
$$

$\Delta A$ and $\Delta B$ are computed independently — they share $A, B$ only through the preconditioners. No joint recombination.

#### Hybrid Picard, coupled — `AdamPolarProductLoRA(-coupled)`, $k \ge 2$ (current $r=64$ winner)

For $j = 1$: identical to the uncoupled case above. For $j \ge 2$:

$$
\begin{aligned}
u_A^\text{eff} &= u_A + \tfrac{1}{\eta}\, B^\top\, \Delta B_\text{prev}\, A &&\text{(or the equivalent form below)} \\
u_B^\text{eff} &= u_B + \tfrac{1}{\eta}\, B\, \Delta A_\text{prev}\, A^\top
\end{aligned}
$$

then run the same uncoupled pipeline ($S^{-1/2}$ whitening, polar, RMS-align) on $(u_A^\text{eff}, u_B^\text{eff})$.

Notes:

- The implementation picks one of two compatibility-equivalent expressions for the $A$-block cross-term: $(1/\eta)\, B^\top \Delta B_\text{prev} A$ or $(1/\eta)\, B\, \Delta A_\text{prev} A^\top$ (they coincide when gradient compatibility holds, diverge under independent per-factor Adam).
- The output rule (per-factor RMS-align to $\eta\|u_A\|_F, \eta\|u_B\|_F$) is unchanged; the rescale uses the *original* $\|u\|_F$, not the augmented $\|u^\text{eff}\|_F$ (`-coupled-endrms` toggles this).
- **Still no joint recombination** — $\Delta A$, $\Delta B$ remain per-factor.

#### `ProductMuonLoRA` / `AdamProductMuonLoRA`

Build one rank-$r$ matrix $D \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ that approximates the merged-weight gradient $\partial L / \partial W$ projected onto the LoRA subspace:

$$D\ =\ \tfrac{r}{\alpha}\, \cdot\, m_B\ \cdot\, A^\top (A A^\top + \delta I)^{-1}$$

where $m_B$ is the EMA-momentum (or Adam-preconditioned moment, for `AdamProductMuonLoRA`) of $\partial L / \partial B$, and the right factor $A^\top (AA^\top + \delta I)^{-1}$ is the damped right pseudoinverse of $A$. Then apply Newton–Schulz polar and Sylvester-recover:

$$
\begin{aligned}
P &= \mathrm{polar}(D), \\
B\, \Delta A + \Delta B\, A &= -\eta\, P, \quad \text{s.t.}\quad B^\top \Delta B = \Delta A\, A^\top \quad \text{(min-Frob gauge)}.
\end{aligned}
$$

Notes:

- $D$ is gauge-invariant under $A \to RA,\ B \to BR^{-1}$ — both factors of $D$ transform compatibly, and the $R$ cancels.
- One operator ($\mathrm{polar}$) on a joint $W$-space matrix; no per-block whitening; Sylvester is the only mechanism producing the factor pair.
- The $\partial L / \partial A$ channel is intentionally not used to build $D$ — it would re-introduce gauge-dependence (see `optim.py:ProductMuonLoRA` docstring).

#### This proposal (clip variant), $k$-Picard iterations

QR factorizations $B = Q_B R_B$ and $A = R_A Q_A^\top$ (computed once per step). With $\mathrm{clip}(M, \tau) = U\,\mathrm{diag}(\min(\sigma_i, \tau))\,V^\top$ for SVD $M = U\Sigma V^\top$:

$$
\begin{aligned}
&\Delta A \leftarrow 0,\ \Delta B \leftarrow 0 \\
&\textbf{for } j = 1, \dots, k: \\
&\quad T_A = -Q_B^\top\, \Delta B\, A, \qquad T_B = -B\, \Delta A\, Q_A \\
&\quad X_A^\text{unc} = T_A - \eta\, R_B^{-\top} u_A, \qquad Y_B^\text{unc} = T_B - \eta\, u_B\, R_A^{-1} \\
&\quad X_A^\star = \mathrm{clip}(X_A^\text{unc},\, \tau_A), \qquad Y_B^\star = \mathrm{clip}(Y_B^\text{unc},\, \tau_B) \\
&\quad (\Delta A,\, \Delta B) = \mathrm{Lift}(X_A^\star,\, Y_B^\star) \quad \text{(Sylvester, min-Frob gauge)}
\end{aligned}
$$

#### Structural contrast

The table answers six questions about each variant. Each row is a design choice; each column shows what that variant picked for it.

| design choice (question) | Hybrid Picard ($k = 1, 2$) | ProductMuon | this proposal |
|---|---|---|---|
| **What rotation, if any, is applied to the linear cost $u$ before the operator runs?** | $S^{-1/2}$ symmetric square-root, applied per block | none — the operator runs on the joint $W$-space matrix, no per-block rotation | $R_B^{-\top}, R_A^{-1}$ from QR, applied per block |
| **Does the operator act on each block separately, or on a single matrix combining both?** | per block — one operator call for $A$, another for $B$ | joint — one operator call on a single $W$-space matrix $D$ | per block |
| **What does the operator do to a singular direction whose value $\sigma_i$ is below the threshold $\tau$?** | amplifies it up to $\tau$ (every $\sigma_i \to \tau$) | replaces it with $\tau$ via $\mathrm{polar}$ (every $\sigma_i \to 1$, then $\tau$ comes from a global scale) | leaves it untouched ($\sigma_i$ stays at $\sigma_i$) |
| **How is the overall step magnitude set?** | post-hoc Frobenius rescale of each factor to $\eta\|u\|_F$ | scalar $-\eta$ multiplying the joint $\mathrm{polar}$ output | natural prox: magnitude drops out of the variational closed form (no extra rescale) |
| **Are $\Delta A$ and $\Delta B$ produced independently, or recombined into a self-consistent pair?** | independently — no recombination; each factor's update is computed without reference to the other's | recombined via Sylvester (min-Frob gauge) | recombined via Sylvester (min-Frob gauge) |
| **Through what mechanism does the off-block ($\Delta B$ for the $A$-subproblem, etc.) influence the on-block?** | only via shared preconditioners $S_A, S_B$ ($k = 1$); plus a cross-term added to $u_A^\text{eff}, u_B^\text{eff}$ ($k = 2$) | implicit — both blocks live inside the joint $W$ matrix $D$ that is never factored | explicit cross-coupling target $T_A, T_B$ ($T = 0$ at $k = 1$; nonzero at $k \ge 2$), plus the Sylvester lift |

#### Where this proposal is genuinely new vs hybrid Picard

Three places, not one:

1. **Operator.** $\mathrm{polar} \to \mathrm{clip}$. The headline change.
2. **Whitening.** $S_B^{-1/2}$ (symmetric) $\to R_B^{-\top}$ (QR). Both are inverse-square-roots of $S_B$ on its column subspace and differ by an orthogonal rotation. Frobenius norm is preserved, but the singular *structure* of $X_A^\text{unc}$ differs — and $\mathrm{clip}$ depends on that structure, so the basis change is not innocuous.
3. **Recombination.** None $\to$ Sylvester min-Frob lift. Hybrid Picard's per-factor RMS-align leaves the pair off the min-Frob gauge surface in general; the lift puts them on it.

The §2.1 variational derivation forces all three changes simultaneously: clip is the closed form *in QR coordinates with the min-Frob lift*. Reverting any one re-introduces the heuristic the derivation replaces.

**Implication for the sweep.** The "polar $\times$ natural-prox" cells in §5 are *not* a reproduction of hybrid Picard. They sit in the new code path with QR whitening + Sylvester lift, both of which `AdamPolarProductLoRA` lacks; they isolate the operator swap with the other two changes held fixed. Cross-checks against the leaderboard (§3) come from the unchanged `AdamPolarProductLoRA` runs, not from a re-run in the new code path. A win at the headline cell tests the joint architecture, not the operator in isolation.

## 4. Open questions

Open empirical questions only. (Design questions — iteration order, init boundary, lift discipline — are resolved upstream: §0.2, §2.7, §8 unit tests respectively.)

**Q1 (headline). Does clip + $k=2$ widen the $r=16$ lead?** Target: best $c$ at $r=16$ gives eval $< 0.7546$. Answered by Phase 1 (§5).

**Q2. Is best-$c$ stable across ranks?** The variational form does not select $c$. Rank-stability is a **necessary** condition for shippability: if best-$c$ already differs between $r=16$ and $r=64$ on this single workload, $c$ is problem-tunable and the proposal fails the goal. Q2 is not sufficient — even if best-$c$ matches across ranks here, broader workload checks (different model, different dataset) are needed before claiming shippability; that is a follow-up campaign, not part of this proposal. Answered by Phase 2.

**Q3. Is the noise-amplification mechanism the right diagnosis?** Treat as speculative — the available diagnostics are only weak evidence either way.

The mechanism claim is about the *singular structure* of the cross-coupling target $T_A$ (small-$\sigma$ directions = noise; polar amplifies them; clip leaves them alone). A clean test would compare large-$\sigma$ vs small-$\sigma$ directions of $T_A$ to a ground-truth signal direction — which we don't have.

What the ride-along diagnostics actually measure (§7):

- **Cross-coupling magnitude** $\|T_A\|_F / \|u_A\|_F$. Tells us whether the cross-term is big, **not** whether it's noisy. A large cross-term can be informative signal (mechanism wrong, polar correctly amplifies it) or noise (mechanism right). Magnitude alone does not distinguish.
- **Clipping fraction** $\#\{\sigma_i(X_{A,\text{unc}}) > \tau\}/r$ and **reachability ratio** $\|X^\star\|_F / \|X_{A,\text{unc}}\|_F$. Tells us how much spectral mass clip is removing. Indirectly informative: if clip wins at $r=16$ AND removes a lot of small-$\sigma$ mass there but little at $r=64$, that's *consistent with* the noise hypothesis but does not prove it (could also be that the small-$\sigma$ directions are signal that just happens to hurt at small rank for a different reason).

A definitive test of the mechanism would require an independent signal estimate for $T_A$ (e.g. cross-seed correlation of singular directions, joint-system reference). Out of scope for this proposal.

**Practical posture:** if clip wins, report the cross-coupling and clipping-fraction diagnostics as supporting evidence consistent with the mechanism, not as confirmation. The mechanism is a guess about *why*; the headline result stands or falls on Q1, not Q3.

**Q4. Does natural-prox magnitude carry, or does Picard-rescale carry?** §2.3 commits to natural-prox as the variational closed form. But the Muon family (Muon, AdaMuon arXiv:2507.11005, NorMuon arXiv:2510.05491) and the existing `AdamPolarProductLoRA` all apply post-spectral-projection rescaling by design — empirically, the magnitude out of the spectral step does not match what learning-rate schedules expect. So Picard-rescale is *not* a heuristic violation to be dismissed; it is a validated pattern across the broader optimizer family. Phase 3 ablation, only if Q1 succeeds. Outcome semantics: natural-prox winning supports the prox formulation; Picard-rescale winning supports the AdaMuon-style design pattern with clip as the direction-shaper. Either is informative.

**Out-of-scope follow-up.** If Q1 fails, the family is hyperparameter-saturated and the next campaign needs a different parametrization (DoRA, periodic adapter merging, rank-adaptive). Not addressed here.

**Note: the Adam-covector compromise is not an open question here.** §2.6 substitutes $u_A, u_B$ for raw gradients, breaking the variational identity $G_A A^\top = B^\top G_B$. The compromise is shared with hybrid Picard (the baseline), AdaMuon, and NorMuon — both clip and polar run under it, so the differential clip-vs-polar effect is orthogonal to whether the substitution is "clean". We do not treat it as a question this experiment can answer.

## 5. Sweep design

**Primary axes:**

| axis | values | meaning |
|---|---|---|
| $r$ | $\{16, 64\}$ | LoRA rank |
| direction | $\{\text{polar},\ \text{clip}\}$ | which $\mathcal{P}$ operator the skeleton uses (§0.3) |
| magnitude | $\{\text{natural-prox},\ \text{Picard-rescale}\}$ | how the step magnitude is set after $\mathcal{P}$ runs (see below) |
| $k$ | $\{1, 2\}$ | number of Picard inner iterations (§0.2) |
| $\eta$ | $\{1\mathrm{e}{-4},\ 3\mathrm{e}{-4},\ 1\mathrm{e}{-3}\}$ (extend on boundary) | learning rate |
| $c$ | $\{0.1, 0.3, 1.0, 3.0\}$ for clip; n/a for polar | clip-radius shape parameter $\tau = c \cdot \sigma_{\max}(X_\text{unc})$ (§0.4) |

**Magnitude axis values.** Both branches start from the skeleton's Lift output $(\Delta A, \Delta B) = \mathrm{Lift}(X_A^\star, Y_B^\star)$. They differ in what is applied to that pair before $A \leftarrow A + \Delta A$, $B \leftarrow B + \Delta B$.

**natural-prox** — apply directly:

$$\Delta A^\text{apply} = \Delta A,\qquad \Delta B^\text{apply} = \Delta B.$$

The variational closed form (§2.3, $\lambda = \eta$) sets the magnitude; no extra rescale.

**Picard-rescale** — Frobenius-rescale each factor to the Adam-covector norm:

$$\Delta A^\text{apply} = \frac{\eta\,\|u_A\|_F}{\|\Delta A\|_F}\,\Delta A,\qquad \Delta B^\text{apply} = \frac{\eta\,\|u_B\|_F}{\|\Delta B\|_F}\,\Delta B.$$

This is the rule the existing `AdamPolarProductLoRA` uses (called "RMS-align" in legacy code, though it is a Frobenius rescale, not an element-wise RMS).

**Effect on the clip constraint.** For the clip branch, Picard-rescale moves the step off the spectral-norm ball:

$$\|X^\star\|_2 \le \tau\quad\text{(by construction of clip)},\qquad\text{but}\qquad\bigl\|R_B^\top \Delta A^\text{apply}\bigr\|_2 \ne \tau\quad\text{after rescale.}$$

So clip $\times$ Picard-rescale is **not** the variational closed form of the prox. Whether that matters empirically is an open question:

- The broader Muon family does exactly this kind of post-spectral-projection rescale by design. **AdaMuon** (Si et al., arXiv:2507.11005) orthogonalizes via Newton–Schulz, then applies $\gamma_t = 0.2\sqrt{mn}/\|\hat O_t\|_F$ to match Adam's empirical RMS scale (eq. 8). **NorMuon** (Li et al., arXiv:2510.05491) orthogonalizes, then applies per-neuron (row-wise) adaptive scaling from second-order statistics. **AdamPolarProductLoRA** does its per-factor rescale to $\eta\|u\|_F$ for the same reason.
- The shared empirical observation: the magnitude that falls out of the spectral-projection step does not match what downstream learning-rate schedules expect, and a rescale to an Adam-comparable scale is what makes the optimizer trainable at standard $\eta$.
- It is therefore plausible that Picard-rescale carries empirically for the clip variant *even though* it breaks the prox formulation. We include both magnitude branches in the sweep for this reason; the natural-prox branch tests the variational claim, the Picard-rescale branch tests the AdaMuon-style design pattern.

A win at clip $\times$ Picard-rescale is a positive result for the clip *direction* combined with a Muon-family rescale; it is not a positive result for the prox formulation as such. Q4 (§4) is the relevant test.

The polar $\times$ Picard-rescale $\times \{k=1,k=2\}$ cells **do not reproduce** the existing `AdamPolarProductLoRA` baseline — they live in the new code path with QR whitening and Sylvester lift, both of which `AdamPolarProductLoRA` lacks (§3 "How this proposal differs"). They serve as an A/B against the clip variant within a fixed architecture, isolating the operator swap. Cross-checks to the existing leaderboard numbers come from the unchanged `AdamPolarProductLoRA` runs in §3, not from the new code path.

**Sequencing.**

1. **Phase 1 — $r=16$, $k=2$, $c$-characterization.** clip $\times$ natural-prox $\times$ $k=2$ $\times$ $r=16$ $\times$ $c$ (4) $\times$ $\eta$ (3) = 12 cells. Extend $\eta$ by one value on boundary. Decisive on Q1.
2. **Phase 2 (conditional).** If Phase 1's best clip cell beats 0.7546, run the same recipe at $r=64$ — checks the no-regress floor (Q1 at $r=64$) and $c$-stability (Q2). If Phase 1 fails, end the campaign.
3. **Phase 3 (conditional, ablations).** Only if Phase 2 succeeds: $k=1$ vs $k=2$ at the best $(c, \eta)$ from Phase 2; Picard-rescale vs natural-prox at the best $(c, \eta, k)$ (Q4).

Q3 diagnostics ride along on every cell from Phase 1 onward (§7).

## 6. Decision rule

Shipped optimizer must satisfy **both** thresholds with **the same $(c, k)$ tuple**:

| rank | threshold |
|---|---|
| $r=16$ | $< 0.7546$ |
| $r=64$ | $\le 0.7382$ |

(Strict at $r=16$ because that's where the family currently has no winning config; non-strict at $r=64$ because the existing winner is already in this family — matching it with the same $(c, k)$ that wins at $r=16$ is the point.)

Outcomes:

- **Both met, same $(c, k)$:** rank-stability holds on this workload — necessary condition for shipping is satisfied. *Not* yet shippable: cross-workload stability (different model/dataset) is a separate follow-up before a ship claim.
- **Both met, different $(c, k)$ per rank:** problem-tunable already on the rank axis — do *not* ship; escalate to a different parametrization campaign.
- **One met, other regresses:** rank-locked — do *not* ship; same escalation.
- **Neither met:** family is hyperparameter-saturated within this formulation; same escalation.

## 7. Diagnostics

Logged per pair every 200 steps. Strict inequalities ($\sigma > \tau$, not $\ge$).

- **Clipping fraction:** $\#\{\sigma_i(X_\text{unc}) > \tau\}\,/\,r$. Where on the $c$-continuum the run sits effectively.
- **Saturation gauge:** $\sigma_{\max}(X_\text{unc})\,/\,\tau$. One-number summary.
- **Reachability ratio:** $\|X^\star\|_F\,/\,\|X_\text{unc}\|_F$. How much the clip is shrinking the step.
- **Cross-coupling magnitude (Q3):** $\|B^\top \Delta B_\text{prev} A\|_F\,/\,\|u_A\|_F$. Mechanism predicts large at $r=16$, small at $r=64$.
- **Finite-step ratio:** $\|\Delta B \Delta A\|_F\,/\,\|B \Delta A + \Delta B A\|_F$. Bilinear second-order term, recorded for cross-run comparison.

## 8. Reproducibility

- Submission via `slurm_scripts/submit.sh` with `SWEEP_SCOPE=ext_compare,polar_family` and explicit purpose string.
- New optimizer registered as `adam-polar-product-lora-clip` in `OPTIMIZER_CHOICES`; entries added to `OPTIM_COLORS` and at least one `OPTIM_FAMILIES` set in `lora_playground/plot_utils.py`.
- Analysis via `lora_playground.loader.load_runs(where=…)`; never hand-typed group lists.
- Unit tests for the new clip block solve (in `tests/test_polar_product.py`):
  1. **Sylvester limit.** $c \to \infty$ recovers the Frobenius-coupled Sylvester closed form to $10^{-5}$ on a synthetic LoRA pair.
  2. **Polar limit.** $c \to 0^+$ recovers Picard's polar *direction* (after Frobenius rescale) to $10^{-3}$ on the same pair.
  3. **Determinism** on a tiny tensor with fixed seed.
  4. **Min-Frobenius gauge:** $\|B^\top \Delta B - \Delta A\, A^\top\|_F \le 10^{-5} \cdot \|\Delta A\, A^\top\|_F$ after the lift.
  5. **Sign:** with $T = 0$ and $\lambda$ large enough that the linear cost dominates, $\langle G_A, \Delta A \rangle < 0$.

These resolve D3 before any GPU time is spent.
