# A core-reduction solver for the joint operator-norm LoRA update

Companion to `docs/notes/polar_coupled_problem.md`. That doc states the
open problem; this doc gives a candidate solution: reduce the joint LoRA
tangent problem to a tiny "forbidden-corner" core, then take a
**projected quotient polar** direction. The solver is cheap, uses only
$(A_t, B_t, \nabla_A L, \nabla_B L, \lambda)$, is symmetric in
$(A, B)$, exact in the one-factor cases, and certifies a deterministic
$1/2$ approximation to the exact joint operator-norm direction. An
optional nuclear-completion polish drives the core to optimality at
small extra cost.

The standing assumptions of the open-problem doc apply throughout —
in particular the gradient-compatibility condition
$(\nabla_A L)\, A_t^\top = B_t^\top (\nabla_B L)$, which is what lets
the linear cost in $(\Delta A, \Delta B)$ be written as a single inner
product against the implicit $G$ on the tangent image.

## 1. Build the active tangent-gradient core

Write $m = d_\text{out}$, $n = d_\text{in}$ and compute thin QR-type
factorizations

$$B_t = Q_L R_L, \qquad A_t = R_R Q_R^\top,$$

with $Q_L \in \mathbb{R}^{m \times r}$, $Q_R \in \mathbb{R}^{n \times r}$
column-orthonormal and $R_L, R_R \in \mathbb{R}^{r \times r}$ invertible
(by the full-rank standing assumption). Let $G_A := \nabla_A L$,
$G_B := \nabla_B L$. From the accessible factor gradients,

$$L_0 := R_L^{-\top} G_A = Q_L^\top G \in \mathbb{R}^{r \times n}, \qquad R_0 := G_B R_R^{-\top} = G\, Q_R \in \mathbb{R}^{m \times r}.$$

(The right-hand identities use the gradients' definition; we never
materialize $G$.) Define the shared $r \times r$ core block

$$C := L_0\, Q_R = Q_L^\top R_0.$$

Both expressions are equal in exact arithmetic by gradient
compatibility; in finite precision, average them.

Strip the row/column support already represented by $(Q_L, Q_R)$:

$$L_\perp := L_0 - C\, Q_R^\top, \qquad R_\perp := R_0 - Q_L\, C.$$

By construction $L_\perp Q_R = 0$ and $Q_L^\top R_\perp = 0$. Take
thin SVDs (or QRs) absorbing the singular-value factor on the inner
side,

$$L_\perp = E\, V^\top, \qquad R_\perp = U\, F,$$

so that $V \in \mathbb{R}^{n \times s}$, $U \in \mathbb{R}^{m \times t}$
have orthonormal columns with $V^\top Q_R = 0$, $U^\top Q_L = 0$, and
$E \in \mathbb{R}^{r \times s}$, $F \in \mathbb{R}^{t \times r}$ for
some $s, t \le r$. Empty blocks are allowed.

### Why this is exactly the active core

The tangent map $J_t[\Delta A, \Delta B] = B_t \Delta A + \Delta B A_t$
factors through the basis change $\widetilde A = R_L \Delta A$,
$\widetilde B = \Delta B\, R_R$:

$$J_t = Q_L \widetilde A + \widetilde B\, Q_R^\top.$$

Decomposing $\widetilde A = X_1 Q_R^\top + X_2$ with $X_2 Q_R = 0$ and
$\widetilde B = Q_L Y_1 + Y_2$ with $Q_L^\top Y_2 = 0$, the image is

$$J_t = Q_L (X_1 + Y_1) Q_R^\top + Q_L X_2 + Y_2 Q_R^\top,$$

i.e.\ the set of matrices with zero $(Q_L^\perp, Q_R^\perp)$ block in
the basis $[Q_L, U] / [Q_R, V]$. For every core matrix

$$\widehat Z = \begin{bmatrix} X & Y \\ W & 0 \end{bmatrix} \in \mathbb{R}^{(r+t) \times (r+s)},$$

the lifted tangent

$$Z = [Q_L, U]\, \widehat Z\, [Q_R, V]^\top$$

satisfies $\|Z\|_2 = \|\widehat Z\|_2$ (orthonormal bases) and
$\langle G_A, \Delta A\rangle + \langle G_B, \Delta B\rangle
= \langle G, Z\rangle = \langle \widehat H, \widehat Z\rangle$ where

$$\widehat H = \begin{bmatrix} C & E \\ F & 0 \end{bmatrix}.$$

The (22) block of $\widehat H$ is undefined as data — but irrelevant,
because the corresponding block of every feasible $\widehat Z$ is
zero. We fix it to $0$ for definiteness.

The joint problem is therefore equivalent to

$$\tau := \sup_{\widehat Z_{22} = 0,\, \|\widehat Z\|_2 \le 1}\, \langle \widehat H, \widehat Z\rangle. \qquad (\dagger)$$

## 2. Cheap solver: projected quotient polar

Let $\Pi$ zero the lower-right block:

$$\Pi\!\begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{bmatrix} = \begin{bmatrix} M_{11} & M_{12} \\ M_{21} & 0 \end{bmatrix}.$$

Compute the compact polar factor of the core,

$$P := \mathrm{polar}(\widehat H) = U_\sigma V_\sigma^\top \quad \text{where} \quad \widehat H = U_\sigma \Sigma V_\sigma^\top.$$

If $\widehat H = 0$, return zero updates. Otherwise project and
renormalize:

$$R := \Pi(P), \qquad \gamma := \|R\|_2, \qquad \widehat Z_+ := R / \gamma.$$

This is the **ascent** direction. The constrained operator-norm
core update is

$$\widehat Z_\text{upd} = -\lambda\, \widehat Z_+.$$

For the squared-penalty form, set
$\widehat \tau := \langle \widehat H, \widehat Z_+\rangle = \|\widehat H\|_* / \gamma$ and use

$$\widehat Z_\text{upd} = -\lambda\, \widehat \tau\, \widehat Z_+.$$

### Cost

A single polar of a matrix of size at most $2r \times 2r$ (Newton-Schulz
or symmetric eigendecomposition; $O(r^3)$). All upstream work — thin
QRs of $B_t, A_t$, formation of $L_0, R_0, C, L_\perp, R_\perp$ —
is $O((m + n) r^2)$. No object of size $d_\text{out} \times d_\text{in}$
is ever formed.

### Guarantee: $\gamma \in [1, 2]$ and $\widehat \tau / \tau \ge 1/2$

Since $\widehat H_{22} = 0$, $\langle \widehat H, M\rangle = \langle \widehat H, \Pi(M)\rangle$ for any $M$. Write $\Pi$ as

$$\Pi(M) = \begin{bmatrix} I & 0 \\ 0 & 0 \end{bmatrix} M + \begin{bmatrix} 0 & 0 \\ 0 & I \end{bmatrix} M \begin{bmatrix} I & 0 \\ 0 & 0 \end{bmatrix}.$$

Each summand has spectral norm $\le \|M\|_2$, so $\|\Pi(M)\|_2 \le 2\|M\|_2$. Hence for any $\|M\|_2 \le 1$,

$$\langle \widehat H, M\rangle = \langle \widehat H, \Pi(M)\rangle = 2 \langle \widehat H, \Pi(M)/2\rangle \le 2\tau,$$

since $\Pi(M)/2$ is feasible for $(\dagger)$. Taking sup over $M$ gives $\|\widehat H\|_* \le 2\tau$. Combined with the trivial $\tau \le \|\widehat H\|_*$ (the feasible set of $(\dagger)$ sits inside the ambient spectral ball),

$$\frac{\|\widehat H\|_*}{2} \le \tau \le \|\widehat H\|_*.$$

The candidate achieves $\widehat \tau = \langle \widehat H, R/\gamma\rangle = \langle \widehat H, P\rangle / \gamma = \|\widehat H\|_* / \gamma$. Lower bound on $\gamma$: by Hölder,

$$\|\widehat H\|_* = \langle \widehat H, P\rangle = \langle \widehat H, \Pi(P)\rangle \le \|\widehat H\|_* \cdot \|\Pi(P)\|_2 = \|\widehat H\|_* \cdot \gamma,$$

so $\gamma \ge 1$ (assuming $\widehat H \ne 0$). Upper bound: $\gamma = \|\Pi(P)\|_2 \le 2 \|P\|_2 = 2$. Therefore

$$\frac{\widehat \tau}{\tau} \ge \frac{\|\widehat H\|_* / \gamma}{\|\widehat H\|_*} = \frac{1}{\gamma} \ge \frac{1}{2}.$$

For the squared-penalty objective, the achieved decrease is at least
$\widehat \tau^2 / \tau^2 \ge 1/4$ of the exact one.

## 3. Optional exact polish via nuclear completion

Strong duality between operator and nuclear norms gives

$$\tau = \min_{D \in \mathbb{R}^{t \times s}}\, \left\|\begin{bmatrix} C & E \\ F & D \end{bmatrix}\right\|_*.$$

If $D_\star$ achieves the minimum and the nuclear norm is differentiable
at $M_\star := \begin{bmatrix} C & E \\ F & D_\star \end{bmatrix}$,
then an exact primal maximizer of $(\dagger)$ is
$\widehat Z_\star = \mathrm{polar}(M_\star)$ — the KKT condition is
exactly $\mathrm{polar}(M_\star)_{22} = 0$.

Subgradient of the nuclear norm at differentiable $M$ is its polar, so

$$\nabla_D\, \left\|\begin{bmatrix} C & E \\ F & D \end{bmatrix}\right\|_* = \mathrm{polar}\!\begin{bmatrix} C & E \\ F & D \end{bmatrix}_{22}.$$

A few L-BFGS steps on the small $t \times s$ variable $D$, initialized
at $D = 0$, drive the (22) block of the polar to zero. Per iteration
cost: one polar of a $\le 2r \times 2r$ matrix, $O(r^3)$.

For each iterate $D$, define

$$P_D := \mathrm{polar}\!\begin{bmatrix} C & E \\ F & D \end{bmatrix}, \qquad \widehat Z_D := \frac{\Pi(P_D)}{\|\Pi(P_D)\|_2},$$

and certificates

$$\mathrm{LB}(D) := \langle \widehat H, \widehat Z_D\rangle, \qquad \mathrm{UB}(D) := \left\|\begin{bmatrix} C & E \\ F & D \end{bmatrix}\right\|_*.$$

By weak duality $\mathrm{LB}(D) \le \tau \le \mathrm{UB}(D)$ for all
$D$, with equality at $D_\star$. Keep the $D = 0$ candidate as a
fallback — it preserves the $1/2$ guarantee unconditionally. The
polish only ever improves on it.

### Menu of cheap-solver choices

The two extremes above (closed-form $D = 0$ projected polar, full
nuclear-completion polish) are the endpoints of a small family. All
share the same core reduction and lift-back; they differ only in how
they solve $(\dagger)$. Recording them so the implementation choice is
explicit:

1. **Closed-form $D = 0$** (Section 2). One polar of $\widehat H$.
   Deterministic $\widehat \tau / \tau \ge 1/2$, no iteration. The
   default unless evidence suggests the bound is loose in practice.

2. **Subgradient L-BFGS on the exact nuclear-completion** (above). One
   polar of size $\le 2r \times 2r$ per step. Subgradient at
   non-differentiable points is any element of $\partial \|\cdot\|_*$;
   in practice the polar branch is fine because degenerate spectra are
   measure-zero.

3. **Smoothed nuclear-completion**
   $\phi_\varepsilon(D) := \mathrm{tr}\bigl( (K(D)^\top K(D) + \varepsilon^2 I)^{1/2} \bigr)$,
   with $K(D) = \begin{bmatrix} C & E \\ F & D \end{bmatrix}$ and
   gradient
   $\nabla_D \phi_\varepsilon(D) = \bigl[ K(D) (K(D)^\top K(D) + \varepsilon^2 I)^{-1/2} \bigr]_{22}$.
   This is the smoothed polar's (22) block. Use when the unsmoothed
   subgradient L-BFGS oscillates near degenerate spectra; pick
   $\varepsilon$ at the level of the smallest singular value you want
   resolved (e.g.\ $\varepsilon = 10^{-6} \|\widehat H\|_2$). At
   $\varepsilon \to 0$ this matches option 2; at $D = 0$ and any
   $\varepsilon$ it matches option 1 modulo $O(\varepsilon)$.

4. **One Newton step in $D$ from $D = 0$.** The smoothed objective is
   convex in $D$ with a structured Hessian (blocks of inverse-square-root
   factors). One step yields a "second-order projected polar" — strictly
   better than option 1 with the same primal/dual certificate machinery,
   at roughly twice the cost. Worth trying if option 1 is consistently
   leaving $\gamma$ near 2.

5. **Frank-Wolfe / conditional gradient on the primal $(\dagger)$.**
   Each step is a polar of $\widehat H$ followed by a line-search on
   the spectral-norm ball intersected with $\widehat Z_{22} = 0$. Same
   per-step cost as options 2-3; converges to the exact $\tau$.

In all cases the LB / UB certificate from Section 3 still applies — at
any iterate, $\mathrm{LB}(D) \le \tau \le \mathrm{UB}(D)$, and the gap
is a free stopping criterion.

### Practical recommendation

Start with option 1. If the certificate gap
$\mathrm{UB}(0) - \mathrm{LB}(0) = \|\widehat H\|_*(1 - 1/\gamma)$ is
small relative to the step size $\lambda$, no polish is needed. If the
gap is consistently large across training steps, switch to option 3
with a fixed budget (e.g.\ 3 L-BFGS iterations). Options 4-5 are
mostly of theoretical interest — option 3 dominates in practice.

## 4. Lift the core update back to $(\Delta A, \Delta B)$

Given the chosen scaled core update

$$\widehat Z_\text{upd} = \begin{bmatrix} X & Y \\ W & 0 \end{bmatrix},$$

use the default min-Frobenius gauge $B_t^\top \Delta B = \Delta A\, A_t^\top$.
With $S_L = B_t^\top B_t = R_L^\top R_L$ and
$S_R = A_t A_t^\top = R_R R_R^\top$, solve the $r \times r$ Sylvester
equation

$$S_L K + K S_R = R_L^\top X R_R^\top$$

for $K \in \mathbb{R}^{r \times r}$ (one symmetric eigendecomposition
per side, $O(r^3)$). Then

$$\Delta A = S_L^{-1}\!\left[ (R_L^\top X - K R_R)\, Q_R^\top + R_L^\top Y\, V^\top \right],$$

$$\Delta B = \left[ Q_L (X R_R^\top - R_L K) + U W R_R^\top \right] S_R^{-1}.$$

These satisfy

$$B_t \Delta A + \Delta B\, A_t = [Q_L, U]\, \widehat Z_\text{upd}\, [Q_R, V]^\top, \qquad B_t^\top \Delta B = \Delta A\, A_t^\top = K.$$

## 5. Sanity checks

**Frobenius replacement reduces to Case 1.** Replacing $\|\cdot\|_2$
by $\|\cdot\|_F$ in $(\star)$, the core problem
$\min_{\widehat Z_{22}=0}\, \langle \widehat H, \widehat Z\rangle + (1/2\lambda) \|\widehat Z\|_F^2$
has unconstrained minimizer $-\lambda \widehat H$, which already has
(22) = 0; the constraint is inactive. Lifting through the gauge
formulas reproduces the Sylvester closed form of the open-problem
doc.

**One-factor restriction reduces to Case 2.** Forcing $\Delta A = 0$
restricts $\widehat Z$ to (12) = (22) = 0, a strip
$\widehat Z = \begin{bmatrix} \tilde C \\ \tilde W \end{bmatrix}[I_r, 0]$.
The relevant data is $\begin{bmatrix} C \\ F \end{bmatrix} = [Q_L, U]^\top G\, Q_R$,
and the maximizer is $\mathrm{polar}([C; F])$. Lifting,

$$Z = -\lambda\, \mathrm{polar}(G Q_R)\, Q_R^\top = -\lambda\, \mathrm{polar}(G_B R_R^{-\top})\, Q_R^\top,$$

and $\Delta B = -\lambda\, \mathrm{polar}(G_B R_R^{-\top})\, R_R^{-1}$,
which is the Case 2 formula. The $\Delta B = 0$ case is symmetric.

**Symmetry in $(A, B)$.** Transposing the construction swaps
$A_t \leftrightarrow B_t^\top$, $G_A \leftrightarrow G_B^\top$,
$E \leftrightarrow F^\top$, $Q_R \leftrightarrow Q_L$, and
$\widehat H \leftrightarrow \widehat H^\top$. Both the polar and its
projection $\Pi$ commute with transposition, so the algorithm
respects the stated $(A, B, G) \leftrightarrow (B^\top, A^\top, G^\top)$
symmetry.

## 6. Preconditioning in core space

The temptation when wiring this solver into a real optimizer is to
precondition each factor gradient independently — e.g.\ run Adam on
$G_A$ and $G_B$ as separate tensors, then feed the preconditioned
$(\widetilde G_A, \widetilde G_B)$ to the core solver. **Don't.** This
breaks gradient compatibility in general: $\widetilde G_A A_t^\top \ne B_t^\top \widetilde G_B$,
the linear cost is no longer a single tangent-space functional, and
the gauge symmetry is no longer respected. Section 5's "averaging $C$"
papers over the symptom but not the deeper issue: factor-Adam adapts
coordinates that the variational problem regards as gauge-redundant.

The right design is: **decide what space the optimizer lives in, then
put momentum / adaptive scaling there.** For this solver, the
principled space is the LoRA tangent — equivalently the core
$\widehat H$. Maintain momentum and/or second-moment statistics in
core space, transport them across steps as the bases rotate, and run
the projected polar on the preconditioned core covector.

### Core-basis transport

The bases $\mathcal U_t = [Q_{L,t}, U_t] \in \mathbb{R}^{m \times (r + t)}$
and $\mathcal V_t = [Q_{R,t}, V_t] \in \mathbb{R}^{n \times (r + s)}$
change every step as $A_t, B_t$ update. To carry a core-space tensor
$\widehat M_{t-1}$ from step $t-1$ to step $t$, compute small overlap
matrices

$$T_L := \mathcal U_t^\top \mathcal U_{t-1}, \qquad T_R := \mathcal V_t^\top \mathcal V_{t-1},$$

(sizes at most $2r \times 2r$) and set

$$\widehat M_{t-1 \to t} := T_L \widehat M_{t-1} T_R^\top.$$

This is a parallel-transport-style approximation: it projects the old
core object onto the new bases and ignores out-of-basis components.
Cost $O(r^3)$ per pair per step.

### Variant ladder

Increasingly adaptive options. Start at the bottom and only climb if
the data demands it.

**1. Pure projected quotient polar.** Raw factor gradients
(compatibility holds automatically), no momentum, no adaptive scaling.
The clean baseline; everything else compares against this.

**2. Core momentum.** Raw factor gradients $G_A, G_B$ → build
$\widehat H_t$ as in Sections 1-2. Maintain a transported EMA in core
space:

$$\widehat M_t = \beta_1\, \Pi(\widehat M_{t-1 \to t}) + (1 - \beta_1)\, \widehat H_t,$$

where $\Pi$ zeros the forbidden (22) block (it's invisible to feasible
$\widehat Z$ anyway). Run the projected polar on $\widehat M_t$
instead of $\widehat H_t$. Lift back via the same Sylvester-gauge
formulas of Section 4. **This is the natural Muon-style LoRA tangent
optimizer.**

**3. Core momentum + scalar RMS.** Add a single scalar second moment
per LoRA pair:

$$s_t = \beta_2\, s_{t-1} + (1 - \beta_2)\, \|\widehat H_t\|_F^2$$

(or $\widehat \tau_t^2$ — equivalent up to a constant), and adapt the
outer step size $\lambda_t = \eta / (\sqrt{s_t} + \varepsilon)$. Adam-
like magnitude normalization without a coordinate-wise diagonal. Safe
first adaptive variant.

**4. Core Shampoo / Kronecker preconditioner.** Maintain transported
left/right second moments in core space:

$$L_t = \beta_2\, T_L L_{t-1} T_L^\top + (1 - \beta_2)\, \widehat H_t \widehat H_t^\top,$$

$$R_t = \beta_2\, T_R R_{t-1} T_R^\top + (1 - \beta_2)\, \widehat H_t^\top \widehat H_t.$$

Precondition the momentum core,

$$\widetilde H_t = (L_t + \varepsilon I)^{-1/4}\, \widehat M_t\, (R_t + \varepsilon I)^{-1/4},$$

zero the forbidden block, and run projected polar on $\widetilde H_t$.
$O(r^3)$ per pair per step. The most principled adaptive option;
respects rotations of the core bases far better than coordinate-wise
RMS.

**5. Core-coordinate elementwise RMS.** Maintain $V_t$ as a transported
EMA of $\widehat H_t \odot \widehat H_t$ and precondition
$\widetilde H_t = \widehat M_t / (\sqrt{V_t} + \varepsilon)$. Closer to
literal Adam, but the residual bases $U, V$ rotate freely so
elementwise RMS in core coordinates is still partly arbitrary. Empirical
variant; not principled.

**6. Factor Adam + compatibility projection.** Run Adam separately on
$G_A, G_B$, project onto the gradient-compatibility subspace before
feeding to the solver. Use **only** as an ablation, to answer "can we
rescue the existing factor-Adam machinery?" Not the conceptual answer.

### Gradient-compatibility diagnostic

For any preconditioned input $(\widetilde G_A, \widetilde G_B)$ (and
even raw inputs, as a sanity check), log

$$\mathrm{compat} := \frac{\|\widetilde G_A A_t^\top - B_t^\top \widetilde G_B\|_F}{\|\widetilde G_A A_t^\top\|_F + \|B_t^\top \widetilde G_B\|_F + \varepsilon}.$$

For raw gradients this should be near machine epsilon. For variants
1-4 the *inputs* to the core construction are raw factor gradients, so
compat stays small — preconditioning happens after compatibility is
secured. For variant 6 it stays small by construction (after the
projection). For variant 5, compat measures how far you've drifted from
the variational regime; large values are a warning, not a fatal error.

### Logging for momentum / adaptive variants

In addition to the $\gamma, \mathrm{LB}, \mathrm{UB}, \mathrm{relgap}$
certificate (Section 7), log:

- $\langle \widehat H_t, \widehat Z_t\rangle$ — alignment of the
  *instantaneous* gradient with the chosen direction. The optimizer
  actually solved a problem on $\widehat M_t$ (or $\widetilde H_t$);
  this number tells you whether that direction still points downhill
  on the raw loss.
- $\langle \widehat M_t, \widehat Z_t\rangle$ (or
  $\langle \widetilde H_t, \widehat Z_t\rangle$) — alignment of the
  preconditioned object with the chosen direction. This is what the
  certificate guarantees.
- $\mathrm{compat}$ as above.

If instantaneous alignment goes negative or near-zero while
momentum/preconditioner alignment stays positive, momentum is stale —
a signal to shrink $\beta_1$ or step size.

## 7. Implementation plan

Concrete order of operations for wiring this into a LoRA optimizer.
Each step references the section that derives it.

### Per-pair step (no polish)

Inputs: $A_t, B_t, G_A := \nabla_A L, G_B := \nabla_B L, \lambda$.

1. **Bases.** Thin QR: $B_t = Q_L R_L$, $A_t = R_R Q_R^\top$
   (Section 1).
2. **Accessible gradient core.** Solve $L_0 = R_L^{-\top} G_A$ and
   $R_0 = G_B R_R^{-\top}$ (triangular solves, $O(r^2 \max(m, n))$).
3. **Shared block.** $C = \tfrac{1}{2}(L_0 Q_R + Q_L^\top R_0)$
   (averaging the two algebraically-equal expressions for finite-
   precision robustness; disagreement between the two is also a free
   diagnostic for gradient-compatibility violation, see Section 6).
4. **Residuals.** $L_\perp = L_0 - C Q_R^\top$,
   $R_\perp = R_0 - Q_L C$.
5. **Thin SVDs.** $L_\perp = E V^\top$ ($V$ in $Q_R^\perp$,
   $E \in \mathbb{R}^{r \times s}$), $R_\perp = U F$ ($U$ in
   $Q_L^\perp$, $F \in \mathbb{R}^{t \times r}$).
6. **Core polar.** $\widehat H = \begin{bmatrix} C & E \\ F & 0 \end{bmatrix}$;
   $P = \mathrm{polar}(\widehat H)$ via Newton-Schulz or symmetric
   eigendecomposition (matrix size $\le 2r \times 2r$).
7. **Project + renormalize.** $R = \Pi(P)$; $\gamma = \|R\|_2$;
   $\widehat Z_+ = R / \gamma$ (Section 2).
8. **Scale.** Constrained: $\widehat Z_\text{upd} = -\lambda \widehat Z_+$.
   Squared-penalty:
   $\widehat Z_\text{upd} = -\lambda (\|\widehat H\|_* / \gamma) \widehat Z_+$.
9. **Sylvester for the gauge.** Eigendecompose $S_L = R_L^\top R_L$
   and $S_R = R_R R_R^\top$ once each; solve
   $S_L K + K S_R = R_L^\top X R_R^\top$ entrywise in the eigenbasis
   ($O(r^3)$).
10. **Lift back.** Apply the $\Delta A, \Delta B$ formulas of
    Section 4 and write into the param tensors.

### Per-pair certificates to log

At iteration zero (or every step, ~free given items already computed):

- $\gamma = \|\Pi(P)\|_2$ — guaranteed in $[1, 2]$.
- $\|\widehat H\|_*$ — sum of singular values of $\widehat H$.
- $\mathrm{LB}(0) = \langle \widehat H, \widehat Z_+\rangle = \|\widehat H\|_* / \gamma$.
- $\mathrm{UB}(0) = \|\widehat H\|_*$.
- $\mathrm{relgap}_0 = 1 - 1/\gamma \in [0, 1/2]$.

These should ride along in `log_optim_diagnostics` from day one (per
the project's "diagnostics by default" policy). The relgap distribution
across layers and steps is the trigger for whether to enable polish.

### Polish (only if needed)

If $\mathrm{relgap}_0$ is consistently above ~0.1 across important
layers/steps, enable the smoothed nuclear-completion polish
(Section 3, option 3 of the menu):

11. **Initialize** $D = 0$.
12. **Compute** $K(D) = \begin{bmatrix} C & E \\ F & D \end{bmatrix}$,
    its smoothed polar
    $K(D)(K(D)^\top K(D) + \varepsilon^2 I)^{-1/2}$, and the
    gradient $\nabla_D \phi_\varepsilon(D)$ as the (22) block.
13. **Run** 2-3 L-BFGS steps on $D$. Each step costs one polar of a
    $\le 2r \times 2r$ matrix, $O(r^3)$.
14. **At each polished $D$** form $\widehat Z_D = \Pi(P_D) / \|\Pi(P_D)\|_2$,
    $\mathrm{LB}(D) = \langle \widehat H, \widehat Z_D\rangle$,
    $\mathrm{UB}(D) = \|K(D)\|_*$. Use $\mathrm{UB}(D) - \mathrm{LB}(D)$
    as a live stopping criterion.
15. **Fallback.** Always retain the $D = 0$ candidate. If polish does
    not improve the lower bound by a meaningful margin, ship the
    closed-form direction — it preserves the unconditional $1/2$
    guarantee.

Suggested $\varepsilon$: $10^{-6} \|\widehat H\|_2$ (resolves any
singular value above $10^{-6}$ of the leading one; below that the
polar's direction is numerically ambiguous anyway).

### Implementation priority

The core solver and the preconditioning ladder (Section 6) should be
developed in this order. Each rung adds one mechanism; certificate
logging rides along from step 1.

1. **Variant 1 — pure projected quotient polar.** Raw factor gradients,
   steps 1-10 above, full certificate logging
   ($\gamma, \mathrm{LB}, \mathrm{UB}, \mathrm{relgap}, \mathrm{compat}$).
   Inspect $\gamma$ / relgap distribution on a 2k-step run.
2. **Polish** (steps 11-15) only if relgap is consistently $> 0.1$ on
   layers and steps that matter. Keep $D = 0$ as the default fallback
   regardless of polish status.
3. **Variant 2 — core momentum.** Add transported core EMA
   $\widehat M_t$ (Section 6); run the solver on $\widehat M_t$. This
   is the first comparison-worthy optimizer; expected to be the main
   "muon-style LoRA tangent" baseline.
4. **Variant 3 — core momentum + scalar RMS.** Adam-like magnitude
   normalization without coordinate-wise diagonal. The safer adaptive
   step.
5. **Variant 4 — core Shampoo.** Left/right transported second moments
   + Kronecker preconditioning. The most principled adaptive option;
   only build if variants 2-3 leave a clear gap.
6. **Variants 5-6** are ablation-only.

At every rung, log all of $\gamma, \mathrm{LB}, \mathrm{UB}, \mathrm{relgap}, \mathrm{compat}$,
plus $\langle \widehat H_t, \widehat Z_t\rangle$ (instantaneous
alignment) for variants 2+.
