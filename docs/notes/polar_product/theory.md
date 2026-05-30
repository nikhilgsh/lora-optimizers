# Polar-product theory: variational form, cases, and Sylvester gauge lift

Combines the per-step variational program with its closed-form solver. The variational framework (problem statements) is in §§Goal–Empirical context; the Sylvester gauge lift (closed-form solver shared across Cases 1, 3, and the adjacent formulation) is in §Sylvester gauge lift.


## Goal

A self-contained statement of the variational problems for LoRA optimizer
design used in this project. The currently live target is the
**adjacent formulation** (§"Adjacent formulation — primary target");
the joint operator-norm program (Case 3) was empirically closed by
experiments E1–E7 in `investigations.md` (best variant
0.7490 at $r=64$, lost to the hybrid Picard baseline 0.7382), and
material on it below is reference only. The clipping-prox proposal
(`proposal.md`) builds directly on the adjacent
formulation.

**E-numbering.** "E1, E2, …" are the experiments tabulated in
`investigations.md` §3 (variants of the joint
operator-norm core solver).

## Setup and notation

LoRA fine-tuning: for each frozen base weight
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, add a trainable
low-rank correction

$$W \;\to\; W + \tfrac{\alpha}{r}\, B\, A,$$

with $A \in \mathbb{R}^{r \times d_\text{in}}$,
$B \in \mathbb{R}^{d_\text{out} \times r}$,
$r \ll \min(d_\text{out}, d_\text{in})$. We set $\alpha/r = 1$ by
absorbing it into the gradient or learning rate. Let
$G := \nabla_W L \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ be
the dense gradient w.r.t. the effective weight. The per-factor
gradients accessible from back-prop are

$$\nabla_A L \;=\; B_t^\top\, G \;\in\; \mathbb{R}^{r \times d_\text{in}}, \qquad \nabla_B L \;=\; G\, A_t^\top \;\in\; \mathbb{R}^{d_\text{out} \times r}.$$

**We never materialize $G$.** The algorithm takes
$(A_t, B_t, \nabla_A L, \nabla_B L, \lambda)$ with $\lambda > 0$ and
returns $(\Delta A, \Delta B)$.

**Standing assumptions.**

1. $A_t, B_t$ have full rank $r$. The $B = 0$ initialization and damping
   are out of scope (see Case 1 note).
2. **Gradient compatibility.** The factor gradients lie in the range of
   $J_t^\star$ for some common $G$, equivalently

   $$(\nabla_A L)\, A_t^\top \;=\; B_t^\top\, (\nabla_B L).$$

   Without this, the objective $(\star)$ below is unbounded below along
   the gauge kernel of $J_t$. Raw autograd through a single $G$
   satisfies the condition automatically; momentum / Adam-preconditioned
   factor gradients in general do not. Implementations that feed
   non-compatible inputs to an algorithm derived from $(\star)$ are
   using the algorithm as a step *shape*, outside the variational
   regime — that is a separate use case not addressed here.

## The variational principle

A modern view of optimizer design — exemplified by the Muon line
(`~/modded-nanogpt/train_gpt.py`) and AdaMuon (Muon with per-coordinate
Adam-style preconditioning before the orthogonalization step) — treats
an optimizer as a variational problem. Two formulations differ by how
magnitude is set:

$$\min_{\|\Delta W\| \le \lambda}\; \langle G,\, \Delta W\rangle \qquad \text{(constrained)}, \qquad \min_{\Delta W}\; \langle G,\, \Delta W\rangle + \frac{1}{2\lambda}\, \rho(\Delta W) \qquad \text{(squared-penalty)}.$$

Norm choice picks the optimizer family:

- **Frobenius**, $\rho = \|\cdot\|_F^2$: penalized solution is plain GD,
  $\Delta W = -\lambda G$.
- **Operator (spectral)** constrained: $\Delta W = -\lambda\, \mathrm{polar}(G)$,
  where $\mathrm{polar}(G) = U V^\top$ for the compact SVD
  $G = U\Sigma V^\top$ ($U, V$ have orthonormal columns). The compact
  polar factor is the canonical tie-breaker when the spectrum is
  degenerate; when $G = 0$ the update is zero. This is the Muon update
  — equalized singular values, magnitude $\lambda$.
- **Operator** squared-penalty: same direction, scaled by $\lambda \|G\|_*$
  (the dual norm of $\|\cdot\|_2$). In the LoRA setting below the
  magnitude factor is restricted to the LoRA tangent and may differ
  from the ambient $\|G\|_*$.

For LoRA we don't have a free $\Delta W$. The actual finite update is
$(B_t + \Delta B)(A_t + \Delta A) - B_t A_t = B_t \Delta A + \Delta B\, A_t + \Delta B\, \Delta A$;
we optimize the first-order tangent update

$$J_t[\Delta A, \Delta B] \;:=\; B_t \Delta A + \Delta B\, A_t,$$

dropping the second-order $\Delta B\, \Delta A$. The variational
problem in factor coordinates is

$$\boxed{\;\min_{\Delta A,\, \Delta B}\; \langle \nabla_A L, \Delta A\rangle + \langle \nabla_B L, \Delta B\rangle + \frac{1}{2\lambda}\, \rho\!\bigl(B_t \Delta A + \Delta B\, A_t\bigr)\;}\quad (\star)$$

(or its constrained analogue with $\|B_t \Delta A + \Delta B\, A_t\| \le \lambda$).
Under gradient compatibility, the linear term equals
$\langle G,\, J_t[\Delta A, \Delta B]\rangle$, so $G$ never appears
alone in the algorithm.

**Gauge.** The objective depends on $(\Delta A, \Delta B)$ only through
$J_t[\Delta A, \Delta B]$. The kernel of $J_t$ is

$$\ker J_t \;=\; \{(S A_t,\, -B_t S) : S \in \mathbb{R}^{r \times r}\}$$

(check: $B_t (S A_t) + (-B_t S) A_t = 0$). The minimizer of $(\star)$
is therefore non-unique in factor coordinates; we ask the algorithm to
return the **minimum-Frobenius representative**, characterized by

$$B_t^\top \Delta B \;=\; \Delta A\, A_t^\top.$$

This fixes the second-order term $\Delta B\, \Delta A$ in the actual
finite LoRA update for reproducibility.

## Adjacent formulation — primary target

**This is the live target for project optimizer design.** "Adjacent"
here means the spectral constraint is imposed *per block*
($\|B \Delta A\|_2$ and $\|\Delta B\, A\|_2$ separately) rather than
jointly on the tangent $\|B \Delta A + \Delta B\, A\|_2$ — i.e.
adjacent to the joint operator-norm program $(\star)$, sharing the
Frobenius coupling term but replacing the joint constraint with two
per-channel ones.

**Per-channel spectral constraints with Frobenius coupling.**

$$\min_{\Delta A,\, \Delta B}\, \langle \nabla_A L,\, \Delta A\rangle + \langle \nabla_B L,\, \Delta B\rangle + \frac{1}{2\lambda}\|B \Delta A + \Delta B\, A\|_F^2 \quad \text{s.t.}\ \|B \Delta A\|_2 \le \lambda,\ \|\Delta B\, A\|_2 \le \lambda.$$

**Why this formulation matters.** Hybrid Picard's block-coordinate
iteration (see glossary) converges to the KKT point of *this* problem
with operator-norm-saturating polar in each block solve, not the
joint operator-norm problem $(\star)$. Replacing the polar block solve
by the exact prox of the per-block subproblem — which is **singular-value
clipping** — gives the principled optimizer for this formulation.

### The exact block solve is clipping, not polar

Each per-factor block subproblem has the form

$$\min_{\|X\|_2 \le \lambda}\ \langle C_X, X\rangle + \frac{1}{2\lambda}\|X\|_F^2$$

for some channel-coordinate data $C_X$ (see clipping-prox proposal §2.2 for the explicit form in terms of $u_A, u_B$ and the QR factors). This is the proximal map of the Frobenius-squared penalty onto the spectral-norm ball — its exact minimizer is the Frobenius projection of $-\lambda C_X$ onto that ball, computed by **clipping the singular values of $-\lambda C_X$ at $\lambda$**:

$$X^\star = U\, \mathrm{diag}\!\bigl(\min(\sigma_i, \lambda)\bigr)\, V^\top, \qquad -\lambda C_X = U\, \mathrm{diag}(\sigma_i)\, V^\top.$$

Hybrid Picard, by contrast, takes $-\lambda \cdot \mathrm{polar}(C_X)$ — saturating *every* active singular mode to magnitude $\lambda$ instead of truncating only the modes that exceed it. Polar is the operator-norm-solve direction (max-inner-product unit-norm matrix); clipping is the actual prox of the squared-penalty subproblem the formulation specifies.

**Consequence.** Polar erases the relative magnitudes of singular modes; clipping preserves sub-threshold modes at their unconstrained magnitudes. This gives a possible explanation for why clipping-like updates could behave differently from polar updates across ranks, but the rank-dependent validation pattern is empirical, not derived from this argument.

## Solved cases of $(\star)$

- **Case 1.** Joint, $\rho = \|\cdot\|_F^2$. Closed form (Sylvester).
- **Case 2.** One-factor restriction, operator norm. Closed form (polar).
- **Case 3.** Joint, operator norm. Empirically closed (E1–E7).

### Case 1. Joint Frobenius — Sylvester closed form

Smoothness of $\|\cdot\|_F^2$ gives normal equations
$J_t^\star J_t [\Delta A, \Delta B] = -\lambda\, J_t^\star [G]$, with
$J_t^\star (M) = (B_t^\top M,\; M\, A_t^\top)$. Writing
$S_L := B_t^\top B_t$ and $S_R := A_t A_t^\top$:

$$\begin{aligned}
S_L\, \Delta A + B_t^\top \Delta B\, A_t &= -\lambda\, \nabla_A L, \\
\Delta B\, S_R + B_t \Delta A\, A_t^\top &= -\lambda\, \nabla_B L.
\end{aligned}$$

Imposing the min-Frobenius gauge $B_t^\top \Delta B = \Delta A\, A_t^\top$
reduces the system to a Sylvester equation in $K \in \mathbb{R}^{r \times r}$:

$$K\, S_R + S_L\, K \;=\; -\lambda\, B_t^\top G\, A_t^\top \;=\; -\lambda\, (\nabla_A L)\, A_t^\top \;=\; -\lambda\, B_t^\top\, (\nabla_B L).$$

Solvable in $O(r^3)$ by separately eigendecomposing $S_L, S_R$ and
solving entrywise in the transformed bases. Then

$$\Delta A^\star = -S_L^{-1}\,(\lambda\, \nabla_A L + K^\star\, A_t), \qquad \Delta B^\star = -(\lambda\, \nabla_B L + B_t K^\star)\, S_R^{-1}.$$

### Case 2. One-factor operator-norm — closed form

Hold $A = A_t$ fixed:

$$\min_{\|\Delta B\, A_t\|_2 \le \lambda}\; \langle \nabla_B L,\, \Delta B\rangle.$$

Via the SVD $A_t = U_R \Sigma_R V_R^\top$, reparameterize
$\Delta B = X\, \Sigma_R^{-1}\, U_R^\top$. Then $\Delta B\, A_t = X\, V_R^\top$
has the same spectrum as $X$, and the problem reduces to

$$\min_{\|X\|_2 \le \lambda}\; \langle (\nabla_B L)\, U_R\, \Sigma_R^{-1},\, X\rangle,$$

with minimizer $X^\star = -\lambda\, \mathrm{polar}\bigl((\nabla_B L)\, U_R\, \Sigma_R^{-1}\bigr)$. Unwinding,

$$\Delta B^\star = -\lambda\, \mathrm{polar}\!\bigl((\nabla_B L)\, U_R\, \Sigma_R^{-1}\bigr)\, \Sigma_R^{-1}\, U_R^\top.$$

The polar projection is computed by Newton-Schulz on a
$d_\text{out} \times r$ matrix; everything else is $r \times r$.

### Case 3. Joint operator-norm — empirically closed

$$\min_{\Delta A,\, \Delta B :\, \|B_t \Delta A + \Delta B\, A_t\|_2 \le \lambda}\; \langle \nabla_A L,\, \Delta A\rangle + \langle \nabla_B L,\, \Delta B\rangle.$$

Case 1's smooth normal-equation derivation does not apply: $\|\cdot\|_2$
is non-smooth, the spectral-norm ball interacts with the low-rank image
of $J_t$ in a non-Euclidean way, and the optimality condition is not a
linear system. A deterministic $\tfrac{1}{2}$-approximation core solver
exists (`theory.md`), but experiments E1–E7 in
`investigations.md` show every variant of it loses to
hybrid Picard on the project workload (best E3: 0.7490 at $r=64$ vs
Picard 0.7382). The project has since pivoted to the adjacent
formulation above; this case is retained as reference for the
variational structure but is not under active development.

## Empirical context

LoRA fine-tuning of a 1B-parameter language model on a code dataset,
2000-step horizon, single seed, single-pass online (so LR tuning on
the eval stream is not an overfitting concern). Final eval cross-entropy
with learning rate selected per variant:

| Variant | $r{=}16$ | $r{=}64$ | $r{=}128$ |
|---|---|---|---|
| Plain Adam | 0.758 | — | — |
| Frobenius block-diagonal | 0.759 | 0.751 | — |
| Frobenius joint, Case 1 | 0.758 | 0.753 | — |
| Operator-norm one-factor, Case 2 | **0.755** | 0.745 | — |
| Hybrid Picard | 0.762 | **0.738** | ≈0.735 |

Two observations: (1) in the Frobenius family, coupling has little
measured benefit (Case 1 vs block-diagonal differ by ≤0.002 across
$r$); (2) hybrid Picard sign-flips across rank — above per-factor polar
at $r=16$ by ~0.007, below it at $r=64$ by ~0.007.

---

## Sylvester gauge lift (the closed-form solver)

Given a target tangent in core coordinates $X \in \mathbb{R}^{r \times n}$ representing a desired first-order change $J = B \Delta A + \Delta B\, A$ to a LoRA pair $(A, B)$, the closed-form **Sylvester gauge lift** returns the min-Frobenius factor representative $(\Delta A, \Delta B)$ — the unique pair of factor updates satisfying $B^\top \Delta B = \Delta A\, A^\top$ and minimizing $\|\Delta A\|_F^2 + \|\Delta B\|_F^2$.

Implementation: `solve_sylvester` in `lora_playground/utils.py`. Used by the joint operator-norm solver (Case 3, E-series), the joint Frobenius solver (Case 1, `AdamLinLoRA`), and the per-block adjacent formulation (target of `proposal.md`).

### Setup

Compute thin QR factorizations of the current factors:

$$B_t = Q_L R_L, \qquad A_t = R_R Q_R^\top,$$

with $Q_L \in \mathbb{R}^{m \times r}$, $Q_R \in \mathbb{R}^{n \times r}$ column-orthonormal and $R_L, R_R \in \mathbb{R}^{r \times r}$ invertible (full-rank standing assumption). Define the spectral preconditioners

$$S_L := B_t^\top B_t = R_L^\top R_L, \qquad S_R := A_t A_t^\top = R_R R_R^\top.$$

### The lift

Given a core-coordinate target $X \in \mathbb{R}^{r \times n}$ for the $A$-block (symmetric construction holds for the $B$-block), solve the small $r \times r$ Sylvester equation

$$S_L K + K S_R = R_L^\top X R_R^\top$$

for $K \in \mathbb{R}^{r \times r}$. One symmetric eigendecomposition per side, $O(r^3)$. Then assemble

$$\Delta A = S_L^{-1}\!\left[ (R_L^\top X - K R_R)\, Q_R^\top \right],$$

$$\Delta B = \left[ Q_L (X R_R^\top - R_L K) \right] S_R^{-1}.$$

These satisfy the tangent identity and the gauge condition:

$$B_t \Delta A + \Delta B\, A_t = Q_L\, X\, Q_R^\top \cdot (\text{rescaled by } R_L, R_R), \qquad B_t^\top \Delta B = \Delta A\, A_t^\top = K.$$

### Sanity check: Frobenius limit

Replacing the per-channel spectral constraint by the unconstrained Frobenius prox in the calling subproblem reduces the core target to $X = -\lambda L_0$ with $L_0 = R_L^{-\top} u_A$ (the [Adam covector](glossary.md#optimizer-concepts) in the QR basis). Lifting through the Sylvester gauge formulas above reproduces the closed-form Frobenius-coupled Sylvester step that the unconstrained ($c \to \infty$) limit of the clipping-prox optimizer is required to match — the unit test in `proposal.md` §6 verifies this to $10^{-5}$ on a synthetic LoRA pair.
