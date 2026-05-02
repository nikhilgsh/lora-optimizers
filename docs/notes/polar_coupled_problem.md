# An efficient algorithm for joint operator-norm LoRA updates — open problem

## Goal

A self-contained statement of an open algorithm-design problem in
low-rank optimizer design: we want a solver we can drop into a LoRA
optimizer.

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

A modern view of optimizer design (Muon / AdaMuon line) treats an
optimizer as a variational problem. Two formulations differ by how
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
finite LoRA update for reproducibility. The variational problem and
its sanity checks are stated on the tangent update; candidates are not
required to optimize the second-order term, but should justify any
deviation from the default gauge.

## Solved cases of $(\star)$

- **Case 1.** Joint, $\rho = \|\cdot\|_F^2$. Closed form (Sylvester).
- **Case 2.** One-factor restriction, operator norm. Closed form (polar).
- **Case 3.** Joint, operator norm. **Open.**

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

**Note on damping.** Implementations that substitute
$S_L + \varepsilon I,\, S_R + \varepsilon I$ into the undamped formulas
above are *not* solving the damped factor-space surrogate
(adding $\tfrac{\varepsilon}{2\lambda}\|\Delta A\|_F^2 + \tfrac{\varepsilon}{2\lambda}\|\Delta B\|_F^2$
removes the gauge degeneracy that the undamped closed form relies on).
The substitution is a numerical regularizer, not a principled surrogate.
Out of scope for this open problem.

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
$d_\text{out} \times r$ matrix; everything else is $r \times r$. The
$B$-fixed analogue is symmetric (apply the same derivation with
$B_t^\top$ in place of $A_t$).

### Case 3. Joint operator-norm — **open**

$$\min_{\Delta A,\, \Delta B :\, \|B_t \Delta A + \Delta B\, A_t\|_2 \le \lambda}\; \langle \nabla_A L,\, \Delta A\rangle + \langle \nabla_B L,\, \Delta B\rangle.$$

Case 1's smooth normal-equation derivation does not apply: $\|\cdot\|_2$
is non-smooth, the spectral-norm ball interacts with the low-rank image
of $J_t$ in a non-Euclidean way, and the optimality condition is not a
linear system.

## A natural-but-flawed iteration: hybrid Picard

Take the Frobenius normal equations of Case 1, move the cross terms to
the RHS as a "coupling correction" on the gradient, and replace each
per-factor solve with the Case-2 operator-norm solve:

$$\Delta A^{(k+1)} = \mathrm{polar\text{-}solve}_A\!\left(\nabla_A L + \tfrac{1}{\lambda}\, B_t^\top \Delta B^{(k)} A_t\right), \quad \Delta B^{(k+1)} = \mathrm{polar\text{-}solve}_B\!\left(\nabla_B L + \tfrac{1}{\lambda}\, B_t \Delta A^{(k)} A_t^\top\right),$$

initialized at zero. The $1/\lambda$ on the cross term is essential.
This is the iteration we currently run. Its fixed point is *not* the
optimality condition of $(\star)$ for any norm-induced penalty we know
— it mixes Frobenius cross-terms with operator-norm per-factor solves.

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

Two observations (motivational, not a benchmark claim):

1. In the Frobenius family, coupling has little measured benefit
   (Case 1 vs block-diagonal differ by ≤0.002 across $r$).
2. The hybrid Picard sign-flips across rank: above per-factor polar at
   $r=16$ by ~0.007, below it at $r=64$ by ~0.007.

A finer ablation at $r=16$, varying hybrid-Picard iteration count
(1 = uncoupled per-factor polar on both factors from zero init,
distinct from Case 2's one-factor restriction; 2 = current default):

| iterations | step-1400 eval cross-entropy |
|---|---|
| 1 | 0.7669 |
| 2 | 0.7726 |
| 3 | 0.7681 |
| 4 | 0.7710 |

The non-monotone pattern is consistent with damped oscillation toward a
fixed point near 0.770 — worse than the uncoupled per-factor variant
(0.767, same metric and step). Driving the iteration to convergence
would land at the bad fixed point, consistent with the
"variationally inconsistent" diagnosis above.

## The open problem

Using only $(A_t, B_t, \nabla_A L, \nabla_B L, \lambda)$ satisfying the
standing assumptions, find a cheap algorithm that approximately solves
one of:

1. **Constrained form** — fix the tangent spectral norm to $\lambda$:

   $$\min_{\Delta A,\, \Delta B :\, \|B_t \Delta A + \Delta B\, A_t\|_2 \le \lambda}\; \langle \nabla_A L,\, \Delta A\rangle + \langle \nabla_B L,\, \Delta B\rangle.$$

2. **Squared-penalty form** — solve $(\star)$ with $\rho = \|\cdot\|_2^2$;
   the optimum has tangent spectral norm scaled by a gradient-dependent
   restricted dual norm, not $\lambda$.

3. **Direction only** — return a maximizing direction, with magnitude
   delegated to the outer optimizer.

These are *not literally equivalent* (1 fixes the magnitude, 2 picks it
from the gradient, 3 leaves it out). They share maximizing directions.
A candidate must state which one it targets and return the min-Frobenius
factor representative $B_t^\top \Delta B = \Delta A\, A_t^\top$ (or
justify a different gauge).

**Scope.** The variational problem is stated on the tangent update
$J_t[\Delta A, \Delta B]$. Finite-step stability and the size of the
dropped second-order term $\Delta B\, \Delta A$ are out of scope —
controlling them requires conditioning bounds on $A_t, B_t$ (e.g.,
$\sigma_{\min}$ bounded away from zero) or an outer step-size policy,
both downstream of this open problem.

### Cost budget (per training step, per LoRA pair)

Allowed:

- $O(1)$ symmetric solves / eigendecompositions of $r \times r$
  matrices ($r \le 256$). Cost $O(r^3)$, negligible.
- $O(1)$ Newton-Schulz polar iterations on $r \times d$ or
  $d \times r$ matrices ($d = \max(d_\text{in}, d_\text{out})$). Each
  step costs $O(d r^2)$.
- $O(1)$ multiplications of compatible $r \times d$, $d \times r$,
  $r \times r$ matrices.

Forbidden:

- Any $d_\text{out} \times d_\text{in}$ dense object that depends on
  $G$ (in particular, an SVD of one).
- Explicit dense $d \times d$ projection matrices (e.g.,
  $Q_B Q_B^\top$), even if $G$-independent. Use skinny factors and
  $r \times r$ cores.
- Iteration counts that grow with $d$ or with the loss landscape.

A slow exact reference solver violating the budget (e.g., proximal
methods, full SVD per step) is fine *as a debugging oracle* — to check
that the $(\star)$-minimizer is in fact a useful update direction, and
to measure how far cheap approximations land. The deliverable is the
cheap algorithm.

### Sanity checks

- **Reduces to Case 1.** Swapping $\|\cdot\|_2$ for $\|\cdot\|_F$
  reproduces the Sylvester closed form (squared-penalty) or its
  direction (constrained).
- **Reduces to Case 2 under explicit one-factor restriction.** If the
  admissible set is restricted to $\Delta A = 0$, recover the displayed
  $A$-fixed $\Delta B$ formula; symmetrically, restricting to
  $\Delta B = 0$ recovers the $B$-fixed $\Delta A$ formula. This check
  concerns the restricted problem, not the min-Frobenius gauge for the
  full joint problem — the joint min-Frobenius representative will not
  generally satisfy $\Delta A = 0$ even when one factor's update is
  small.
- **Symmetric in $(A, B)$.** $(\star)$ is invariant under
  $(A, B, G) \leftrightarrow (B^\top, A^\top, G^\top)$; the algorithm
  should be too.

## Adjacent formulation: the implicit hybrid-Picard objective

**Per-channel-tangent-contribution constraints, Frobenius coupling.**
Constrain each factor's contribution to the tangent separately,
keeping the Frobenius coupling:

$$\min_{\Delta A,\, \Delta B}\, \langle \nabla_A L,\, \Delta A\rangle + \langle \nabla_B L,\, \Delta B\rangle + \frac{1}{2\lambda}\|B_t \Delta A + \Delta B\, A_t\|_F^2 \quad \text{s.t.}\ \|B_t \Delta A\|_2 \le \lambda,\ \|\Delta B\, A_t\|_2 \le \lambda.$$

This is the coherent variational problem whose alternating-minimization
fixed point the **hybrid Picard iteration of Section "A natural-but-flawed
iteration" is implicitly trying to solve.** Each per-factor sub-problem
is a Case-2 operator-norm solve on $B_t \Delta A$ or $\Delta B\, A_t$.
The doc's "variational inconsistency" diagnosis is precisely that
hybrid Picard mixes Frobenius cross-coupling with operator-norm
per-channel constraints — the alternation converges, but to the KKT
point of *this* problem, not of $(\star)$ with a joint operator norm.

This is **not** the open problem — Case 3 constrains the joint tangent
spectral norm, which is gauge-invariant and tight. It is useful as an
empirical control: if a clean implementation of this problem
(block-coordinate ascent driven to convergence, explicit multipliers)
outperforms hybrid Picard, the failure mode is the iteration; if it
matches hybrid Picard's bad fixed point, the failure mode is the
formulation itself.

**Side note: the clean block solve is singular-value clipping, not
polar.** Each block sub-problem of the formulation above is
$\min_{\|X\|_2 \le \lambda}\, \langle C_X, X\rangle + (1/2\lambda)\|X\|_F^2$
for some channel data $C_X$. The exact minimizer is the Frobenius
projection of $-\lambda C_X$ onto the spectral-norm ball:

$$X^+ = U\, \mathrm{diag}\!\bigl(\min(\sigma_i, \lambda)\bigr)\, V^\top \quad \text{where} \quad -\lambda C_X = U\, \mathrm{diag}(\sigma_i)\, V^\top.$$

Hybrid Picard instead applies $-\lambda \cdot \mathrm{polar}(C_X)$,
which saturates *every* active singular direction to magnitude
$\lambda$ rather than clipping only the modes that exceed it. That is
the "operator-norm solve" (the unit-norm direction maximizing the
inner product) but it is *not* the prox of the squared-penalty
sub-problem, which is what the alternation actually wants. Replacing
polar with clip in each block is the minimal fix that makes hybrid
Picard's iteration converge to the KKT point of its own implicit
objective. Recorded as a note; not a priority for this project — Case
3 (joint tangent spectral) is the target.
