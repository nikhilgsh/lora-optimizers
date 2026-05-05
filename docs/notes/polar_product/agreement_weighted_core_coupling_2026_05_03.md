# Proposal: agreement-weighted core coupling (AWC)

*2026-05-03*

A candidate algorithm for the rank-stable-calibration open problem in `tikhonov_picard_collapse_2026_05_03.md`. Suggested externally; written up here for evaluation.

## Motivation

The doc identifies a rank-dependent failure: the optimal Picard iteration count $k$ in the polar-product LoRA optimizer is $k = 1$ at rank 16 and $k = 3$ at rank 64, with no logged statistic distinguishing the two regimes. The §5.1 anti-pattern "faster convergence to the same target" rules out solving the existing self-consistency problem more accurately, since $k \to \infty$ is precisely what the rank-16 evidence says we should not approach.

AWC takes the alternative path: keep the per-block prox machinery, but **replace the global $k$ knob with a modewise reliability coefficient $\rho$ computed from how well the $A$-side and $B$-side Adam-preconditioned directions agree about the shared $r \times r$ core update.** When the two factor-side views agree, the algorithm couples them (large-$k$ behavior); when they disagree, it doesn't (uncoupled / $k = 1$ behavior). The agreement check is per mode, not global.

Because the cross-coupling weight is modulated by $\rho$ rather than driven to its self-consistent fixed point, AWC solves a different objective from the existing Picard formulation — it doesn't reduce to $k \to \infty$ in any limit unless $\rho \equiv 1$.

## Setup

Per layer pair $(A, B)$, after Adam produces $u_A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $u_B \in \mathbb{R}^{d_{\text{out}} \times r}$. Define Gram matrices and their inverse roots (all $r \times r$, cheap):
$$
G_A = A A^\top + \delta I, \qquad G_B = B^\top B + \delta I.
$$

### Decompose Adam directions into "exclusive" and "shared-core" parts

Project $u_A$ along the row-space of $A$:
$$
u_A^{\text{core}} \ =\ (u_A A^\top)\, G_A^{-1}\, A, \qquad u_A^\circ \ =\ u_A - u_A^{\text{core}}.
$$
Symmetrically project $u_B$ along the column-space of $B$:
$$
u_B^{\text{core}} \ =\ B\, G_B^{-1}\, (B^\top u_B), \qquad u_B^\circ \ =\ u_B - u_B^{\text{core}}.
$$
$u_A^\circ$ is the part of $u_A$ that does not interact with $A$'s existing row content (cannot reach the shared LoRA subspace through $A$ alone); $u_B^\circ$ is symmetric. The "core" pieces $u_A^{\text{core}}, u_B^{\text{core}}$ live in the part of factor space that affects $BA$ in directions both factors can see.

The decomposition is gauge-invariant: under $A \to R A,\ B \to B R^{-1}$, the projections $G_A^{-1} A$ and $B G_B^{-1}$ transform compatibly so $u_A^\circ, u_B^\circ$ are unchanged.

### Whitened core views

Define
$$
S_A \ =\ G_B^{-1/2}\, (u_A A^\top)\, G_A^{-1/2}, \qquad S_B \ =\ G_B^{-1/2}\, (B^\top u_B)\, G_A^{-1/2}.
$$
Both are $r \times r$. They are the two factor-side views of what the *shared* dense LoRA correction should be, whitened by the per-side Gram metric so they are comparable.

In a noise-free linearization $u_A \propto B^\top G,\ u_B \propto G A^\top$ for a common dense gradient $G$, $S_A$ and $S_B$ are equal. They diverge when:

- Adam's elementwise preconditioning treats the two factor sides asymmetrically;
- batch noise contaminates the two factor gradients independently;
- the linearization breaks down.

### Modewise agreement coefficient

Symmetrize and take an SVD:
$$
S_+ = \tfrac{1}{2}(S_A + S_B), \qquad S_+ = U \Sigma V^\top.
$$
Express both views in this basis:
$$
\widehat S_A = U^\top S_A V, \qquad \widehat S_B = U^\top S_B V.
$$
For each entry $(i, j)$, define the agreement coefficient
$$
\rho_{ij} \ =\ \mathrm{clip}_{[0,1]}\!\left( \frac{2\, (\widehat S_A)_{ij}\, (\widehat S_B)_{ij}}{ (\widehat S_A)_{ij}^2 + (\widehat S_B)_{ij}^2 + \varepsilon_\rho } \right) \ \in\ [0, 1].
$$
$\rho_{ij} \approx 1$ when the two views point the same direction with similar magnitude in mode $(i, j)$; $\rho_{ij} \approx 0$ when they disagree, cancel, or only one side has meaningful signal there. The reliability operator
$$
\mathcal{R}(X) = U \big(\rho \odot (U^\top X V)\big) V^\top
$$
zeros out modes the two views disagree on and passes through modes they agree on.

## Algorithm

**1. Exclusive update (uncoupled, like $k = 1$).** Apply the standard polar-product pipeline to $u_A^\circ$ and $u_B^\circ$:
$$
P_A^\circ = \mathrm{polar}_{\text{NS-}j}\!\big(G_B^{-1/2} u_A^\circ\big), \qquad P_B^\circ = \mathrm{polar}_{\text{NS-}j}\!\big(u_B^\circ G_A^{-1/2}\big),
$$
$$
\widehat Z = G_B^{-1/2} P_A^\circ, \qquad \widehat Y = P_B^\circ G_A^{-1/2},
$$
$$
Z = -\eta\, \frac{\lVert u_A^\circ \rVert_F}{\lVert \widehat Z \rVert_F + \varepsilon}\, \widehat Z, \qquad Y = -\eta m\, \frac{\lVert u_B^\circ \rVert_F}{\lVert \widehat Y \rVert_F + \varepsilon}\, \widehat Y.
$$

**2. Shared-core update (agreement-weighted).** Solve, in the whitened core variables $X_A, X_B \in \mathbb{R}^{r \times r}$:
$$
\min_{X_A, X_B}\ \langle S_A, X_A\rangle + \langle S_B, X_B\rangle + \frac{1}{2\eta}\Big(\lVert X_A\rVert_F^2 + \lVert X_B\rVert_F^2 + 2\,\langle X_A, \mathcal{R}(X_B)\rangle\Big),
$$
subject to per-block spectral caps $\lVert X_A\rVert_2 \le \tau_A$, $\lVert X_B\rVert_2 \le \tau_B$ (or polar substitution, as in Algorithm 1 of the main doc).

The cross-term $\langle X_A, \mathcal{R}(X_B)\rangle$ is the only term coupling the two core updates; it is modulated entry-wise by $\rho$. With $\rho \equiv 0$ the problem decouples (per-block, $k = 1$-like). With $\rho \equiv 1$ the cross-coupling is fully active (joint, $k \to \infty$-like).

A practical no-new-knob choice for the spectral caps:
$$
\tau_A = \eta\, \frac{\lVert S_A\rVert_F}{\sqrt{r}}, \qquad \tau_B = \eta m\, \frac{\lVert S_B\rVert_F}{\sqrt{r}}.
$$

The problem is convex, $r \times r$, and small. Solve by projected gradient descent:
$$
\nabla_{X_A} = S_A + \eta^{-1}\big(X_A + \mathcal{R}(X_B)\big), \qquad \nabla_{X_B} = S_B + \eta^{-1}\big(X_B + \mathcal{R}(X_A)\big),
$$
with step size $s = \eta / (1 + \rho_{\max})$ where $\rho_{\max} = \max_{ij} \rho_{ij}$, projection $\Pi_{\lVert X\rVert_2 \le \tau}$ via singular-value clip (or polar). Run to tolerance.

**3. Combine.** Unwhiten the core solution
$$
C_A = G_B^{-1/2} X_A G_A^{-1/2}, \qquad C_B = G_B^{-1/2} X_B G_A^{-1/2},
$$
and apply
$$
\Delta A = Z + C_A A, \qquad \Delta B = Y + B C_B.
$$
No outer Picard loop, no min-Frobenius lift, no post-combine rescale.

## Why it might recover both ranks

- **$\rho$ low in important core modes** → cross-term suppressed → core update behaves as if uncoupled → $k = 1$-like behavior. Hypothesis: rank 16.
- **$\rho$ high in important core modes** → cross-term active → core update couples both factors self-consistently → $k \to \infty$-like behavior on those modes. Hypothesis: rank 64.

The mechanism is rank-free in its own statement; the rank-dependence emerges (if it does) from how Adam-preconditioned views happen to agree at small vs large $r$.

## Diagnostics that test the mechanism, independent of running AWC

These can be logged on existing $k$-sweep runs without implementing the full algorithm. They tell us whether the AWC story is consistent with the data before committing to the implementation.

- **Per-layer cosine of the two core views.** $\cos(S_A, S_B) = \langle S_A, S_B\rangle / (\lVert S_A\rVert_F \lVert S_B\rVert_F + \varepsilon)$. Prediction: higher at rank 64 than rank 16 in the layers where $k = 3$ helps the most.
- **Agreement vs disagreement energy.** $E_+ = \lVert S_+\rVert_F^2,\ E_- = \lVert (S_A - S_B)/2\rVert_F^2,\ q_{\text{agree}} = E_+ / (E_+ + E_-)$. Prediction: rank 64 has more $q_{\text{agree}}$, especially in the "core" piece.
- **Energy-weighted modewise $\bar\rho$.** $\bar\rho_+ = \sum_{ij} \rho_{ij} (\widehat S_+)_{ij}^2 / \sum_{ij} (\widehat S_+)_{ij}^2$. Should distinguish ranks more cleanly than the four scalar diagnostics in §4.3 of the main doc.
- **Where does $k=3$ minus $k=1$ live?** Decompose $J^{(3)} - J^{(1)}$ into shared-core ($X_C = X_A + X_B$) and ownership ($X_D = X_A - X_B$) parts. If the $k$-flip is a shared-core ownership issue, the difference should be dominated by shared-core movement, not by the exclusive $(Z, Y)$ pieces.
- **Modewise localization of the Picard correction.** In the $(U, V)$ basis of $S_+$, ask whether the $k=3$ minus $k=1$ correction at rank 64 lives in high-$\rho$ modes (consistent with AWC story), and whether the rank-16 harmful correction lives in low-$\rho$ modes.
- **Counterfactual closeness.** Compute the AWC update offline at logged checkpoints (for $k = 1$ and $k = 3$ runs); measure $\lVert J_{\text{AWC}} - J_{k=1}\rVert_F / \lVert J_{k=1}\rVert_F$ and similarly for $k = 3$. Predict closer to $k = 1$ at rank 16, closer to $k = 3$ at rank 64.

## Falsification

The proposal collapses if any of the following:

- The $\rho$ distributions overlap across ranks at the same level the four §4.3 statistics do.
- $J^{(3)} - J^{(1)}$ is dominated by exclusive (non-core) movement, not shared-core.
- Rank 64's $k = 3$ gain comes from low-$\rho$ modes (then $\rho$ is the wrong gate).
- The offline AWC update is not closer to $k = 1$ at rank 16 or to $k = 3$ at rank 64.

## Open issues with the construction

- $\rho_{ij}$ is computed from current-step $S_A, S_B$, which are functions of single-step Adam outputs. Adam smooths via $\beta_1, \beta_2$ but $\rho$ is still a per-step quantity; the noise floor of these views has not been characterized.
- The core/exclusive split projects on $A$'s row-space and $B$'s column-space. At small $r$ these spaces are narrow; at large $r$ wider. The mass distribution between "core" and "exclusive" pieces may itself be rank-dependent in a way that interacts with the AWC mechanism.
- $\varepsilon_\rho$ in the agreement-coefficient denominator is a regularizer that, while small, is a numerical knob. Whether AWC is sensitive to it has not been tested.
- The "exclusive" pieces $u_A^\circ, u_B^\circ$ are updated like $k = 1$. If at large $r$ those pieces also benefit from coupling, this part of the design is itself an assumption.
- The construction is more complex than a single-statistic gate, with multiple places where the implementation can drift from the spec.
