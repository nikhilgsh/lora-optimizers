# The polar-product LoRA optimizer

The polar-product family is this project's current best LoRA optimizer. This document derives it from a variational program: state the program, solve it exactly by block-coordinate descent, and then make two named substitutions that take the exact solver to the actual algorithm. The full algorithm is stated at the end (§8). The document is purely expository.

## 1. Setup

We fine-tune a pretrained transformer by adding a low-rank correction to each frozen weight matrix. For a frozen $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the LoRA correction is

$$
\Delta W \;=\; \frac{\alpha}{r}\, B A,
\qquad A \in \mathbb{R}^{r \times d_{\text{in}}},\ \ B \in \mathbb{R}^{d_{\text{out}} \times r},
$$

with $r \ll \min(d_{\text{in}}, d_{\text{out}})$ the LoRA rank and $\alpha$ a fixed scaling constant; we absorb $\alpha/r$ into the learning rate. In all experiments reported here we set $\alpha = r$, so $\alpha/r = 1$ and the effective per-step scale is rank-independent — this is what keeps the optimal $\eta$ stable across $r \in \{16, 32, 64, 128, 256\}$ in our sweeps. Only $A$ and $B$ are trained, and backpropagation produces factor gradients $g_A, g_B$ at each step. Adam preconditioning maps these to bias-corrected directions $u_A, u_B$ in the standard way. The variational derivation below takes $u_A, u_B$ as given; the explicit Adam update is restated in Algorithm 1 (§8.2).

*Remark (initialization).* Our default LoRA initialization follows the PEFT default — random $A$ and $B = 0$ — call this **Init[A]**. The recent µA paper (Chen, Villar, Hayou, arXiv:2602.06204) argues that the alternative **Init[B]** (random $B$, $A = 0$) with effective multiplier 1 is the principled choice: it produces a rank-invariant optimal learning rate that also matches the full-finetuning optimum, enabling LoRA→FFT transfer. Under Init[A] + $\alpha = r$ (our current setup) the same paper predicts an $r^{-1/2}$ drift in optimal $\eta$; we have not seen this drift empirically across $r \in \{16, \ldots, 256\}$, but the rank range may be too narrow to resolve it. A switch to Init[B] is under consideration; for now we keep Init[A] for continuity with prior leaderboard runs and treat the init regime as a planned ablation. See Appendix B for the µA-style derivation specialized to the polar-product step.

**Definition 1 (joint tangent).** A perturbation $(\Delta A, \Delta B)$ of the factors changes the merged weight $W + B A$ by

$$
J \;:=\; B\, \Delta A + \Delta B\, A \;\in\; \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}.
$$

The joint tangent is the only object the loss sees: any pair $(\Delta A, \Delta B)$ producing the same $J$ produces the same first-order change in loss. This makes per-factor reasoning subtle — the two factors interact through $J$ rather than as independent parameter blocks.

## 2. The variational problem

A single optimizer step on a layer pair targets the **per-block operator-norm program**:

$$
\min_{\Delta A,\, \Delta B}\ \underbrace{\langle u_A,\, \Delta A\rangle + \langle u_B,\, \Delta B\rangle}_{\text{linear cost}} \;+\; \underbrace{\frac{1}{2\eta}\, \lVert B\, \Delta A + \Delta B\, A \rVert_F^2}_{\text{trust region on $J$}}
\quad\text{s.t.}\quad
\lVert B\, \Delta A \rVert_2 \le \tau, \ \ \lVert \Delta B\, A \rVert_2 \le \tau.
\tag{1}
$$

The three pieces:

- **Linear cost.** $u_A, u_B$ are the Adam-preconditioned descent directions on each factor; minimizing $\langle u_A, \Delta A\rangle + \langle u_B, \Delta B\rangle$ is the per-factor analogue of "step in the direction of $-u$."
- **Coupling penalty.** The Frobenius term $\tfrac{1}{2\eta}\lVert J\rVert_F^2$ is the only term coupling $\Delta A$ and $\Delta B$. A step in $\Delta A$ alone is meaningful through its image $B\Delta A$; the penalty acknowledges that the two factors share this image.
- **Per-block spectral constraints.** $\lVert B \Delta A\rVert_2 \le \tau$ and $\lVert \Delta B\, A\rVert_2 \le \tau$ separately cap the operator norm of each factor's contribution to the merged update. This is the LoRA analogue of Muon's spectral cap on the dense weight update.

The "per-block" qualifier names the constraint structure: $B \Delta A$ and $\Delta B A$ are constrained *separately*, not their sum.

The threshold $\tau$ is the only undefined hyperparameter in (1); it has no workload-independent default. We will eliminate it in §6.

*Remark (exact chord vs tangent).* The full change in the merged weight from a step $(\Delta A, \Delta B)$ is

$$
\Delta W \;=\; (B + \Delta B)(A + \Delta A) - BA \;=\; J + \Delta B\,\Delta A,
$$

so $J$ is the tangent linearization at $(A, B)$, dropping the second-order $\Delta B\,\Delta A$ term. The block-coordinate decomposition of §3 extends to the exact chord: holding $\Delta B$ fixed, $\Delta W = (B + \Delta B)\,\Delta A + \Delta B\, A$ is still linear in $\Delta A$, so Lemmas 1–2 and Proposition 1 carry over verbatim with $B + \Delta B$ in place of $B$ (and $A + \Delta A$ in place of $A$ for the symmetric $B$-subproblem). The spectral preconditioner $S_B^{-1/2}$ must then be recomputed each Picard iterate as $\Delta B$ changes; at LoRA rank sizes ($r \le 256$) this is cheap via a Newton-style iterative matrix inverse square root, so the cached form in §8 is a convenience, not a real cost saving. Since $\Delta B\,\Delta A$ is $O(\eta^2)$ in step size, the difference is a second-order correction. We work with $J$ here for simplicity and leave the exact-chord variant as a future ablation.

## 3. Block-coordinate decomposition

The Frobenius coupling in (1) is bilinear in $(\Delta A, \Delta B)$, so block-coordinate descent — fix one factor, optimize the other — reduces (1) to single-factor subproblems. The bilinearity collapses neatly:

**Lemma 1 (cross-coupling collapse).** Holding $\Delta B$ fixed in (1), the $A$-subproblem becomes

$$
\min_{\Delta A}\ \bigl\langle \tilde{u}_A,\, \Delta A\bigr\rangle \;+\; \frac{1}{2\eta}\,\bigl\langle \Delta A,\, B^\top B\, \Delta A \bigr\rangle
\quad\text{s.t.}\quad \lVert B\Delta A\rVert_2 \le \tau,
\tag{2}
$$

with **corrected linear cost** $\tilde{u}_A \;:=\; u_A + \tfrac{1}{\eta}\, B^\top\, \Delta B\, A$. By symmetry, fixing $\Delta A$ yields a $B$-subproblem with $\tilde u_B := u_B + \tfrac{1}{\eta}\, B\, \Delta A\, A^\top$.

*Proof.* Expand the Frobenius coupling:

$$
\lVert B \Delta A + \Delta B\, A \rVert_F^2 \;=\; \lVert B\Delta A\rVert_F^2 \;+\; 2\,\bigl\langle B\Delta A,\, \Delta B\, A\bigr\rangle \;+\; \lVert \Delta B\, A\rVert_F^2.
$$

The third term is constant in $\Delta A$. The cross term $\langle B\Delta A, \Delta B A\rangle = \langle \Delta A,\, B^\top \Delta B\, A\rangle$ is linear in $\Delta A$ and folds into the linear cost via $\tilde u_A$. The first term gives the $B^\top B$ regularizer. The constraint is unchanged. ∎

The shift $\tilde u_A - u_A = \tfrac{1}{\eta} B^\top \Delta B A$ is the **cross-coupling correction** — the only place the two factors interact in the per-block subproblem.

## 4. Exact solution of the per-block subproblem

The $A$-subproblem (2) has a non-identity quadratic in $B^\top B$ and a constraint on $B\Delta A$. Whitening removes both.

**Definition 2 (whitened objects).** Let $S_B := B^\top B + \delta I$ ($r \times r$, SPD; $\delta I$ damping handles rank-deficient $B$). The **whitened update** and **whitened cost** are

$$
Y_A \;:=\; S_B^{1/2}\, \Delta A,
\qquad
c_A \;:=\; S_B^{-1/2}\, \tilde u_A.
$$

The map $\Delta A \leftrightarrow Y_A$ is invertible by $\Delta A = S_B^{-1/2} Y_A$.

**Lemma 2 (whitened subproblem).** In coordinates $(Y_A, c_A)$, the $A$-subproblem (2) is

$$
\min_{Y_A}\ \bigl\langle c_A,\, Y_A\bigr\rangle \;+\; \frac{1}{2\eta}\,\lVert Y_A\rVert_F^2
\quad\text{s.t.}\quad \lVert Y_A\rVert_2 \le \tau.
\tag{3}
$$

*Proof.* Substitute and approximate $B^\top B \approx S_B$ (valid up to the $\delta$ regularizer): $\langle \tilde u_A, \Delta A\rangle = \langle c_A, Y_A\rangle$, $\langle \Delta A, S_B \Delta A\rangle = \lVert Y_A\rVert_F^2$, and $\lVert B\Delta A\rVert_2 \approx \lVert Y_A\rVert_2$. ∎

**Proposition 1 (per-block clip prox).** For any threshold $\tau > 0$, the unique minimizer of (3) is $Y_A^\star(\tau) = \mathrm{clip}_\tau(-\eta\, c_A)$, and in original coordinates

$$
\boxed{\quad
\Delta A^\star(\tau) \;=\; S_B^{-1/2}\, \mathrm{clip}_\tau\!\bigl(-\eta\, c_A\bigr),
\quad}
\tag{4}
$$

where $\mathrm{clip}_\tau\!\bigl(U\Sigma V^\top\bigr) := U\,\min(\Sigma, \tau)\,V^\top$ is the singular-value clip.

*Proof.* Complete the square in (3):

$$
\langle c_A, Y_A\rangle + \frac{1}{2\eta}\lVert Y_A\rVert_F^2
\;=\;
\frac{1}{2\eta}\,\lVert Y_A - (-\eta\, c_A)\rVert_F^2 \;-\; \tfrac{\eta}{2}\,\lVert c_A\rVert_F^2.
$$

The constrained problem is therefore the Frobenius projection of $-\eta\, c_A$ onto $\{\lVert Y_A\rVert_2 \le \tau\}$, whose closed form (Mirsky) is the singular-value clip. Undo the whitening. ∎

The $B$-side is symmetric: with $c_B := \tilde u_B \, S_A^{-1/2}$ and $Y_B := \Delta B\, S_A^{1/2}$,

$$
\Delta B^\star(\tau) \;=\; \mathrm{clip}_\tau\!\bigl(-\eta\, c_B\bigr)\, S_A^{-1/2}.
\tag{4'}
$$

Equations (4) and (4′) define a per-block prox map: from a corrected linear cost ($\tilde u_A$ or $\tilde u_B$) and a threshold $\tau$, produce the optimal factor update. Substitutions 1 and 2 in §§6–7 act on this map.

## 5. Picard outer loop and the exact reference algorithm

The two subproblems share state — $\tilde u_A$ depends on $\Delta B$ and vice versa — so block-coordinate descent on the joint problem (1) alternates the two solves.

**Algorithm R** (exact clip-based reference solver). Given Adam directions $u_A, u_B$, current factors $A, B$, threshold $\tau > 0$, and Picard count $k$:

1. Initialize $\Delta A^{(0)} = \Delta B^{(0)} = 0$.
2. For $n = 1, \ldots, k$:
    a. **Cross-coupling correction.**
        $$
        \tilde u_A^{(n)} \;=\; u_A + \tfrac{1}{\eta}\, B^\top\, \Delta B^{(n-1)}\, A,
        \qquad
        \tilde u_B^{(n)} \;=\; u_B + \tfrac{1}{\eta}\, B\, \Delta A^{(n-1)}\, A^\top.
        $$
    b. **Per-block clip prox** (Proposition 1):
        $$
        \Delta A^{(n)} \;=\; S_B^{-1/2}\,\mathrm{clip}_\tau\!\bigl(-\eta\, S_B^{-1/2}\,\tilde u_A^{(n)}\bigr),
        \qquad
        \Delta B^{(n)} \;=\; \mathrm{clip}_\tau\!\bigl(-\eta\,\tilde u_B^{(n)}\,S_A^{-1/2}\bigr)\,S_A^{-1/2}.
        $$
3. Return $(\Delta A^{(k)}, \Delta B^{(k)})$.

Algorithm R is the exact block-coordinate solver of (1) at fixed threshold $\tau$. Its only departure from the algorithm we ultimately want (Algorithm 1, §8) is that it (i) carries an undefined hyperparameter $\tau$ and (ii) uses clip rather than polar. Substitutions 1 and 2 in §§6–7 remove these in turn.

**Proposition 2 (joint solver in the limit).** Every fixed point of Algorithm R is a global optimum of (1). When the iteration contracts, $(\Delta A^{(k)}, \Delta B^{(k)}) \to$ joint optimum as $k \to \infty$.

*Proof sketch.* At a fixed point, Proposition 1 makes $\Delta A^\infty$ optimal for (2) at $\tilde u_A^{(\infty)}$, and symmetrically for $\Delta B^\infty$. Block optimality plus convexity of (1) in $(\Delta A, \Delta B)$ imply joint optimality. ∎

Two limits:

- $k = 1$: with $\Delta B^{(0)} = 0$, $\tilde u_A = u_A$ and the cross-coupling correction is dropped — each factor steps as if the other were fixed.
- $k \to \infty$ (under contraction): the iterates converge to the joint solution of (1).

## 6. Substitution 1: adaptive $\tau$

The threshold $\tau$ in (1) has no workload-independent default. Substitute it with a runtime quantity that we already have — Adam's own magnitude $\lVert u_A\rVert_F$ — by choosing $\tau$ at each per-block step to match it.

The per-block prox $\Delta A^\star(\tau)$ from Proposition 1 depends on the current $\tilde u_A$ via $c_A = S_B^{-1/2}\,\tilde u_A$, and $\tilde u_A$ is updated each Picard iterate as $\Delta B$ changes (§5). The substitution $\tau \leftarrow \tau^\star_A$ below is therefore *per-iterate*: at iterate $n$ of Algorithm R, $\tilde u_A^{(n)}$ determines $c_A^{(n)}$, the function $f$ in the statement below is computed against this $c_A^{(n)}$, and $\tau^\star_A$ takes a new value $\tau_A^{\star\,(n)}$. The Picard iteration of §5 with this rule produces a sequence $\tau_A^{\star\,(1)}, \tau_A^{\star\,(2)}, \ldots$, which converges along with $(\Delta A^{(k)}, \Delta B^{(k)})$ if the iteration contracts.

**Proposition 3 (Substitution 1: adaptive threshold).** Fix the corrected cost $\tilde u_A$ (and hence $c_A$). Let $f(\tau) := \lVert \Delta A^\star(\tau)\rVert_F$. Then:

(a) $f$ is continuous and non-decreasing on $[0, \infty)$, strictly increasing on $\bigl(0,\, \eta\,\sigma_{\max}(c_A)\bigr]$, and constant on $\bigl[\eta\,\sigma_{\max}(c_A),\, \infty\bigr)$.

(b) $f(0) = 0$ and $f(\infty) = \eta\,\lVert S_B^{-1}\,\tilde u_A\rVert_F$.

Provided $\eta\,\lVert u_A\rVert_F \le f(\infty)$, there is a unique $\tau^\star_A \in \bigl(0,\, \eta\,\sigma_{\max}(c_A)\bigr]$ satisfying $f(\tau^\star_A) = \eta\,\lVert u_A\rVert_F$. Algorithm R with $\tau \leftarrow \tau^\star_A$ at each per-block step is then a block-coordinate solver of (1) at adaptive threshold.

*Proof.* In Appendix A. ∎

The substitution pins the per-block update's Frobenius magnitude to Adam's natural scale; the hyperparameter $\tau$ is gone with no new hyperparameter replacing it. A symmetric $\tau^\star_B$ governs the $B$-side prox (4′).

*Remark (admissibility hypothesis).* The condition $\eta\,\lVert u_A\rVert_F \le f(\infty)$ asks that the unconstrained per-block solution have Frobenius magnitude at least Adam's. It can fail when the trust region in (1) is so tight that even the cap-free solution is smaller than Adam's natural scale; in that regime the magnitude rule cannot be satisfied at any $\tau$, and the algorithm of §8 — which always rescales to $\eta\,\lVert u_A\rVert_F$ — diverges from the variational interpretation.

### 6.1 Alternative (Substitution $1'$): pick $\tau$ from a chord-spectral trust region

Substitution 1 above pins the per-block update's Frobenius magnitude to Adam's. That choice is convenient (no new hyperparameter, no per-step σ estimate) but has no clean variational source — the per-block-contraction caps $\tau$ in (1) are absorbed into a Frobenius rule that doesn't follow from (1)'s constraints. It also controls only the Frobenius norm of the update, not the spectral norm of the factors themselves. In practice we observe that $\sigma_{\max}(B)$ can grow $\sim10\times$ over a run while $\lVert B\rVert_F$ barely changes; once that happens, a step at $\tau=\eta\,\lVert u_A\rVert_F$ is too large in operator norm and the Picard iteration stops contracting.

A different rule picks $\tau$ at each per-block step so that the *actual merged-weight change* satisfies a spectral-norm trust region:

$$
\bigl\lVert \Delta W \bigr\rVert_2 \;=\; \bigl\lVert (B+\Delta B)(A+\Delta A) - BA \bigr\rVert_2 \;=\; \bigl\lVert B\,\Delta A + \Delta B\,A + \Delta B\,\Delta A \bigr\rVert_2 \;\le\; \eta.
$$

This is a meaningful trust region directly on the chord — the actual quantity the loss sees — using the operator norm rather than Frobenius. The threshold $\eta$ here plays the role of a *spectral step size*: maximum allowed singular value of the merged-weight change per step.

By submultiplicativity (Spectron, Janson et al. 2026, eq 13–15), if the per-block update satisfies $\lVert \Delta A\rVert_2 \le \rho$ and $\lVert \Delta B\rVert_2 \le \rho$, then

$$
\lVert \Delta W\rVert_2 \;\le\; \sigma_{\max}(B)\cdot\rho + \sigma_{\max}(A)\cdot\rho + \rho^2 \;=\; \rho\bigl(\sigma_{\max}(A) + \sigma_{\max}(B) + \rho\bigr).
$$

Setting the right-hand side $\le \eta$ and using $\rho \le 1$ gives the sufficient condition

$$
\boxed{\quad
\rho \;:=\; \frac{\eta}{\sigma_{\max}(A) + \sigma_{\max}(B) + 1}.
\quad}
\tag{6}
$$

**Substitution $1'$ (chord-spectral magnitude rule).** Compute $\rho$ via (6) at each per-block step, and rescale the per-block direction $\widetilde{\mathrm dA} := S_B^{-1/2}\,\mathrm{polar}(c_A)$ from Theorem 1 to operator norm $\rho$ rather than Frobenius norm $\eta\,\lVert u_A\rVert_F$:

$$
\Delta A \;:=\; -\rho \, \frac{\widetilde{\mathrm dA}}{\bigl\lVert \widetilde{\mathrm dA}\bigr\rVert_2}, \qquad \Delta B \;:=\; -\rho \, \frac{\widetilde{\mathrm dB}}{\bigl\lVert \widetilde{\mathrm dB}\bigr\rVert_2}.
$$

This guarantees $\lVert \Delta A\rVert_2 = \lVert \Delta B\rVert_2 = \rho$ and hence $\lVert \Delta W\rVert_2 \le \eta$ by construction.

*Honest variational status.* The cross-coupling and per-block direction $\widetilde{\mathrm dA}$ come from program (1) (Frobenius coupling + per-block-contribution caps; algorithm.md §§3–4). The magnitude rule (6) comes from a different program (Spectron's per-factor op-norm caps + chord trust region). These are two distinct variational sources; a single program that derives both with a clean closed-form per-block prox does not obviously exist. Substitution $1'$ is therefore a heuristic combination, in the same spirit as Substitution 2 (clip → polar), which is also variationally inconsistent outside the saturating regime.

*Effects.*

- **Automatic brake when $\sigma_{\max}$ grows.** When $\sigma_{\max}(B)$ grows, $\rho$ shrinks and the update gets smaller on its own. This targets the empirical failure observed at $r=128$ step $\sim$2200 in `polar_k3_4k_rsweep`, where $\sigma_{\max}(B)$ had grown $\sim10\times$ from initialization.
- **Hyperparameter $\eta$ is reinterpreted.** Under Substitution 1, $\eta\!\approx\!3{\times}10^{-4}$ is a Frobenius-rate scale; under Substitution $1'$, $\eta$ is a spectral-norm rate. Empirically these differ by a factor on the order of $\sigma_{\max}(A)+\sigma_{\max}(B)$, so the optimal $\eta$ for Substitution $1'$ is expected to be $\sim$10–30$\times$ larger. Per-rank lr retuning is required.
- **Rest of the algorithm unchanged.** Lemma 1 (cross-coupling correction), Lemma 2 (whitening), and polar via Newton–Schulz all carry over; only the final magnitude rescale of the per-block step is replaced.

The cost of computing $\rho$: one power-iteration step on each factor per optimizer step (∼ free at LoRA's matrix sizes).

## 7. Substitution 2: clip $\to$ polar

After Substitution 1, the per-block prox is $\mathrm{clip}_{\tau^\star_A}$. Substitution 2 replaces clip with the polar map. The clean reading is:

> **Pretend the saturating regime holds.** Act as if every singular value of $-\eta\,c_A$ exceeds $\tau^\star_A$ — so the clip flattens all of them to a common value and equals polar up to scale.

Whenever the pretense is correct, the resulting algorithm is variationally exact; otherwise, it is a uniform-spectrum approximation imposed by acting as if the pretense were correct.

**Definition 3 (polar map).** For $X = U\Sigma V^\top$, $\mathrm{polar}(X) := U V^\top$ — every singular value mapped to $1$, singular vectors preserved.

**Proposition 4 (clip = polar in the saturating regime).** If $\tau \le \eta\,\sigma_{\min}(c_A)$ — every singular value of $-\eta\, c_A$ exceeds the cap — then

$$
\mathrm{clip}_\tau\!\bigl(-\eta\, c_A\bigr) \;=\; -\tau \cdot \mathrm{polar}(c_A).
$$

*Proof.* When all singular values of $-\eta c_A$ exceed $\tau$, $\mathrm{clip}_\tau$ replaces each by $\tau$ and preserves singular vectors; the result is $\tau\,U V^\top = \tau\,\mathrm{polar}(-\eta c_A)$. Since $\mathrm{polar}$ is invariant under positive scaling and odd under negation, $\mathrm{polar}(-\eta c_A) = -\mathrm{polar}(c_A)$. ∎

Substituting Proposition 4 into the per-block prox (4) at $\tau = \tau^\star_A$ — under the pretense — and using Proposition 3's magnitude rule to fix $\tau^\star_A$ in terms of $\lVert u_A\rVert_F$:

**Theorem 1 (the polar-product update).** Define

$$
\boxed{\quad
\Delta A \;:=\; -\eta \, \frac{\lVert u_A \rVert_F}{\bigl\lVert S_B^{-1/2}\, \mathrm{polar}(c_A)\bigr\rVert_F} \cdot S_B^{-1/2}\, \mathrm{polar}(c_A).
\quad}
\tag{5}
$$

Like Proposition 3, this is a per-iterate statement: at each Picard iterate $n$, the current $\tilde u_A^{(n)}$ determines $c_A^{(n)}$, $\tau^{\star\,(n)}_A$, and the right-hand side of (5). In the saturating regime of Proposition 4 at $\tau = \tau^{\star\,(n)}_A$, equation (5) at iterate $n$ coincides with the exact solver $\Delta A^\star(\tau^{\star\,(n)}_A)$ of the $A$-subproblem (2) with $\Delta B = \Delta B^{(n-1)}$.

*Proof.* Suppose the hypothesis of Proposition 4 holds at threshold $\tau^\star_A$. Then $\mathrm{clip}_{\tau^\star_A}(-\eta\, c_A) = -\tau^\star_A\,\mathrm{polar}(c_A)$, so by Proposition 1,

$$
\Delta A^\star(\tau^\star_A) \;=\; S_B^{-1/2}\,\mathrm{clip}_{\tau^\star_A}(-\eta\, c_A) \;=\; -\tau^\star_A\, S_B^{-1/2}\,\mathrm{polar}(c_A).
$$

Taking Frobenius norms and applying the magnitude rule of Proposition 3,

$$
\eta\,\lVert u_A\rVert_F \;=\; \lVert \Delta A^\star(\tau^\star_A)\rVert_F \;=\; \tau^\star_A\,\bigl\lVert S_B^{-1/2}\,\mathrm{polar}(c_A)\bigr\rVert_F,
$$

which solves to $\tau^\star_A = \eta\,\lVert u_A\rVert_F\,/\,\lVert S_B^{-1/2}\,\mathrm{polar}(c_A)\rVert_F$. Substituting back gives the right-hand side of (5). ∎

*Remark (outside the saturating regime).* Equation (5) defines $\Delta A$ unconditionally. Whenever the saturating-regime hypothesis fails — some $\sigma_i(-\eta\,c_A) < \tau^\star_A$ — the right-hand side of (5) is no longer equal to the exact solver $\Delta A^\star(\tau^\star_A)$ of (1): the clip would have left the small singular directions untouched, while polar flattens them. The whitened update $Y_A = S_B^{1/2}\,\Delta A$ has uniform spectrum by construction (since $\mathrm{polar}(c_A)$ is semi-orthogonal), with Frobenius magnitude $\eta\,\lVert u_A\rVert_F$. So $\Delta A$ is the uniform-spectrum step at Adam's magnitude — the Muon-style prior layered onto the variational target rather than a solution to it.

The $B$-side is symmetric, with $c_B = \tilde u_B\, S_A^{-1/2}$:

$$
\Delta B \;:=\; -\eta\, \frac{\lVert u_B\rVert_F}{\bigl\lVert \mathrm{polar}(c_B)\, S_A^{-1/2}\bigr\rVert_F}\cdot \mathrm{polar}(c_B)\, S_A^{-1/2}.
\tag{5'}
$$

Equations (5) and (5′) are the per-block updates Algorithm 1 implements.

## 8. The polar-product algorithm

The algorithm splits cleanly into two pieces: a **core** that produces a per-block direction $(\widetilde{\mathrm dA}, \widetilde{\mathrm dB})$ from $(u_A, u_B, A, B)$, and a **magnitude rule** that turns that direction into the applied update $(\mathrm dA, \mathrm dB)$. The core comes from the variational program (1) (Lemmas 1–2, Substitution 2) and is shared by all variants. The magnitude rule is the swap point: Substitution 1 (Theorem 1) and Substitution $1'$ (§6.1) plug in here.

### 8.1 The polar map via Newton–Schulz

**Algorithm 0** — $\mathrm{polar}_{\text{NS-}j}(M)$:

1. $X_0 \gets M / \lVert M \rVert_F$.
2. For $i = 0, \ldots, j-1$: $\quad X_{i+1} \gets \tfrac{3}{2} X_i - \tfrac{1}{2} X_i X_i^\top X_i$.
3. Return $X_j$.

Default $j = 5$. The iteration drives every singular value of $X_0$ towards one cubically; five iterations suffice on the matrices arising here.

### 8.2 Core

**Hyperparameters:** Adam $\beta_1, \beta_2, \varepsilon$; Picard count $k$; Newton–Schulz iters $j$; preconditioner regularizer $\delta$.

**Persistent state:** Adam moments $(m_A, v_A, m_B, v_B)$; step counter $t$.

**Algorithm 1 (core)** — one step on layer pair $(A, B)$, up to magnitude rescale:

1. **Adam preconditioning.** Update first and second moments and form bias-corrected directions:
    $$
    m_A \gets \beta_1 m_A + (1-\beta_1) g_A, \qquad v_A \gets \beta_2 v_A + (1-\beta_2) g_A \odot g_A,
    $$
    $$
    u_A \;=\; \frac{m_A / (1-\beta_1^{t})}{\sqrt{v_A / (1-\beta_2^{t})} + \varepsilon},
    $$
    and symmetrically for $B$.

2. **Spectral preconditioners** (LoRA Gram matrices' inverse square roots, both $r \times r$; cached and refreshed periodically):
    $$
    S_A^{-1/2} \;=\; (A A^\top + \delta I)^{-1/2}, \qquad S_B^{-1/2} \;=\; (B^\top B + \delta I)^{-1/2}.
    $$

3. **Picard cross-coupling loop.** Initialize $\mathrm dA = 0$, $\mathrm dB = 0$. For $n = 1, \ldots, k$:
    - **Cross-coupling correction** (Lemma 1; no correction at $n = 1$):
        $$
        \tilde u_A \;=\; u_A + \tfrac{1}{\eta}\, B^\top\, (\mathrm dB)\, A, \qquad \tilde u_B \;=\; u_B + \tfrac{1}{\eta}\, B\, (\mathrm dA)\, A^\top.
        $$
    - **Whiten** (Definition 2):
        $$
        c_A \;=\; S_B^{-1/2}\, \tilde u_A, \qquad c_B \;=\; \tilde u_B\, S_A^{-1/2}.
        $$
    - **Polar prox** (Substitution 2; Algorithm 0 for the polar map):
        $$
        P_A \;=\; \mathrm{polar}_{\text{NS-}j}(c_A), \qquad P_B \;=\; \mathrm{polar}_{\text{NS-}j}(c_B).
        $$
    - **Undo whitening** (produces the per-block direction):
        $$
        \widetilde{\mathrm dA} \;=\; S_B^{-1/2}\, P_A, \qquad \widetilde{\mathrm dB} \;=\; P_B\, S_A^{-1/2}.
        $$
    - **Magnitude rule** (swap point — see §8.3):
        $$
        (\mathrm dA, \mathrm dB) \;\gets\; \mathcal M\bigl(\widetilde{\mathrm dA}, \widetilde{\mathrm dB};\; u_A, u_B, A, B\bigr).
        $$

4. **Apply:** $A \gets A + \mathrm dA$, $\ B \gets B + \mathrm dB$.

### 8.3 Magnitude rule $\mathcal M$

The core in §8.2 produces $(\widetilde{\mathrm dA}, \widetilde{\mathrm dB})$ — un-rescaled per-block directions. The magnitude rule $\mathcal M$ turns these into the applied $(\mathrm dA, \mathrm dB)$. Two choices, one extra hyperparameter (a learning rate $\eta$):

**Variant A — Frobenius rule (Substitution 1; Theorem 1).** Pin Frobenius magnitude to Adam's:
$$
\mathrm dA \;=\; -\eta\, \frac{\lVert u_A\rVert_F}{\lVert \widetilde{\mathrm dA}\rVert_F}\,\widetilde{\mathrm dA},
\qquad
\mathrm dB \;=\; -\eta\, \frac{\lVert u_B\rVert_F}{\lVert \widetilde{\mathrm dB}\rVert_F}\,\widetilde{\mathrm dB}.
$$
Here $\eta$ is a Frobenius-rate scale; typical value $\eta\!\approx\!3{\times}10^{-4}$.

**Variant B — chord-spectral rule (Substitution $1'$; §6.1).** Set per-block operator-norm magnitude to a chord trust-region radius:
$$
\rho \;=\; \frac{\eta}{\sigma_{\max}(A) + \sigma_{\max}(B) + 1},
\qquad
\mathrm dA \;=\; -\rho\, \frac{\widetilde{\mathrm dA}}{\lVert \widetilde{\mathrm dA}\rVert_2},
\qquad
\mathrm dB \;=\; -\rho\, \frac{\widetilde{\mathrm dB}}{\lVert \widetilde{\mathrm dB}\rVert_2}.
$$
Here $\eta$ is a spectral-norm rate (cap on $\lVert \Delta W\rVert_2$); empirically $\sim$10–30$\times$ larger than Variant A's $\eta$. $\sigma_{\max}(A), \sigma_{\max}(B)$ via one power-iteration step each.

The two variants share everything in §8.2 — Adam preconditioning, spectral preconditioners, cross-coupling correction, whiten, polar prox, undo whitening — and differ only in the final magnitude rescale. Switching between them is a one-function swap of $\mathcal M$.

*Why Variant B was introduced.* Variant A is the original polar-product algorithm and remains the default at LoRA ranks where it has been stable. At $r=128$ on the 4k-step sweep, we observed an instability around step $\sim$2200: $\sigma_{\max}(B)$ had grown $\sim10\times$ from initialization while $\lVert B\rVert_F$ barely changed, and a step at the Frobenius-pinned magnitude became too large in operator norm, breaking the Picard contraction. Variant B was introduced to address this — by capping the operator norm of the per-block update directly, it self-attenuates as $\sigma_{\max}(A), \sigma_{\max}(B)$ grow. Early single-seed results suggest Variant B is more stable at $r=128$, but a full sweep is in progress; treat the choice between A and B as open until those results are in.

## 9. Recap: Algorithm 1 line by line

| Algorithm 1 step | Named object | Role | Source | Variant-dependent? |
|---|---|---|---|---|
| Adam preconditioning | $u_A, u_B$ | Linear cost in (1) | §2 | shared |
| Cross-coupling correction | $\tilde u_A = u_A + \tfrac{1}{\eta} B^\top\,\mathrm dB\, A$ | Other factor's contribution to first-order condition | Lemma 1 | shared |
| Whiten | $c_A = S_B^{-1/2}\,\tilde u_A$ | Reduces (2) to (3) in whitened coordinates | Lemma 2 | shared |
| Polar via Newton–Schulz | $P_A = \mathrm{polar}(c_A)$ | Exact in saturating regime, uniform-spectrum prior otherwise | Substitution 2; Prop 4 | shared |
| Undo whitening | $\widetilde{\mathrm dA} = S_B^{-1/2}\,P_A$ | Per-block direction in original coordinates | §4 | shared |
| Magnitude rule $\mathcal M$ | $\mathrm dA = -\eta\,\tfrac{\lVert u_A\rVert_F}{\lVert\widetilde{\mathrm dA}\rVert_F}\,\widetilde{\mathrm dA}$ *or* $\mathrm dA = -\rho\,\widetilde{\mathrm dA}/\lVert\widetilde{\mathrm dA}\rVert_2$ | Sets step magnitude | Substitution 1 or $1'$ | **swap point** |
| Picard outer loop, $k$ iterations | $\Delta A^{(n)}, \Delta B^{(n)}$ | Block-coordinate descent on (1) | Prop 2 | shared |

The headline construction (Theorem 1): the variational program (1) has an exact block-coordinate solver (4) via whitening + clip; substituting clip by polar (Substitution 2) gives the shared core in §8.2, and a magnitude rule (Substitution 1 or $1'$) eliminates the cap threshold $\tau$ as an explicit hyperparameter. The polar substitution is exact when the spectral cap saturates uniformly and a uniform-spectrum prior otherwise.

## Appendix A. Proof of Proposition 3

*Recall.* $f(\tau) := \lVert \Delta A^\star(\tau)\rVert_F$, where $\Delta A^\star(\tau) = S_B^{-1/2}\,\mathrm{clip}_\tau(-\eta\, c_A)$ and $c_A = S_B^{-1/2}\,\tilde u_A$. The claim is that $f$ is continuous on $[0, \infty)$, non-decreasing, strictly increasing on $(0,\,\eta\,\sigma_{\max}(c_A)]$, with $f(0) = 0$ and $f(\infty) = \eta\,\lVert S_B^{-1}\,\tilde u_A\rVert_F$; and that the equation $f(\tau^\star_A) = \eta\,\lVert u_A\rVert_F$ has a unique solution provided $\eta\,\lVert u_A\rVert_F \le f(\infty)$.

*Proof.* Let $-\eta\, c_A = U_M\,\Sigma_M\,V_M^\top$ be a thin SVD with $\sigma_i := \sigma_i(-\eta\, c_A) = \eta\,\sigma_i(c_A)$ and $V_M^\top V_M = I$. Then

$$
\mathrm{clip}_\tau(-\eta\, c_A) \;=\; U_M\,\mathrm{diag}(m(\tau))\,V_M^\top,
\qquad m_i(\tau) \;:=\; \min(\sigma_i,\, \tau).
$$

Each $m_i$ is continuous and non-decreasing, strictly increasing on $(0,\, \sigma_i)$, and constant on $[\sigma_i, \infty)$.

Using $\Delta A^\star(\tau) = S_B^{-1/2}\, U_M\,\mathrm{diag}(m(\tau))\, V_M^\top$ and the Frobenius identity $\lVert S_B^{-1/2}\,X\rVert_F^2 = \mathrm{tr}(X^\top\, S_B^{-1}\, X)$:

$$
f(\tau)^2 \;=\; \mathrm{tr}\!\bigl(V_M\,\mathrm{diag}(m)\,U_M^\top\, S_B^{-1}\,U_M\,\mathrm{diag}(m)\,V_M^\top\bigr)
\;=\; \sum_i m_i(\tau)^2\,Q_{ii},
$$

where $Q := U_M^\top\, S_B^{-1}\, U_M$ and the second equality uses cyclicity of trace and $V_M^\top V_M = I$. Since $S_B^{-1}$ is positive definite and $U_M$ has orthonormal columns, $Q_{ii} = \langle U_M^{(i)},\, S_B^{-1}\, U_M^{(i)}\rangle > 0$ strictly.

Therefore $f(\tau)^2$ is a sum of non-decreasing terms with strictly positive weights — continuous, non-decreasing, and strictly increasing wherever some $m_i$ is, i.e. on $(0,\, \sigma_{\max}]$. The boundary values:

- $f(0) = 0$: every $m_i(0) = 0$.
- $f(\infty) = f(\sigma_{\max})$: at $\tau \ge \sigma_{\max}$, $m_i(\tau) = \sigma_i$, so $f(\infty)^2 = \sum_i \sigma_i^2\,Q_{ii} = \mathrm{tr}((-\eta\,c_A)^\top\, S_B^{-1}\,(-\eta\,c_A)) = \eta^2\,\lVert S_B^{-1}\,\tilde u_A\rVert_F^2$.

Continuity and strict monotonicity of $f$ on $[0,\,\sigma_{\max}]$, plus $\eta\,\lVert u_A\rVert_F \le f(\sigma_{\max}) = f(\infty)$, give a unique $\tau^\star_A$ by the intermediate-value theorem. Admissibility (every $\tau > 0$ defines a non-empty feasible set in (1)) makes $\Delta A^\star(\tau^\star_A)$ the exact minimizer of (2) at that threshold. ∎

## Appendix B. µA-style scaling for the polar-product step

This appendix specializes the µA derivation of Chen, Villar, Hayou (arXiv:2602.06204) to the polar-product update. The conclusion: under their stylized model, **Variant A inherits the same rank-scaling laws as the paper's SignSGD analysis**, while **Variant B's $\eta$ has a different unit (spectral step size, not Frobenius rate) and is rank-invariant under Init[B] in particular**. The σ-drift failure mode that motivates Variant B is invisible to this $\Theta(\cdot)$ feature-scale calculation — it lives in conditioning constants the model does not track.

### B.1 Setup and assumptions

Consider one square LoRA layer of width $n$ and rank $r$, with $r \le n$. Adopt the paper's convention $W = W^\star + \alpha BA$ with effective multiplier $\alpha$. In our experiments $\alpha = 1$ ($\alpha_{\text{PEFT}} = r$ giving $\alpha_{\text{PEFT}}/r = 1$).

The feature update decomposes (Eq. (1) of the paper) as

$$
\Delta Z_B^t \;=\; \underbrace{\alpha B_{t-1}\, \Delta Z_A^t}_{\delta_1} \;+\; \underbrace{\alpha\, \Delta B_t\, Z_A^{t-1}}_{\delta_2} \;+\; \underbrace{\alpha\, \Delta B_t\, \Delta Z_A^t}_{\delta_3},
\qquad Z_A^t = A_t Z, \quad Z_B^t = \alpha B_t Z_A^t.
$$

Stable feature learning means $\Delta Z_B^t = \Theta(1)$ per coordinate over fixed $t$, with each $\delta_k = O(1)$.

**Assumptions** (carried throughout this appendix; flagged because they are not all self-evidently tight):

- **(A1) Adam-as-SignSGD proxy.** The Adam-preconditioned directions $u_A, u_B$ have $O(1)$ entries aligned with the gradient sign. This is what the paper assumes for its calculation; the polar map preserves the property up to $\Theta(\sqrt{rn})$ Frobenius scale.
- **(A2) Rank-one local model.** The single-sample factor gradients $g_A \propto (B^\top d\bar Z) \otimes Z$ and $g_B \propto d\bar Z \otimes Z_A$ are rank-one outer products. Mini-batches and Adam moment averaging give higher rank in practice; the dimension counting still goes through but constants change.
- **(A3) Isotropic whitening.** $S_A^{-1/2}, S_B^{-1/2}$ change constants but not powers of $n, r$. This is the assumption that breaks at the σ-drift failure (§B.4).

### B.2 Per-step entry scale of the polar-product step

For a rank-one $u_A = p q^\top$ with $p \in \mathbb R^r$, $q \in \mathbb R^n$ entries $\Theta(1)$:

$$
\lVert u_A\rVert_F = \Theta(\sqrt{rn}), \qquad \mathrm{polar}(u_A) = (p/\lVert p\rVert)(q/\lVert q\rVert)^\top, \qquad \lVert \mathrm{polar}(u_A)\rVert_F = 1.
$$

**Variant A (Frobenius rule).** $\Delta A = -\eta_F \cdot \lVert u_A\rVert_F \cdot \mathrm{polar}(u_A)/\lVert\widetilde{\mathrm dA}\rVert_F$ — entries of $\Delta A$ are $\Theta(\eta_F)$. Aligned with $Z$, so $\Delta A\, Z = \Theta(\eta_F\, n)$ per coordinate. Symmetrically $\Delta B\, x = \Theta(\eta_F\, r\, \zeta)$ for $x$ with entries of scale $\zeta$. **Same recursions as SignSGD** in the 2602 paper.

**Variant B (chord-spectral rule).** $\Delta A$ has operator norm $\rho = \eta_S/(\sigma_{\max}(A) + \sigma_{\max}(B) + 1)$. Entries are $\Theta(\rho/\sqrt{rn})$, and $\Delta A\, Z = \Theta(\rho\sqrt{n/r})$ per coordinate; $\Delta B\, x = \Theta(\rho\sqrt{r/n}\,\zeta)$. **Different recursion**: $\eta_S$ is a spectral step, not a Frobenius rate.

### B.3 Variant A: rank-scaling laws (recovers 2602)

Let $\beta_t = \Theta(B_t)$ per entry and $\zeta_t = \Theta(Z_A^t)$ per coordinate. Plugging the Variant A scalings into $\delta_1, \delta_2, \delta_3$ with the paper's independence assumption on the multiplicative term:

$$
\delta_1^t = \Theta(\alpha\, \eta_F\, n\, \sqrt r\, \beta_{t-1}), \qquad
\delta_2^t = \Theta(\alpha\, \eta_F\, r\, \zeta_{t-1}), \qquad
\delta_3^t = \Theta(\alpha\, \eta_F^2\, n\, \sqrt r).
$$

**Init[A]** ($A_0 \sim \mathcal N(0, 1/n)$, $B_0 = 0$): $Z_A^0 = \Theta(1)$, $B_t = \Theta(\eta_F)$. Combining,

$$
\Delta Z_B^t = \max\bigl(\Theta(\alpha\, \eta_F\, r),\, \Theta(\alpha\, \eta_F^2\, rn)\bigr).
$$

For $\alpha = r^{-\gamma}$, the largest stable $\eta_F$ is $\Theta(n^{-1/2}\, r^{-(1-\gamma)/2})$. **Our setting** $\alpha = 1$ ($\gamma = 0$):

$$
\boxed{\eta_F^{\text{Init[A]},\,\alpha=1} \;=\; \Theta(n^{-1/2}\, r^{-1/2}).}
$$

**Init[B]** ($B_0 \sim \mathcal N(0, 1/r)$, $A_0 = 0$): $\beta_0 = \Theta(r^{-1/2})$, $Z_A^0 = 0$, $Z_A^t = \Theta(\eta_F\, n)$.

$$
\Delta Z_B^t = \max\bigl(\Theta(\alpha\, \eta_F\, n),\, \Theta(\alpha\, \eta_F^2\, nr)\bigr).
$$

For $\alpha = 1$: $\eta_F = \Theta(n^{-1})$, **rank-invariant** in $r$. With this choice $\delta_1 = \Theta(1)$, $\delta_2 = \Theta(r/n)$, $\delta_3 = \Theta(\sqrt r/n)$, and $Z_A^t = \Theta(1)$. Learning happens primarily through $A$. This is the paper's "transferable" regime and matches FFT.

$$
\boxed{\eta_F^{\text{Init[B]},\,\alpha=1} \;=\; \Theta(n^{-1}), \quad \text{rank-invariant.}}
$$

### B.4 Variant B: spectral-rate scaling

Variant B applies operator-norm $\rho$ rescaling, so the "per-entry $\Theta(\eta)$" property of (A1) is replaced by "per-entry $\Theta(\rho/\sqrt{rn})$".

**Init[B]**, $\alpha = 1$ (the cleanest case). At init, $B_0$ entries $\Theta(r^{-1/2})$ give $\sigma_{\max}(B_0) = \Theta(\sqrt{n/r})$ for $r \ll n$ (Marchenko–Pastur edge). The Variant B denominator gives

$$
\rho \;=\; \Theta\!\bigl(\eta_S\, \sqrt{r/n}\bigr).
$$

Then $\Delta Z_A = \Theta(\rho\sqrt{n/r}) = \Theta(\eta_S)$, and the leading term is $\delta_1 = B_0\, \Delta Z_A = \Theta(\eta_S)$ (the $\sqrt{n/r}$ scale of $B_0$ cancels against $\rho$'s denominator). Lower-order terms $\delta_2 = \Theta(\eta_S^2\, r/n)$, $\delta_3 = \Theta(\eta_S^2\sqrt r/n)$ are negligible.

$$
\boxed{\eta_S^{\text{Init[B]},\,\alpha=1} \;=\; \Theta(1), \quad \text{rank-invariant in $r$ and $n$.}}
$$

This is consistent in *spirit* with the paper's "transferable regime" but different in *unit*: it is a spectral step size, not a Frobenius rate. Numerical $\eta_S$ values do not transfer from the $\eta_F$ tunings; they should be re-tuned. The empirical $\sim$10–30× ratio noted in §6.1 is a consequence of this unit shift.

**Init[A]**, $\alpha = 1$: the rank-one model is awkward at step 1 ($B_0 = 0$), but for $t$ large enough that the random $A_0 Z$ initialization is overtaken by the update, the same dimension counting suggests $\rho = \Theta(1)$, hence $\eta_S = \Theta(1)$ with $\sigma_{\max}(A) = O(1)$. Less precise than the Init[B] result.

### B.5 The σ-drift failure is invisible to (A3)

The 2602 calculation tracks only Frobenius/coordinate-level scaling — singular values are absorbed into constants. Our observed Variant A failure at $r = 128$ around step $\sim 2200$ is exactly an (A3) violation: $\lVert B\rVert_F$ stays $\Theta(1)$ while $\sigma_{\max}(B)$ grows $\sim 10\times$. The submultiplicative bound

$$
\lVert \Delta W\rVert_2 \;\le\; \sigma_{\max}(B)\, \lVert \Delta A\rVert_2 + \sigma_{\max}(A)\, \lVert \Delta B\rVert_2 + \lVert \Delta B\rVert_2\, \lVert \Delta A\rVert_2
$$

makes the failure visible: Variant A controls the Frobenius norms on the right, not the operator norms or $\sigma_{\max}$ prefactors. Variant B fixes this by construction (§6.1).

### B.6 Summary

| Variant | Init | $\alpha$ | Optimal $\eta$ scaling | Unit |
|---|---|---|---|---|
| A (Frobenius) | Init[A] | 1 | $\Theta(n^{-1/2}\, r^{-1/2})$ | per-entry / Frobenius rate |
| A (Frobenius) | Init[B] | 1 | $\Theta(n^{-1})$ | per-entry / Frobenius rate |
| B (chord-spectral) | Init[B] | 1 | $\Theta(1)$, rank-invariant | spectral step size |
| B (chord-spectral) | Init[A] | 1 | $\Theta(1)$ (less precise) | spectral step size |

**Caveats.** All scalings are leading-order under (A1)–(A3). The σ-drift failure of §B.5 is a violation of (A3) that the calculation does not see. Empirically, our Init[A] + $\alpha = 1$ + Variant A sweeps show approximately stable optimal $\eta$ across $r \in \{16, \ldots, 256\}$, which is in tension with the predicted $r^{-1/2}$ drift; the rank range may be too narrow to resolve a factor of 4× change in $1/\sqrt r$, or the rank-one stylization (A2) may break in our workload's actual regime. Confirming or refuting the predicted scaling is a planned ablation.

## References

- **LoRA.** E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
- **Adam.** D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv:1412.6980.
- **AdamW.** I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:1711.05101.
- **Muon.** K. Jordan et al. *Muon: An optimizer for hidden layers in neural networks.* 2024. Source of the Newton–Schulz polar iteration in Algorithm 0 and the spectral-cap design philosophy on dense updates.
- **Spectron.** A. Janson et al. *Spectron: a unified spectral framework for low-rank optimizer design.* 2026. arXiv:2602.12429. Submultiplicativity bound used in §6.1, equations (13)–(15).
- **µA (Maximal-Update Adaptation).** N. Chen, S. Villar, S. Hayou. *Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning.* 2026. arXiv:2602.06204. Source of the rank-scaling laws specialized in Appendix B and the Init[B] + $\alpha = 1$ recommendation discussed in §1.
- **Mirsky's theorem.** L. Mirsky. *Symmetric gauge functions and unitarily invariant norms.* Quarterly Journal of Mathematics 11 (1960), 50–59. Closed form for the Frobenius projection onto an operator-norm ball used in Proposition 1.
- **Newton–Schulz polar iteration.** N. J. Higham. *Functions of Matrices: Theory and Computation*, SIAM 2008, Chapter 8. Cubic convergence of the iteration in Algorithm 0.
