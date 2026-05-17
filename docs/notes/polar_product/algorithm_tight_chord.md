# A polar-product LoRA optimizer

This document derives a low-rank adaptation (LoRA) optimizer from a variational program. The goal is exposition: state the program, solve it, and identify each piece of the resulting algorithm with a piece of the program. Every step — whitening, the cross-coupling correction, the block-Jacobi outer loop, the polar map, the magnitude radius — is derived; nothing is an empirical patch on top of a simpler algorithm.

The variational program is a single program in operator-norm geometry: a Frobenius coupling on the merged-weight tangent plus per-block-contribution caps on the operator norms of each factor's contribution to the tangent. The user-facing magnitude hyperparameter is a single spectral step size $\eta$ — a per-step cap on the operator norm of the **tangent** $J := B\,\Delta A + \Delta B\, A$, the first-order linearization of the merged-weight change $\Delta W$. The constraints inside the program are *not* the user-facing tangent cap: instead, the program's per-block-contribution caps $\tau_A, \tau_B$ are derived from $\eta$ through a **chain of tight implications**,

$$
\underbrace{\lVert J\rVert_2 \le \eta}_{\text{user-facing (tangent)}}
\;\overset{\text{Prop 3}}{\Longleftarrow}\;
\underbrace{\lVert\Delta A\rVert_2, \lVert\Delta B\rVert_2 \le \rho}_{\text{per-factor}}
\;\overset{\text{Lemma 3 under (H)}}{\Longleftarrow}\;
\underbrace{\lVert S_B^{1/2}\Delta A\rVert_2 \le \tau_A,\ \lVert\Delta B\, S_A^{1/2}\rVert_2 \le \tau_B}_{\text{program's caps}}
$$

— each cap is the tightest cap (in its own geometry) that implies the next-higher-level cap. The chain runs:

1. **$\eta \to \rho$ (Proposition 3).** Choose $\rho$ so that the per-factor caps $\lVert\Delta A\rVert_2, \lVert\Delta B\rVert_2 \le \rho$ imply the tangent cap via submultiplicativity: $s\rho \le \eta$ where $s = \sigma_{\max}(A) + \sigma_{\max}(B)$, giving $\rho = \eta / s$.
2. **$\rho \to (\tau_A, \tau_B)$ (Lemma 3 under (H)).** Under a saturating-regime hypothesis (H) on the block-Jacobi trajectory, a partial-isometry identity says the operator norm of each factor update is $\tau_A \cdot \lVert S_B^{-1/2}\rVert_2$ resp. $\tau_B \cdot \lVert S_A^{-1/2}\rVert_2$. Setting these to $\rho$ pins $\tau_A = \rho\sqrt{\sigma_{\min}(B)^2 + \delta_B}$ and $\tau_B = \rho\sqrt{\sigma_{\min}(A)^2 + \delta_A}$.

The chord $\Delta W = J + \Delta B\,\Delta A$ differs from $J$ by the bilinear term, whose operator norm is bounded by $\rho^2 = (\eta/s)^2 = (\eta/s^2)\,\eta$. Empirically $s^2/\eta \gg 1$ throughout training in our setting (≳ 90 at init, ≳ 750 late; see Appendix A), so the chord agrees with the tangent up to a relative correction $\eta/s^2$ that is $\le 1\%$ throughout. Capping $J$ directly is the cleaner derivation; the chord is bounded by $\eta\,(1 + \eta/s^2)$ as a consequence.

The derivation proceeds in two stages.

1. **Exact clipped solver (§§3–6).** Block-coordinate descent on the program derives the cross-coupling correction (Lemma 1) and the whitening change of variable (Lemma 2). The clip prox solves the resulting whitened subproblem exactly (Proposition 1). Block-Jacobi iteration on the joint problem (§6) gives **Algorithm 1**, an exact solver of the program for any choice of the per-block-contribution caps.
2. **Saturating-regime simplification (§§7–8).** Under (H), clip becomes polar (Proposition 2), Lemma 3 makes the per-factor norm a state-only function of $\tau$, and the chain above pins $(\tau_A, \tau_B)$ from $\eta$ alone. This produces **Algorithm 2**, the polar variant — exact block-Jacobi BCD on the program at the chain-pinned caps. The per-factor and tangent caps are auto-satisfied properties of its iterates, not separate constraints. This is what we run.

§2 begins with a **warmup**: the simplest spectral-cap LoRA program — per-factor caps with a linear cost and nothing else — whose closed-form solution is one polar map per factor and is essentially Muon applied independently to each LoRA factor. The warmup makes contact with what is already familiar; §§3–8 are what is gained by adding the Frobenius coupling and the tangent-implied chain.

§9 states Algorithm 2 (what we run) alongside Algorithm 1 (what it simplifies from).

## 1. Setup

Fine-tune a pretrained transformer by adding a low-rank correction to each frozen weight matrix. For a frozen $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the LoRA correction is

$$
\Delta W \;=\; \frac{\alpha}{r}\, B A,
\qquad A \in \mathbb{R}^{r \times d_{\text{in}}},\ \ B \in \mathbb{R}^{d_{\text{out}} \times r},
$$

with $r \ll \min(d_{\text{in}}, d_{\text{out}})$ the LoRA rank and $\alpha$ a fixed scaling constant; absorb $\alpha/r$ into the learning rate. Only $A, B$ are trained, and backpropagation produces factor gradients $g_A, g_B$. Adam preconditioning maps these to bias-corrected directions $u_A, u_B$ in the standard way; the variational derivation below takes $u_A, u_B$ as given.

**Definition 1 (tangent and chord).** A perturbation $(\Delta A, \Delta B)$ produces a *tangent* change to the merged weight

$$
J \;:=\; B\,\Delta A + \Delta B\, A,
$$

and an exact *chord* change

$$
\Delta W \;=\; (B+\Delta B)(A+\Delta A) - BA \;=\; J + \Delta B\,\Delta A.
$$

The chord is what the loss sees. The tangent $J$ is its first-order linearization. The bilinear term $\Delta B\,\Delta A$ is $O(\rho^2)$ in the per-factor radius and is bounded relative to $J$ by $\rho/s$ where $s = \sigma_{\max}(A) + \sigma_{\max}(B)$; we cap $J$ directly via the program (so the bilinear term cannot exceed $\rho^2$ by construction) and verify a-posteriori that $\rho/s = \eta/s^2 \ll 1$ in the relevant operating regime (Appendix A). When that ratio approaches 1 the tangent stops being a faithful proxy for $\Delta W$ and the user-facing semantic would need to be re-examined.

## 2. Warmup: per-factor Muon-style program

The simplest variational program with operator-norm geometry asks for the linear-cost minimizer subject to per-factor operator-norm caps:

$$
\min_{\Delta A, \Delta B}\ \langle u_A, \Delta A\rangle + \langle u_B, \Delta B\rangle
\quad\text{s.t.}\quad
\lVert \Delta A\rVert_2 \le \rho_A, \quad \lVert \Delta B\rVert_2 \le \rho_B.
\tag{W}
$$

The two factors are independent: no coupling term, no shared constraint. Each subproblem is

$$
\min_{X}\ \langle u, X\rangle \quad\text{s.t.}\quad \lVert X\rVert_2 \le \rho.
$$

The closed form follows from a result of Mirsky (1960) on Frobenius projection onto operator-norm balls, which we use repeatedly below.

**Lemma 0 (Mirsky on the operator-norm ball).** Let $M \in \mathbb{R}^{m \times n}$ and $\tau, \rho > 0$. Both optimization problems below admit closed-form minimizers, sharing a common reduction via von Neumann's trace inequality.

*(a) Frobenius projection.*

$$
\min_{X \in \mathbb{R}^{m \times n}}\ \lVert X - M\rVert_F^2 \quad\text{s.t.}\quad \lVert X\rVert_2 \le \tau
$$

has unique minimizer $X^\star = \mathrm{clip}_\tau(M)$, where for $M = U \Sigma V^\top$ (SVD) the **singular-value clip** is $\mathrm{clip}_\tau(M) := U\,\min(\Sigma, \tau)\,V^\top$ (singular values capped at $\tau$, singular vectors preserved).

*(b) Linear cost.*

$$
\min_{X \in \mathbb{R}^{m \times n}}\ \langle u, X\rangle \quad\text{s.t.}\quad \lVert X\rVert_2 \le \rho
$$

attains minimum $-\rho\,\lVert u\rVert_*$ (nuclear norm) at $X^\star = -\rho\,\mathrm{polar}(u)$, where $\mathrm{polar}(U\Sigma V^\top) := U V^\top$. The minimizer is unique on the row and column spaces of $u$; on the kernel directions any singular values in $[0, \rho]$ achieve the minimum.

*Proof.* Both proofs reduce to scalar problems on singular values via the same step. Von Neumann's trace inequality (Mirsky 1960) states

$$
\langle X, A\rangle \;\le\; \sum_i \sigma_i(X)\,\sigma_i(A) \tag{$\ast$}
$$

for any $X, A \in \mathbb{R}^{m \times n}$, with equality iff $X$ and $A$ share singular vectors with matching ordering of singular values.

*(a)* Let $M = U \Sigma V^\top$ be the SVD of $M$. Expand $\lVert X - M\rVert_F^2 = \lVert X\rVert_F^2 - 2\langle X, M\rangle + \lVert M\rVert_F^2$. The middle term is upper-bounded by $(\ast)$ with $A = M$, so for any $X$ with prescribed singular values $\sigma_i(X)$, the objective is minimized by aligning singular vectors with $M$. Hence the minimizer has SVD $X = U\, D\, V^\top$ with $D \succeq 0$ diagonal (sharing $M$'s singular vectors), and the problem reduces to

$$
\min_{D \succeq 0}\ \sum_i (D_{ii} - \sigma_i(M))^2 \quad\text{s.t.}\quad \max_i D_{ii} \le \tau,
$$

uniquely solved by $D_{ii} = \min(\sigma_i(M), \tau)$. Hence $X^\star = \mathrm{clip}_\tau(M)$.

*(b)* Apply $(\ast)$ to $\langle u, -X\rangle \le \sum_i \sigma_i(u)\,\sigma_i(X)$, i.e. $\langle u, X\rangle \ge -\sum_i \sigma_i(u)\,\sigma_i(X)$, with equality iff $-X$ and $u$ share singular vectors. Under $\sigma_i(X) \le \rho$, the right side is minimized by $\sigma_i(X) = \rho$ on every $i$ with $\sigma_i(u) > 0$. Aligning singular vectors with $u$ gives $X^\star = -\rho\,\mathrm{polar}(u)$, with value $-\rho \sum_i \sigma_i(u) = -\rho\,\lVert u\rVert_*$. ∎

Applying part (b) to each factor, program (W) yields

$$
\Delta A^\star \;=\; -\rho_A\,\mathrm{polar}(u_A), \qquad \Delta B^\star \;=\; -\rho_B\,\mathrm{polar}(u_B).
$$

This is **Muon** (Jordan et al. 2024) applied independently to each LoRA factor: Adam direction, polar map, scale by an operator-norm radius. The radii $\rho_A, \rho_B$ are externally specified hyperparameters with no closed-form derivation from (W) itself.

What program (W) **does not** capture:

- *No coupling.* The two factors share an image: any $(\Delta A, \Delta B)$ producing the same tangent $J = B\Delta A + \Delta B A$ produces the same first-order change in loss. Program (W) does not see this — it minimizes a sum of two unrelated linear costs. In particular, an update can have small per-factor cost yet produce a large or wasteful $J$, or vice versa.
- *No whitening.* The constraint $\lVert\Delta A\rVert_2 \le \rho_A$ controls the factor itself, not the merged-weight contribution $B\,\Delta A$. The latter is the geometrically meaningful quantity (it lives in the same space as the loss); $B$ and the constraint live in incompatible coordinates.
- *No tangent control.* The radii $\rho_A, \rho_B$ are not connected to the merged-weight change. With $\rho_A = \rho_B = \rho$, submultiplicativity gives only $\lVert J\rVert_2 \le s\rho$ (and $\lVert\Delta W\rVert_2 \le s\rho + \rho^2$), which is loose unless $\rho$ is chosen with this bound in mind.

The remainder of this document repairs these three deficiencies in turn. The Frobenius coupling on $J$ (§3) couples the two factors; the per-block-contribution constraints (§3) and Lemma 2 (§5) put the operator-norm cap on the right object; the tangent trust region (§8) connects the radii to $J$.

## 3. The direction program

A single optimizer step on a layer pair targets the **per-block operator-norm program**:

$$
\min_{\Delta A,\, \Delta B}\ \langle u_A, \Delta A\rangle + \langle u_B, \Delta B\rangle
\;+\; \frac{1}{2\eta}\,\lVert B\, \Delta A + \Delta B\, A \rVert_F^2
\quad\text{s.t.}\quad
\lVert B\, \Delta A \rVert_2 \le \tau_A, \ \ \lVert \Delta B\, A \rVert_2 \le \tau_B.
\tag{1}
$$

Three pieces:

- **Linear cost.** Same as (W).
- **Frobenius coupling.** $\frac{1}{2\eta}\lVert J\rVert_F^2$ is the only term coupling $\Delta A$ and $\Delta B$. It treats $J$ as the primary object: a step that produces a small first-order change in loss but a large $J$ pays a quadratic penalty.
- **Per-block-contribution caps.** Each factor's contribution to the tangent — $B\Delta A$ and $\Delta B\,A$ — is capped separately in operator norm, with independent caps $\tau_A, \tau_B$. This is the geometrically natural cap: the relevant object is what the merged-weight update receives from each side, not the bare factor.

The caps $\tau_A, \tau_B$ are free hyperparameters of program (1). §§4–6 give an exact solver of (1) for any choice of them (Algorithm 1). §§7–8 derive $(\tau_A, \tau_B)$ from a single user-facing tangent step size $\eta$ (a cap on $\lVert J\rVert_2$) via the chain $\eta \to \rho \to (\tau_A, \tau_B)$ outlined in §1 (Proposition 3 + Lemma 3 under (H)), yielding Algorithm 2.

## 4. Block-coordinate decomposition

The Frobenius coupling is bilinear in $(\Delta A, \Delta B)$, so block-coordinate descent reduces (1) to single-factor subproblems with a corrected linear cost.

**Lemma 1 (cross-coupling correction).** Fix $\Delta B$. The Frobenius coupling in (1) splits as

$$
\lVert B\,\Delta A + \Delta B\, A\rVert_F^2 \;=\; \lVert B\,\Delta A\rVert_F^2 \;+\; 2\langle B\,\Delta A,\, \Delta B\, A\rangle \;+\; \lVert\Delta B\, A\rVert_F^2.
$$

The third term is constant in $\Delta A$. The cross term is linear in $\Delta A$:

$$
2\langle B\,\Delta A,\, \Delta B\, A\rangle \;=\; 2\langle \Delta A,\, B^\top\,\Delta B\, A\rangle,
$$

so it absorbs into the linear cost as a shift. Define the **corrected linear cost**

$$
\tilde u_A \;:=\; u_A \;+\; \tfrac{1}{\eta}\, B^\top\, \Delta B\, A.
$$

Then the $A$-subproblem of (1) is

$$
\min_{\Delta A}\ \langle \tilde{u}_A,\, \Delta A\rangle \;+\; \frac{1}{2\eta}\,\langle \Delta A,\, B^\top B\, \Delta A\rangle
\quad\text{s.t.}\quad \lVert B\,\Delta A\rVert_2 \le \tau_A.
\tag{2}
$$

By symmetry, fixing $\Delta A$ gives the $B$-subproblem with corrected cost $\tilde u_B := u_B + \tfrac{1}{\eta}\, B\, \Delta A\, A^\top$. ∎

The shift $\tilde u_A - u_A = \tfrac{1}{\eta}\, B^\top \Delta B\, A$ is the **cross-coupling correction**: the only place the two factors interact in the per-block subproblem.

## 5. Whitening and the per-block clip prox

The $A$-subproblem (2) has a non-identity quadratic in $B^\top B$ and a constraint on $B\,\Delta A$. A linear change of variable removes both.

**Definition 2 (whitened objects).** Let $S_B := B^\top B$ ($r \times r$, PSD). The **whitened update** and **whitened cost** are

$$
Y \;:=\; S_B^{1/2}\,\Delta A, \qquad c \;:=\; S_B^{-1/2}\,\tilde u_A.
$$

The map $\Delta A \leftrightarrow Y$ is invertible by $\Delta A = S_B^{-1/2} Y$.

**Lemma 2 (whitened subproblem).** In coordinates $(Y, c)$, the $A$-subproblem (2) is

$$
\min_{Y}\ \langle c,\, Y\rangle \;+\; \frac{1}{2\eta}\,\lVert Y\rVert_F^2
\quad\text{s.t.}\quad \lVert Y\rVert_2 \le \tau_A.
\tag{3}
$$

*Proof.* Substitute $\Delta A = S_B^{-1/2} Y$: the linear cost becomes $\langle\tilde u_A, S_B^{-1/2} Y\rangle = \langle c, Y\rangle$; the quadratic becomes $\langle Y, Y\rangle$; and $\lVert B\Delta A\rVert_2 = \lVert U_B \Sigma_B V_B^\top \Delta A\rVert_2 = \lVert\Sigma_B V_B^\top \Delta A\rVert_2 = \lVert S_B^{1/2}\Delta A\rVert_2 = \lVert Y\rVert_2$ (where $B = U_B \Sigma_B V_B^\top$). ∎

The whitening is *forced* by the program: it is the unique linear change of variable that simultaneously diagonalizes the $B^\top B$ quadratic and turns the constraint on $\lVert B\Delta A\rVert_2$ into a constraint on $\lVert Y\rVert_2$.

**Proposition 1 (per-block clip prox).** For any $\tau_A > 0$, the unique minimizer of (3) is $Y^\star(\tau_A) = \mathrm{clip}_{\tau_A}(-\eta\, c)$ where $\mathrm{clip}_\tau(U \Sigma V^\top) := U\,\min(\Sigma, \tau)\,V^\top$ (singular-value clip), and in original coordinates

$$
\Delta A^\star(\tau_A) \;=\; S_B^{-1/2}\,\mathrm{clip}_{\tau_A}\!\bigl(-\eta\, S_B^{-1/2}\,\tilde u_A\bigr).
\tag{4}
$$

*Proof.* Complete the square in (3): $\langle c, Y\rangle + \tfrac{1}{2\eta}\lVert Y\rVert_F^2 = \tfrac{1}{2\eta}\lVert Y - (-\eta\, c)\rVert_F^2 - \tfrac{\eta}{2}\lVert c\rVert_F^2$. The constrained problem is the Frobenius projection of $-\eta\, c$ onto the operator-norm ball of radius $\tau_A$, which by Lemma 0 is the singular-value clip. ∎

The $B$-side is symmetric, with $S_A := A A^\top$:

$$
\Delta B^\star(\tau_B) \;=\; \mathrm{clip}_{\tau_B}\!\bigl(-\eta\, \tilde u_B\, S_A^{-1/2}\bigr)\, S_A^{-1/2}.
\tag{4'}
$$

Equations (4) and (4′) define the per-block prox: from a corrected linear cost and a threshold, produce the optimal factor update.

## 6. Block-Jacobi outer loop: Algorithm 1

The two subproblems share state — $\tilde u_A$ depends on $\Delta B$ and vice versa — so the joint problem (1) is solved by alternating the two clip-prox solves. Both updates at outer iteration $n$ use the previous iterate $(n-1)$ of the *other* block (simultaneous rather than sequential); this is **block-Jacobi**, not block-Gauss-Seidel.

**Algorithm 1** (exact clipped solver). Given $u_A, u_B, A, B, \tau_A, \tau_B$, and block-Jacobi sweep count $k$:

1. $\Delta A^{(0)} = \Delta B^{(0)} = 0$.

2. For $n = 1, \ldots, k$:

   - **Cross-coupling correction.**

     $$
     \tilde u_A^{(n)} = u_A + \tfrac{1}{\eta}\, B^\top\, \Delta B^{(n-1)}\, A,
     \qquad
     \tilde u_B^{(n)} = u_B + \tfrac{1}{\eta}\, B\, \Delta A^{(n-1)}\, A^\top.
     $$

   - **Per-block clip prox** (Proposition 1):

     $$
     \Delta A^{(n)} = S_B^{-1/2}\,\mathrm{clip}_{\tau_A}\!\bigl(-\eta\, S_B^{-1/2}\,\tilde u_A^{(n)}\bigr),
     \qquad
     \Delta B^{(n)} = \mathrm{clip}_{\tau_B}\!\bigl(-\eta\,\tilde u_B^{(n)}\, S_A^{-1/2}\bigr)\, S_A^{-1/2}.
     $$

3. Return $(\Delta A^{(k)}, \Delta B^{(k)})$.

Every fixed point is a global minimum of (1); when the iteration contracts, $k \to \infty$ converges to the joint optimum. The block-Jacobi outer loop is *derived* — it is BCD on (1), not an additional algorithmic choice — and Algorithm 1 is the **exact single-program solver of (1)** at any fixed $(\tau_A, \tau_B)$. What remains, in §§7–8, is to close the program by picking $(\tau_A, \tau_B)$.

## 7. Closing the program: the saturating regime

Algorithm 1 is an exact solver of (1) at any given $(\tau_A, \tau_B)$. We want to *pin* these caps to a single spectral step size $\eta$ that controls the tangent $\lVert J\rVert_2$ (and hence the chord up to a bilinear correction; see Appendix A). For that closure to work, we need $\lVert\Delta A^\star(\tau_A)\rVert_2$ to be a simple function of $\tau_A$ times something computable from $(A, B)$ alone — free of the block-Jacobi iterate. In general it is not: the clip prox output magnitude depends on the corrected $\tilde u_A^{(n)}$, which depends on $\Delta B^{(n-1)}$. Under a saturating hypothesis on the trajectory, it is.

Write

$$
c_A^{(n)} \;:=\; S_B^{-1/2}\,\tilde u_A^{(n)}, \qquad c_B^{(n)} \;:=\; \tilde u_B^{(n)}\, S_A^{-1/2}
$$

for the whitened costs at block-Jacobi iterate $n$.

**Saturating-regime hypothesis (H).** For all $n = 1, \ldots, k$,

$$
\tau_A \;\le\; \eta\,\sigma_{\min}\!\bigl(c_A^{(n)}\bigr), \qquad
\tau_B \;\le\; \eta\,\sigma_{\min}\!\bigl(c_B^{(n)}\bigr).
\tag{H}
$$

This is a hypothesis on the *trajectory* of the block-Jacobi iteration, not on the initial inputs alone — both sides of each inequality move as $n$ advances. We assume (H) holds along the trajectory we care about ($n = 1, \ldots, k$) and derive the simplification it permits; what happens when it fails is discussed at the end of this section.

**Definition 3 (polar map).** For $X = U \Sigma V^\top$, $\mathrm{polar}(X) := U V^\top$ — every singular value mapped to one, singular vectors preserved.

**Proposition 2 (clip $\to$ polar under H).** If (H) holds at iterate $n$, then $\mathrm{clip}_{\tau_A}(-\eta\, c_A^{(n)}) = -\tau_A\,\mathrm{polar}(c_A^{(n)})$. Symmetrically for the $B$-side.

*Proof.* Under (H), $\mathrm{clip}_{\tau_A}$ flattens every singular value of $-\eta\, c_A^{(n)}$ to $\tau_A$ and preserves singular vectors, giving $\tau_A\, U V^\top$. Polar is invariant under positive scaling and odd under negation. ∎

Substituting into the clip-prox expressions of Proposition 1 gives, under (H), the iterate-wise solution

$$
\boxed{\quad
\Delta A^\star(\tau_A) \;=\; -\tau_A\, D_A^{(n)}, \qquad \Delta B^\star(\tau_B) \;=\; -\tau_B\, D_B^{(n)},
\quad}
\tag{5}
$$

with directions

$$
D_A^{(n)} \;:=\; S_B^{-1/2}\,\mathrm{polar}\bigl(c_A^{(n)}\bigr), \qquad D_B^{(n)} \;:=\; \mathrm{polar}\bigl(c_B^{(n)}\bigr)\, S_A^{-1/2}.
$$

The directions still depend on the iterate $n$ via $c_A^{(n)}, c_B^{(n)}$ — that is, on the cross-coupling correction — but the *operator norms* of these directions do not.

**Lemma 3 (factor-norm collapse).** Under (H), the operator norms of $D_A^{(n)}, D_B^{(n)}$ are state-only (independent of the block-Jacobi iterate):

$$
\lVert D_A^{(n)}\rVert_2 \;=\; \lVert S_B^{-1/2}\rVert_2 \;=\; \bigl(\sigma_{\min}(B)^2 + \delta_B\bigr)^{-1/2},
\qquad
\lVert D_B^{(n)}\rVert_2 \;=\; \lVert S_A^{-1/2}\rVert_2 \;=\; \bigl(\sigma_{\min}(A)^2 + \delta_A\bigr)^{-1/2}.
$$

*Proof.* The polar factor $\mathrm{polar}(c_A^{(n)}) \in \mathbb{R}^{r \times d_{\text{in}}}$ has $r \le d_{\text{in}}$ and orthonormal rows, so $\mathrm{polar}(c_A^{(n)})\,\mathrm{polar}(c_A^{(n)})^\top = I_r$. Then

$$
\lVert S_B^{-1/2}\,\mathrm{polar}(c_A^{(n)})\rVert_2^2 \;=\; \sigma_{\max}\!\bigl(S_B^{-1/2}\,\mathrm{polar}\,\mathrm{polar}^\top\,S_B^{-1/2}\bigr) \;=\; \sigma_{\max}(S_B^{-1}) \;=\; \lVert S_B^{-1/2}\rVert_2^2.
$$

Symmetric for the $B$-side: $\mathrm{polar}(c_B^{(n)}) \in \mathbb{R}^{d_{\text{out}} \times r}$ has $r \le d_{\text{out}}$ and orthonormal columns, so $\mathrm{polar}(c_B^{(n)})^\top\,\mathrm{polar}(c_B^{(n)}) = I_r$, and the same calculation gives $\lVert D_B^{(n)}\rVert_2 = \lVert S_A^{-1/2}\rVert_2$. The damped-spectrum forms follow from $S_B = B^\top B + \delta_B I$ and likewise for $S_A$. ∎

Lemma 3 is the key observation. The iterate-dependence of $D_A^{(n)}, D_B^{(n)}$ — driven by the cross-coupling correction — sits entirely in their *singular vectors* under (H), not in their norms. Consequently, $\lVert\Delta A^\star(\tau_A)\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$ at every iterate, a state-only function of $\tau_A$. This is exactly the property we needed: it lets the tangent constraint be folded into program (1) as state-only caps, in §8.

**When (H) fails.** Outside the saturating regime — when some singular direction of $c_A^{(n)}$ falls below $\tau_A/\eta$ — clip is no longer polar on that direction (clip leaves small singular values alone; polar lifts them to one). Lemma 3 then fails: $\lVert D_A^{(n)}\rVert_2$ can be strictly larger than $\lVert S_B^{-1/2}\rVert_2$ because polar inflates the small directions. Algorithm 2 (§9) uses polar unconditionally; in the non-saturating regime it ceases to be the exact solver of (1) and instead applies a uniform-spectrum prior on the whitened cost. The directions remain coherent; the exact single-program identification is what is lost.

The polar map is computed via Newton–Schulz iteration: $X_0 = M / \lVert M\rVert_F$, $X_{i+1} = \tfrac{3}{2} X_i - \tfrac{1}{2} X_i X_i^\top X_i$. The iteration drives every singular value of $X_0$ toward one cubically; a small fixed number of iterations suffice on matrices of LoRA size.

## 8. The chain: $\eta \to \rho \to (\tau_A, \tau_B)$

§§3–6 left the program's caps $(\tau_A, \tau_B)$ as free hyperparameters. §7 showed that under (H), the per-block-contribution norm $\lVert\Delta A^\star(\tau_A)\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$ is a state-only function of $\tau_A$ (Lemma 3), and similarly for the $B$-side. We now use this to derive $(\tau_A, \tau_B)$ from a single user-facing magnitude hyperparameter — the spectral step size $\eta$ — via a chain of tight implications. The chain has two steps.

### Step 1: $\eta \to \rho$ (Proposition 3)

The user-facing constraint is the tangent cap $\lVert J\rVert_2 \le \eta$ — a per-step bound on the operator norm of the first-order change to the merged weight. We derive a per-factor radius $\rho$ such that $\lVert\Delta A\rVert_2 \le \rho$ and $\lVert\Delta B\rVert_2 \le \rho$ implies the tangent cap.

Setting $\lVert\Delta A\rVert_2 = \lVert\Delta B\rVert_2 = \rho$, submultiplicativity gives

$$
\lVert J\rVert_2 \;=\; \lVert B\,\Delta A + \Delta B\, A\rVert_2
\;\le\; \sigma_B\,\rho + \sigma_A\,\rho \;=\; s\rho,
\qquad s := \sigma_{\max}(A) + \sigma_{\max}(B).
$$

**Proposition 3 (tight-tangent radius).** The largest $\rho \ge 0$ with $s\rho \le \eta$ is

$$
\rho \;=\; \frac{\eta}{s}.
\tag{6}
$$

*Proof.* Linear; $s\rho = \eta$ at the boundary. ∎

This is the **tightest** $\rho$ for the implication "per-factor caps $\rho$ $\Rightarrow$ tangent cap $\eta$" via submultiplicativity: at $\rho_\star = \eta/s$ the inequality binds with equality $s\rho_\star = \eta$. Larger $\rho$ would violate the tangent cap on worst-case-aligned $(\Delta A, \Delta B)$.

The chord $\Delta W = J + \Delta B\,\Delta A$ then satisfies $\lVert\Delta W\rVert_2 \le \eta + \rho^2 = \eta\,(1 + \eta/s^2)$, i.e. the bilinear correction is $\eta/s^2$ in relative size. Appendix A reports $\eta/s^2 \le 0.011$ throughout a full r=64 training run; capping $J$ directly is the cleaner program, and the chord is automatically bounded.

### Step 2: $\rho \to (\tau_A, \tau_B)$ (Lemma 3 under (H))

The program's caps are on the per-block-contribution norms $\lVert S_B^{1/2}\Delta A\rVert_2 \le \tau_A$ and $\lVert\Delta B\, S_A^{1/2}\rVert_2 \le \tau_B$, not on the per-factor norms. We derive $(\tau_A, \tau_B)$ such that solving (1) at those caps yields iterates with $\lVert\Delta A\rVert_2 \le \rho$ and $\lVert\Delta B\rVert_2 \le \rho$.

Under (H), Lemma 3 gives $\lVert\Delta A^\star(\tau_A)\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$. Setting this equal to $\rho$ pins

$$
\boxed{\quad
\tau_A \;=\; \rho\,\lVert S_B^{-1/2}\rVert_2^{-1} \;=\; \rho\,\sqrt{\sigma_{\min}(B)^2 + \delta_B},
\qquad
\tau_B \;=\; \rho\,\sqrt{\sigma_{\min}(A)^2 + \delta_A}.
\quad}
\tag{7}
$$

As in Step 1, this is the **tightest** $(\tau_A, \tau_B)$ for the implication "program's caps $\Rightarrow$ per-factor caps $\rho$" — the per-factor norm binds with equality $\lVert\Delta A^\star\rVert_2 = \rho$ along the saturating-regime ray under (H).

### The pinned program

Program (1) with state-fixed caps (7) is what Algorithm 2 solves. The user-facing tangent cap and the intermediate per-factor cap are *consequences* of the chain — properties guaranteed by the iterates Algorithm 2 produces — not constraints inside the program. The applied update at outer iteration $n$ is

$$
\boxed{\quad
\mathrm dA^{(n)} \;=\; -\tau_A\, D_A^{(n)} \;=\; -\rho\,\frac{D_A^{(n)}}{\lVert D_A^{(n)}\rVert_2},
\qquad
\mathrm dB^{(n)} \;=\; -\tau_B\, D_B^{(n)} \;=\; -\rho\,\frac{D_B^{(n)}}{\lVert D_B^{(n)}\rVert_2}.
\quad}
\tag{8}
$$

The two equalities in each line are identical under (H) by Lemma 3. The form on the right (explicit normalize-then-scale-by-$\rho$) is what the implementation uses, since it does not require computing $\sigma_{\min}(B), \sigma_{\min}(A)$ separately — the normalization absorbs the state-only norm of $D^{(n)}$, however that norm is computed. There is no rescaling heuristic: under (H), normalize-then-scale-by-$\rho$ is identically the polar update at the chain-pinned $\tau$.

### Sufficient condition for (H)

Combining (7) with the iterate-wise statement of (H):

$$
\rho \;\le\; \eta\,\min\!\Bigl(\sigma_{\min}\!\bigl(c_A^{(n)}\bigr)\,\lVert S_B^{-1/2}\rVert_2,\ \ \sigma_{\min}\!\bigl(c_B^{(n)}\bigr)\,\lVert S_A^{-1/2}\rVert_2\Bigr)
\quad\text{for } n = 1, \ldots, k.
$$

When this holds along the trajectory, Algorithm 2 is the exact solver of program (1) with state-fixed caps (7), and Lemma 3 makes the per-factor cap auto-binding (Step 2 of the chain is tight). When it fails, the discussion at the end of §7 applies — the directions remain coherent but the chain's tightness is lost on those small-singular modes, and Algorithm 2 ceases to be the exact (1)-solver.

### Simpler variant: direct cap on per-block tangent contributions (no $\rho$)

The chain $\eta \to \rho \to (\tau_A, \tau_B)$ above goes through the per-factor radius $\rho$ in two steps. A more direct route uses the triangle inequality on $J$ together with the caps that are already in program (1):

$$
\lVert J\rVert_2 \;=\; \lVert B\,\Delta A + \Delta B\, A\rVert_2 \;\le\; \lVert B\,\Delta A\rVert_2 + \lVert\Delta B\, A\rVert_2 \;\le\; \tau_A + \tau_B.
$$

Setting $\tau_A + \tau_B \le \eta$ implies the tangent cap directly, with no ρ in between. The symmetric choice is

$$
\tau_A \;=\; \tau_B \;=\; \tau, \qquad \tau \;=\; \eta/2,
$$

equivalently (absorbing the $1/2$ into the user-facing knob) "cap each per-block tangent contribution at a single step size $\tau$." Under (H), Proposition 2 still applies and the update at iterate $n$ is

$$
\mathrm dA^{(n)} \;=\; -\tau\, D_A^{(n)}, \qquad
\mathrm dB^{(n)} \;=\; -\tau\, D_B^{(n)}.
$$

This drops the dependence on $\sigma_{\max}(A), \sigma_{\max}(B)$ entirely — no power iteration, no $\rho$ rescale, no normalize-then-scale step. It is the "naive step size" companion to Algorithm 2: same direction, scalar magnitude $\tau$.

**Differences from the $\rho$-routed form (7):**

1. **Per-factor norms are unbounded by the program.** $\lVert\Delta A\rVert_2 = \tau \cdot \lVert S_B^{-1/2}\rVert_2$ can be large when $B$ is ill-conditioned. Program (1) caps only the per-block tangent contribution; the per-factor cap was an artifact of routing through $\rho$.
2. **Bilinear-term worst case is looser.** $\lVert\Delta B\,\Delta A\rVert_2 \le \tau^2 / \sqrt{(\sigma_{\min}(A)^2 + \delta_A)(\sigma_{\min}(B)^2 + \delta_B)}$, bounded only by the damping $\varepsilon_{\text{rel}}$. With $\varepsilon_{\text{rel}} = 10^{-2}$ and $\tau = \eta/2$, the worst-case factor relative to $\eta$ is $\sim 1/(\varepsilon_{\text{rel}} s^2/4) \cdot \eta = \mathcal{O}(\eta/s^2) \cdot \varepsilon_{\text{rel}}^{-1}$ — i.e. $100\times$ the $\rho$-routed bound in worst-case alignment. Empirically smaller, but the guarantee weakens.
3. **The equipartition $\tau_A = \tau_B$ is a choice, not a derivation.** Any split with $\tau_A + \tau_B \le \eta$ implies the tangent cap; the symmetric split is the simplest and aligns with the symmetric treatment of $A$ and $B$ in program (1), but is not forced.

This is the natural one-knob ablation to the $\rho$-routed pinning: same program, same directions, simpler magnitude. Comparing the two on a controlled sweep tests whether the $\rho$-routed pinning's extra structure (state-dependent $\rho$ via $\sigma_{\max}$, geometrically meaningful per-factor norm) buys anything beyond a constant rescale.

## 9. Algorithm 2 (the polar variant — what we run)

Algorithm 2 is exact block-Jacobi BCD on program (1) at the chain-pinned caps (7) of §8, with the clip-to-polar substitution of Proposition 2. Under hypothesis (H), it is the exact solver of (1) at those caps; the per-factor and tangent caps are auto-satisfied properties of its iterates, not constraints enforced by a separate program. We present it directly in normalize-then-scale form (the right-hand side of (8)), which absorbs the state-only norm of $D^{(n)}$ via normalization rather than computing $\sigma_{\min}(A), \sigma_{\min}(B)$ explicitly.

**Hyperparameters:** Adam $\beta_1, \beta_2, \varepsilon$; block-Jacobi sweep count $k$; Newton–Schulz iters $j$; preconditioner regularizer $\varepsilon_{\text{rel}}$; spectral step size $\eta$.

**Persistent state:** Adam moments $(m_A, v_A, m_B, v_B)$; step counter $t$; warm-started top singular vectors for $A, B$ (for power iteration).

**Algorithm 2.** One step on layer pair $(A, B)$:

1. **Adam preconditioning.** Update first and second moments and form bias-corrected directions $u_A, u_B$ in the standard way.

2. **Spectral preconditioners** (refreshed periodically; both $r \times r$):
   $$
   S_A^{-1/2} = (A A^\top + \delta_A I)^{-1/2},
   \qquad
   S_B^{-1/2} = (B^\top B + \delta_B I)^{-1/2}.
   $$
   The §5 derivation corresponds to $\delta_A = \delta_B = 0$; in practice $A, B$ can be near-singular (especially at init), so the implementation damps each side. We use the **scale-invariant parameterization**
   $$
   \delta_A \;=\; \varepsilon_{\text{rel}}\,\sigma_{\max}(A A^\top), \qquad \delta_B \;=\; \varepsilon_{\text{rel}}\,\sigma_{\max}(B^\top B),
   $$
   with a single dimensionless hyperparameter $\varepsilon_{\text{rel}} \in [0, 1)$. Eigenvalues of $A A^\top$ (resp. $B^\top B$) below $\varepsilon_{\text{rel}} \cdot \sigma_{\max}$ are effectively floored — interpreted as the spectrum fraction below which factor directions are treated as noise rather than signal. The parameterization is invariant under $A \to cA, B \to cB$ and carries the same meaning across LoRA rank $r$ and across training time, neither of which absolute $\delta$ achieves.

3. **Top singular values** via warm-started power iteration:
   $$
   \sigma_A \gets \sigma_{\max}(A), \qquad \sigma_B \gets \sigma_{\max}(B).
   $$

4. **Tight-tangent radius:**
   $$
   s \gets \sigma_A + \sigma_B, \qquad
   \rho \gets \eta / s.
   $$

5. **Block-Jacobi cross-coupling loop.** Initialize $\mathrm dA = \mathrm dB = 0$. For $n = 1, \ldots, k$, run the three sub-steps below.

   - **Cross-coupling correction.**

     $$
     \tilde u_A \;=\; u_A + \tfrac{1}{\eta}\, B^\top\, \mathrm dB\, A,
     \qquad
     \tilde u_B \;=\; u_B + \tfrac{1}{\eta}\, B\, \mathrm dA\, A^\top.
     $$

   - **Direction** (whiten, polar map via Newton–Schulz with $j$ iterations, unwhiten — composed):

     $$
     D_A \;=\; S_B^{-1/2}\,\mathrm{polar}_{\text{NS-}j}\!\bigl(S_B^{-1/2}\,\tilde u_A\bigr),
     \qquad
     D_B \;=\; \mathrm{polar}_{\text{NS-}j}\!\bigl(\tilde u_B\, S_A^{-1/2}\bigr)\, S_A^{-1/2}.
     $$

   - **Tight-tangent rescale** (per (8)).

     $$
     \mathrm dA \;=\; -\rho\,\frac{D_A}{\lVert D_A\rVert_2},
     \qquad
     \mathrm dB \;=\; -\rho\,\frac{D_B}{\lVert D_B\rVert_2}.
     $$

6. **Apply.** $A \gets A + \mathrm dA$, $\quad B \gets B + \mathrm dB$.

The line-by-line correspondence with the variational program:

| Algorithm 2 step | Variational source |
|---|---|
| Adam preconditioning ($u_A, u_B$) | Linear cost in (1) |
| Spectral preconditioners ($S_A^{-1/2}, S_B^{-1/2}$) | Whitening forced by Lemma 2 |
| Tight-tangent radius ($\rho$) | Tangent constraint $\lVert J\rVert_2 \le \eta$ + Proposition 3 |
| Cross-coupling correction ($\tilde u_A, \tilde u_B$) | Lemma 1 |
| Directions $D_A^{(n)}, D_B^{(n)}$ (whiten + polar + unwhiten) | Lemma 2 + Prop 1 + Prop 2 under (H); equation (5) |
| Tight-tangent rescale | Lemma 3 + state-only caps (7); equation (8) |
| Block-Jacobi outer loop | BCD on (1) — Algorithm 1 |

**Algorithm 1 (reference, not run).** Replace the polar step in 5b with the clip prox of Proposition 1 and drop the rescale in 5c (which is then redundant — the clip prox already returns the correct magnitude at the given $\tau$). Algorithm 1 is the exact single-program solver of (1) at any user-chosen $(\tau_A, \tau_B)$; we run Algorithm 2 because under (H) it coincides with Algorithm 1 at the tangent-saturating caps (7), with the polar form avoiding an SVD per inner step.

## 10. Algorithm 2′ — the implemented variant at $k \ge 2$

`magnitude_rule = "spectral_chord_tight"` in `lora_playground/optim.py` coincides with Algorithm 2 at $k = 1$ and differs at $k \ge 2$. The code **pre-normalizes the Adam updates by the operator norm of the whitened direction** before entering the Picard loop, so the polar map sees a whitened direction of unit norm at iter 1. This subsection states the modified algorithm and quantifies its effect on the polar input.

### 10.1. Notation

Let $u_A, u_B$ be the bias-corrected Adam updates on $A, B$; $s := \sigma_{\max}(A) + \sigma_{\max}(B)$; $\rho$ the tight-tangent radius from §8. Define the whitened Adam direction and the whitened cross-coupling at iter $n$:

$$
X_A := S_B^{-1/2}\,u_A, \qquad C_A^{(n)} := S_B^{-1/2}\,B^\top\,\mathrm dB^{(n-1)}\,A.
$$

Symmetrically $X_B := u_B\,S_A^{-1/2}$, $C_B^{(n)} := B\,\mathrm dA^{(n-1)}\,A^\top\,S_A^{-1/2}$.

### 10.2. The two algorithms

**Algorithm 2 (block-Jacobi).** Init $\mathrm dA^{(0)} = \mathrm dB^{(0)} = 0$. For $n = 1, \ldots, k$:

$$
\begin{aligned}
D_A^{(n)} &= S_B^{-1/2}\,\mathrm{polar}\!\bigl(X_A + \tfrac{1}{\eta}\,C_A^{(n)}\bigr), &
\mathrm dA^{(n)} &= -\rho\,D_A^{(n)} / \lVert D_A^{(n)}\rVert_2, \\[2pt]
D_B^{(n)} &= \mathrm{polar}\!\bigl(X_B + \tfrac{1}{\eta}\,C_B^{(n)}\bigr)\,S_A^{-1/2}, &
\mathrm dB^{(n)} &= -\rho\,D_B^{(n)} / \lVert D_B^{(n)}\rVert_2.
\end{aligned}
$$

**Algorithm 2′ (code).** Same as Algorithm 2, with the Adam updates pre-rescaled before the loop:

$$
u_A \;\leftarrow\; u_A / \sigma_{\max}(X_A),
\qquad
u_B \;\leftarrow\; u_B / \sigma_{\max}(X_B).
$$

*Geometric motivation.* Program (1) places its trust region in operator-norm geometry — $\lVert B\,\Delta A\rVert_2 \le \tau_A$, equivalently $\lVert Y\rVert_2 \le \tau_A$ in the whitened variable $Y = S_B^{1/2}\,\Delta A$ (Lemma 2). The linear cost in the whitened frame is $\langle X_A, Y\rangle$ with $X_A = S_B^{-1/2}\,u_A$. The pre-rescale normalizes $X_A$ to unit operator norm, putting the dual input and the primal trust region on the same spectral scale. This corresponds to a modified program with linear cost $\langle X_A/\sigma_{\max}(X_A),\, Y\rangle$; Algorithm 2′ is block-Jacobi on the modified program.

*Implementation note.* The shipped code multiplies the cross-coupling coefficient by an additional factor of $2$, replacing $1/\eta$ with $2/(\rho s) \approx 2/\eta$. This doubling does not change the qualitative analysis below and is treated as an empirical implementation choice.

### 10.3. The polar-input ratio $R^{(n)}$

Both algorithms compute polar on an argument of the form $\beta\,X_A + \gamma\,C_A^{(n)}$ for positive scalars $\beta, \gamma$ that are constant in the Picard iterate $n$. Polar is invariant under positive uniform scaling, so the polar output depends only on the **polar-input ratio**:

$$
R^{(n)} \;:=\; \frac{(\gamma/\beta)\,\lVert C_A^{(n)}\rVert_2}{\sigma_{\max}(X_A)}.
$$

When $R^{(n)} \ll 1$ the cross-coupling has no effect on the polar output and Picard is inert; when $R^{(n)} = \Theta(1)$ the cross-coupling enters the polar output and Picard's iterates move.

The coefficients:

|  | $\beta$ | $\gamma$ | $\gamma/\beta$ |
|---|---|---|---|
| Algorithm 2  | $1$                    | $1/\eta$  | $1/\eta$ |
| Algorithm 2′ | $1/\sigma_{\max}(X_A)$ | $1/\eta$  | $\sigma_{\max}(X_A)/\eta$ |

Substituting:

$$
R_2^{(n)} \;=\; \frac{\lVert C_A^{(n)}\rVert_2}{\eta\,\sigma_{\max}(X_A)},
\qquad
R_{2'}^{(n)} \;=\; \frac{\lVert C_A^{(n)}\rVert_2}{\eta}.
$$

The factor $\sigma_{\max}(X_A)$ cancels in $R_{2'}^{(n)}$. The pre-rescale was chosen precisely to remove this state-dependent factor from the ratio.

### 10.4. Bounds

**Lemma 4 ($R_2$ is suppressed by ill-conditioning of $B$).** Let $w \in \mathbb{R}^{r}$ be a unit right-singular vector of $B$ at $\sigma_{\min}(B)$, and $\alpha := \lVert w^\top u_A\rVert_2$. In the tight-tangent linear regime $\rho s \approx \eta$, with relative damping $\delta_B = \varepsilon_{\text{rel}}\,\sigma_{\max}(B)^2$,

$$
R_2^{(n)} \;\le\; \frac{\sigma_{\min}(B) + \sqrt{\varepsilon_{\text{rel}}}\,\sigma_{\max}(B)}{\alpha}.
$$

*Proof.* $S_B^{-1/2}\,B^\top = \mathrm{polar}(B^\top)$ has unit operator norm, and $\lVert \mathrm dB^{(n-1)}\rVert_2 = \rho$ from the previous rescale, so

$$
\lVert C_A^{(n)}\rVert_2 \;\le\; \rho\,\sigma_{\max}(A),
\qquad
\frac{1}{\eta}\,\lVert C_A^{(n)}\rVert_2 \;\le\; \frac{\sigma_{\max}(A)}{s} \;\le\; 1
$$

via $\rho \approx \eta/s$. For the denominator, $S_B^{-1/2}$ has largest eigenvalue $1/\sqrt{\sigma_{\min}(B)^2 + \delta_B}$ along $w$, so $\sigma_{\max}(X_A) \ge \alpha/\sqrt{\sigma_{\min}(B)^2 + \delta_B}$. Combine, apply $\sqrt{a^2 + b^2} \le a + b$, substitute relative damping. $\square$

Let $\kappa(B) := \sigma_{\max}(B)/\sigma_{\min}(B)$:

- *$B$ ill-conditioned* ($\kappa(B) \gg 1/\sqrt{\varepsilon_{\text{rel}}}$): the bound reduces to $\sqrt{\varepsilon_{\text{rel}}}\,\sigma_{\max}(B)/\alpha$.
- *$B$ well-conditioned* ($\kappa(B) \sim 1$): the bound is $\sigma_{\max}(B)/\alpha$.

LoRA init ($B = 0$, $\kappa(B) = \infty$) and many singular directions during training fall in the ill-conditioned regime.

**Estimating $\sigma_{\max}(B)/\alpha$.** Under bias-corrected Adam, the entries of $u_A$ have magnitudes concentrated near $1$ (sign-of-gradient with small fluctuations). For unit $w \in \mathbb{R}^r$, each entry of the row $w^\top u_A \in \mathbb{R}^{d_{\text{in}}}$ is a weighted sum of $\pm 1$-like values with weights $w$, $\|w\| = 1$; by concentration each entry is $O(1)$, and the row has Euclidean norm

$$
\alpha \;\approx\; \sqrt{d_{\text{in}}}.
$$

Substituting into the ill-conditioned bound:

$$
R_2^{(n)} \;\lesssim\; \sqrt{\frac{\varepsilon_{\text{rel}}}{d_{\text{in}}}}\,\sigma_{\max}(B).
$$

With $\varepsilon_{\text{rel}} = 10^{-2}$ and $d_{\text{in}} = O(10^3)$ (transformer hidden dim), the prefactor is $O(10^{-3})$; the cross-coupling is invisible to polar for any modest $\sigma_{\max}(B)$.

**Lemma 5 ($R_{2'}$ is bounded by a dimensionless constant).** In the tight-tangent linear regime $\rho s \approx \eta$,

$$
R_{2'}^{(n)} \;\le\; \frac{\sigma_{\max}(A)}{s} \;\le\; 1.
$$

*Proof.* By the same numerator bound, $\lVert C_A^{(n)}\rVert_2 \le \rho\,\sigma_{\max}(A) \approx \eta\,\sigma_{\max}(A)/s$. Then $R_{2'}^{(n)} = \lVert C_A^{(n)}\rVert_2/\eta \le \sigma_{\max}(A)/s$, and $\sigma_{\max}(A) \le s$. $\square$

The bound is a dimensionless ratio of $A$'s leading singular value to the combined $A$-plus-$B$ scale. The cross-coupling coefficient $1/\eta$ exactly cancels the $\eta$-scaling in the numerator, leaving a state-only quantity in $[0, 1]$.

A formal lower bound on $R_{2'}^{(n)}$ would require alignment assumptions on $\mathrm dB^{(n-1)}$ relative to $A$'s leading singular direction; in general neither $R_{2'}^{(n)} \to 0$ nor $R_{2'}^{(n)} \to 1$ is generic. The upper bound suffices to claim $R_{2'}^{(n)} = O(1)$ in contrast to Lemma 4's $R_2 \ll 1$ for typical $B$.

### 10.5. Comparison

From the two ratios in §10.3:

$$
\boxed{\quad
\frac{R_{2'}^{(n)}}{R_2^{(n)}} \;=\; \sigma_{\max}(X_A).
\quad}
$$

The lifting factor is exactly $\sigma_{\max}(X_A)$ — the same quantity that appears in Lemma 4's denominator bound and drives $R_2 \ll 1$ in the ill-conditioned regime. Algorithm 2′ absorbs $1/\sigma_{\max}(X_A)$ into the base term's scaling, so the suppression that makes $R_2$ small is exactly the lift that makes $R_{2'}$ order $1$.

At $k = 1$, $C_A^{(1)} = 0$, both algorithms evaluate $\mathrm{polar}(X_A)$, and the trajectories coincide. The divergence appears at $k \ge 2$ and is controlled by $\sigma_{\max}(X_A)$ along the trajectory.

## Appendix A. Properties of the tight-tangent radius and chord-vs-tangent gap

$$
\boxed{\quad \rho \;=\; \eta / s \quad}
$$

**Monotonicity.** $\rho$ increases in $\eta$, decreases in $s = \sigma_{\max}(A) + \sigma_{\max}(B)$. Larger factor singular values $\Rightarrow$ smaller step. The rule self-attenuates as $A, B$ grow.

**Boundary.** When $\rho = \eta/s$, the tangent bound binds with equality: $s\rho = \eta$.

**Chord-vs-tangent gap (the bilinear correction).** At $\rho = \eta/s$ the chord satisfies $\lVert\Delta W\rVert_2 \le \eta + \rho^2 = \eta\,(1 + \eta/s^2)$. The dimensionless quantity $\eta/s^2$ controls when the chord and the tangent diverge:

- $\eta/s^2 \ll 1$: bilinear term negligible; chord $\approx$ tangent and capping $J$ is equivalent to capping $\Delta W$.
- $\eta/s^2 \sim 1$: bilinear and tangent contributions to the chord are comparable; the program's tangent semantic is no longer a faithful proxy for the chord and the derivation in §8 would need to be revisited.

**Quadratic form.** A stricter derivation caps the chord $\lVert\Delta W\rVert_2 \le \eta$ directly via submultiplicativity:

$$
s\rho + \rho^2 \le \eta \quad\Longrightarrow\quad \rho = \tfrac{1}{2}\bigl(-s + \sqrt{s^2 + 4\eta}\bigr).
$$

Limits:

$$
\rho \;\to\; \eta/s \quad (\eta \ll s^2), \qquad \rho \;\to\; \sqrt{\eta} - s/2 \quad (\eta \gg s^2).
$$

The implementation uses the quadratic form throughout.

## Appendix B. Newton–Schulz polar iteration

The polar map is computed iteratively. Given $M$, set $X_0 = M / \lVert M\rVert_F$, then iterate

$$
X_{i+1} \;=\; \tfrac{3}{2} X_i - \tfrac{1}{2} X_i X_i^\top X_i.
$$

If $X_i$ has SVD $X_i = U \Sigma V^\top$, then $X_{i+1} = U\,(\tfrac{3}{2}\Sigma - \tfrac{1}{2}\Sigma^3)\, V^\top$. The polynomial $p(\sigma) = \tfrac{3}{2}\sigma - \tfrac{1}{2}\sigma^3$ has $p(1) = 1$ and $p'(1) = 0$, so the fixed point at $\sigma = 1$ is super-attracting; convergence is cubic in a neighborhood. The Frobenius normalization at $X_0$ ensures all singular values lie in the basin of attraction $(0, \sqrt{3})$. A small fixed number of iterations (typically 5) drives every singular value to within machine precision of one.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Kingma & Ba, *Adam.* arXiv:1412.6980.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization (AdamW).* arXiv:1711.05101.
- Jordan et al., *Muon: An optimizer for hidden layers in neural networks.* 2024. Source of the Newton–Schulz polar iteration and the spectral-cap design philosophy on dense weight updates.
- Mirsky, *Symmetric gauge functions and unitarily invariant norms.* Quart. J. Math. 11 (1960), 50–59. Closed form for the Frobenius projection onto an operator-norm ball used in Proposition 1.
- Higham, *Functions of Matrices: Theory and Computation.* SIAM 2008, Ch. 8. Cubic convergence of the Newton–Schulz iteration.
