# A polar-product LoRA optimizer

This document derives a low-rank adaptation (LoRA) optimizer from a variational program. The goal is exposition: state the program, solve it, and identify each piece of the resulting algorithm with a piece of the program. Every step — whitening, the cross-coupling correction, the block-Jacobi outer loop, the polar map, the magnitude radius — is derived; nothing is an empirical patch on top of a simpler algorithm.

The variational program is a single program in operator-norm geometry: a Frobenius coupling on the merged-weight tangent plus per-block-contribution caps on the operator norms of each factor's contribution. The derivation proceeds in two stages.

1. **Exact clipped solver (§§3–6).** Block-coordinate descent on the program derives the cross-coupling correction (Lemma 1) and the whitening change of variable (Lemma 2). The clip prox solves the resulting whitened subproblem exactly (Proposition 1). Block-Jacobi iteration on the joint problem (§6) gives **Algorithm 1**, an exact solver of the program for any choice of the per-block-contribution caps $\tau_A, \tau_B$.
2. **Saturating-regime simplification (§§7–8).** Under a saturating hypothesis on the block-Jacobi trajectory (§7), clip becomes polar (Proposition 2), and a partial-isometry identity (Lemma 3) collapses the operator norms of the per-block contributions to *state-only* quantities depending on factor spectra alone. This lets $\tau_A, \tau_B$ be pinned a priori to enforce a chord trust region $\lVert\Delta W\rVert_2 \le \eta$ (Proposition 3). The closure produces **Algorithm 2**, the polar variant whose only magnitude hyperparameter is the spectral step size $\eta$. This is what we run.

§2 begins with a **warmup**: the simplest spectral-cap LoRA program — per-factor caps with a linear cost and nothing else — whose closed-form solution is one polar map per factor and is essentially Muon applied independently to each LoRA factor. The warmup makes contact with what is already familiar; §§3–8 are what is gained by adding the Frobenius coupling and the chord trust region.

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

The chord is what the loss sees. The tangent $J$ is its first-order linearization. The bilinear term $\Delta B\,\Delta A$ is $O(\eta^2)$ in step size but its operator norm matters: when factor singular values drift far from initialization, $\sigma_{\max}(\Delta B\,\Delta A)$ can become comparable to $\sigma_{\max}(J)$ and the tangent linearization stops being a useful proxy for $\Delta W$.

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
- *No chord control.* The radii $\rho_A, \rho_B$ are not connected to the merged-weight change. With $\rho_A = \rho_B = \rho$, submultiplicativity gives only $\lVert\Delta W\rVert_2 \le \rho(\sigma_{\max}(A)+\sigma_{\max}(B)) + \rho^2$, which is loose unless $\rho$ is chosen with this bound in mind.

The remainder of this document repairs these three deficiencies in turn. The Frobenius coupling on $J$ (§3) couples the two factors; the per-block-contribution constraints (§3) and Lemma 2 (§5) put the operator-norm cap on the right object; the chord trust region (§8) connects the radii to $\Delta W$.

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

The caps $\tau_A, \tau_B$ are free hyperparameters of program (1). §§4–6 give an exact solver of (1) for any choice of them (Algorithm 1). §§7–8 close the program: under a saturating-regime hypothesis (§7), the chord trust region $\lVert\Delta W\rVert_2 \le \eta$ pins $(\tau_A, \tau_B)$ to state-only functions of $(A, B, \eta)$ (§8), yielding Algorithm 2.

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

Algorithm 1 is an exact solver of (1) at any given $(\tau_A, \tau_B)$. We want to *pin* these caps to a single spectral step size $\eta$ that controls the chord $\lVert\Delta W\rVert_2$. For that closure to work, we need $\lVert\Delta A^\star(\tau_A)\rVert_2$ to be a simple function of $\tau_A$ times something computable from $(A, B)$ alone — free of the block-Jacobi iterate. In general it is not: the clip prox output magnitude depends on the corrected $\tilde u_A^{(n)}$, which depends on $\Delta B^{(n-1)}$. Under a saturating hypothesis on the trajectory, it is.

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

Lemma 3 is the key observation. The iterate-dependence of $D_A^{(n)}, D_B^{(n)}$ — driven by the cross-coupling correction — sits entirely in their *singular vectors* under (H), not in their norms. Consequently, $\lVert\Delta A^\star(\tau_A)\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$ at every iterate, a state-only function of $\tau_A$. This is exactly the property we needed: it lets the chord constraint be folded into program (1) as state-only caps, in §8.

**When (H) fails.** Outside the saturating regime — when some singular direction of $c_A^{(n)}$ falls below $\tau_A/\eta$ — clip is no longer polar on that direction (clip leaves small singular values alone; polar lifts them to one). Lemma 3 then fails: $\lVert D_A^{(n)}\rVert_2$ can be strictly larger than $\lVert S_B^{-1/2}\rVert_2$ because polar inflates the small directions. Algorithm 2 (§9) uses polar unconditionally; in the non-saturating regime it ceases to be the exact solver of (1) and instead applies a uniform-spectrum prior on the whitened cost. The directions remain coherent; the exact single-program identification is what is lost.

The polar map is computed via Newton–Schulz iteration: $X_0 = M / \lVert M\rVert_F$, $X_{i+1} = \tfrac{3}{2} X_i - \tfrac{1}{2} X_i X_i^\top X_i$. The iteration drives every singular value of $X_0$ toward one cubically; a small fixed number of iterations suffice on matrices of LoRA size.

## 8. The tight-chord radius and state-only caps

Lemma 3 gives $\lVert\Delta A^\star(\tau_A)\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$ and $\lVert\Delta B^\star(\tau_B)\rVert_2 = \tau_B\,\lVert S_A^{-1/2}\rVert_2$ — state-only functions of $\tau_A, \tau_B$. We now use this to fold a chord trust region $\lVert\Delta W\rVert_2 \le \eta$ into program (1).

**Submultiplicative bound.** Set the per-factor operator norms equal: $\lVert\Delta A\rVert_2 = \lVert\Delta B\rVert_2 = \rho$. Then

$$
\lVert\Delta W\rVert_2 \;=\; \lVert B\Delta A + \Delta B A + \Delta B\,\Delta A\rVert_2
\;\le\; \sigma_B\,\rho + \sigma_A\,\rho + \rho^2 \;=\; s\rho + \rho^2,
\qquad s := \sigma_{\max}(A) + \sigma_{\max}(B).
$$

**Proposition 3 (tight-chord radius).** The largest $\rho \ge 0$ with $s\rho + \rho^2 \le \eta$ is

$$
\rho \;=\; \frac{-s + \sqrt{s^2 + 4\eta}}{2}.
\tag{6}
$$

*Proof.* Solve the quadratic $\rho^2 + s\rho - \eta = 0$ for the positive root. ∎

The hyperparameter $\eta$ is a per-step cap on the operator norm of the merged-weight change — a spectral step size.

**State-only caps.** Demanding $\lVert\Delta A^\star(\tau_A)\rVert_2 = \lVert\Delta B^\star(\tau_B)\rVert_2 = \rho$ and applying Lemma 3 pins

$$
\boxed{\quad
\tau_A \;=\; \rho\,\lVert S_B^{-1/2}\rVert_2^{-1} \;=\; \rho\,\sqrt{\sigma_{\min}(B)^2 + \delta_B},
\qquad
\tau_B \;=\; \rho\,\sqrt{\sigma_{\min}(A)^2 + \delta_A}.
\quad}
\tag{7}
$$

Both $\rho$ (via Prop 3) and the right-hand sides of (7) depend only on the factor spectra $\sigma_{\max}(A), \sigma_{\max}(B), \sigma_{\min}(A), \sigma_{\min}(B)$ and the hyperparameters $\eta, \delta_A, \delta_B$. They are *iterate-independent*. Hence under (H), program (1) with these state-fixed caps is a **single program in $\eta$** whose exact solver is Algorithm 1 with the clip-to-polar substitution of Proposition 2 — i.e., Algorithm 2 (§9). The chord trust region $\lVert\Delta W\rVert_2 \le \eta$ is automatically saturated at every iterate.

The applied update at outer iteration $n$ is

$$
\boxed{\quad
\mathrm dA^{(n)} \;=\; -\tau_A\, D_A^{(n)} \;=\; -\rho\,\frac{D_A^{(n)}}{\lVert D_A^{(n)}\rVert_2},
\qquad
\mathrm dB^{(n)} \;=\; -\tau_B\, D_B^{(n)} \;=\; -\rho\,\frac{D_B^{(n)}}{\lVert D_B^{(n)}\rVert_2}.
\quad}
\tag{8}
$$

The two equalities in each line are identical under (H) by Lemma 3. The form on the right (explicit normalize-then-scale-by-$\rho$) is what the implementation uses, since it does not require computing $\sigma_{\min}(B), \sigma_{\min}(A)$ separately — the normalization absorbs the state-only norm of $D^{(n)}$, however that norm is computed.

**Sufficient condition for (H).** Combining (7) with the iterate-wise statement of (H):

$$
\rho \;\le\; \eta\,\min\!\Bigl(\sigma_{\min}\!\bigl(c_A^{(n)}\bigr)\,\lVert S_B^{-1/2}\rVert_2,\ \ \sigma_{\min}\!\bigl(c_B^{(n)}\bigr)\,\lVert S_A^{-1/2}\rVert_2\Bigr)
\quad\text{for } n = 1, \ldots, k.
$$

When this holds along the trajectory, Algorithm 2 is the exact solver of program (1) with state-fixed caps (7). When it fails, the discussion at the end of §7 applies — the directions remain coherent but the exact single-program identification is lost.

## 9. Algorithm 2 (the polar variant — what we run)

Algorithm 2 is Algorithm 1 with clip replaced by polar (Prop 2) and with $(\tau_A, \tau_B)$ pinned via Prop 3 + Lemma 3 to state-only functions of $(A, B, \eta)$ — equation (7) of §8. Under hypothesis (H), it is the exact solver of program (1) at those caps. We present it directly in normalize-then-scale form (the right-hand side of (8)), which makes the chord-saturation invariant $\lVert\mathrm dA\rVert_2 = \lVert\mathrm dB\rVert_2 = \rho$ manifest and does not require computing $\sigma_{\min}(A), \sigma_{\min}(B)$ explicitly.

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

4. **Tight-chord radius:**
   $$
   s \gets \sigma_A + \sigma_B, \qquad
   \rho \gets \tfrac{1}{2}\bigl(-s + \sqrt{s^2 + 4\eta}\bigr).
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

   - **Tight-chord rescale** (per (8)).

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
| Tight-chord radius ($\rho$) | Chord constraint $\lVert\Delta W\rVert_2 \le \eta$ + Proposition 3 |
| Cross-coupling correction ($\tilde u_A, \tilde u_B$) | Lemma 1 |
| Directions $D_A^{(n)}, D_B^{(n)}$ (whiten + polar + unwhiten) | Lemma 2 + Prop 1 + Prop 2 under (H); equation (5) |
| Tight-chord rescale | Lemma 3 + state-only caps (7); equation (8) |
| Block-Jacobi outer loop | BCD on (1) — Algorithm 1 |

**Algorithm 1 (reference, not run).** Replace the polar step in 5b with the clip prox of Proposition 1 and drop the rescale in 5c (which is then redundant — the clip prox already returns the correct magnitude at the given $\tau$). Algorithm 1 is the exact single-program solver of (1) at any user-chosen $(\tau_A, \tau_B)$; we run Algorithm 2 because under (H) it coincides with Algorithm 1 at the chord-saturating caps (7), with the polar form avoiding an SVD per inner step.

## Appendix A. Properties of the tight-chord radius

$$
\boxed{\quad \rho \;=\; \frac{-s + \sqrt{s^2 + 4\eta}}{2} \quad}
$$

**Two regimes.**

$$
\rho \;\approx\;
\begin{cases}
\eta / s & \text{if } \eta \ll s^2 \quad \text{(linear regime — bilinear term negligible)} \\
\sqrt{\eta} - s/2 & \text{if } \eta \gg s^2 \quad \text{(square-root regime — bilinear term dominates)}
\end{cases}
$$

**Monotonicity.** $\rho$ increases in $\eta$, decreases in $s$. Larger factor singular values $\Rightarrow$ smaller step. The rule self-attenuates as $A, B$ grow.

**Boundary.** When $\rho = \rho_\star := (-s+\sqrt{s^2+4\eta})/2$, the chord bound holds with equality: $s\rho_\star + \rho_\star^2 = \eta$.

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
