# A polar-product LoRA optimizer

This document derives a low-rank adaptation (LoRA) optimizer from a variational program. The goal is exposition: state the program, solve it, and identify each piece of the resulting algorithm with a piece of the program. Every step — whitening, the cross-coupling correction, the Picard outer loop, the polar map, the magnitude radius — is derived; nothing is an empirical patch on top of a simpler algorithm.

The derivation has two complementary halves, both stated in operator-norm geometry:

1. A **direction program** (§§3–7) — a Frobenius coupling on the merged-weight tangent plus per-block-contribution caps on the operator norms of each factor's contribution. Block-coordinate descent on this program derives the cross-coupling correction (Lemma 1) and the whitening change of variable (Lemma 2). The clip prox solves the resulting whitened subproblem exactly (Proposition 1); a saturating-regime substitution (Proposition 2) replaces clip with polar. The Picard outer loop is the BCD iteration on the joint problem.
2. A **magnitude program** (§8) — per-factor operator-norm caps on $\Delta A, \Delta B$ together with a chord trust region on the actual merged-weight change $\Delta W$. Submultiplicativity closes the program: the largest admissible per-factor radius $\rho$ is the unique positive root of $\rho^2 + s\rho = \eta$ where $s = \sigma_{\max}(A) + \sigma_{\max}(B)$.

§2 begins with a **warmup**: the simplest spectral-cap LoRA program — per-factor caps with a linear cost and nothing else — whose closed-form solution is one polar map per factor and is essentially Muon applied independently to each LoRA factor. The warmup makes contact with what is already familiar; §§3–8 are what is gained by adding the Frobenius coupling and the chord trust region.

§9 states the algorithm. §10 returns to the question of whether the two programs of §§3–7 and §8 can be unified into one program, and identifies the structural obstruction.

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

The caps $\tau_A, \tau_B$ are left unspecified here. The §§3–7 derivation will produce one-parameter families of solutions $\Delta A^\star(\tau_A), \Delta B^\star(\tau_B)$, each proportional to a fixed unit direction; §8 will pick $\tau_A, \tau_B$ implicitly via a complementary program that pins the per-factor operator norms of $\Delta A, \Delta B$ on the chord trust region.

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

## 6. Picard outer loop

The two subproblems share state — $\tilde u_A$ depends on $\Delta B$ and vice versa — so block-coordinate descent on the joint problem (1) alternates the solves.

**Algorithm R** (exact reference solver). Given $u_A, u_B, A, B, \tau_A, \tau_B$, and Picard count $k$:

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

Every fixed point is a global minimum of (1); when the iteration contracts, $k \to \infty$ converges to the joint optimum. Picard is *derived*: it is the BCD outer loop on (1), not an additional algorithmic choice.

## 7. Polar substitution: clip $\to$ polar

The clip prox (4) solves (2) exactly for any $\tau_A > 0$, and similarly (4′) for any $\tau_B > 0$. We narrow attention to the **saturating regime**: each cap small enough that every singular value of the corresponding whitened cost exceeds it. There, clip becomes polar.

Write

$$
c_A \;:=\; S_B^{-1/2}\,\tilde u_A, \qquad c_B \;:=\; \tilde u_B\, S_A^{-1/2}
$$

for the whitened costs of the $A$- and $B$-subproblems (cf. (4), (4′)).

**Definition 3 (polar map).** For $X = U \Sigma V^\top$, $\mathrm{polar}(X) := U V^\top$ — every singular value mapped to one, singular vectors preserved.

**Proposition 2.** If $\tau_A \le \eta\,\sigma_{\min}(c_A)$, then $\mathrm{clip}_{\tau_A}(-\eta\, c_A) = -\tau_A\,\mathrm{polar}(c_A)$. Symmetrically for $\tau_B \le \eta\,\sigma_{\min}(c_B)$.

*Proof.* Under the hypothesis, $\mathrm{clip}_{\tau_A}$ flattens every singular value of $-\eta\, c_A$ to $\tau_A$ and preserves singular vectors, giving $\tau_A\, U V^\top$. Polar is invariant under positive scaling and odd under negation. ∎

Substituting into (4) and (4′) gives the saturating-regime exact solution

$$
\boxed{\quad
\Delta A^\star(\tau_A) \;=\; -\tau_A\, D_A, \qquad \Delta B^\star(\tau_B) \;=\; -\tau_B\, D_B,
\quad}
\tag{5}
$$

with directions

$$
D_A \;:=\; S_B^{-1/2}\,\mathrm{polar}(c_A), \qquad D_B \;:=\; \mathrm{polar}(c_B)\, S_A^{-1/2}.
$$

**Reading (5) geometrically.** The solutions $\{\Delta A^\star(\tau_A) : \tau_A > 0\}$ and $\{\Delta B^\star(\tau_B) : \tau_B > 0\}$ trace **rays** in directions $-D_A, -D_B$. The caps $\tau_A, \tau_B$ are positions along the rays; the directions $D_A, D_B$ are independent of them. So §§3–7 have produced

- the **directions** $D_A, D_B$ (fully determined by the program), and
- a **family of magnitudes** parameterized by $\tau_A, \tau_B$ (the program leaves these open).

§8 will pick a point on the ray.

The polar map is computed via Newton–Schulz iteration: $X_0 = M / \lVert M\rVert_F$, $X_{i+1} = \tfrac{3}{2} X_i - \tfrac{1}{2} X_i X_i^\top X_i$. The iteration drives every singular value of $X_0$ toward one cubically; a small fixed number of iterations suffice on matrices of LoRA size.

## 8. The magnitude program: tight chord

§7 produced a ray $\Delta A^\star(\tau) = -\tau\, D_A$ — direction fixed, magnitude free in $\tau$. We still need to pick where on the ray to land. The natural quantity to control is the **chord** $\Delta W$ — the actual change in the merged weight, including the bilinear term $\Delta B\,\Delta A$ — since this is what the loss sees.

**Magnitude program (T).** Set per-factor operator-norm caps and demand that the chord respect a trust region:

$$
\boxed{\quad
\lVert \Delta A\rVert_2 \le \rho, \quad \lVert \Delta B\rVert_2 \le \rho, \quad \lVert \Delta W\rVert_2 \le \eta.
\quad}
\tag{6}
$$

The hyperparameter $\eta$ is a per-step cap on the operator norm of the merged-weight change — a spectral step size.

**Proposition 3 (tight-chord radius).** The largest $\rho \ge 0$ satisfying (6), given $\sigma_A := \sigma_{\max}(A)$ and $\sigma_B := \sigma_{\max}(B)$, is

$$
\rho \;=\; \frac{-s + \sqrt{s^2 + 4\eta}}{2},
\qquad s := \sigma_A + \sigma_B.
\tag{7}
$$

*Proof.* By submultiplicativity,

$$
\lVert\Delta W\rVert_2 \;=\; \lVert B\Delta A + \Delta B A + \Delta B\,\Delta A\rVert_2
\;\le\; \sigma_B\,\lVert\Delta A\rVert_2 + \sigma_A\,\lVert\Delta B\rVert_2 + \lVert\Delta B\rVert_2\,\lVert\Delta A\rVert_2.
$$

With $\lVert\Delta A\rVert_2 = \lVert\Delta B\rVert_2 = \rho$ this simplifies to $s\rho + \rho^2$. Set this equal to $\eta$ and solve the quadratic $\rho^2 + s\rho - \eta = 0$ for the positive root. ∎

**Picking a point on the rays.** §7 produced 1-parameter families $\Delta A^\star(\tau_A) = -\tau_A\, D_A$ and $\Delta B^\star(\tau_B) = -\tau_B\, D_B$ of saturating-regime solutions. Demand the per-factor caps of (T): $\lVert \Delta A\rVert_2 = \lVert \Delta B\rVert_2 = \rho$. Since $\lVert \Delta A^\star(\tau_A)\rVert_2 = \tau_A\,\lVert D_A\rVert_2$ and similarly for $B$, this fixes

$$
\tau_A(\rho) \;=\; \frac{\rho}{\lVert D_A\rVert_2}, \qquad \tau_B(\rho) \;=\; \frac{\rho}{\lVert D_B\rVert_2}.
$$

**$\tau_A, \tau_B$ are functions of $\rho$.** The §3 caps were left free; §8 closes the program by setting $\tau_A = \tau_A(\rho), \tau_B = \tau_B(\rho)$. The applied update is

$$
\boxed{\quad
\mathrm dA \;=\; -\rho\,\frac{D_A}{\lVert D_A\rVert_2},
\qquad
\mathrm dB \;=\; -\rho\,\frac{D_B}{\lVert D_B\rVert_2}.
\quad}
\tag{8}
$$

This guarantees $\lVert\mathrm dA\rVert_2 = \lVert\mathrm dB\rVert_2 = \rho$ exactly, hence $\lVert\Delta W\rVert_2 \le \eta$ by submultiplicativity, with no slack.

**Saturating-regime check.** Prop 2's hypotheses were $\tau_A \le \eta\,\sigma_{\min}(c_A)$ and $\tau_B \le \eta\,\sigma_{\min}(c_B)$. Under $\tau_A = \tau_A(\rho), \tau_B = \tau_B(\rho)$ these become a state-dependent condition on $\rho$:

$$
\rho \;\le\; \eta\,\min\bigl(\sigma_{\min}(c_A)\,\lVert D_A\rVert_2,\ \sigma_{\min}(c_B)\,\lVert D_B\rVert_2\bigr).
$$

When this holds, (8) is the exact saturating-regime solver of program (1) at $(\tau_A(\rho), \tau_B(\rho))$ together with program (T). When it fails, $D_A, D_B$ remain coherent directions (uniform-spectrum prior in whitened coordinates) but are no longer the exact clip-prox solvers — the small singular directions of $c_A, c_B$ that clip would have left untouched are flattened by polar instead.

The hyperparameter $\eta$ has the meaning of a spectral step size: the user's bound on $\lVert\Delta W\rVert_2$ per step.

## 9. The algorithm

**Hyperparameters:** Adam $\beta_1, \beta_2, \varepsilon$; Picard count $k$; Newton–Schulz iters $j$; preconditioner regularizer $\delta$; spectral step size $\eta$.

**Persistent state:** Adam moments $(m_A, v_A, m_B, v_B)$; step counter $t$; warm-started top singular vectors for $A, B$ (for power iteration).

**Algorithm 1.** One step on layer pair $(A, B)$:

1. **Adam preconditioning.** Update first and second moments and form bias-corrected directions $u_A, u_B$ in the standard way.

2. **Spectral preconditioners** (refreshed periodically; both $r \times r$):
   $$
   S_A^{-1/2} = (A A^\top + \delta I)^{-1/2},
   \qquad
   S_B^{-1/2} = (B^\top B + \delta I)^{-1/2}.
   $$
   The damping $\delta I$ is an implementation detail handling rank-deficient factors; the §5 derivation corresponds to $\delta = 0$.

3. **Top singular values** via warm-started power iteration:
   $$
   \sigma_A \gets \sigma_{\max}(A), \qquad \sigma_B \gets \sigma_{\max}(B).
   $$

4. **Tight-chord radius:**
   $$
   s \gets \sigma_A + \sigma_B, \qquad
   \rho \gets \tfrac{1}{2}\bigl(-s + \sqrt{s^2 + 4\eta}\bigr).
   $$

5. **Picard cross-coupling loop.** Initialize $\mathrm dA = \mathrm dB = 0$. For $n = 1, \ldots, k$, run the three sub-steps below.

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

| Algorithm 1 step | Variational source |
|---|---|
| Adam preconditioning ($u_A, u_B$) | Linear cost in (1) and (T) |
| Spectral preconditioners ($S_A^{-1/2}, S_B^{-1/2}$) | Whitening forced by Lemma 2 |
| Tight-chord radius ($\rho$) | Magnitude program (T), Proposition 3 |
| Cross-coupling correction ($\tilde u_A, \tilde u_B$) | Lemma 1 |
| Directions $D_A, D_B$ (whiten + polar + unwhiten, composed) | Lemma 2 + Prop 1 + Prop 2 (saturating regime); rays of (5) |
| Tight-chord rescale | (8) |
| Picard outer loop | BCD on (1) |

## 10. Toward a single program

Algorithm 1 has three desirable features:

- **(W)** *Whitening* — preconditioning by $S_A^{-1/2}, S_B^{-1/2}$.
- **(C)** *Cross-coupling* — Picard correction via Lemma 1.
- **(Ch)** *Chord control* — magnitude tied to $\lVert\Delta W\rVert_2 \le \eta$.

The derivation uses **two** programs: (1) gives (W) and (C); (T) gives (Ch). Can a **single** program give all three?

### The structural obstruction

Two op-norm constraints are at play, on different objects:

$$
\lVert B\,\Delta A\rVert_2 \;\le\; \cdot \qquad\text{vs.}\qquad \lVert \Delta A\rVert_2 \;\le\; \cdot
$$

Each is needed for a different reason:

- **Whitening (Lemma 2)** turns the constraint $\lVert B\,\Delta A\rVert_2 \le \tau$ in $\Delta A$-coords into $\lVert Y_A\rVert_2 \le \tau$ in whitened coords (clean op-norm ball ⇒ Mirsky's projection ⇒ clip prox). It needs the constraint on $\lVert B\,\Delta A\rVert_2$.
- **Chord submultiplicativity** $\lVert\Delta W\rVert_2 \le \sigma_B \rho + \sigma_A\rho + \rho^2$ (Prop 3 proof) needs bounds on $\lVert\Delta A\rVert_2$ and $\lVert\Delta B\rVert_2$ to control the bilinear term $\lVert\Delta B\,\Delta A\rVert_2 \le \lVert\Delta B\rVert_2\,\lVert\Delta A\rVert_2$. The chord cannot be bounded without per-factor op-norm constraints.

The two constraints are related ($\lVert B\,\Delta A\rVert_2 \le \sigma_{\max}(B)\,\lVert\Delta A\rVert_2$) but not interchangeable. No single op-norm constraint serves both roles.

### Single-program options

You have to give up at least one of (W), (C), (Ch):

| Program | (W) | (C) | (Ch) | What it gives |
|---|---|---|---|---|
| **(1) + tangent t.r.** $\lVert J\rVert_2 \le \eta$ | ✓ | ✓ | ✗ | $\Delta A = -(\eta/2)\,D_A$ |
| **(W)** of §2 (per-factor caps) | ✗ | ✗ | ✓ | $\Delta A = -\rho\,\mathrm{polar}(u_A)$ |
| **(1) + per-factor cap** $\lVert\Delta A\rVert_2 \le \rho$ | partial | ✓ | ✓ | no closed form |

**Tangent trust region.** Add $\lVert J\rVert_2 \le \eta$ to (1). By triangle inequality, $\lVert J\rVert_2 \le \lVert B\,\Delta A\rVert_2 + \lVert\Delta B\,A\rVert_2 \le \tau_A + \tau_B$, so $\tau_A = \tau_B = \eta/2$ enforces it. Substituting into (5) of §7 gives the explicit update

$$
\mathrm dA \;=\; -\tfrac{\eta}{2}\, D_A \;=\; -\tfrac{\eta}{2}\, S_B^{-1/2}\,\mathrm{polar}(c_A),
\qquad
\mathrm dB \;=\; -\tfrac{\eta}{2}\, D_B \;=\; -\tfrac{\eta}{2}\, \mathrm{polar}(c_B)\, S_A^{-1/2}.
$$

The cross-coupling correction (Lemma 1, via Picard) and whitening are unchanged — only the magnitude rule of step 5's last bullet is replaced. Single program, closed form, keeps (W) and (C). But it bounds the *tangent* $J$, not the chord $\Delta W = J + \Delta B\,\Delta A$. The bilinear term $\Delta B\,\Delta A$ is uncontrolled. Worse, the per-factor op-norm $\lVert\mathrm dA\rVert_2 = (\eta/2)\,\lVert D_A\rVert_2$ scales with $\sigma_{\max}(S_B^{-1/2}) \approx 1/\sigma_{\min}(B)$, so when $B$ becomes ill-conditioned the actual update can grow without bound — exactly the failure mode (Ch) was introduced to prevent.

**Program (W).** Drop the Frobenius coupling. Two decoupled subproblems give $\Delta A = -\rho\,\mathrm{polar}(u_A)$ in closed form (Mirsky on a per-factor op-norm ball). $\rho$ pinned by tight-chord submultiplicativity. Single program with (Ch), but no (W) or (C) — this is Muon applied independently to each LoRA factor.

**(1) with per-factor cap.** Replace the per-block-contribution cap by $\lVert\Delta A\rVert_2 \le \rho$ directly. Frobenius coupling still gives Lemmas 1 and 2 — but in whitened coords the constraint becomes $\lVert S_B^{-1/2} Y_A\rVert_2 \le \rho$, which is not an op-norm ball, so the clip prox of Proposition 1 no longer applies. No closed-form direction.

### What we picked

Algorithm 1 prefers (W) + (C) + (Ch), so it accepts two programs. The price is variational compactness; the gain is the chord-aware adaptive magnitude $\rho$. A unifying single program — perhaps a Riemannian metric on the LoRA manifold inducing both geometries simultaneously, or a primal-dual reformulation in $J$-space with a chord constraint and gauge-fixing regularizer — is left open.

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
