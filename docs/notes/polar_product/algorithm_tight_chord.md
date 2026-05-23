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

The derivation proceeds in three stages.

1. **Whitened residual program and LMO (§§3–4).** The local model is the residual objective $\tfrac{1}{2\eta}\lVert J + \eta\,G\rVert_F^2$ under per-block-contribution caps. The whitening change of variable $Y_A := S_B^{1/2}\,\Delta A$, $Y_B := \Delta B\,S_A^{1/2}$ (Definition 2) diagonalizes the operator-norm caps and exposes the whitened gradients $\widetilde G_A, \widetilde G_B$ as duals (Lemma 2). Adam supplies surrogate directions, calibrated to unit operator norm (§4.1). The per-block solver is the linear-minimization oracle (LMO) over the operator-norm ball, which is the polar map (§4.2; Lemma 0).
2. **Frank-Wolfe outer loop (§§5–6).** The whitened residual quadratic couples the two blocks; each Picard iter is a Frank-Wolfe step that linearizes the quadratic at the current iterate and applies the polar LMO to the linearization. **Anchored FW** (§5) drops the self-terms $Y_A^{(n)}, Y_B^{(n)}$ from the linearization; in factor coordinates this recovers the cross-coupling correction (Lemma 1) and matches the production code. **Full-residual FW** (§6) keeps the self-terms — one extra matmul per block per Picard iter; logged as a variant, not currently implemented.
3. **Closure of the program (§§7–8).** Under a saturating-regime hypothesis on the FW trajectory, the per-block-contribution norm $\lVert\mathrm dA^\star(\tau_A)\rVert_2$ is state-only (Lemma 3) and the polar map coincides with the exact clip-prox solver (Proposition 2). The chain $\eta \to \rho \to (\tau_A, \tau_B)$ in §8 then pins the program's caps from a single user-facing spectral step size. This produces **Algorithm 2** (§9, the polar variant — what we run); §10 states **Algorithm 2′**, the implemented variant at $k \ge 2$ that bakes in the Adam calibration of §4.1.

§2 begins with a **warmup**: the simplest spectral-cap LoRA program — per-factor caps with a linear cost and nothing else — whose closed-form solution is the operator-norm LMO applied independently to each LoRA factor, and is essentially Muon. The warmup is the LMO of §4.2 without coupling; §§3–10 are what is gained by adding the Frobenius coupling, the whitening change of variable, and the tangent-implied chain.

§9 states Algorithm 2 (what we run) alongside Algorithm 1 (the exact clip-prox block-Jacobi solver Algorithm 2 specializes from under (H); see the Remark at the end of §7).

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

The remainder of this document repairs these three deficiencies in turn. The Frobenius coupling on $J$ (§3) couples the two factors; the per-block-contribution constraints (§3) and Lemma 2 (§4) put the operator-norm cap on the right object; the tangent trust region (§8) connects the radii to $J$.

## 3. The residual program

A single optimizer step on a layer pair targets the **whitened residual program**. Let $G := \nabla f(W + BA)$ be the dense gradient with respect to the merged weight. A factor perturbation $(\Delta A, \Delta B)$ produces the tangent $J = B\,\Delta A + \Delta B\,A$, and the first-order local model of the loss is

$$
\langle G,\,J\rangle + \frac{1}{2\eta}\,\lVert J\rVert_F^2.
$$

Up to a constant in $(\Delta A, \Delta B)$, this is the **residual form**

$$
\frac{1}{2\eta}\,\lVert J + \eta\,G\rVert_F^2,
$$

which asks for a tangent update $J$ that approximates the dense gradient step $-\eta\,G$ in merged-weight space. Adding per-block-contribution operator-norm caps gives the program:

$$
\min_{\Delta A,\,\Delta B}\ \frac{1}{2\eta}\,\lVert B\,\Delta A + \Delta B\,A + \eta\,G\rVert_F^2
\quad\text{s.t.}\quad
\lVert B\,\Delta A\rVert_2 \le \tau_A,\ \ \lVert\Delta B\,A\rVert_2 \le \tau_B.
\tag{1}
$$

Three pieces:

- **Residual cost.** $\lVert J + \eta\,G\rVert_F^2$ packages the Frobenius coupling and the linear cost in a single squared norm. It treats $J$ as the primary object: the step optimizes how well the tangent approximates the dense gradient step $-\eta\,G$.
- **Per-block-contribution caps.** Each factor's contribution to the tangent — $B\,\Delta A$ and $\Delta B\,A$ — is capped separately in operator norm. This is the geometrically natural cap: the relevant object is what the merged-weight update receives from each side, not the bare factor.
- **Practical replacement of $G$.** The dense gradient $G$ is not what optimizers see directly. Backpropagation produces factor gradients $g_A = B^\top G$, $g_B = G\,A^\top$, and Adam preconditioning yields directions $u_A, u_B$. These are the inputs to the algorithm. §4 connects them to $G$ via the whitening change of variable and the calibration of Adam as a directional preconditioner.

The caps $\tau_A, \tau_B$ are free hyperparameters of program (1). §4 changes variables and identifies the per-block solver as the operator-norm-ball LMO. §§5–6 give the Frank-Wolfe outer loop: the **anchored** variant (§5) is what production runs, and the **full-residual** variant (§6) is a one-line untested alternative. §7 closes the program by deriving the per-block-contribution norm under a saturating-regime hypothesis. §8 derives $(\tau_A, \tau_B)$ from a single user-facing tangent step size $\eta$ via the chain $\eta \to \rho \to (\tau_A, \tau_B)$.

## 4. Whitening and the operator-norm LMO

Program (1)'s constraint is on $\lVert B\,\Delta A\rVert_2$, not on $\lVert\Delta A\rVert_2$, and the quadratic couples the two factors through $\lVert J\rVert_F^2$. A single linear change of variable diagonalizes both.

**Definition 2 (whitened objects).** Let $S_A := AA^\top$ and $S_B := B^\top B$ ($r \times r$ PSD; here assumed rank $r$, with damping deferred to §9). Define the **column-orthonormal projector** of $B$ and the **row-orthonormal projector** of $A$,

$$
U_B \;:=\; B\,S_B^{-1/2} \in \mathbb{R}^{d_{\text{out}} \times r},
\qquad
V_A \;:=\; S_A^{-1/2}\,A \in \mathbb{R}^{r \times d_{\text{in}}},
$$

satisfying $U_B^\top U_B = I_r$ and $V_A V_A^\top = I_r$. The **whitened factor updates** and **whitened gradients** are

$$
Y_A \;:=\; S_B^{1/2}\,\Delta A, \qquad Y_B \;:=\; \Delta B\,S_A^{1/2},
$$

$$
\widetilde G_A \;:=\; S_B^{-1/2}\,g_A \;=\; S_B^{-1/2}\,B^\top G, \qquad \widetilde G_B \;:=\; g_B\,S_A^{-1/2} \;=\; G\,A^\top\,S_A^{-1/2}.
$$

The maps $\Delta A \leftrightarrow Y_A$ and $\Delta B \leftrightarrow Y_B$ are invertible. In these coordinates the tangent factors cleanly:

$$
J \;=\; B\,\Delta A + \Delta B\,A \;=\; U_B\,Y_A + Y_B\,V_A,
$$

and the operator-norm caps become caps on the whitened variables: $\lVert B\,\Delta A\rVert_2 = \lVert U_B\,Y_A\rVert_2 = \lVert Y_A\rVert_2$ (since $U_B$ is an isometry on its column space), and symmetrically $\lVert\Delta B\,A\rVert_2 = \lVert Y_B\rVert_2$.

**Lemma 2 (whitened residual program).** The linear term $\langle G,\,J\rangle$ decomposes as

$$
\langle G,\,J\rangle \;=\; \langle U_B^\top G,\,Y_A\rangle + \langle G\,V_A^\top,\,Y_B\rangle \;=\; \langle \widetilde G_A,\,Y_A\rangle + \langle \widetilde G_B,\,Y_B\rangle.
$$

In whitened coordinates, program (1) reads

$$
\min_{Y_A,\,Y_B}\ \langle \widetilde G_A,\,Y_A\rangle + \langle \widetilde G_B,\,Y_B\rangle + \frac{1}{2\eta}\,\lVert U_B\,Y_A + Y_B\,V_A\rVert_F^2
\quad\text{s.t.}\quad \lVert Y_A\rVert_2 \le \tau_A,\ \lVert Y_B\rVert_2 \le \tau_B.
\tag{2}
$$

*Proof.* $U_B^\top G = S_B^{-1/2}\,B^\top G = \widetilde G_A$ and $G\,V_A^\top = G\,A^\top\,S_A^{-1/2} = \widetilde G_B$. Substitution is immediate. ∎

The whitening is *forced* by the program: it is the unique linear change of variable that simultaneously diagonalizes the operator-norm caps and exhibits the whitened gradients as the dual objects. The cross-coupling correction (Lemma 1, below) and Algorithm 1's clip-prox solver, both obtainable by completing the square on a per-block subproblem of (1), are recovered as corollaries — see §5.3 and the Remark at the end of §7. The cleaner spine is to take program (2) as the working objective and apply Frank-Wolfe iteration directly.

### 4.1 Adam calibration

Optimizers do not have $G$ on hand. They have Adam-preconditioned factor directions $u_A, u_B$ in place of $g_A, g_B$. Substituting gives whitened Adam directions

$$
q_A \;:=\; S_B^{-1/2}\,u_A, \qquad q_B \;:=\; u_B\,S_A^{-1/2}.
$$

Adam is a **directional** preconditioner — its elementwise rescaling reshapes the gradient direction — but its absolute magnitude is not a faithful estimate of $\lVert\widetilde G_A\rVert$. Using $q_A$ as if it were $\widetilde G_A$ would inject a state-dependent scale into the residual quadratic, breaking the geometry. The fix is to **calibrate** to unit operator norm:

$$
Q_A \;:=\; \frac{q_A}{\lVert q_A\rVert_2}
\;=\; \frac{S_B^{-1/2}\,u_A}{\lVert S_B^{-1/2}\,u_A\rVert_2},
\qquad
Q_B \;:=\; \frac{q_B}{\lVert q_B\rVert_2}
\;=\; \frac{u_B\,S_A^{-1/2}}{\lVert u_B\,S_A^{-1/2}\rVert_2}.
$$

With optional per-block strengths $\gamma_A, \gamma_B > 0$ (default $\gamma_A = \gamma_B = 1$), the Adam-surrogate whitened residual program substitutes $\widetilde G_A \rightsquigarrow \gamma_A\,Q_A$ and $\widetilde G_B \rightsquigarrow \gamma_B\,Q_B$ into (2). **This is the program Algorithm 2′ solves**, and the pre-rescale step in `algorithm_clean_implementation.md` §2.5 (dividing $X_A$ by $\sigma_{\max}(X_A)$) is exactly this calibration. The default $\gamma = 1$ puts the dual input and the primal trust region on the same spectral scale.

### 4.2 Operator-norm LMO

The per-block solver is the linear-minimization oracle (LMO) over the operator-norm ball, which is part (b) of Lemma 0 from the warmup §2:

$$
\arg\min_{\lVert Y\rVert_2 \le \tau}\ \langle C,\,Y\rangle \;=\; -\tau\,\mathrm{polar}(C),
\qquad
\min\text{-value} \;=\; -\tau\,\lVert C\rVert_*.
$$

**Definition 3 (polar map).** For $C = U\,\Sigma\,V^\top$, $\mathrm{polar}(C) := U\,V^\top$ — every singular value mapped to one, singular vectors preserved.

This is the only per-block tool the rest of the derivation needs. §5 applies it to a linearization of (2) at the current FW iterate. The clip-prox alternative — Frobenius projection of $-\eta\,C$ onto the same ball — solves a different per-block subproblem and is discussed as a Remark at the end of §7.

## 5. Anchored Frank-Wolfe — the production iteration

The whitened residual quadratic in (2) couples $Y_A$ and $Y_B$ through $\lVert U_B\,Y_A + Y_B\,V_A\rVert_F^2$. Frank-Wolfe iteration handles the coupling by **linearizing the quadratic at the current iterate** and applying the LMO of §4.2 to the resulting linear program. The linearization point is a modeling choice; this section uses the **anchored** linearization, which drops self-terms and produces the production iteration. §6 covers the alternative full linearization.

### 5.1 FW linear costs

Write $\Phi(Y_A, Y_B)$ for the Adam-surrogate whitened residual objective from §4.1:

$$
\Phi(Y_A, Y_B) \;=\; \langle \gamma_A\,Q_A,\,Y_A\rangle + \langle \gamma_B\,Q_B,\,Y_B\rangle + \frac{1}{2\eta}\,\lVert U_B\,Y_A + Y_B\,V_A\rVert_F^2.
$$

Its block gradients, using $U_B^\top U_B = I_r$ and $V_A V_A^\top = I_r$, are

$$
\nabla_{Y_A}\Phi \;=\; \gamma_A\,Q_A + \tfrac{1}{\eta}\bigl(Y_A + U_B^\top\,Y_B\,V_A\bigr),
\qquad
\nabla_{Y_B}\Phi \;=\; \gamma_B\,Q_B + \tfrac{1}{\eta}\bigl(U_B\,Y_A\,V_A^\top + Y_B\bigr).
$$

### 5.2 Anchored linearization

Define the **anchored** Frank-Wolfe linearization: at iterate $(Y_A^{(n)}, Y_B^{(n)})$, linearize the $A$-block at $(0,\,Y_B^{(n)})$ — i.e. anchor the linearized variable at zero while keeping the other block at its current value — and symmetrically linearize the $B$-block at $(Y_A^{(n)},\,0)$. This drops the self-terms $Y_A^{(n)}, Y_B^{(n)}$ from the gradients and gives the per-block linear costs

$$
C_A^{(n)} \;:=\; \gamma_A\,Q_A + \tfrac{1}{\eta}\,U_B^\top\,Y_B^{(n)}\,V_A,
\qquad
C_B^{(n)} \;:=\; \gamma_B\,Q_B + \tfrac{1}{\eta}\,U_B\,Y_A^{(n)}\,V_A^\top.
$$

Applying the LMO of §4.2 with full Frank-Wolfe step ($\alpha_n = 1$):

$$
Y_A^{(n+1)} \;=\; -\tau_A\,\mathrm{polar}\bigl(C_A^{(n)}\bigr),
\qquad
Y_B^{(n+1)} \;=\; -\tau_B\,\mathrm{polar}\bigl(C_B^{(n)}\bigr).
\tag{3}
$$

Initialization $Y_A^{(0)} = Y_B^{(0)} = 0$ makes $C_A^{(0)} = \gamma_A\,Q_A$ and $C_B^{(0)} = \gamma_B\,Q_B$, so the $n = 0$ vertex is the polar of the (calibrated) Adam direction — exactly the per-block Muon update of the warmup (§2), now derived from the joint program.

### 5.3 Recovery of original variables — Lemma 1 falls out

Translating back via $\Delta A = S_B^{-1/2}\,Y_A$ and $\Delta B = Y_B\,S_A^{-1/2}$:

$$
\mathrm dA^{(n+1)} \;=\; -\tau_A\,\underbrace{S_B^{-1/2}\,\mathrm{polar}(C_A^{(n)})}_{=:\, D_A^{(n)}},
\qquad
\mathrm dB^{(n+1)} \;=\; -\tau_B\,\underbrace{\mathrm{polar}(C_B^{(n)})\,S_A^{-1/2}}_{=:\, D_B^{(n)}}.
$$

The polar argument $C_A^{(n)}$ can be re-expressed in factor coordinates by factoring $S_B^{-1/2}$ out:

$$
C_A^{(n)}
\;=\; \gamma_A\,\frac{S_B^{-1/2}\,u_A}{\lVert S_B^{-1/2}\,u_A\rVert_2}
\;+\; \tfrac{1}{\eta}\,S_B^{-1/2}\,B^\top\,\mathrm dB^{(n)}\,A
\;=\; S_B^{-1/2}\,\bigl[\,\hat u_A \;+\; \tfrac{1}{\eta}\,B^\top\,\mathrm dB^{(n)}\,A\,\bigr],
$$

where $\hat u_A := \gamma_A\,u_A / \lVert S_B^{-1/2}\,u_A\rVert_2$ is the calibrated Adam direction in factor coordinates. Symmetrically,

$$
C_B^{(n)} \;=\; \bigl[\,\hat u_B \;+\; \tfrac{1}{\eta}\,B\,\mathrm dA^{(n)}\,A^\top\,\bigr]\,S_A^{-1/2}.
$$

**Lemma 1 (cross-coupling correction, FW reading).** The anchored FW linear cost in factor coordinates is exactly the corrected linear cost obtained by completing the square on the $A$-subproblem of (1) after fixing $\Delta B$:

$$
\tilde u_A \;=\; u_A \;+\; \tfrac{1}{\eta}\,B^\top\,\mathrm dB^{(n)}\,A,
$$

and symmetrically $\tilde u_B = u_B + \tfrac{1}{\eta}\,B\,\mathrm dA^{(n)}\,A^\top$. (The full $\hat u_A$ here includes the $\gamma_A$ strength and the Adam calibration; setting $\gamma_A = 1$ and substituting $u_A$ for $\hat u_A$ recovers the bare $\tilde u_A$.) The cross-coupling correction has two equivalent derivations: completing the square on the block-coordinate quadratic, or anchored-FW linearization of the joint program. They give the same linear cost. ∎

The recovered direction $D_A^{(n)} = S_B^{-1/2}\,\mathrm{polar}\bigl(S_B^{-1/2}\,\tilde u_A\bigr) = S_B^{-1/2}\,\mathrm{polar}\bigl(C_A^{(n)}\bigr)$ is exactly what Algorithm 2 (§9) and Algorithm 2′ (§10) use, and is what `_chord_tight_clean_polar_pipeline` (`lora_playground/optim.py:3408`) computes; see `algorithm_clean_implementation.md` §2 for the implementation walkthrough.

## 6. Full-residual Frank-Wolfe (variant, not currently implemented)

The **full** Frank-Wolfe linearization at the current iterate $(Y_A^{(n)}, Y_B^{(n)})$ — without anchoring — uses the full block gradients of §5.1, retaining the self-terms:

$$
C_A^{(n),\,\text{full}} \;:=\; \gamma_A\,Q_A + \tfrac{1}{\eta}\bigl(Y_A^{(n)} + U_B^\top\,Y_B^{(n)}\,V_A\bigr),
\qquad
C_B^{(n),\,\text{full}} \;:=\; \gamma_B\,Q_B + \tfrac{1}{\eta}\bigl(U_B\,Y_A^{(n)}\,V_A^\top + Y_B^{(n)}\bigr).
$$

The FW vertex is $Y_A^{(n+1)} = -\tau_A\,\mathrm{polar}(C_A^{(n),\,\text{full}})$ as before. Factoring $S_B^{-1/2}$ from the $A$-side — using $Y_A^{(n)} = S_B^{1/2}\,\mathrm dA^{(n)}$, so $S_B^{-1/2}\cdot S_B\,\mathrm dA^{(n)} = S_B^{1/2}\,\mathrm dA^{(n)} = Y_A^{(n)}$ — gives the polar input in mixed factor/whitened form

$$
C_A^{(n),\,\text{full}} \;=\; S_B^{-1/2}\,\bigl[\,\hat u_A \;+\; \tfrac{1}{\eta}\,\bigl(S_B\,\mathrm dA^{(n)} \;+\; B^\top\,\mathrm dB^{(n)}\,A\bigr)\,\bigr],
$$

and symmetrically $C_B^{(n),\,\text{full}} = \bigl[\,\hat u_B + \tfrac{1}{\eta}\bigl(B\,\mathrm dA^{(n)}\,A^\top + \mathrm dB^{(n)}\,S_A\bigr)\,\bigr]\,S_A^{-1/2}$.

The only difference from the anchored form (§5.3) is the presence of the **self-terms** $S_B\,\mathrm dA^{(n)}$ and $\mathrm dB^{(n)}\,S_A$ inside the brackets. Geometrically, anchored FW models each block's polar input as if the previous iterate of that block contributed nothing to the residual; full FW retains the contribution.

**Cost.** Two additional $(r{\times}r)\cdot(r{\times}d)$ matmuls per Picard iter — well under $10\%$ of the per-step optimizer FLOP budget at production shapes (see `algorithm_clean_implementation.md` §3).

**Status.** Untested. The anchored variant matches the production code; the full variant has not been implemented or swept. Logged here as a one-line specification, not a proposed sweep.

## 7. Saturating regime: per-block-contribution norms are state-only

Algorithm 2 (§9) is the anchored FW iteration of §5 at the chain-pinned caps (7) of §8. To close the chain we need $\lVert\mathrm dA^{(n+1)}\rVert_2$ — the per-factor operator norm of the FW vertex — to be a simple, state-only function of $\tau_A$ (independent of the FW iterate $n$). From (3) and the definition of $D_A^{(n)}$, $\mathrm dA^{(n+1)} = -\tau_A\,D_A^{(n)}$ with $D_A^{(n)} = S_B^{-1/2}\,\mathrm{polar}(C_A^{(n)})$; the polar factor is a partial isometry, and this is enough to make its operator norm a state-only quantity.

**Lemma 3 (factor-norm collapse).** For any FW iterate $n$, the operator norms of $D_A^{(n)}, D_B^{(n)}$ are state-only — independent of the FW iterate, and in particular independent of $C_A^{(n)}, C_B^{(n)}$:

$$
\lVert D_A^{(n)}\rVert_2 \;=\; \lVert S_B^{-1/2}\rVert_2 \;=\; \bigl(\sigma_{\min}(B)^2 + \delta_B\bigr)^{-1/2},
\qquad
\lVert D_B^{(n)}\rVert_2 \;=\; \lVert S_A^{-1/2}\rVert_2 \;=\; \bigl(\sigma_{\min}(A)^2 + \delta_A\bigr)^{-1/2}.
$$

*Proof.* The polar factor $\mathrm{polar}(C_A^{(n)}) \in \mathbb{R}^{r \times d_{\text{in}}}$ has $r \le d_{\text{in}}$ and orthonormal rows, so $\mathrm{polar}(C_A^{(n)})\,\mathrm{polar}(C_A^{(n)})^\top = I_r$. Then

$$
\lVert S_B^{-1/2}\,\mathrm{polar}(C_A^{(n)})\rVert_2^2 \;=\; \sigma_{\max}\!\bigl(S_B^{-1/2}\,\mathrm{polar}\,\mathrm{polar}^\top\,S_B^{-1/2}\bigr) \;=\; \sigma_{\max}(S_B^{-1}) \;=\; \lVert S_B^{-1/2}\rVert_2^2.
$$

Symmetric for the $B$-side: $\mathrm{polar}(C_B^{(n)}) \in \mathbb{R}^{d_{\text{out}} \times r}$ has $r \le d_{\text{out}}$ and orthonormal columns, so $\mathrm{polar}(C_B^{(n)})^\top\,\mathrm{polar}(C_B^{(n)}) = I_r$, and the same calculation gives $\lVert D_B^{(n)}\rVert_2 = \lVert S_A^{-1/2}\rVert_2$. The damped-spectrum forms follow from $S_B = B^\top B + \delta_B I$ and likewise for $S_A$. ∎

Lemma 3 is the key observation. The FW iterate-dependence of $D_A^{(n)}, D_B^{(n)}$ — driven by the cross-coupling — sits entirely in their *singular vectors*, not in their norms. Consequently $\lVert\mathrm dA^{(n+1)}\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$ at every iterate, a state-only function of $\tau_A$. This is exactly the property needed in §8 to fold the tangent constraint into program (1) as state-only caps.

**Saturating-regime hypothesis (H).** Writing $c_A^{(n)} := S_B^{-1/2}\,\tilde u_A^{(n)}$ for the anchored polar input in factor coordinates (so that $C_A^{(n)} = c_A^{(n)}$ when $\gamma_A = 1$ and $\hat u_A = u_A$, i.e. before Adam calibration), the saturating-regime hypothesis is

$$
\tau_A \;\le\; \eta\,\sigma_{\min}\!\bigl(c_A^{(n)}\bigr), \qquad
\tau_B \;\le\; \eta\,\sigma_{\min}\!\bigl(c_B^{(n)}\bigr)
\qquad\text{for } n = 1,\ldots,k.
\tag{H}
$$

This is a hypothesis on the FW trajectory, not on the initial inputs alone — both sides of each inequality move as $n$ advances. Under (H), every singular direction of $c_A^{(n)}$ has $\sigma > \tau_A/\eta$, and the polar map coincides with the clip-prox alternative discussed in the Remark below.

**Proposition 2 (clip $\to$ polar under H).** If (H) holds at iterate $n$, then $\mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A^{(n)}\bigr) = -\tau_A\,\mathrm{polar}\bigl(c_A^{(n)}\bigr)$, where $\mathrm{clip}_\tau(U\Sigma V^\top) := U\,\min(\Sigma, \tau)\,V^\top$. Symmetrically for the $B$-side.

*Proof.* Under (H), $\mathrm{clip}_{\tau_A}$ flattens every singular value of $-\eta\,c_A^{(n)}$ to $\tau_A$ and preserves singular vectors, giving $\tau_A\,U V^\top$. Polar is invariant under positive scaling and odd under negation. ∎

**When (H) fails.** Outside the saturating regime — when some singular direction of $c_A^{(n)}$ falls below $\tau_A/\eta$ — clip and polar diverge on that direction (clip leaves small singular values alone; polar lifts them to one). Algorithm 2 uses polar unconditionally; in the non-saturating regime its update is still well-defined and the partial-isometry guarantee of Lemma 3 still holds (so the per-factor norm is still state-only), but its identification with the exact clip-prox solver of (1) — Algorithm 1, in the Remark below — is lost. The directions remain coherent; what is lost is exactness as an (1)-solver, replaced by a uniform-spectrum prior on the whitened cost.

**Remark (Algorithm 1, the exact clip-prox solver — not run).** Frank-Wolfe is one way to solve the whitened residual program (2); another is **block-coordinate descent** with the per-block subproblem solved exactly. The per-block subproblem of (2), fixing the other block, has the form

$$
\min_{Y_A}\ \langle c_A^{(n)},\,Y_A\rangle + \tfrac{1}{2\eta}\,\lVert Y_A\rVert_F^2
\quad\text{s.t.}\quad \lVert Y_A\rVert_2 \le \tau_A.
$$

Completing the square gives $\tfrac{1}{2\eta}\lVert Y_A - (-\eta\,c_A^{(n)})\rVert_F^2$ up to a constant, so this is the Frobenius projection of $-\eta\,c_A^{(n)}$ onto the operator-norm ball of radius $\tau_A$, whose solution is the singular-value **clip** (Lemma 0(a), Mirsky 1960):

$$
Y_A^\star(\tau_A) \;=\; \mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A^{(n)}\bigr).
$$

Block-Jacobi iteration with this clip-prox at every inner step is **Algorithm 1**, the exact single-program solver of (1) at any user-chosen $(\tau_A, \tau_B)$. Under (H), Proposition 2 says clip $=$ polar at the chain-pinned caps, so Algorithm 2 and Algorithm 1 produce identical updates on the saturating-regime ray; Algorithm 2 substitutes polar (a uniform-spectrum prior on the whitened cost) for clip unconditionally, which is what makes it computable via Newton–Schulz without an SVD per inner step. Algorithm 1 is a reference point, not the run algorithm.

The polar map (FW vertex) is computed via Newton–Schulz iteration: $X_0 = M / \lVert M\rVert_F$, $X_{i+1} = \tfrac{3}{2}\,X_i - \tfrac{1}{2}\,X_i\,X_i^\top\,X_i$. The iteration drives every singular value of $X_0$ toward one cubically; a small fixed number of iterations suffice on matrices of LoRA size. See Appendix B.

## 8. The chain: $\eta \to \rho \to (\tau_A, \tau_B)$

§§3–6 left the program's caps $(\tau_A, \tau_B)$ as free hyperparameters. §7 showed that the per-block-contribution norm $\lVert\mathrm dA^\star(\tau_A)\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$ is a state-only function of $\tau_A$ at every FW iterate (Lemma 3), and similarly for the $B$-side. We now use this to derive $(\tau_A, \tau_B)$ from a single user-facing magnitude hyperparameter — the spectral step size $\eta$ — via a chain of tight implications. The chain has two steps.

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

Algorithm 2 is **anchored Frank-Wolfe** (§5) on the whitened residual program (2) at the chain-pinned caps (7) of §8, with the polar LMO of §4.2 at each step. Under hypothesis (H), Proposition 2 identifies this iteration with the exact block-Jacobi clip-prox solver of (1) at those caps (Algorithm 1 in the §7 Remark); the per-factor and tangent caps are auto-satisfied properties of its iterates, not constraints enforced by a separate program. We present it directly in normalize-then-scale form (the right-hand side of (8)), which absorbs the state-only norm of $D^{(n)}$ via normalization rather than computing $\sigma_{\min}(A), \sigma_{\min}(B)$ explicitly.

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
| Adam preconditioning ($u_A, u_B$) | Surrogate for $G$ in (1); calibrated via §4.1 |
| Spectral preconditioners ($S_A^{-1/2}, S_B^{-1/2}$) | Whitening of Definition 2; forced by Lemma 2 |
| Tight-tangent radius ($\rho$) | Tangent constraint $\lVert J\rVert_2 \le \eta$ + Proposition 3 |
| Cross-coupling correction ($\tilde u_A, \tilde u_B$) | Lemma 1 (recovered as anchored-FW linear cost; §5.3) |
| Directions $D_A^{(n)}, D_B^{(n)}$ (whiten + polar + unwhiten) | LMO over operator-norm ball (§4.2) applied to anchored FW cost; equation (3) |
| Tight-tangent rescale | Lemma 3 + state-only caps (7); equation (8) |
| Block-Jacobi outer loop | Anchored Frank-Wolfe on (2) — §5 |

**Algorithm 1 (reference, not run).** Replace the polar step in 5b with the clip prox of the §7 Remark and drop the rescale in 5c (which is then redundant — the clip prox already returns the correct magnitude at the given $\tau$). Algorithm 1 is the exact single-program solver of (1) at any user-chosen $(\tau_A, \tau_B)$; we run Algorithm 2 because under (H) it coincides with Algorithm 1 at the tangent-saturating caps (7), with the polar form avoiding an SVD per inner step.

## 10. Algorithm 2′ — the implemented variant at $k \ge 2$

`magnitude_rule = "spectral_chord_tight"` in `lora_playground/optim.py` coincides with Algorithm 2 at $k = 1$ and differs at $k \ge 2$. The code **pre-normalizes the Adam updates by the operator norm of the whitened direction** before entering the Picard loop, so the polar map sees a whitened direction of unit norm at iter 1. This is exactly the **Adam calibration** of §4.1: replacing $q_A = S_B^{-1/2}\,u_A$ with $Q_A = q_A / \lVert q_A\rVert_2$ so the dual input sits at unit operator norm. This subsection states the modified algorithm and quantifies its effect on the polar input.

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

*Geometric motivation.* Program (1) places its trust region in operator-norm geometry — $\lVert B\,\Delta A\rVert_2 \le \tau_A$, equivalently $\lVert Y_A\rVert_2 \le \tau_A$ in the whitened variable $Y_A = S_B^{1/2}\,\Delta A$ (Lemma 2). The linear cost in the whitened frame is $\langle X_A, Y_A\rangle$ with $X_A = S_B^{-1/2}\,u_A$. The pre-rescale normalizes $X_A$ to unit operator norm, putting the dual input and the primal trust region on the same spectral scale. This corresponds to a modified program with linear cost $\langle X_A/\sigma_{\max}(X_A),\, Y_A\rangle$ — i.e. $\langle Q_A, Y_A\rangle$ in the §4.1 notation — and Algorithm 2′ is anchored Frank-Wolfe on the modified (calibrated) program.

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
- Mirsky, *Symmetric gauge functions and unitarily invariant norms.* Quart. J. Math. 11 (1960), 50–59. Closed form for the Frobenius projection onto an operator-norm ball used in Lemma 0(a) and the §7 Remark (Algorithm 1's clip-prox per-block solver).
- Higham, *Functions of Matrices: Theory and Computation.* SIAM 2008, Ch. 8. Cubic convergence of the Newton–Schulz iteration.
