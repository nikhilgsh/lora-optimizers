# A polar-product LoRA optimizer — Frank-Wolfe / polar variant

This document derives the Frank-Wolfe (FW) variant of the polar-product LoRA optimizer: state the variational program, solve it via FW iteration on the per-block whitened residual, and use the polar map as the FW vertex. The result is **Algorithm 2** (anchored linearization) and **Algorithm 2′** (the same iteration with Adam calibration baked in). §8 presents a related variant — full-gradient linearization with the self-term kept — and analyzes the structure of its polar input.

A companion document `algorithm_tight_chord.md` derives the block-coordinate-descent variant of the same program (Algorithm 1, clip-prox per-block solver). The two documents are independent; this one stands on its own.

**Reading guide.**

- **Notation reference (below)** — every symbol grounded in one place.
- **§§1–4** — setup, the warmup Muon-style program (§2), the residual program (§3), and its whitened form (§4).
- **§5** — calibrating the Adam direction relative to the FW correction term.
- **§6** — operator-norm LMO: polar is the FW vertex.
- **§7** — anchored Frank-Wolfe iteration, recovers Lemma 1.
- **§8** — full-gradient linearization variant; derives the structure of its polar input at the first inner iteration.
- **§§9–10** — closure: saturating-regime hypothesis (H), Lemma 3, chain $\eta \to \rho \to (\tau_A, \tau_B)$.
- **§11** — Algorithm 2 pseudocode and correspondence with the variational program.
- **§12** — when the inner Picard iteration moves at $k \ge 2$ (calibration analysis).


## Notation reference

All symbols used below are grouped here. Shapes assume a single LoRA layer pair.

**The factors.** Only $A$ and $B$ are trained; the optimizer produces updates $\mathrm{d}A, \mathrm{d}B$:

| symbol | shape | meaning |
|---|---|---|
| $A$ | $(r, d_{\text{in}})$ | LoRA "down" factor |
| $B$ | $(d_{\text{out}}, r)$ | LoRA "up" factor |
| $\mathrm{d}A, \mathrm{d}B$ | same as $A, B$ | proposed factor updates this step |
| $G$ | $(d_{\text{out}}, d_{\text{in}})$ | gradient on the merged weight $W + BA$ |
| $g_A = B^\top G,\ g_B = G\, A^\top$ | same as $A, B$ | factor gradients from backprop |
| $u_A, u_B$ | same as $A, B$ | bias-corrected Adam directions |

**Gram matrices and their square roots** ($r \times r$):

| symbol | definition | role |
|---|---|---|
| $S_A := A\,A^\top$ | $(r, r)$, PSD | "size" of $A$'s rows |
| $S_B := B^\top B$ | $(r, r)$, PSD | "size" of $B$'s columns |
| $S_A^{1/2}, S_B^{1/2}$ | PSD square roots | used to whiten |
| $S_A^{-1/2}, S_B^{-1/2}$ | inverse PSD square roots (damped via $\delta_A, \delta_B$) | used to un-whiten |
| $\delta_A, \delta_B$ | scalars | damping; $\varepsilon_{\text{rel}}\,\sigma_{\max}(\cdot)^2$ in the scale-invariant parameterization |

**Direction-only projectors:**

| symbol | definition | property |
|---|---|---|
| $U_B := B\,S_B^{-1/2}$ | $(d_{\text{out}}, r)$ | orthonormal columns: $U_B^\top U_B = I_r$ |
| $V_A := S_A^{-1/2}\,A$ | $(r, d_{\text{in}})$ | orthonormal rows: $V_A V_A^\top = I_r$ |

**Whitened quantities:**

| symbol | definition | shape | role |
|---|---|---|---|
| $Y_A := S_B^{1/2}\,\mathrm{d}A$ | whitened $A$-update | $(r, d_{\text{in}})$ | primal variable in program (2) |
| $Y_B := \mathrm{d}B\,S_A^{1/2}$ | whitened $B$-update | $(d_{\text{out}}, r)$ | primal variable in program (2) |
| $X_A := S_B^{-1/2}\,u_A$ | whitened Adam direction (A-side) | $(r, d_{\text{in}})$ | uncalibrated dual |
| $X_B := u_B\,S_A^{-1/2}$ | whitened Adam direction (B-side) | $(d_{\text{out}}, r)$ | uncalibrated dual |
| $\widetilde{G}_A := S_B^{-1/2}\,g_A$ | whitened gradient on $A$ | $(r, d_{\text{in}})$ | exact dual; replaced by $X_A$ in practice |
| $\widetilde{G}_B := g_B\,S_A^{-1/2}$ | whitened gradient on $B$ | $(d_{\text{out}}, r)$ | exact dual; replaced by $X_B$ in practice |
| $Q_A := X_A / \sigma_{\max}(X_A)$ | calibrated unit-op-norm whitened Adam (A) | $(r, d_{\text{in}})$ | linear cost in calibrated Algorithm 2′ (§5) |
| $Q_B := X_B / \sigma_{\max}(X_B)$ | calibrated unit-op-norm whitened Adam (B) | $(d_{\text{out}}, r)$ | linear cost in calibrated Algorithm 2′ |

**Magnitude knobs and derived radii:**

| symbol | meaning |
|---|---|
| $\eta$ | user-facing spectral step size — caps $\lVert J\rVert_2$ |
| $s := \sigma_{\max}(A) + \sigma_{\max}(B)$ | combined factor scale |
| $\rho := \eta/s$ | per-factor radius (cap on $\lVert\mathrm{d}A\rVert_2, \lVert\mathrm{d}B\rVert_2$) |
| $\tau_A := \rho\,\sqrt{\sigma_{\min}(B)^2 + \delta_B}$ | program's cap on $\lVert Y_A\rVert_2$ |
| $\tau_B := \rho\,\sqrt{\sigma_{\min}(A)^2 + \delta_A}$ | program's cap on $\lVert Y_B\rVert_2$ |

**The tangent and chord:**

| symbol | definition | meaning |
|---|---|---|
| $J := B\,\mathrm{d}A + \mathrm{d}B\,A$ | first-order linearization of the merged-weight change | algorithm controls $\lVert J\rVert_2$ |
| $\Delta W := (B+\mathrm{d}B)(A+\mathrm{d}A) - BA = J + \mathrm{d}B\,\mathrm{d}A$ | exact chord | what the loss sees |

**Frank-Wolfe iterates and per-block linear costs:**

| symbol | meaning |
|---|---|
| $n = 0, 1, \ldots, k-1$ | Picard iteration index ($k$ = total inner iters) |
| $\mathrm{d}A^{(n)}, \mathrm{d}B^{(n)}$ | factor updates at FW iterate $n$ (initialized to zero at $n=0$) |
| $Y_A^{(n)}, Y_B^{(n)}$ | whitened iterates |
| $C_A^{(n)}$ (anchored, §7) | $Q_A + (1/\eta)\,U_B^\top Y_B^{(n)} V_A$ — polar input at iter $n+1$ (whitened coords) |
| $C_A^{(n),\text{full}}$ (§8) | $Q_A + (1/\eta)\bigl(Y_A^{(n)} + U_B^\top Y_B^{(n)} V_A\bigr)$ — polar input with self-term kept |
| $c_A^{(n)} := S_B^{-1/2}\,\tilde u_A^{(n)}$ | polar input at iter $n+1$ (factor coords), where $\tilde u_A^{(n)} := u_A + (1/\eta)\,B^\top\,\mathrm dB^{(n)}\,A$ is the cross-coupling-corrected Adam direction |
| $D_A^{(n)} := S_B^{-1/2}\,\mathrm{polar}(C_A^{(n)})$ | factor-space direction (unwhitened) |

**Matrix functions:**

| symbol | definition |
|---|---|
| $\mathrm{polar}(M)$ | for $M = U \Sigma V^\top$, returns $U V^\top$ (singular values $\to 1$, vectors preserved) |
| $\mathrm{clip}_\tau(M)$ | returns $U\,\min(\Sigma, \tau)\,V^\top$ (caps singular values at $\tau$) |
| $\sigma_{\max}(\cdot), \sigma_{\min}(\cdot)$ | largest / smallest singular value |
| $\lVert\cdot\rVert_2, \lVert\cdot\rVert_F, \lVert\cdot\rVert_*$ | operator, Frobenius, nuclear norm |

**Algorithm labels used below:**

| label | what it is |
|---|---|
| **Algorithm 1** | exact clip-prox block-Jacobi solver — derived in the companion `algorithm_tight_chord.md`. Referenced here as a contrastive point in §9 (where clip coincides with polar under (H)). |
| **Algorithm 2** | polar FW, anchored, uncalibrated (§11). |
| **Algorithm 2′** | Algorithm 2 with the §5 pre-rescale $u_A \leftarrow u_A/\sigma_{\max}(X_A)$ baked in. §12 explains why this matters at $k \ge 2$. |

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

The chord is what the loss sees. The tangent $J$ is its first-order linearization. The bilinear term $\Delta B\,\Delta A$ is $O(\rho^2)$ in the per-factor radius and is bounded relative to $J$ by $\rho/s$ where $s = \sigma_{\max}(A) + \sigma_{\max}(B)$; we cap $J$ directly via the program (so the bilinear term cannot exceed $\rho^2$ by construction) and verify a-posteriori that $\rho/s = \eta/s^2 \ll 1$ in the relevant operating regime (Appendix A).

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

has unique minimizer $X^\star = \mathrm{clip}_\tau(M)$, where for $M = U \Sigma V^\top$ (SVD) the **singular-value clip** is $\mathrm{clip}_\tau(M) := U\,\min(\Sigma, \tau)\,V^\top$.

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

*(a)* Let $M = U \Sigma V^\top$ be the SVD of $M$. Expand $\lVert X - M\rVert_F^2 = \lVert X\rVert_F^2 - 2\langle X, M\rangle + \lVert M\rVert_F^2$. The middle term is upper-bounded by $(\ast)$ with $A = M$, so for any $X$ with prescribed singular values $\sigma_i(X)$, the objective is minimized by aligning singular vectors with $M$. Hence the minimizer has SVD $X = U\, D\, V^\top$ with $D \succeq 0$ diagonal, and the problem reduces to

$$
\min_{D \succeq 0}\ \sum_i (D_{ii} - \sigma_i(M))^2 \quad\text{s.t.}\quad \max_i D_{ii} \le \tau,
$$

uniquely solved by $D_{ii} = \min(\sigma_i(M), \tau)$.

*(b)* Apply $(\ast)$ to $\langle u, -X\rangle$ to get $\langle u, X\rangle \ge -\sum_i \sigma_i(u)\,\sigma_i(X)$, with equality iff $-X$ and $u$ share singular vectors. Under $\sigma_i(X) \le \rho$, the right side is minimized by $\sigma_i(X) = \rho$ on every $i$ with $\sigma_i(u) > 0$. Aligning singular vectors with $u$ gives $X^\star = -\rho\,\mathrm{polar}(u)$, with value $-\rho\,\lVert u\rVert_*$. ∎

Applying part (b) to each factor, program (W) yields

$$
\Delta A^\star \;=\; -\rho_A\,\mathrm{polar}(u_A), \qquad \Delta B^\star \;=\; -\rho_B\,\mathrm{polar}(u_B).
$$

This is **Muon** (Jordan et al. 2024) applied independently to each LoRA factor.

What program (W) **does not** capture:

- *No coupling.* Any $(\Delta A, \Delta B)$ producing the same tangent produces the same first-order change in loss.
- *No whitening.* The constraint controls the bare factor, not the merged-weight contribution $B\,\Delta A$.
- *No tangent control.* The radii are not connected to $J$ via the program itself.

The remainder of this document repairs these three deficiencies in turn.

## 3. The residual program

Let $G := \nabla f(W + BA)$ be the dense gradient with respect to the merged weight. The first-order local model of the loss is

$$
\langle G,\,J\rangle + \frac{1}{2\eta}\,\lVert J\rVert_F^2,
$$

equivalently (up to a constant) the **residual form**

$$
\frac{1}{2\eta}\,\lVert J + \eta\,G\rVert_F^2.
$$

Adding per-block-contribution operator-norm caps gives:

$$
\min_{\Delta A,\,\Delta B}\ \frac{1}{2\eta}\,\lVert B\,\Delta A + \Delta B\,A + \eta\,G\rVert_F^2
\quad\text{s.t.}\quad
\lVert B\,\Delta A\rVert_2 \le \tau_A,\ \ \lVert\Delta B\,A\rVert_2 \le \tau_B.
\tag{1}
$$

The caps $\tau_A, \tau_B$ are pinned in §10.

## 4. Whitening

Program (1)'s constraint is on $\lVert B\,\Delta A\rVert_2$, not on $\lVert\Delta A\rVert_2$, and the quadratic couples the two factors through $\lVert J\rVert_F^2$. A single linear change of variable diagonalizes both.

**Definition 2 (whitened objects).** Let $S_A := AA^\top$ and $S_B := B^\top B$ ($r \times r$ PSD; here assumed rank $r$, with damping deferred to §11). Define

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
\widetilde G_A \;:=\; S_B^{-1/2}\,g_A, \qquad \widetilde G_B \;:=\; g_B\,S_A^{-1/2}.
$$

The maps $\Delta A \leftrightarrow Y_A$ and $\Delta B \leftrightarrow Y_B$ are invertible. In these coordinates the tangent factors cleanly:

$$
J \;=\; U_B\,Y_A + Y_B\,V_A,
$$

and the operator-norm caps become caps on the whitened variables: $\lVert B\,\Delta A\rVert_2 = \lVert Y_A\rVert_2$, $\lVert\Delta B\,A\rVert_2 = \lVert Y_B\rVert_2$.

**Lemma 2 (whitened residual program).** $\langle G,\,J\rangle = \langle \widetilde G_A,\,Y_A\rangle + \langle \widetilde G_B,\,Y_B\rangle$. In whitened coordinates, program (1) reads

$$
\min_{Y_A,\,Y_B}\ \langle \widetilde G_A,\,Y_A\rangle + \langle \widetilde G_B,\,Y_B\rangle + \frac{1}{2\eta}\,\lVert U_B\,Y_A + Y_B\,V_A\rVert_F^2
\quad\text{s.t.}\quad \lVert Y_A\rVert_2 \le \tau_A,\ \lVert Y_B\rVert_2 \le \tau_B.
\tag{2}
$$

This change of variable diagonalizes the operator-norm caps and surfaces $\widetilde G_A, \widetilde G_B$ as the linear-cost duals.

## 5. Adam calibration

Optimizers do not have $G$ on hand. They have Adam-preconditioned factor directions $u_A, u_B$ in place of $g_A, g_B$. Substituting gives the whitened Adam directions $X_A := S_B^{-1/2}\,u_A$ and $X_B := u_B\,S_A^{-1/2}$ (Notation reference).

The polar map is scale-invariant, so the only thing that matters about the dual input is its direction *and* its scale **relative to the other terms in the FW linear cost** (§7). Looking ahead: the FW polar input at Picard iter $n$ has the form

$$
C_A^{(n)} \;=\; \gamma_A\,Q_A \;+\; \tfrac{1}{\eta}\,U_B^\top\,Y_B^{(n)}\,V_A,
$$

a base term $\gamma_A\,Q_A$ plus a correction term. Choosing $\gamma_A$ and the normalization of $Q_A$ is equivalent to choosing the scale at which the polar map sees the base term relative to the correction.

**Natural scale of the correction term.** Anticipating the chain $\eta \to \rho \to (\tau_A, \tau_B)$ derived in §10 — where $\rho = \eta/s$ and $\tau_B = \rho\,\sqrt{\sigma_{\min}(A)^2 + \delta_A}$ — the correction term in $C_A^{(n)}$ is bounded in operator norm by

$$
\Bigl\lVert\tfrac{1}{\eta}\,U_B^\top\,Y_B^{(n)}\,V_A\Bigr\rVert_2
\;\le\; \frac{\lVert Y_B^{(n)}\rVert_2}{\eta}
\;\le\; \frac{\tau_B}{\eta}
\;=\; \frac{\sqrt{\sigma_{\min}(A)^2 + \delta_A}}{s}
\;\le\; \frac{\sigma_{\max}(A)}{s}
\;\le\; 1.
$$

The inequalities use: $U_B, V_A$ partial isometries; the op-norm cap on $Y_B^{(n)}$ from program (1); the chain (7) with $\tau_B = \rho\sqrt{\sigma_{\min}(A)^2 + \delta_A}$ and $\rho = \eta/s$; $\sigma_{\min}(A) \le \sigma_{\max}(A)$; and $\sigma_{\max}(A) \le s$. Symmetrically for the $B$-side. **The correction term lives at op-norm scale $O(1)$, ceilinged at $1$.**

This pins the calibration. For the polar map to weigh the base and correction comparably, $\gamma_A\,Q_A$ must sit at op-norm scale $1$:

$$
Q_A \;:=\; \frac{X_A}{\sigma_{\max}(X_A)},
\qquad
\gamma_A \;=\; 1,
$$

and symmetrically $Q_B = X_B / \sigma_{\max}(X_B)$, $\gamma_B = 1$. The Adam-surrogate program substitutes $\widetilde G_A \rightsquigarrow Q_A$ and $\widetilde G_B \rightsquigarrow Q_B$ into (2). Equivalently, the pre-rescale $u_A \leftarrow u_A / \sigma_{\max}(X_A)$ is hoisted in front of the whitening.

The §8 variant adds a self-term that obeys the same bound under (7); the calibration carries over.

## 6. Operator-norm LMO

The per-block solver is the linear-minimization oracle (LMO) over the operator-norm ball, which is part (b) of Lemma 0:

$$
\arg\min_{\lVert Y\rVert_2 \le \tau}\ \langle C,\,Y\rangle \;=\; -\tau\,\mathrm{polar}(C),
\qquad
\min\text{-value} \;=\; -\tau\,\lVert C\rVert_*.
$$

**Definition 3 (polar map).** For $C = U\,\Sigma\,V^\top$, $\mathrm{polar}(C) := U\,V^\top$ — every singular value mapped to one, singular vectors preserved.

This is the only per-block tool the rest of the derivation needs. §7 applies it to a linearization of (2) at the current FW iterate. The clip-prox alternative (Lemma 0a) is the per-block solver for block-coordinate descent on (1); under hypothesis (H) of §9, clip and polar coincide at the chain-pinned caps — see §9 for the contrast and the companion document for the BCD derivation.

## 7. Frank-Wolfe derivation of Algorithm 2

The whitened residual quadratic in (2) couples $Y_A$ and $Y_B$ through $\lVert U_B\,Y_A + Y_B\,V_A\rVert_F^2$. Frank-Wolfe handles the coupling by **linearizing the quadratic at the current iterate** and applying the LMO of §6 to the resulting linear program. The linearization point is a modeling choice. This section makes the choice that produces the run algorithm: drop the self-terms $Y_A^{(n)}, Y_B^{(n)}$ from the linearization. §8 keeps them.

### 7.1 FW linear costs

Write $\Phi(Y_A, Y_B)$ for the Adam-surrogate whitened residual objective (with $\gamma_A = \gamma_B = 1$ fixed by §5's correction-scale argument):

$$
\Phi(Y_A, Y_B) \;=\; \langle Q_A,\,Y_A\rangle + \langle Q_B,\,Y_B\rangle + \frac{1}{2\eta}\,\lVert U_B\,Y_A + Y_B\,V_A\rVert_F^2.
$$

Its block gradients, using $U_B^\top U_B = I_r$ and $V_A V_A^\top = I_r$, are

$$
\nabla_{Y_A}\Phi \;=\; Q_A + \tfrac{1}{\eta}\bigl(Y_A + U_B^\top\,Y_B\,V_A\bigr),
\qquad
\nabla_{Y_B}\Phi \;=\; Q_B + \tfrac{1}{\eta}\bigl(U_B\,Y_A\,V_A^\top + Y_B\bigr).
$$

### 7.2 FW linearization (self-terms dropped)

At iterate $(Y_A^{(n)}, Y_B^{(n)})$, linearize the $A$-block at $(0,\,Y_B^{(n)})$ and the $B$-block at $(Y_A^{(n)},\,0)$. This drops the self-terms and gives

$$
C_A^{(n)} \;:=\; Q_A + \tfrac{1}{\eta}\,U_B^\top\,Y_B^{(n)}\,V_A,
\qquad
C_B^{(n)} \;:=\; Q_B + \tfrac{1}{\eta}\,U_B\,Y_A^{(n)}\,V_A^\top.
$$

Applying the LMO of §6 with full Frank-Wolfe step ($\alpha_n = 1$):

$$
Y_A^{(n+1)} \;=\; -\tau_A\,\mathrm{polar}\bigl(C_A^{(n)}\bigr),
\qquad
Y_B^{(n+1)} \;=\; -\tau_B\,\mathrm{polar}\bigl(C_B^{(n)}\bigr).
\tag{3}
$$

Initialization $Y_A^{(0)} = Y_B^{(0)} = 0$ makes $C_A^{(0)} = Q_A$ and $C_B^{(0)} = Q_B$, so the $n = 0$ vertex is the polar of the calibrated Adam direction — exactly the per-block Muon update of §2, now derived from the joint program.

### 7.3 Recovery of original variables — Lemma 1 falls out

Translating back via $\Delta A = S_B^{-1/2}\,Y_A$ and $\Delta B = Y_B\,S_A^{-1/2}$:

$$
\mathrm dA^{(n+1)} \;=\; -\tau_A\,\underbrace{S_B^{-1/2}\,\mathrm{polar}(C_A^{(n)})}_{=:\, D_A^{(n)}},
\qquad
\mathrm dB^{(n+1)} \;=\; -\tau_B\,\underbrace{\mathrm{polar}(C_B^{(n)})\,S_A^{-1/2}}_{=:\, D_B^{(n)}}.
$$

The polar argument $C_A^{(n)}$ can be re-expressed in factor coordinates by factoring $S_B^{-1/2}$ out:

$$
C_A^{(n)}
\;=\; S_B^{-1/2}\,\bigl[\,\hat u_A \;+\; \tfrac{1}{\eta}\,B^\top\,\mathrm dB^{(n)}\,A\,\bigr],
$$

where $\hat u_A := u_A / \lVert S_B^{-1/2}\,u_A\rVert_2$. Symmetrically,

$$
C_B^{(n)} \;=\; \bigl[\,\hat u_B \;+\; \tfrac{1}{\eta}\,B\,\mathrm dA^{(n)}\,A^\top\,\bigr]\,S_A^{-1/2}.
$$

**Lemma 1 (cross-coupling correction).** The FW linear cost in factor coordinates is exactly the corrected linear cost obtained by completing the square on the $A$-subproblem of (1) after fixing $\Delta B$:

$$
\tilde u_A \;=\; u_A \;+\; \tfrac{1}{\eta}\,B^\top\,\mathrm dB^{(n)}\,A,
$$

and symmetrically $\tilde u_B = u_B + \tfrac{1}{\eta}\,B\,\mathrm dA^{(n)}\,A^\top$. ∎

The recovered direction $D_A^{(n)} = S_B^{-1/2}\,\mathrm{polar}\bigl(S_B^{-1/2}\,\tilde u_A\bigr)$ is what Algorithm 2 (§11) uses.

## 8. Variant: FW linearization with self-terms retained

The alternative FW linearization at the current iterate keeps the self-terms — i.e. uses the full block gradients of §7.1:

$$
C_A^{(n),\,\text{self}} \;:=\; Q_A + \tfrac{1}{\eta}\bigl(Y_A^{(n)} + U_B^\top\,Y_B^{(n)}\,V_A\bigr),
\qquad
C_B^{(n),\,\text{self}} \;:=\; Q_B + \tfrac{1}{\eta}\bigl(U_B\,Y_A^{(n)}\,V_A^\top + Y_B^{(n)}\bigr).
$$

In factor coordinates, the only difference from §7.3 is the **self-terms** $S_B\,\mathrm dA^{(n)}$ and $\mathrm dB^{(n)}\,S_A$ inside the brackets — §7 dropped them; this variant retains them.

**Correction-term scale.** The extra self-term $\tfrac{1}{\eta}\,Y_A^{(n)}$ has operator norm $\le \tau_A/\eta = \sqrt{\sigma_{\min}(B)^2 + \delta_B}/s \le 1$ under (7), by the same argument as in §5 for the cross-coupling.

**Cost.** Two additional $(r{\times}r)\cdot(r{\times}d)$ matmuls per Picard iter.

### 8.1 The self-term at inner iter $n=1$ collapses to $-\alpha\,\mathrm{polar}(Q_A)$

Using the §7.2 convention that $Y_A^{(n+1)}$ is the output of the FW step at iter $n$ from input $C_A^{(n)}$, initialize $Y_A^{(0)} = Y_B^{(0)} = 0$. Then:

- **Iter $n=0$.** $C_A^{(0),\text{full}} = Q_A$ since both self and cross terms vanish. The FW vertex is $Y_A^{(1)} = -\tau_A\,\mathrm{polar}(Q_A)$, equivalently in factor coordinates (chain-pinned form (8))
  $$
  \mathrm dA^{(1)} \;=\; -\rho\,\frac{S_B^{-1/2}\,\mathrm{polar}(Q_A)}{\lVert S_B^{-1/2}\rVert_2}.
  $$
- **Iter $n=1$.** The polar input is $C_A^{(1),\text{full}} = Q_A + (1/\eta)\bigl(Y_A^{(1)} + U_B^\top Y_B^{(1)} V_A\bigr)$. The self-term contribution is $(1/\eta)\,Y_A^{(1)}$. Recover $Y_A^{(1)} = S_B^{1/2}\,\mathrm dA^{(1)}$; the $S_B^{1/2}$ and $S_B^{-1/2}$ cancel:
  $$
  Y_A^{(1)} \;=\; -\rho\,\sqrt{\sigma_{\min}(B)^2 + \delta_B}\,\mathrm{polar}(Q_A).
  $$

Multiply by $1/\eta$ and use $\rho/\eta = 1/s$:

$$
\boxed{\quad
\tfrac{1}{\eta}\,Y_A^{(1)} \;=\; -\alpha\,\mathrm{polar}(Q_A),
\qquad
\alpha \;:=\; \frac{\sqrt{\sigma_{\min}(B)^2 + \delta_B}}{s}.
\quad}
$$

The self-term at the first non-trivial inner iteration is a scaled copy of $\mathrm{polar}(Q_A)$, with scaling $\alpha$ determined entirely by the current factor spectrum.

### 8.2 Polar-input ratio for the self-term

Against the calibrated base $\lVert Q_A\rVert_2 = 1$:

$$
\frac{\lVert (1/\eta)\,Y_A^{(1)}\rVert_2}{\lVert Q_A\rVert_2}
\;=\; \alpha
\;=\; \frac{\sqrt{\sigma_{\min}(B)^2 + \delta_B}}{s}.
$$

The expression is exact under chain-pinning (Lemma 3): the bound is achieved with equality, not strict. The cross-coupling bound $\sigma_{\max}(A)/s$ of §5 is, by contrast, generically not saturated.

### 8.3 Polar-input structure of the §8 variant

Combining §8.1 with the cross-coupling term, the polar input at inner iter $n=1$ is

$$
C_A^{(1),\text{full}}
\;=\; Q_A \;-\; \alpha\,\mathrm{polar}(Q_A) \;+\; \tfrac{1}{\eta}\,U_B^\top\,Y_B^{(1)}\,V_A.
$$

Two structural observations relative to the anchored §7 polar input $Q_A + (1/\eta)\,U_B^\top\,Y_B^{(1)}\,V_A$:

- **The self-term shares singular vectors with $Q_A$.** $\mathrm{polar}(Q_A)$ has the SVD $U V^\top$ on the same $U, V$ that diagonalize $Q_A$, so $Q_A - \alpha\,\mathrm{polar}(Q_A)$ has the same singular vectors as $Q_A$ with singular values shifted uniformly down by $\alpha$. The cross-coupling lives in singular directions determined by $B$'s column space and $A$'s row space, generically not aligned with $Q_A$'s singular vectors.
- **The self-term magnitude is chain-pinned, not iterate-driven.** The per-iter rescale (Lemma 3) forces $\lVert\mathrm dA^{(n)}\rVert_2 = \rho$ exactly, so $\alpha$ depends only on the current factor spectrum.

## 9. Saturating regime: per-block-contribution norms are state-only

Algorithm 2 (§11) is the FW iteration of §7 at the chain-pinned caps (7) of §10. To close the chain we need $\lVert\mathrm dA^{(n+1)}\rVert_2$ — the per-factor operator norm of the FW vertex — to be a simple, state-only function of $\tau_A$ (independent of the FW iterate $n$).

**Lemma 3 (factor-norm collapse).** For any FW iterate $n$, the operator norms of $D_A^{(n)}, D_B^{(n)}$ are state-only:

$$
\lVert D_A^{(n)}\rVert_2 \;=\; \bigl(\sigma_{\min}(B)^2 + \delta_B\bigr)^{-1/2},
\qquad
\lVert D_B^{(n)}\rVert_2 \;=\; \bigl(\sigma_{\min}(A)^2 + \delta_A\bigr)^{-1/2}.
$$

*Proof.* The polar factor $\mathrm{polar}(C_A^{(n)}) \in \mathbb{R}^{r \times d_{\text{in}}}$ has $r \le d_{\text{in}}$ and orthonormal rows, so $\mathrm{polar}(C_A^{(n)})\,\mathrm{polar}(C_A^{(n)})^\top = I_r$. Then

$$
\lVert S_B^{-1/2}\,\mathrm{polar}(C_A^{(n)})\rVert_2^2 \;=\; \sigma_{\max}(S_B^{-1}) \;=\; \lVert S_B^{-1/2}\rVert_2^2.
$$

Symmetric for the $B$-side. The damped-spectrum forms follow from $S_B = B^\top B + \delta_B I$ and likewise for $S_A$. ∎

The FW iterate-dependence of $D_A^{(n)}, D_B^{(n)}$ sits entirely in their *singular vectors*, not in their norms.

**Saturating-regime hypothesis (H).** Writing $c_A^{(n)} := S_B^{-1/2}\,\tilde u_A^{(n)}$ for the polar input in factor coordinates,

$$
\tau_A \;\le\; \eta\,\sigma_{\min}\!\bigl(c_A^{(n)}\bigr), \qquad
\tau_B \;\le\; \eta\,\sigma_{\min}\!\bigl(c_B^{(n)}\bigr)
\qquad\text{for } n = 1,\ldots,k.
\tag{H}
$$

**Proposition 2 (clip $\to$ polar under H).** If (H) holds at iterate $n$, then $\mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A^{(n)}\bigr) = -\tau_A\,\mathrm{polar}\bigl(c_A^{(n)}\bigr)$.

*Proof.* Under (H), $\mathrm{clip}_{\tau_A}$ flattens every singular value of $-\eta\,c_A^{(n)}$ to $\tau_A$ and preserves singular vectors, giving $\tau_A\,U V^\top$. Polar is invariant under positive scaling and odd under negation. ∎

**Connection to Algorithm 1 (BCD, companion doc).** The companion document derives the exact block-coordinate-descent solver of (2) with clip-prox per block. Under (H), Proposition 2 says clip and polar coincide at the chain-pinned caps, so Algorithm 2 and Algorithm 1 produce identical updates on the saturating-regime ray; Algorithm 2 (here) substitutes polar (a uniform-spectrum prior on the whitened cost) for clip unconditionally, which is what makes it computable via Newton–Schulz without an SVD per inner step. Outside (H), clip and polar diverge on directions with $\sigma_i(c_A^{(n)}) < \tau_A/\eta$ (clip leaves small singular values alone; polar lifts them to one).

The polar map is computed via Newton–Schulz iteration (Appendix B).

## 10. The chain: $\eta \to \rho \to (\tau_A, \tau_B)$

We now derive $(\tau_A, \tau_B)$ from a single user-facing magnitude hyperparameter $\eta$ via a chain of tight implications:

$$
\underbrace{\lVert J\rVert_2 \le \eta}_{\text{user-facing (tangent)}}
\;\overset{\text{Prop 3}}{\Longleftarrow}\;
\underbrace{\lVert\Delta A\rVert_2, \lVert\Delta B\rVert_2 \le \rho}_{\text{per-factor}}
\;\overset{\text{Lemma 3 under (H)}}{\Longleftarrow}\;
\underbrace{\lVert S_B^{1/2}\Delta A\rVert_2 \le \tau_A,\ \lVert\Delta B\, S_A^{1/2}\rVert_2 \le \tau_B}_{\text{program's caps}}
$$

### Step 1: $\eta \to \rho$ (Proposition 3)

Setting $\lVert\Delta A\rVert_2 = \lVert\Delta B\rVert_2 = \rho$, submultiplicativity gives $\lVert J\rVert_2 \le s\rho$. Proposition 3 sets

$$
\rho \;=\; \frac{\eta}{s}.
\tag{6}
$$

The chord satisfies $\lVert\Delta W\rVert_2 \le \eta + \rho^2 = \eta\,(1 + \eta/s^2)$ (Appendix A).

### Step 2: $\rho \to (\tau_A, \tau_B)$ (Lemma 3 under (H))

Under (H), Lemma 3 gives $\lVert\Delta A^\star(\tau_A)\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$. Setting this equal to $\rho$ pins

$$
\boxed{\quad
\tau_A \;=\; \rho\,\sqrt{\sigma_{\min}(B)^2 + \delta_B},
\qquad
\tau_B \;=\; \rho\,\sqrt{\sigma_{\min}(A)^2 + \delta_A}.
\quad}
\tag{7}
$$

### The pinned program

Program (1) with state-fixed caps (7) is what Algorithm 2 solves. The applied update at outer iteration $n$ is

$$
\boxed{\quad
\mathrm dA^{(n)} \;=\; -\tau_A\, D_A^{(n)} \;=\; -\rho\,\frac{D_A^{(n)}}{\lVert D_A^{(n)}\rVert_2},
\qquad
\mathrm dB^{(n)} \;=\; -\tau_B\, D_B^{(n)} \;=\; -\rho\,\frac{D_B^{(n)}}{\lVert D_B^{(n)}\rVert_2}.
\quad}
\tag{8}
$$

The two equalities in each line are identical under (H) by Lemma 3. The right-hand normalize-then-scale-by-$\rho$ form absorbs the state-only norm of $D^{(n)}$ without explicitly computing $\sigma_{\min}(B), \sigma_{\min}(A)$.

### Sufficient condition for (H)

$$
\rho \;\le\; \eta\,\min\!\Bigl(\sigma_{\min}\!\bigl(c_A^{(n)}\bigr)\,\lVert S_B^{-1/2}\rVert_2,\ \ \sigma_{\min}\!\bigl(c_B^{(n)}\bigr)\,\lVert S_A^{-1/2}\rVert_2\Bigr)
\quad\text{for } n = 1, \ldots, k.
$$

When this holds, Algorithm 2 is the exact solver of (1) at the chain-pinned caps. When it fails, Algorithm 2's update is still well-defined; the directions remain coherent but the identification with the exact (1)-solver is lost.

## 11. Algorithm 2 (the polar variant)

Algorithm 2 is the anchored FW iteration on (2) at the chain-pinned caps (7), with the polar LMO at each step and the §5 Adam calibration hoisted in front of the loop.

**Hyperparameters:** Adam $\beta_1, \beta_2, \varepsilon$; block-Jacobi sweep count $k$; Newton–Schulz iters $j$; preconditioner regularizer $\varepsilon_{\text{rel}}$; spectral step size $\eta$.

**Persistent state:** Adam moments $(m_A, v_A, m_B, v_B)$; step counter $t$; warm-started top singular vectors for $A, B$.

**Algorithm 2.** One step on layer pair $(A, B)$:

1. **Spectral preconditioners** (refreshed periodically; both $r \times r$):
   $$
   S_A^{-1/2} = (A A^\top + \delta_A I)^{-1/2},
   \qquad
   S_B^{-1/2} = (B^\top B + \delta_B I)^{-1/2},
    $$
   with the scale-invariant damping
   $$
   \delta_A \;=\; \varepsilon_{\text{rel}}\,\sigma_{\max}(A A^\top), \qquad \delta_B \;=\; \varepsilon_{\text{rel}}\,\sigma_{\max}(B^\top B).
   $$

2. **Adam preconditioning and calibration.** Form bias-corrected $u_A, u_B$. Then calibrate to unit op-norm in the whitened frame (§5):
   $$
   u_A \;\leftarrow\; u_A / \sigma_{\max}(S_B^{-1/2}\,u_A), \qquad u_B \;\leftarrow\; u_B / \sigma_{\max}(u_B\,S_A^{-1/2}).
   $$

3. **Top singular values** via warm-started power iteration:
   $$
   \sigma_A \gets \sigma_{\max}(A), \qquad \sigma_B \gets \sigma_{\max}(B).
   $$

4. **Tight-tangent radius:** $s \gets \sigma_A + \sigma_B$, $\rho \gets \eta / s$.

5. **Block-Jacobi cross-coupling loop.** Initialize $\mathrm dA = \mathrm dB = 0$. For $n = 1, \ldots, k$:

   - **Cross-coupling correction:**
     $$
     \tilde u_A \;=\; u_A + \tfrac{1}{\eta}\, B^\top\, \mathrm dB\, A,
     \qquad
     \tilde u_B \;=\; u_B + \tfrac{1}{\eta}\, B\, \mathrm dA\, A^\top.
     $$

   - **Direction** (whiten, polar via Newton–Schulz with $j$ iters, unwhiten):
     $$
     D_A \;=\; S_B^{-1/2}\,\mathrm{polar}_{\text{NS-}j}\!\bigl(S_B^{-1/2}\,\tilde u_A\bigr),
     \qquad
     D_B \;=\; \mathrm{polar}_{\text{NS-}j}\!\bigl(\tilde u_B\, S_A^{-1/2}\bigr)\, S_A^{-1/2}.
     $$

   - **Tight-tangent rescale:**
     $$
     \mathrm dA \;=\; -\rho\,\frac{D_A}{\lVert D_A\rVert_2},
     \qquad
     \mathrm dB \;=\; -\rho\,\frac{D_B}{\lVert D_B\rVert_2}.
     $$

6. **Apply.** $A \gets A + \mathrm dA$, $B \gets B + \mathrm dB$.

The line-by-line correspondence with the variational program:

| Algorithm 2 step | Variational source |
|---|---|
| Spectral preconditioners | Whitening of Definition 2; forced by Lemma 2 |
| Adam preconditioning + calibration | Surrogate for $G$ with $\lVert Q_A\rVert_2 = \lVert Q_B\rVert_2 = 1$; §5 |
| Tight-tangent radius ($\rho$) | Tangent constraint $\lVert J\rVert_2 \le \eta$ + Proposition 3 |
| Cross-coupling correction | Lemma 1 (recovered as FW linear cost; §7.3) |
| Directions $D_A^{(n)}, D_B^{(n)}$ | LMO over operator-norm ball (§6) applied to FW cost; equation (3) |
| Tight-tangent rescale | Lemma 3 + state-only caps (7); equation (8) |
| Block-Jacobi outer loop | FW iteration on (2) — §7 |

## 12. Remark: when the Picard iteration moves

At $k = 1$ the §5 calibration is moot: the loop has no correction term yet ($\mathrm dB^{(0)} = 0$, so $C_A^{(1)} = Q_A$), polar is scale-invariant, and only the direction of $Q_A$ enters. The calibration starts to matter at $k \ge 2$, when the polar input becomes a sum of the base term and a non-zero correction.

Without the calibration, the base term $X_A = S_B^{-1/2}\,u_A$ has $\sigma_{\max}(X_A)$ potentially much larger than $1$ when $B$ is near-singular (e.g. at LoRA init when $B = 0$, $S_B^{-1/2}$ is dominated by the damping floor). The correction sits at op-norm $O(1)$ under (7). The polar map is scale-invariant and sees only the ratio, so the correction is suppressed by $\sigma_{\max}(X_A)$ and Picard becomes inert.

The §5 calibration normalizes the base term to op-norm $1$, the same scale as the correction. After it, the polar-input ratio is bounded by $\sigma_{\max}(A)/s \le 1$ and generically $\Theta(1)$ — the loop's iterates move.

This is the only $k \ge 2$ effect of the calibration: at $k = 1$ trajectories with and without it coincide; at $k \ge 2$ the un-calibrated form's correction is suppressed by $\sigma_{\max}(X_A)$ relative to where the calibrated form's iterates land.

## Appendix A. Properties of the tight-tangent radius and chord-vs-tangent gap

$$
\boxed{\quad \rho \;=\; \eta / s \quad}
$$

**Monotonicity.** $\rho$ increases in $\eta$, decreases in $s = \sigma_{\max}(A) + \sigma_{\max}(B)$. Larger factor singular values $\Rightarrow$ smaller step. The rule self-attenuates as $A, B$ grow.

**Boundary.** When $\rho = \eta/s$, the tangent bound binds with equality: $s\rho = \eta$.

**Chord-vs-tangent gap.** At $\rho = \eta/s$ the chord satisfies $\lVert\Delta W\rVert_2 \le \eta + \rho^2 = \eta\,(1 + \eta/s^2)$. The dimensionless quantity $\eta/s^2$ controls when chord and tangent diverge:

- $\eta/s^2 \ll 1$: bilinear term negligible; chord $\approx$ tangent.
- $\eta/s^2 \sim 1$: the program's tangent semantic is no longer a faithful proxy for the chord; the derivation in §10 would need to be revisited.

**Quadratic form.** A stricter derivation caps the chord directly:

$$
s\rho + \rho^2 \le \eta \quad\Longrightarrow\quad \rho = \tfrac{1}{2}\bigl(-s + \sqrt{s^2 + 4\eta}\bigr).
$$

Limits: $\rho \to \eta/s$ as $\eta \ll s^2$, $\rho \to \sqrt{\eta} - s/2$ as $\eta \gg s^2$.

## Appendix B. Newton–Schulz polar iteration

The polar map is computed iteratively. Given $M$, set $X_0 = M / \lVert M\rVert_F$, then iterate

$$
X_{i+1} \;=\; \tfrac{3}{2} X_i - \tfrac{1}{2} X_i X_i^\top X_i.
$$

If $X_i$ has SVD $X_i = U \Sigma V^\top$, then $X_{i+1} = U\,(\tfrac{3}{2}\Sigma - \tfrac{1}{2}\Sigma^3)\, V^\top$. The polynomial $p(\sigma) = \tfrac{3}{2}\sigma - \tfrac{1}{2}\sigma^3$ has $p(1) = 1$ and $p'(1) = 0$; convergence is cubic in a neighborhood. The Frobenius normalization at $X_0$ ensures all singular values lie in the basin of attraction $(0, \sqrt{3})$. A small fixed number of iterations (typically 5) drives every singular value to within machine precision of one.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Kingma & Ba, *Adam.* arXiv:1412.6980.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization (AdamW).* arXiv:1711.05101.
- Jordan et al., *Muon: An optimizer for hidden layers in neural networks.* 2024. Source of the Newton–Schulz polar iteration and the spectral-cap design philosophy.
- Mirsky, *Symmetric gauge functions and unitarily invariant norms.* Quart. J. Math. 11 (1960), 50–59. Closed form for the Frobenius projection (Lemma 0a) and linear LMO (Lemma 0b) on operator-norm balls.
- Higham, *Functions of Matrices: Theory and Computation.* SIAM 2008, Ch. 8. Cubic convergence of Newton–Schulz.
