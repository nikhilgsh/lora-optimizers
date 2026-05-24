# A polar-product LoRA optimizer — block-coordinate-descent / clip variant

This document derives the block-coordinate-descent (BCD) variant of the polar-product LoRA optimizer: state the variational program, solve it via block-Jacobi iteration with the per-block subproblem solved exactly by clip-prox. The result is **Algorithm 1**, the exact single-program solver of (1) at any user-chosen $(\tau_A, \tau_B)$, specialized to the chain-pinned caps of §7.

A companion document `algorithm_tight_chord_fw.md` derives the Frank-Wolfe / polar variant of the same program (Algorithm 2).

**Reading guide.**

- **Notation reference (below)** — every symbol grounded in one place.
- **§§1–4** — setup, the warmup Muon-style program (§2), the residual program (§3), and its whitened form (§4).
- **§5** — block-coordinate descent: per-block subproblem of (2) is solved exactly by clip-prox (Lemma 0a). No normalization step, no calibration of the Adam direction needed.
- **§§6–7** — closure: saturating-regime hypothesis (H), Lemma 3 (per-block-contribution norms are state-only), chain $\eta \to \rho \to (\tau_A, \tau_B)$.
- **§8** — Algorithm 1 pseudocode and correspondence with the variational program.


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

Intuition: $B = U_B \cdot S_B^{1/2}$ is a polar decomposition (direction $\times$ size), and similarly $A = S_A^{1/2}\,V_A$.

**Whitened quantities:**

| symbol | definition | shape | role |
|---|---|---|---|
| $Y_A := S_B^{1/2}\,\mathrm{d}A$ | whitened $A$-update | $(r, d_{\text{in}})$ | primal variable in program (2) |
| $Y_B := \mathrm{d}B\,S_A^{1/2}$ | whitened $B$-update | $(d_{\text{out}}, r)$ | primal variable in program (2) |
| $X_A := S_B^{-1/2}\,u_A$ | whitened Adam direction (A-side) | $(r, d_{\text{in}})$ | linear-cost dual |
| $X_B := u_B\,S_A^{-1/2}$ | whitened Adam direction (B-side) | $(d_{\text{out}}, r)$ | linear-cost dual |
| $\widetilde{G}_A := S_B^{-1/2}\,g_A$ | whitened gradient on $A$ | $(r, d_{\text{in}})$ | exact dual; replaced by $X_A$ in practice |
| $\widetilde{G}_B := g_B\,S_A^{-1/2}$ | whitened gradient on $B$ | $(d_{\text{out}}, r)$ | exact dual; replaced by $X_B$ in practice |

Why whiten? The constraint $\lVert B\,\mathrm{d}A\rVert_2 \le \tau_A$ becomes simply $\lVert Y_A\rVert_2 \le \tau_A$, because $\lVert B\,\mathrm{d}A\rVert_2 = \lVert U_B\,Y_A\rVert_2 = \lVert Y_A\rVert_2$.

**Magnitude knobs and derived radii:**

| symbol | meaning |
|---|---|
| $\eta$ | user-facing spectral step size — caps $\lVert J\rVert_2$, the operator norm of the tangent |
| $s := \sigma_{\max}(A) + \sigma_{\max}(B)$ | combined factor scale |
| $\rho := \eta/s$ | per-factor radius (cap on $\lVert\mathrm{d}A\rVert_2, \lVert\mathrm{d}B\rVert_2$) |
| $\tau_A := \rho\,\sqrt{\sigma_{\min}(B)^2 + \delta_B}$ | program's cap on $\lVert Y_A\rVert_2$ |
| $\tau_B := \rho\,\sqrt{\sigma_{\min}(A)^2 + \delta_A}$ | program's cap on $\lVert Y_B\rVert_2$ |

The chain $\eta \to \rho \to (\tau_A, \tau_B)$ is derived in §7.

**The tangent and chord:**

| symbol | definition | meaning |
|---|---|---|
| $J := B\,\mathrm{d}A + \mathrm{d}B\,A$ | first-order linearization of the merged-weight change | the algorithm controls $\lVert J\rVert_2$ |
| $\Delta W := (B+\mathrm{d}B)(A+\mathrm{d}A) - BA = J + \mathrm{d}B\,\mathrm{d}A$ | exact chord | what the loss actually sees |

**Block-coordinate iterates and per-block linear costs:**

| symbol | meaning |
|---|---|
| $n = 0, 1, \ldots, k-1$ | inner iteration index inside the per-step block-Jacobi loop ($k$ = total iters) |
| $\mathrm{d}A^{(n)}, \mathrm{d}B^{(n)}$ | factor updates at iterate $n$ (initialized to zero at $n=0$) |
| $Y_A^{(n)}, Y_B^{(n)}$ | whitened iterates |
| $\tilde u_A^{(n)} := u_A + (1/\eta)\,B^\top\,\mathrm dB^{(n)}\,A$ | cross-coupling-corrected Adam direction at iter $n$ (Lemma 1); shape $(r, d_{\text{in}})$ |
| $\tilde u_B^{(n)} := u_B + (1/\eta)\,B\,\mathrm dA^{(n)}\,A^\top$ | cross-coupling-corrected Adam direction at iter $n$ (Lemma 1); shape $(d_{\text{out}}, r)$ |
| $c_A^{(n)} := S_B^{-1/2}\,\tilde u_A^{(n)}$ | whitened linear cost (A-side); shape $(r, d_{\text{in}})$ |
| $c_B^{(n)} := \tilde u_B^{(n)}\,S_A^{-1/2}$ | whitened linear cost (B-side); shape $(d_{\text{out}}, r)$ |

**Matrix functions:**

| symbol | definition |
|---|---|
| $\mathrm{clip}_\tau(M)$ | for $M = U \Sigma V^\top$, returns $U\,\min(\Sigma, \tau)\,V^\top$ (singular values capped at $\tau$, vectors preserved) |
| $\mathrm{polar}(M)$ | for $M = U \Sigma V^\top$, returns $U V^\top$ (singular values $\to 1$) — referenced in §6 only, where Proposition 2 shows clip and polar coincide under (H) |
| $\sigma_{\max}(\cdot), \sigma_{\min}(\cdot)$ | largest / smallest singular value |
| $\lVert\cdot\rVert_2, \lVert\cdot\rVert_F, \lVert\cdot\rVert_*$ | operator, Frobenius, nuclear norm |

**Algorithm labels used below:**

| label | what it is |
|---|---|
| **Algorithm 1** | the exact clip-prox block-Jacobi solver of program (1) at the chain-pinned caps. The subject of this document. |

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

The two factors are independent: no coupling term, no shared constraint. Each subproblem is a linear minimization on the operator-norm ball.

The closed form follows from a result of Mirsky (1960).

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

attains minimum $-\rho\,\lVert u\rVert_*$ (nuclear norm) at $X^\star = -\rho\,\mathrm{polar}(u)$, where $\mathrm{polar}(U\Sigma V^\top) := U V^\top$.

*Proof.* Both proofs reduce to scalar problems on singular values via the same step. Von Neumann's trace inequality (Mirsky 1960) states

$$
\langle X, A\rangle \;\le\; \sum_i \sigma_i(X)\,\sigma_i(A) \tag{$\ast$}
$$

with equality iff $X$ and $A$ share singular vectors with matching ordering of singular values.

*(a)* Let $M = U \Sigma V^\top$. Expand $\lVert X - M\rVert_F^2 = \lVert X\rVert_F^2 - 2\langle X, M\rangle + \lVert M\rVert_F^2$. The middle term is upper-bounded by $(\ast)$, so for any $X$ with prescribed singular values, the objective is minimized by aligning singular vectors with $M$. Hence the minimizer has SVD $X = U\, D\, V^\top$ with $D \succeq 0$ diagonal, and the problem reduces to

$$
\min_{D \succeq 0}\ \sum_i (D_{ii} - \sigma_i(M))^2 \quad\text{s.t.}\quad \max_i D_{ii} \le \tau,
$$

uniquely solved by $D_{ii} = \min(\sigma_i(M), \tau)$.

*(b)* Apply $(\ast)$ to $\langle u, -X\rangle \le \sum_i \sigma_i(u)\,\sigma_i(X)$, i.e. $\langle u, X\rangle \ge -\sum_i \sigma_i(u)\,\sigma_i(X)$, with equality iff $-X$ and $u$ share singular vectors. Under $\sigma_i(X) \le \rho$, the right side is minimized by $\sigma_i(X) = \rho$ on every $i$ with $\sigma_i(u) > 0$. Aligning singular vectors with $u$ gives $X^\star = -\rho\,\mathrm{polar}(u)$. ∎

Applying part (b) to each factor, program (W) yields

$$
\Delta A^\star \;=\; -\rho_A\,\mathrm{polar}(u_A), \qquad \Delta B^\star \;=\; -\rho_B\,\mathrm{polar}(u_B).
$$

This is **Muon** (Jordan et al. 2024) applied independently to each LoRA factor. The radii $\rho_A, \rho_B$ are externally specified hyperparameters with no closed-form derivation from (W) itself.

What program (W) **does not** capture:

- *No coupling.* The two factors share an image: any $(\Delta A, \Delta B)$ producing the same tangent $J$ produces the same first-order change in loss. Program (W) minimizes a sum of two unrelated linear costs.
- *No whitening.* The constraint $\lVert\Delta A\rVert_2 \le \rho_A$ controls the bare factor, not the merged-weight contribution $B\,\Delta A$.
- *No tangent control.* The radii are not connected to the merged-weight change.

The remainder of this document repairs these three deficiencies.

## 3. The residual program

A single optimizer step on a layer pair targets the **whitened residual program**. Let $G := \nabla f(W + BA)$ be the dense gradient with respect to the merged weight. A factor perturbation produces the tangent $J = B\,\Delta A + \Delta B\,A$, and the first-order local model of the loss is

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

The dense gradient $G$ is not what optimizers see directly. Backpropagation produces factor gradients $g_A = B^\top G$, $g_B = G\,A^\top$, and Adam preconditioning yields directions $u_A, u_B$ — these are the inputs to the algorithm. The caps $\tau_A, \tau_B$ are pinned in §7.

## 4. Whitening

Program (1)'s constraint is on $\lVert B\,\Delta A\rVert_2$, not on $\lVert\Delta A\rVert_2$, and the quadratic couples the two factors through $\lVert J\rVert_F^2$. A single linear change of variable diagonalizes both.

**Definition 2 (whitened objects).** Let $S_A := AA^\top$ and $S_B := B^\top B$ ($r \times r$ PSD; here assumed rank $r$, with damping deferred to §8). Define the **column-orthonormal projector** of $B$ and the **row-orthonormal projector** of $A$,

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

This change of variable diagonalizes the operator-norm caps and surfaces $\widetilde G_A, \widetilde G_B$ as the linear-cost duals.

**Adam substitution.** Optimizers do not have $G$ on hand. They have Adam-preconditioned factor directions $u_A, u_B$ in place of $g_A, g_B$. We substitute $\widetilde G_A \rightsquigarrow X_A := S_B^{-1/2}\,u_A$ and $\widetilde G_B \rightsquigarrow X_B := u_B\,S_A^{-1/2}$ directly into (2). No further rescaling of $X_A, X_B$ is needed: the per-block clip-prox of §5 solves a quadratic-plus-linear cost over the op-norm ball, and clip preserves singular values.

## 5. Block-coordinate descent with clip-prox

Program (2) is solved by **block-Jacobi iteration**: at each inner step fix one block and solve the per-block subproblem exactly. Lemma 0a closes the per-block subproblem.

### 5.1 Per-block subproblem

Fix $Y_B = Y_B^{(n)}$. The $A$-subproblem of (2), expanding the quadratic using $\lVert U_B Y_A + Y_B V_A\rVert_F^2 = \lVert Y_A\rVert_F^2 + 2\langle U_B^\top Y_B^{(n)} V_A,\,Y_A\rangle + \text{const}$ (with $U_B^\top U_B = I_r$), is

$$
\min_{Y_A}\ \langle c_A^{(n)},\,Y_A\rangle + \tfrac{1}{2\eta}\,\lVert Y_A\rVert_F^2
\quad\text{s.t.}\quad \lVert Y_A\rVert_2 \le \tau_A,
$$

where the whitened linear cost combines the Adam direction with the cross-coupling correction from the previous iterate:

$$
c_A^{(n)} \;:=\; X_A \;+\; \tfrac{1}{\eta}\,U_B^\top\,Y_B^{(n)}\,V_A.
$$

Completing the square gives $\tfrac{1}{2\eta}\lVert Y_A - (-\eta\,c_A^{(n)})\rVert_F^2 + \text{const}$, so the subproblem is the **Frobenius projection of $-\eta\,c_A^{(n)}$ onto the operator-norm ball of radius $\tau_A$**.

### 5.2 Closed-form solution: clip-prox

By Lemma 0a:

$$
Y_A^{(n+1)} \;=\; \mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A^{(n)}\bigr).
$$

Symmetrically for the $B$-side:

$$
Y_B^{(n+1)} \;=\; \mathrm{clip}_{\tau_B}\bigl(-\eta\,c_B^{(n)}\bigr),
\qquad c_B^{(n)} \;:=\; X_B + \tfrac{1}{\eta}\,U_B\,Y_A^{(n)}\,V_A^\top.
$$


### 5.3 Recovery of original variables — Lemma 1

Translating back via $\mathrm{d}A = S_B^{-1/2}\,Y_A$ and $\mathrm{d}B = Y_B\,S_A^{-1/2}$, the polar input in factor coordinates is

$$
c_A^{(n)} \;=\; S_B^{-1/2}\,\bigl[\,u_A \;+\; \tfrac{1}{\eta}\,B^\top\,\mathrm dB^{(n)}\,A\,\bigr]
\;=\; S_B^{-1/2}\,\tilde u_A^{(n)},
$$

where $\tilde u_A^{(n)} := u_A + (1/\eta)\,B^\top\,\mathrm dB^{(n)}\,A$ is the **cross-coupling-corrected Adam direction**. Symmetrically $c_B^{(n)} = \tilde u_B^{(n)}\,S_A^{-1/2}$ with $\tilde u_B^{(n)} := u_B + (1/\eta)\,B\,\mathrm dA^{(n)}\,A^\top$.

**Lemma 1 (cross-coupling correction).** The block-coordinate linear costs at iterate $n$ have the closed forms

$$
\tilde u_A^{(n)} \;=\; u_A \;+\; \tfrac{1}{\eta}\,B^\top\,\mathrm dB^{(n)}\,A,
\qquad
\tilde u_B^{(n)} \;=\; u_B \;+\; \tfrac{1}{\eta}\,B\,\mathrm dA^{(n)}\,A^\top,
$$

obtained by completing the square on each subproblem of (1) after fixing the other block. At iter $n=0$ with $\mathrm dA^{(0)} = \mathrm dB^{(0)} = 0$ the corrections vanish: $\tilde u_A^{(0)} = u_A$, $\tilde u_B^{(0)} = u_B$. ∎

### 5.4 Block-Jacobi iteration

Initialize $\mathrm dA^{(0)} = \mathrm dB^{(0)} = 0$. For $n = 0, 1, \ldots, k-1$:

1. Compute the cross-coupling-corrected Adam directions $\tilde u_A^{(n)}, \tilde u_B^{(n)}$ via Lemma 1.
2. Form the whitened linear costs $c_A^{(n)} = S_B^{-1/2}\,\tilde u_A^{(n)}$, $c_B^{(n)} = \tilde u_B^{(n)}\,S_A^{-1/2}$.
3. Apply clip-prox per block:
   $$
   Y_A^{(n+1)} \;=\; \mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A^{(n)}\bigr),
   \qquad
   Y_B^{(n+1)} \;=\; \mathrm{clip}_{\tau_B}\bigl(-\eta\,c_B^{(n)}\bigr).
   $$
4. Unwhiten: $\mathrm dA^{(n+1)} = S_B^{-1/2}\,Y_A^{(n+1)}$, $\mathrm dB^{(n+1)} = Y_B^{(n+1)}\,S_A^{-1/2}$.

After $k$ inner iters, apply: $A \leftarrow A + \mathrm dA^{(k)}$, $B \leftarrow B + \mathrm dB^{(k)}$.

At $k = 1$ this reduces to per-block Muon with whitening: clip-prox of the whitened Adam direction at chain-pinned magnitude. For $k \ge 2$ the cross-coupling correction (Lemma 1) couples the blocks.

## 6. Saturating regime: per-block-contribution norms are state-only

Algorithm 1 (§8) is the block-Jacobi iteration of §5 at the chain-pinned caps (7) of §7. Under a saturating-regime hypothesis, the per-block-contribution norm $\lVert\mathrm dA^{(n+1)}\rVert_2$ is a simple, state-only function of $\tau_A$ (independent of the inner iter $n$).

**Lemma 3 (factor-norm collapse, clip-prox form).** Under hypothesis (H) below, the per-block update operator norms are state-only:

$$
\lVert\mathrm dA^{(n+1)}\rVert_2 \;=\; \tau_A\,\bigl(\sigma_{\min}(B)^2 + \delta_B\bigr)^{-1/2},
\qquad
\lVert\mathrm dB^{(n+1)}\rVert_2 \;=\; \tau_B\,\bigl(\sigma_{\min}(A)^2 + \delta_A\bigr)^{-1/2}.
$$

*Proof under (H).* Under (H), $\mathrm{clip}_{\tau_A}(-\eta\,c_A^{(n)})$ flattens every singular value of $-\eta\,c_A^{(n)}$ to $\tau_A$, giving $Y_A^{(n+1)} = \tau_A\,U V^\top$ where $U V^\top$ is the polar factor of $-c_A^{(n)}$. Then $\mathrm dA^{(n+1)} = S_B^{-1/2}\,Y_A^{(n+1)} = \tau_A\,S_B^{-1/2}\,U V^\top$ has operator norm

$$
\lVert\mathrm dA^{(n+1)}\rVert_2 \;=\; \tau_A\,\lVert S_B^{-1/2}\,U V^\top\rVert_2 \;=\; \tau_A\,\lVert S_B^{-1/2}\rVert_2,
$$

since $U V^\top$ is a partial isometry: $U V^\top (U V^\top)^\top = U U^\top$ projects onto the column space of $U$, so $\sigma_{\max}(S_B^{-1/2}\,U V^\top) = \sigma_{\max}(S_B^{-1/2})$. Symmetric for the $B$-side. ∎

**Saturating-regime hypothesis (H).** With $c_A^{(n)} := S_B^{-1/2}\,\tilde u_A^{(n)}$ the polar input in factor coordinates,

$$
\tau_A \;\le\; \eta\,\sigma_{\min}\!\bigl(c_A^{(n)}\bigr), \qquad
\tau_B \;\le\; \eta\,\sigma_{\min}\!\bigl(c_B^{(n)}\bigr)
\qquad\text{for } n = 1,\ldots,k.
\tag{H}
$$

This is a hypothesis on the block-Jacobi trajectory, not on the initial inputs alone — both sides of each inequality move as $n$ advances.

**Proposition 2 (clip $=$ polar under H).** If (H) holds at iterate $n$, then

$$
\mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A^{(n)}\bigr) \;=\; -\tau_A\,\mathrm{polar}\bigl(c_A^{(n)}\bigr).
$$

*Proof.* Under (H), $\mathrm{clip}_{\tau_A}$ flattens every singular value of $-\eta\,c_A^{(n)}$ to $\tau_A$ and preserves singular vectors, giving $\tau_A\,U V^\top$. Polar is invariant under positive scaling and odd under negation. ∎

Substituting polar for clip yields the same update under (H); this is the connection to the FW variant in the companion document.

**When (H) fails.** Outside the saturating regime — when some singular direction of $c_A^{(n)}$ falls below $\tau_A/\eta$ — clip leaves that singular value unchanged at $\eta\,\sigma_i(c_A^{(n)}) < \tau_A$. The polar coincidence breaks (polar would have lifted that value to $\tau_A$, mis-spending the budget on a direction with weak signal). Lemma 3's identity $\lVert\mathrm dA^{(n+1)}\rVert_2 = \tau_A\,\lVert S_B^{-1/2}\rVert_2$ becomes a *bound* rather than equality.

## 7. The chain: $\eta \to \rho \to (\tau_A, \tau_B)$

§§3–5 left the program's caps $(\tau_A, \tau_B)$ as free hyperparameters. §6 showed that the per-block-contribution norm is a state-only function of $\tau_A$ under (H). We now derive $(\tau_A, \tau_B)$ from a single user-facing magnitude hyperparameter — the spectral step size $\eta$ — via a chain of tight implications:

$$
\underbrace{\lVert J\rVert_2 \le \eta}_{\text{user-facing (tangent)}}
\;\overset{\text{Prop 3}}{\Longleftarrow}\;
\underbrace{\lVert\Delta A\rVert_2, \lVert\Delta B\rVert_2 \le \rho}_{\text{per-factor}}
\;\overset{\text{Lemma 3 under (H)}}{\Longleftarrow}\;
\underbrace{\lVert S_B^{1/2}\Delta A\rVert_2 \le \tau_A,\ \lVert\Delta B\, S_A^{1/2}\rVert_2 \le \tau_B}_{\text{program's caps}}
$$

### Step 1: $\eta \to \rho$ (Proposition 3)

Setting $\lVert\Delta A\rVert_2 = \lVert\Delta B\rVert_2 = \rho$, submultiplicativity gives

$$
\lVert J\rVert_2 \;=\; \lVert B\,\Delta A + \Delta B\, A\rVert_2
\;\le\; \sigma_{\max}(B)\,\rho + \sigma_{\max}(A)\,\rho \;=\; s\rho.
$$

**Proposition 3 (tight-tangent radius).** The largest $\rho \ge 0$ with $s\rho \le \eta$ is

$$
\rho \;=\; \frac{\eta}{s}.
\tag{6}
$$

*Proof.* Linear; $s\rho = \eta$ at the boundary. ∎

The chord $\Delta W = J + \Delta B\,\Delta A$ then satisfies $\lVert\Delta W\rVert_2 \le \eta + \rho^2 = \eta\,(1 + \eta/s^2)$. The bilinear correction $\eta/s^2$ is small whenever $\eta \ll s^2$ (see Appendix A for the regime characterization).

### Step 2: $\rho \to (\tau_A, \tau_B)$ (Lemma 3 under (H))

The program's caps are on the per-block-contribution norms $\lVert S_B^{1/2}\Delta A\rVert_2 \le \tau_A$ and $\lVert\Delta B\, S_A^{1/2}\rVert_2 \le \tau_B$. We derive $(\tau_A, \tau_B)$ such that solving (1) at those caps yields iterates with $\lVert\Delta A\rVert_2 \le \rho$ and $\lVert\Delta B\rVert_2 \le \rho$.

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

Program (1) with state-fixed caps (7) is what Algorithm 1 solves. The user-facing tangent cap and the intermediate per-factor cap are *consequences* of the chain — properties guaranteed by the iterates Algorithm 1 produces, not separately-enforced constraints. The applied update at outer iteration $n$ is

$$
\boxed{\quad
\mathrm dA^{(n)} \;=\; S_B^{-1/2}\,\mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A^{(n-1)}\bigr),
\qquad
\mathrm dB^{(n)} \;=\; \mathrm{clip}_{\tau_B}\bigl(-\eta\,c_B^{(n-1)}\bigr)\,S_A^{-1/2}.
\quad}
\tag{8}
$$

Under (H), this update has factor norm exactly $\rho$ (Lemma 3 + chain Step 2). Outside (H), the clip preserves small singular directions of $c_A^{(n-1)}$ rather than lifting them to $\tau_A$ — the iterate's factor norm can fall below $\rho$ on those directions, which is the correct behavior (we don't spend the budget on directions with weak signal).

### Sufficient condition for (H)

Combining (7) with the iterate-wise statement of (H):

$$
\rho \;\le\; \eta\,\min\!\Bigl(\sigma_{\min}\!\bigl(c_A^{(n)}\bigr)\,\lVert S_B^{-1/2}\rVert_2,\ \ \sigma_{\min}\!\bigl(c_B^{(n)}\bigr)\,\lVert S_A^{-1/2}\rVert_2\Bigr)
\quad\text{for } n = 1, \ldots, k.
$$

When this holds along the trajectory, Algorithm 1 is the exact solver of program (1) with state-fixed caps (7).

### Simpler variant: direct cap on per-block tangent contributions (no $\rho$)

The chain $\eta \to \rho \to (\tau_A, \tau_B)$ goes through the per-factor radius $\rho$ in two steps. A more direct route uses the triangle inequality on $J$:

$$
\lVert J\rVert_2 \;\le\; \lVert B\,\Delta A\rVert_2 + \lVert\Delta B\, A\rVert_2 \;\le\; \tau_A + \tau_B.
$$

Setting $\tau_A + \tau_B \le \eta$ implies the tangent cap directly. The symmetric choice $\tau_A = \tau_B = \eta/2$ drops the dependence on $\sigma_{\max}(A), \sigma_{\max}(B)$ entirely — no power iteration needed. This is the "naive step size" companion to Algorithm 1: same direction, simpler magnitude. The $\rho$-routed pinning uses state-dependent $\sigma_{\max}$ information; this variant does not.

## 8. Algorithm 1 (the clip-prox variant)

Algorithm 1 is the block-Jacobi iteration of §5 on the whitened residual program (2) at the chain-pinned caps (7) of §7, with the clip-prox per-block solver. Under hypothesis (H) it is the exact solver of (1) at those caps; outside (H), see §6.

**Hyperparameters:** Adam $\beta_1, \beta_2, \varepsilon$; block-Jacobi sweep count $k$; preconditioner regularizer $\varepsilon_{\text{rel}}$; spectral step size $\eta$.

**Persistent state:** Adam moments $(m_A, v_A, m_B, v_B)$; step counter $t$; warm-started top singular vectors for $A, B$ (for power iteration).

**Algorithm 1.** One step on layer pair $(A, B)$:

1. **Spectral preconditioners** (refreshed periodically; both $r \times r$):
   $$
   S_A^{-1/2} = (A A^\top + \delta_A I)^{-1/2},
   \qquad
   S_B^{-1/2} = (B^\top B + \delta_B I)^{-1/2}.
    $$
   With the scale-invariant parameterization
   $$
   \delta_A \;=\; \varepsilon_{\text{rel}}\,\sigma_{\max}(A A^\top), \qquad \delta_B \;=\; \varepsilon_{\text{rel}}\,\sigma_{\max}(B^\top B).
   $$
   Eigenvalues below $\varepsilon_{\text{rel}} \cdot \sigma_{\max}$ are floored at the damping threshold.

2. **Adam preconditioning.** Update first and second moments and form bias-corrected directions $u_A, u_B$ in the standard way. No further rescaling (cf. §4).

3. **Top singular values** via warm-started power iteration:
   $$
   \sigma_A \gets \sigma_{\max}(A), \qquad \sigma_B \gets \sigma_{\max}(B).
   $$

4. **Tight-tangent radius and program's caps:**
   $$
   s \gets \sigma_A + \sigma_B, \qquad
   \rho \gets \eta / s, \qquad
   \tau_A \gets \rho\,\sqrt{\sigma_{\min}(B)^2 + \delta_B}, \qquad
   \tau_B \gets \rho\,\sqrt{\sigma_{\min}(A)^2 + \delta_A}.
   $$

5. **Block-Jacobi cross-coupling loop.** Initialize $\mathrm dA = \mathrm dB = 0$. For $n = 1, \ldots, k$:

   - **Cross-coupling correction** (Lemma 1):
     $$
     \tilde u_A \;=\; u_A + \tfrac{1}{\eta}\, B^\top\, \mathrm dB\, A,
     \qquad
     \tilde u_B \;=\; u_B + \tfrac{1}{\eta}\, B\, \mathrm dA\, A^\top.
     $$

   - **Whitened linear cost:**
     $$
     c_A \;=\; S_B^{-1/2}\,\tilde u_A,
     \qquad
     c_B \;=\; \tilde u_B\,S_A^{-1/2}.
     $$

   - **Per-block clip-prox solve** (each requires one thin SVD):
     $$
     Y_A \;=\; \mathrm{clip}_{\tau_A}\bigl(-\eta\,c_A\bigr),
     \qquad
     Y_B \;=\; \mathrm{clip}_{\tau_B}\bigl(-\eta\,c_B\bigr).
     $$
     See Appendix B for the SVD computation; $Y_A$ is shape $(r, d_{\text{in}})$ so the thin SVD costs $O(r^2 d_{\text{in}})$.

   - **Unwhiten:**
     $$
     \mathrm dA \;=\; S_B^{-1/2}\,Y_A,
     \qquad
     \mathrm dB \;=\; Y_B\,S_A^{-1/2}.
     $$

6. **Apply.** $A \gets A + \mathrm dA$, $\quad B \gets B + \mathrm dB$.

The line-by-line correspondence with the variational program:

| Algorithm 1 step | Variational source |
|---|---|
| Adam preconditioning ($u_A, u_B$) | Surrogate for $G$ in (1) |
| Spectral preconditioners ($S_A^{-1/2}, S_B^{-1/2}$) | Whitening of Definition 2; forced by Lemma 2 |
| Tight-tangent radius ($\rho$) | Tangent constraint $\lVert J\rVert_2 \le \eta$ + Proposition 3 |
| Program's caps ($\tau_A, \tau_B$) | Chain Step 2 + Lemma 3 under (H); equation (7) |
| Cross-coupling correction ($\tilde u_A, \tilde u_B$) | Lemma 1 — exact per-block linear cost after completing the square on (1) |
| Per-block clip-prox solve ($Y_A, Y_B$) | Frobenius prox on operator-norm ball, Lemma 0a |
| Block-Jacobi outer loop | BCD on (2) — §5 |

**Cost note.** The thin SVD per inner iter is the cost driver of Algorithm 1: $O(r^2 d)$ per pair per inner iter, where $d \in \{d_{\text{in}}, d_{\text{out}}\}$. The FW polar variant in the companion document uses the Newton–Schulz iteration instead, replacing the SVD with $O(r d)$ matmuls.

## Appendix A. Properties of the tight-tangent radius and chord-vs-tangent gap

$$
\boxed{\quad \rho \;=\; \eta / s \quad}
$$

**Monotonicity.** $\rho$ increases in $\eta$, decreases in $s = \sigma_{\max}(A) + \sigma_{\max}(B)$. Larger factor singular values $\Rightarrow$ smaller step. The rule self-attenuates as $A, B$ grow.

**Boundary.** When $\rho = \eta/s$, the tangent bound binds with equality: $s\rho = \eta$.

**Chord-vs-tangent gap.** At $\rho = \eta/s$ the chord satisfies $\lVert\Delta W\rVert_2 \le \eta + \rho^2 = \eta\,(1 + \eta/s^2)$. The dimensionless quantity $\eta/s^2$ controls when chord and tangent diverge:

- $\eta/s^2 \ll 1$: bilinear term negligible; chord $\approx$ tangent.
- $\eta/s^2 \sim 1$: the program's tangent semantic is no longer a faithful proxy for the chord; the derivation in §7 would need to be revisited.

**Quadratic form.** A stricter derivation caps the chord $\lVert\Delta W\rVert_2 \le \eta$ directly:

$$
s\rho + \rho^2 \le \eta \quad\Longrightarrow\quad \rho = \tfrac{1}{2}\bigl(-s + \sqrt{s^2 + 4\eta}\bigr).
$$

Limits: $\rho \to \eta/s$ as $\eta \ll s^2$, $\rho \to \sqrt{\eta} - s/2$ as $\eta \gg s^2$.

## Appendix B. Clip-prox via thin SVD

The clip operator $\mathrm{clip}_\tau(M)$ requires the singular values and vectors of $M$. For $M \in \mathbb{R}^{r \times d}$ with $r \le d$, the thin SVD $M = U\,\Sigma\,V^\top$ with $U \in \mathbb{R}^{r \times r}$, $\Sigma \in \mathbb{R}^{r \times r}$, $V \in \mathbb{R}^{r \times d}$ is computed in $O(r^2 d)$ floating-point operations. Then

$$
\mathrm{clip}_\tau(M) \;=\; U\,\min(\Sigma, \tau)\,V^\top,
$$

where $\min(\Sigma, \tau)$ caps every diagonal entry of $\Sigma$ at $\tau$.

In Algorithm 1 the clip is applied to $-\eta\,c_A^{(n-1)} \in \mathbb{R}^{r \times d_{\text{in}}}$ on the $A$-side and to $-\eta\,c_B^{(n-1)} \in \mathbb{R}^{d_{\text{out}} \times r}$ on the $B$-side; both are tall-skinny relative to $r$, so the thin SVD is the correct cost model.

Alternatives to a direct thin SVD include:

- A truncated SVD when only the leading singular directions are above $\tau$ (saturating regime). Power iteration with deflation finds them one at a time at $O(r d)$ per iter; the iteration terminates when the next candidate singular value falls below $\tau$. Cost depends on the actual number of saturating directions.
- Compute the $r \times r$ Gram matrix $M\,M^\top$, eigendecompose to get $\Sigma^2$ and $U$, then recover $V$ from $V = \Sigma^{-1}\,U^\top\,M$ on the non-zero-singular-value subspace. Cost: one matmul to form the Gram, one $r \times r$ eigendecomposition, one matmul to recover $V$.

The Newton–Schulz polar iteration used by the FW variant in the companion document does not produce singular values; it lifts every singular value to $1$. There is no fast Newton–Schulz analog for the clip operator, because clip needs the singular *values* to test against $\tau$.

## Appendix C. Shape control: a second degree of freedom in the polar map

The chord-tight derivation in §§1–8 fixes the polar map's *magnitude* via the chain $\eta \to \rho \to (\tau_A, \tau_B)$. Under (H), the magnitude rule fully determines the update: $\mathrm dA = -\tau_A\,S_B^{-1/2}\,\mathrm{polar}(c_A)$, with every singular direction of $\mathrm{polar}(c_A)$ treated identically. This appendix steps back from (H) and asks what additional freedom is available in the polar-side map when the saturating assumption is relaxed.

### C.1. The shape degree of freedom

After whitening and pre-rescaling so that $\sigma_{\max}(X) = 1$ (where $X = S_B^{-1/2}\,u_A$ is the whitened Adam direction), any map $\phi: \mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$ acting on the singular values of $X$ defines a candidate polar-side direction

$$
\Phi(X) \;=\; U\,\mathrm{diag}\bigl(\phi(\sigma_i)\bigr)\,V^\top,
$$

followed by a scalar magnitude rule that lifts $\lVert\Phi(X)\rVert_{op}$ to the chord-tight budget $\rho$. Two scalars are then exposed independently:

- **Operator norm.** $\lVert\Phi(X)\rVert_{op} = \phi(1)$ before rescale, $\rho$ after. Always fixed by the magnitude rule.
- **Frobenius norm.** $\lVert\Phi(X)\rVert_F^2 = \sum_i \phi(\sigma_i)^2$, equivalently $\rho^2\,\mathrm{srank}(\Phi(X))$ after rescale. *Not* fixed by the magnitude rule; depends on $\phi$ and on the spectrum of $X$.

The chord-tight derivation's choice — exact polar — corresponds to $\phi(\sigma) = 1$, so $\lVert\Phi\rVert_F^2 = r$ after rescale and every direction is lifted to the same magnitude. Other choices of $\phi$ preserve the operator-norm budget $\rho$ but redistribute Frobenius energy across directions.

### C.2. A two-budget program

A natural way to expose the shape choice as a structured optimization is to add an explicit Frobenius constraint to the per-block program of §5.1:

$$
\Delta A \;\in\; \mathrm{argmin}_{Y}\;\bigl\langle Y, -\eta\,c_A\bigr\rangle
\quad\text{s.t.}\quad
\lVert Y\rVert_{op} \le \rho,
\;\;
\lVert Y\rVert_F \le T.
\tag{C1}
$$

The op-norm cap $\rho$ comes from the chord-tight magnitude rule and scales with $\eta$. The F-norm cap $T$ is a separate parameter, introduced as a second budget on the update's total spectral energy. Setting $T = \rho\,\sqrt{r}$ recovers exact polar (the F-constraint becomes inactive); setting $T < \rho\,\sqrt{r}$ deviates the update toward a more concentrated singular spectrum.

By von-Neumann's trace inequality (Lemma 0b), the LMO aligns $Y$'s singular vectors with those of $X$. Substituting $Y = U\,\mathrm{diag}(y_i)\,V^\top$ reduces (C1) to a scalar problem on the singular values with closed form

$$
y_i \;=\; \min\!\bigl(\rho,\ \lambda\,\sigma_i\bigr),
\qquad
\lambda \text{ chosen so } \sum_i y_i^2 \;=\; \min\!\bigl(T^2,\ r\,\rho^2\bigr).
\tag{C2}
$$

This is **spectral water-filling**: $\lambda$ is the dual variable for the F-norm constraint. Directions with strong signal $\sigma_i$ saturate at the op-norm ceiling $\rho$ first; once their share of the F-budget is used, the remainder is distributed across weaker directions proportional to their $\sigma_i$. Three regimes are visible in (C2):

- **F-constraint inactive** ($T \ge \rho\,\sqrt{r}$): every $y_i = \rho$, recovering exact polar.
- **F-constraint binding everywhere** ($T \le \rho\,\sqrt{\mathrm{srank}(X)}$): no $y_i$ reaches $\rho$; $y_i = \lambda\,\sigma_i$ with $\lambda = T/\lVert X\rVert_F$ — a pure F-rescaled identity.
- **Intermediate**: top directions saturate at $\rho$; tail directions are F-rescaled.

The intermediate regime is the substantive one: it interpolates smoothly between exact polar and the unprocessed direction, with the interpolation governed by $T$.

Cost of (C2) depends on the polar backend the SSC step is composed into:

- *Clip-prox path (Algorithm 1).* The thin SVD of $X$ is already paid by clip-prox itself, so (C2) adds only the $O(r)$ bisection for $\lambda$ — essentially free.
- *Newton–Schulz polar path (Algorithm 2′, the canonical implementation).* The pipeline does **not** compute an SVD; it uses gram-form NS on $X$ (`algorithm_clean_implementation.md` §5). To realize (C2) here, one must form the $r \times r$ gram $G_X := X X^\top$ (one $r \times d$ matmul) and obtain its eigenvalues. Two cost regimes apply:
  - **Per-polar-call eigvalsh is expensive.** `torch.linalg.eigvalsh` is launch-bound on small $r \times r$ matrices on Blackwell — `walltime_profile.md` records `eigh` whitening at ~700 ms/step opt vs Higham's ~160 ms at r=128, which is why this project moved the whitening preconditioner off eigh. Calling `eigvalsh` once per polar call per pair would re-incur that gap.
  - **Amortized eigvalsh is cheap.** If $c$ is refreshed on a schedule (matching `precond_refresh_every`, i.e. once per N steps and held constant between refreshes), the eigvalsh cost is amortized and falls back to the same FLOP class as the existing Higham refresh. This is the right cost model for the state-dependent SSC use case in §C.5, where we expect $c$ to track slow changes in the workload's spectrum, not jitter step-to-step.

  In either regime, the SSC application itself still uses `_ssc_misr_batched` with the solved $c$; the SVD never enters Algorithm 2′.

### C.3. A smooth implementation: Soft Spectral Clipping

The hard $\min$ in (C2) has a kink at $\sigma_i = \rho/\lambda$. Soft Spectral Clipping (Bertelli et al., *SPECTRA*, arXiv:2603.14315) replaces it with a smooth saturation

$$
H_c(X) \;=\; (I + X X^\top / c^2)^{-1/2}\,X
\;=\;
U\,\mathrm{diag}\!\bigl(h_c(\sigma_i)\bigr)\,V^\top,
\qquad
h_c(\sigma) := \frac{\sigma}{\sqrt{1 + (\sigma/c)^2}}.
$$

$h_c$ is the gradient of the convex potential $c^2\,\sqrt{1 + \sigma^2/c^2}$, so $H_c$ is a smooth proximal-like operator. For $\sigma \ll c$, $h_c(\sigma) \approx \sigma$ (passthrough); for $\sigma \gg c$, $h_c(\sigma) \approx c$ (saturation). The crossover scale $c$ plays the role of the hard threshold $\rho/\lambda$ in (C2).

Composing $H_c$ with the chord-tight magnitude rule gives

$$
\mathrm dA \;=\; \beta\,H_c(X),
\qquad
\beta \;=\; \rho / h_c(1),
$$

so that $\lVert\mathrm dA\rVert_{op} = \rho$ matches the chord-tight budget while the *shape* of the output spectrum is set by $c$. Two implementation routes are available:

- **`_ssc_misr_batched` (production, no SVD).** A matmul-only Newton–Schulz-style routine based on the matrix-inverse-square-root iteration of SPECTRA Algorithm 2, applied to the $r \times r$ gram $I + X X^\top / c^2$. Cubic convergence near the saturation regime; assumes pre-rescaled input ($\sigma_{\max}(X) \approx 1$), which holds after §2.5. This is what `_chord_tight_clean_polar_pipeline` calls when `polar_method="ssc"` — Algorithm 2′ stays SVD-free end-to-end. Cost is one $r \times r$ NS recurrence per polar call, same FLOP class as gram-NS polar.
- **`_ssc_svd` (reference only, do not put in the production path).** Thin SVD of $X$ + apply $h_c$ to the singular values directly. Used to validate MISR convergence on snapshots; introducing it into Algorithm 2′ would replace the gram-NS polar's $O(r^2 D)$ matmul work with a fresh $O(r^2 D)$ SVD per call, which is a significant wall-time regression (SVDs are launch-heavy and not on tensor cores), not a free composition.

### C.4. Relation to fixed-$K$ Newton–Schulz

Fixed-iteration NS at iteration count $K$ also defines a shape map: $\phi_K(\sigma) \to 1$ as $K \to \infty$ for $\sigma$ in the basin of attraction, with finite-$K$ output spectra that interpolate between the input and the polar limit. At matched $(\sigma_{\max}, \lVert\cdot\rVert_F)$, $H_c$ and $\mathrm{NS}_K$ define different but related families:

- **NS** lifts every singular value toward $1$; the F-norm growth is controlled by how aggressively each $\sigma_i$ is moved upward at iteration count $K$.
- **SSC** caps singular values at $\approx c$ from above while leaving small singular values approximately untouched; the F-norm growth is controlled by where the saturation knee sits.

Both families have a single shape parameter ($K$ or $c$). At any given input spectrum, one can find $(K, c)$ pairs that produce updates with matched op-norm and matched Frobenius norm; the residual difference is in the *shape* of the output spectrum (NS distributes lift across all directions; SSC saturates the top and preserves the tail). Whether shape at matched norms is empirically relevant is an open question that this appendix does not resolve.

### C.5. Two interpretations of $c$

The shape parameter $c$ admits two distinct interpretations, with different implications for how it should be set.

**$c$ tied to $\eta$ (state-dependent).** In the water-filling formulation (C2), $\lambda$ — and therefore the effective crossover scale $\rho/\lambda$ — depends on $\rho$ and on the spectrum of $X$. In the SSC analog, this corresponds to choosing $c$ per step by solving a monotone equation that ties it to $\rho$ via the F-budget $T$ (equivalently, via a rank-normalized energy target $\kappa := \tfrac{1}{r}\sum_i (h_c(s_i)/h_c(1))^2$ with $s_i = \sigma_i(X)/\sigma_{\max}(X)$). The user-facing parameter is $T$ (or $\kappa$); $c$ is computed as a state-dependent function. Under this interpretation, $c$ scales with $\eta$ and the shape adapts to the magnitude. The realization cost is the spectrum of $X$ (per the gram-eigvalsh route in C.2 — amortize via a refresh schedule to keep this off the per-step hot path; per-call `eigvalsh` is launch-bound at production $r$ on Blackwell, see `walltime_profile.md`) plus a scalar bisection in $\log c$. The SSC application itself stays on the existing MISR path with the solved $c$, so Algorithm 2′ remains SVD-free.

**$c$ fixed (η-independent).** Alternatively, $c$ can be exposed directly as a hyperparameter, in the pre-rescaled coordinates where $\sigma_{\max}(X) = 1$ holds by construction. In this frame, $c$ no longer references $\eta$ or $\rho$: it sets a fixed knee location on the workload's intrinsic singular-value distribution. Directions with $\sigma_i > c$ are saturated toward a common magnitude (signal-like treatment); directions with $\sigma_i < c$ are left near their natural amplitude (tail-like treatment). The magnitude rule still scales the output by $\rho$ via $\beta$, so the op-norm budget remains tied to $\eta$. What is η-independent under this interpretation is the *partition* of $X$'s spectrum into "treated as signal" vs "left alone", which is read as a property of the workload's pre-rescaled spectrum rather than of the user's step-size choice.

The two interpretations are not equivalent. The first answers "how should the shape adapt as $\eta$ varies?" — the answer being that the F-budget $T$ stays constant and $c$ tracks $\rho$. The second answers "where is the knee of the workload's spectrum?" — the answer being a number determined by the model, dataset, and layer geometry, not by $\eta$. The two cases coincide only if the workload's pre-rescaled spectrum is itself $\eta$-independent over the relevant range and the relevant value of $T$ corresponds to a $c$ that does not vary much across $\eta$.

### C.6. Open questions

1. **Choice of dual norm.** F-norm is one natural pair with the op-norm cap; alternatives include nuclear norm $\lVert\cdot\rVert_*$ (rank-aware) or general Schatten-$p$. F-norm is the cheapest to evaluate and admits the water-filling closed form, but it is not the unique principled choice.

2. **Absolute vs relative $T$.** Whether the F-budget should live in absolute units or be scaled by a state-dependent quantity ($\lVert A\rVert_F$, $\rho\,\sqrt{r}$, etc.) bears on transferability across model sizes and ranks. The choice influences whether a single $T$ generalizes, or whether the user-facing knob should be a unitless ratio.

3. **Hard vs smooth saturation.** Whether the smoothness of $H_c$ around the saturation knee matters in practice, beyond what hard water-filling (C2) provides, is open. Both maps share the same singular-vector alignment and the same op-norm cap; they differ only in the local behavior near the knee.

4. **Interaction with the Picard loop.** The shape map applies per polar step inside the block-Jacobi / Frank–Wolfe loop. At $k \ge 2$, the shape choice on the current sub-iterate feeds into the cross-coupling correction of the next; whether the loop's fixed-point properties depend on the choice of $\phi$ (beyond magnitude) is worth working out.

5. **Empirical operating curves.** The appendix is silent on which choice of $\phi$ — exact polar, fixed-$K$ NS, water-filling (C2), or fixed-$c$ SSC — performs best on the target workload, and whether the answer is sensitive to rank, optimizer hyperparameters, or learning-rate placement within a sweep. These are empirical questions; the role of this appendix is to lay out the structural options.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Kingma & Ba, *Adam.* arXiv:1412.6980.
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization (AdamW).* arXiv:1711.05101.
- Jordan et al., *Muon: An optimizer for hidden layers in neural networks.* 2024. Source of the spectral-cap design philosophy referenced in §2.
- Mirsky, *Symmetric gauge functions and unitarily invariant norms.* Quart. J. Math. 11 (1960), 50–59. Closed form for the Frobenius projection (Lemma 0a) and linear LMO (Lemma 0b) on operator-norm balls.
