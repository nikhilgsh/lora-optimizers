# Fixed-$\kappa$ SSC: the LoRA optimizer this repo runs

This note derives the repo-default LoRA optimizer in a self-contained sequence of Definitions, Lemmas, and Propositions. The structure:

- **§§1–3** set up the LoRA residual program and its whitening.
- **§4** specifies the outer loop as an *anchored block Frank-Wolfe iteration* on the whitened program, and identifies its cross-coupling correction (Lemma 3).
- **§5** specifies the per-block linear oracle, generalizing the polar map to a two-budget oracle parameterized by a stable-rank target $\kappa$, and gives its exact closed form (Proposition 2: hard water-filling).
- **§6** replaces the exact oracle with a smooth surrogate (SSC) that admits an SVD-free implementation.
- **§7** assembles the algorithm; **§8** lists the approximations.

Every symbol used below is defined in this doc.

## 1. Setup

A LoRA adapter has trainable factors $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$, contributing $BA$ to the merged weight $W_0 + BA$. An optimizer step proposes updates $\mathrm dA, \mathrm dB$. The merged-weight change is

$$
\Delta W \;=\; (B + \mathrm dB)(A + \mathrm dA) - BA
\;=\; J \;+\; \mathrm dB\,\mathrm dA,
\qquad
J \;:=\; B\,\mathrm dA + \mathrm dB\,A.
$$

We call $J$ the **tangent**; the bilinear term $\mathrm dB\,\mathrm dA$ is the **chord–tangent gap**. The user-facing magnitude knob is a spectral step size $\eta > 0$. The algorithm targets $\lVert J\rVert_2 \le \eta$ via the per-factor cap:

**Definition 1 (tight-tangent radius).**
$$
\rho \;:=\; \frac{\eta}{\sigma_{\max}(A) + \sigma_{\max}(B)}.
$$

**Lemma 1 (tangent bound).** *If $\lVert\mathrm dA\rVert_2 \le \rho$ and $\lVert\mathrm dB\rVert_2 \le \rho$, then $\lVert J\rVert_2 \le \eta$ and the chord–tangent gap satisfies $\lVert\mathrm dB\,\mathrm dA\rVert_2 \le \rho^2$.*

*Proof.* By submultiplicativity, $\lVert B\,\mathrm dA\rVert_2 + \lVert\mathrm dB\,A\rVert_2 \le \sigma_{\max}(B)\,\rho + \sigma_{\max}(A)\,\rho = (\sigma_{\max}(A) + \sigma_{\max}(B))\,\rho = \eta$, and likewise for the bilinear bound. ∎

So the chord–tangent gap is $O(\rho^2)$ — negligible whenever $\eta$ is small relative to $\sigma_{\max}(A) + \sigma_{\max}(B)$. The algorithm designs against $J$.

## 2. The residual program

Let $g$ denote the gradient of the loss with respect to the merged weight. The local quadratic model of the loss as a function of $J$ is $\langle g, J\rangle + \tfrac{1}{2\eta}\,\lVert J\rVert_F^2$, equivalently (up to a constant) the **residual form** $\tfrac{1}{2\eta}\,\lVert J + \eta\,g\rVert_F^2$. Adding per-block operator-norm caps that enforce $\lVert J\rVert_2 \le \eta$ gives the **residual program**:

$$
\min_{\mathrm dA,\,\mathrm dB}\ \frac{1}{2\eta}\,\bigl\lVert\,B\,\mathrm dA + \mathrm dB\,A + \eta\,g\,\bigr\rVert_F^2
\quad\text{s.t.}\quad
\lVert B\,\mathrm dA\rVert_2,\ \lVert \mathrm dB\,A\rVert_2 \;\text{ bounded.}
\tag{P}
$$

The cap values are placeholders; the per-block solver absorbs them into the rescale of algorithm step 5f. The caps are on the *per-block contributions* $B\,\mathrm dA$ and $\mathrm dB\,A$ (not on $\mathrm dA, \mathrm dB$ directly) so they convert into clean caps in whitened coordinates (§3).

**Remark (Adam).** A real optimizer does not have $g$ on hand. It has Adam-preconditioned factor-space directions $u_A, u_B$ (computed from the running moments of the factor gradients $B^\top g$ and $g\,A^\top$). The algorithm substitutes $u_A, u_B$ for the implicit $B^\top g, g\,A^\top$ that would arise from $g$ in (P). This is a surrogate; it does not change the structure of the program.

## 3. Whitening

**Definition 2 (spectral preconditioners).** Assume $A, B$ have full row/column rank $r$. Define

$$
P_A \;:=\; (A A^\top)^{-1/2}, \qquad P_B \;:=\; (B^\top B)^{-1/2},
$$

both $r \times r$ symmetric positive definite. (In the algorithm of §7 these are damped by $\delta I$ for numerical stability; the derivation below works in the full-rank limit.)

**Definition 3 (whitened factor updates).** Under the invertible substitution
$$
\mathrm dA \;=\; P_B\,Y_A, \qquad \mathrm dB \;=\; Y_B\,P_A,
$$
the variables $(Y_A, Y_B) \in \mathbb{R}^{r \times d_{\text{in}}} \times \mathbb{R}^{d_{\text{out}} \times r}$ are the *whitened factor updates*.

**Lemma 2 (whitened caps).** *$\lVert B\,\mathrm dA\rVert_2 = \lVert Y_A\rVert_2$ and $\lVert \mathrm dB\,A\rVert_2 = \lVert Y_B\rVert_2$.*

*Proof.* Let $B = U_B\,\Sigma_B\,V_B^\top$ be the thin SVD. Then $B\,P_B = U_B\,\Sigma_B\,V_B^\top\,V_B\,\Sigma_B^{-1}\,V_B^\top = U_B\,V_B^\top$, the partial isometry of $B$, which preserves operator norm. So $\lVert B\,P_B\,Y_A\rVert_2 = \lVert Y_A\rVert_2$. Symmetrically for $P_A\,A$. ∎

Under Definition 3, the residual program (P) becomes:

**The whitened residual program.**
$$
\min_{Y_A,\,Y_B}\ \bigl\langle P_B\,u_A,\, Y_A\bigr\rangle + \bigl\langle u_B\,P_A,\, Y_B\bigr\rangle + \frac{1}{2\eta}\,\lVert J\rVert_F^2
\quad\text{s.t.}\quad
\lVert Y_A\rVert_2,\ \lVert Y_B\rVert_2 \;\text{bounded,}
\tag{P'}
$$
with $J = B\,P_B\,Y_A + Y_B\,P_A\,A$ (and $g \rightsquigarrow$ Adam already applied). The linear cost decouples into two block-additive terms; the quadratic still couples through $\lVert J\rVert_F^2$.

---

The next three sections (§§4–6) develop the per-block linear-minimization oracle in the abstract, as properties of a constrained maximum problem over $r \times m$ matrices. They are independent of the LoRA setting and (P'). §7 plugs them into the algorithm.

## 4. The LMO on the operator-norm ball: polar

**Definition 4 (linear-minimization oracle).** For an input matrix $C$ and cap $\tau > 0$,
$$
\mathrm{LMO}(C, \tau) \;:=\; \arg\max_{R}\ \langle C, R\rangle
\quad\text{s.t.}\quad
\lVert R\rVert_2 \le \tau.
$$

**Definition 5 (polar map).** For $C = U\,\mathrm{diag}(s_i)\,V^\top$,
$$
\mathrm{polar}(C) \;:=\; U\,V^\top.
$$

**Proposition 1 (polar solves the LMO).** *For any $C$, $\mathrm{LMO}(C, 1) = \mathrm{polar}(C)$, with optimum value $\sum_i s_i = \lVert C\rVert_*$ (nuclear norm).*

*Proof.* By von Neumann's trace inequality, $\langle C, R\rangle \le \sum_i s_i\,\sigma_i(R)$ with equality iff singular vectors are aligned. Subject to $\sigma_i(R) \le 1$, the maximum is $\sum_i s_i$, attained at $\sigma_i(R) \equiv 1$. ∎

**Observation (wastefulness of polar).** Only the leading singular direction is *required* to saturate the op-norm cap to attain the leading contribution $s_1\,\sigma_1$. Lifting trailing directions to $\sigma_i(R) = 1$ contributes $s_i \ll 1$ to the inner product while adding $1$ to $\lVert R\rVert_F^2$ — a poor signal-to-noise trade. §5 fixes this by adding a second constraint that caps the Frobenius energy.

## 5. The two-budget LMO: fixed-$\kappa$ and hard water-filling

### 5.1 Definition and exact solution

**Definition 6 (fixed-$\kappa$ LMO).** For $\kappa \in (1/r,\,1]$ and input $X$ with $\lVert X\rVert_2 = 1$ (so $s_1 = 1$),
$$
\boxed{\quad
R_\kappa(X) \;\in\; \arg\max_R\ \langle X, R\rangle
\quad\text{s.t.}\quad
\lVert R\rVert_2 \le 1, \quad
\tfrac{1}{r}\,\lVert R\rVert_F^2 \le \kappa.
\quad}
$$
The parameter $\kappa$ is the **normalized stable rank** of the output: $\mathrm{srank}(R)/r = \lVert R\rVert_F^2/(r\,\lVert R\rVert_2^2)$.

**Proposition 2 (hard spectral water-filling).** *Let $X = U\,\mathrm{diag}(s_i)\,V^\top$ with $s_1 = 1$. The unique (up to ties) solution to Definition 6 is*
$$
R_\kappa(X) \;=\; U\,\mathrm{diag}(y_i)\,V^\top,
\qquad
\boxed{\quad y_i \;=\; \min(1,\ \lambda\,s_i),\quad}
$$
*where $\lambda \ge 1$ is the unique multiplier such that $\tfrac{1}{r}\sum_i y_i^2 = \kappa$ when the Frobenius constraint binds, and $\lambda = \infty$ (i.e. $y_i \equiv 1$) otherwise.*

*Proof.* Both constraints are unitarily invariant, so by von Neumann (as in Proposition 1) the maximizer shares singular vectors with $X$ and the problem reduces to
$$
\max_{y_i \in [0,1]}\ \sum_i s_i\,y_i \quad\text{s.t.}\quad \tfrac{1}{r}\sum_i y_i^2 \le \kappa.
$$
KKT conditions split per index: $y_i = 1$ (upper-box active) requires $s_i \ge 1/\lambda$; $y_i \in (0,1)$ (interior) requires $y_i = \lambda\,s_i$. Combining: $y_i = \min(1, \lambda s_i)$. The Frobenius budget is monotone in $\lambda$, pinning $\lambda$ uniquely. ∎

**Corollary 1 (polar limit).** *At $\kappa = 1$, $R_\kappa(X) = \mathrm{polar}(X)$.*

*Proof.* The constraint $\tfrac{1}{r}\lVert R\rVert_F^2 \le 1$ is implied by $\lVert R\rVert_2 \le 1$ and binds only when every $\sigma_i(R) = 1$. Proposition 2 then forces $\lambda \to \infty$, $y_i \equiv 1$. ∎

**Cost.** Evaluating $R_\kappa$ exactly requires the SVD of $X$ and a 1-D root-find for $\lambda$.

**Remark ($\kappa$ is a modeling choice, not an approximation).** Definition 6 *restricts* the feasible set of Definition 4 by adding a Frobenius cap. At $\kappa < 1$, $R_\kappa(X) \ne \mathrm{polar}(X)$ in general, and the achieved objective $\langle X, R_\kappa\rangle$ is *strictly less than* $\lVert X\rVert_*$. So $R_\kappa$ does not approximate polar — it is a different oracle. The choice $\kappa < 1$ is a regularization motivated by the wastefulness observation; it has the same status as choosing a particular norm to constrain. The full ledger in §8 treats $\kappa$ as a modeling choice.

## 6. Smooth surrogate: Soft Spectral Clipping

The hard $\min$ in Proposition 2 has a kink at $s_i = 1/\lambda$, and the SVD per evaluation is expensive on small $r \times r$ inputs. We replace $\min$ with a smooth saturation that admits an SVD-free matrix form.

**Definition 7 (SSC scalar and matrix maps).** For $c > 0$,
$$
h_c(s) \;:=\; \frac{s}{\sqrt{1 + (s/c)^2}},
\qquad
H_c(X) \;:=\; \bigl(I + X X^\top / c^2\bigr)^{-1/2}\,X.
$$
$H_c$ has the same singular vectors as $X$ and singular values $h_c(s_i)$.

**Definition 8 (SSC oracle).** For $\kappa \in (1/r,\,1]$ and input $X$ with $\lVert X\rVert_2 = 1$,
$$
\boxed{\quad
R_\kappa^{\mathrm{SSC}}(X) \;:=\; \frac{H_c(X)}{h_c(1)},
\qquad
c\ \text{chosen so that}\ \ \tfrac{1}{r}\sum_i \Bigl(\frac{h_c(s_i)}{h_c(1)}\Bigr)^2 \;=\; \kappa.
\quad}
$$

**Lemma 3 (SSC properties).** *(i) $\lVert R_\kappa^{\mathrm{SSC}}(X)\rVert_2 = 1$ for any $c$. (ii) The matching equation in Definition 8 has a unique solution $c \in (0, \infty)$ when $\tfrac{1}{r}\sum s_i^2 < \kappa$ (otherwise no $c$ achieves $\kappa$ and we fall back to $c = \infty$, recovering the unprocessed input). (iii) $\tfrac{1}{r}\lVert R_\kappa^{\mathrm{SSC}}(X)\rVert_F^2 = \kappa$ at the chosen $c$.*

*Proof.* (i) The leading singular value of $H_c(X)$ is $h_c(1)$; dividing returns it to $1$. (ii) As $c \to \infty$, $h_c(s_i)/h_c(1) \to s_i$, so the ratio in the matching equation tends to $\tfrac{1}{r}\sum s_i^2$; as $c \to 0^+$, $h_c(s_i)/h_c(1) \to 1$ for $s_i > 0$, so the ratio tends to $\mathrm{rank}(X)/r$. The map is continuous and monotone in this range; existence and uniqueness follow. (iii) By construction. ∎

**Cost.** The $(\cdot)^{-1/2}$ in Definition 7 acts on an $r \times r$ symmetric matrix and is computed by a short Newton–Schulz iteration; no SVD enters the evaluation of $H_c$ itself. However, the matching equation in Definition 8 references the singular values $s_i$ — naively an $r \times r$ `eigvalsh` on $X X^\top$. §6.1 removes this cost.

### 6.1 The one-spike-plus-flat-tail ansatz: closed-form $c$ without `eigvalsh`

The matching equation in Definition 8 requires the full spectrum $\{s_i\}$ in general. We adopt a structural ansatz on that spectrum under which $c$ depends on only the *scalar* $\mu := \lVert X\rVert_F^2 / r$ (the mean squared singular value of $X$).

**The ansatz: one spike plus a flat tail.** Replace the spectrum $(1, s_2, \dots, s_r)$ of $X$ (with $s_1 = 1$ by construction) by
$$
\bigl(1,\ \underbrace{m^{1/2}, \dots, m^{1/2}}_{r-1\ \text{copies}}\bigr),
\qquad
m \;:=\; \frac{r\,\mu - 1}{r - 1}.
$$
This is the unique spectrum of the assumed shape that matches $X$ in both leading singular value and total Frobenius energy: $\sigma_1^2 + (r-1)\,m = r\,\mu = \lVert X\rVert_F^2$. The scalar $m$ is the mean squared singular value *of the tail*.

**The tail target $\kappa_{\text{tail}}$ — why the tail alone?** SSC normalizes by $h_c(1)$ so that the leading output singular value equals $1$ for every $c$ (Lemma 3 (i)). Hence $y_1 = 1$ is *locked* and contributes exactly $1/r$ to $\tfrac{1}{r}\lVert R_\kappa^{\mathrm{SSC}}\rVert_F^2$ regardless of how we choose $c$. The free scalar $c$ acts only on the tail values $y_2, \dots, y_r$. So the $\kappa$ matching equation
$$
\frac{1}{r}\Bigl(\underbrace{1}_{\text{locked}} \;+\; \sum_{i \ge 2} y_i^2\Bigr) \;=\; \kappa
$$
is equivalent to the tail-only condition
$$
\frac{1}{r-1}\sum_{i \ge 2} y_i^2 \;=\; \frac{r\,\kappa - 1}{r - 1} \;=:\; \kappa_{\text{tail}}.
$$
We are not *choosing* to apply $\kappa$ to the tail alone; the spike is fixed by the SSC normalization, and the tail is all $c$ controls. $\kappa_{\text{tail}}$ inherits the same $(1/r, 1]$-style range as $\kappa$ (with $\kappa_{\text{tail}} \to 0$ as $\kappa \to 1/r$ and $\kappa_{\text{tail}} = 1$ at $\kappa = 1$).

**Closed form for $c$.** Under the ansatz, SSC produces output singular values $(1, h_c(\sqrt m)/h_c(1), \dots)$, and the matching equation $\tfrac{1}{r}\sum y_i^2 = \kappa$ reduces to a *scalar* condition on the tail ratio:
$$
\Bigl(\frac{h_c(\sqrt m)}{h_c(1)}\Bigr)^{\!2} \;=\; \kappa_{\text{tail}}.
$$
Substituting $h_c(s)^2 = s^2/(1 + s^2/c^2)$ and solving for $c^2$ yields the closed form used in production:

**Proposition 3 (stable-rank closed form).**
$$
\boxed{\quad
c^2 \;=\; \frac{m\,(1 - \kappa_{\text{tail}})}{\kappa_{\text{tail}} - m},
\qquad
m = \frac{r\,\mu - 1}{r - 1},
\quad
\kappa_{\text{tail}} = \frac{r\,\kappa - 1}{r - 1},
\quad
\mu = \tfrac{1}{r}\lVert X\rVert_F^2.
\quad}
$$
*Valid whenever the ansatz tail $m$ lies strictly below the target tail $\kappa_{\text{tail}}$; in the degenerate regimes ($m \ge \kappa_{\text{tail}}$ or $\kappa_{\text{tail}} \to 1$) the implementation saturates $c$ at $c_{\text{hi}}$ or $c_{\text{lo}}$ respectively.*

*Proof.* From $(h_c(\sqrt m)/h_c(1))^2 = \kappa_{\text{tail}}$:
$$
\frac{m\,/\,(1 + m/c^2)}{1\,/\,(1 + 1/c^2)} \;=\; \kappa_{\text{tail}}
\quad\Longleftrightarrow\quad
\frac{m\,(c^2 + 1)}{c^2 + m} \;=\; \kappa_{\text{tail}}.
$$
Cross-multiplying: $m\,c^2 + m = \kappa_{\text{tail}}\,c^2 + \kappa_{\text{tail}}\,m$, i.e. $c^2(m - \kappa_{\text{tail}}) = m(\kappa_{\text{tail}} - 1)$, giving the boxed expression. ∎

**Cost under the ansatz.** Evaluating $c$ now requires one Frobenius-norm reduction on $X$ (a sum-of-squares), a few scalar operations, and no eigendecomposition. The full SSC oracle is then $H_c(X) / h_c(1)$ via short Newton–Schulz on the $r \times r$ Gram. The entire spectral path is SVD-free and `eigvalsh`-free, and the estimator is stateless — no warm-start or cache of $c$ across iterations.

**Why this works.** The hard $\min$ in Proposition 2 already concentrates Frobenius energy on directions above the water level. The SSC smoothing of §6 further blurs the per-direction values. Together, the output spectrum is far less sensitive to the *individual* small singular values of $X$ than to their *aggregate* energy. The one-spike-plus-flat-tail ansatz captures this: it gets the two structural numbers ($\sigma_{\max}$ and total energy) right and pays only at the level of finer spectral detail, which the smoothing already averages over.

---

## 7. The algorithm: anchored FW on (P')

This section plugs the SSC oracle of §6 into an outer iteration on (P'). The outer iteration is **Frank-Wolfe with anchored linearization**: at each step, the per-block objective is linearized about zero in its own variable (not about the current iterate), and the per-block LMO is applied with vertex-replacement step size.

**Definition 9 (anchored block FW iteration on (P')).** Let $\Phi(Y_A, Y_B)$ denote the objective of (P'). Initialize $Y_A^{(0)} = Y_B^{(0)} = 0$. At iteration $n \ge 0$, compute the anchored partial gradients
$$
\nabla_{Y_A}\Phi\bigl|_{(0,\,Y_B^{(n)})}, \qquad \nabla_{Y_B}\Phi\bigl|_{(Y_A^{(n)},\,0)},
$$
apply the per-block LMO (Definition 4) with the input rescaled to unit op-norm and the SSC oracle (Definition 8) substituted for the polar choice, and update $(Y_A^{(n+1)}, Y_B^{(n+1)})$ by vertex replacement. After $k$ iterations, recover $(\mathrm dA^{(k)}, \mathrm dB^{(k)})$ via Definition 3 and rescale to per-factor radius $\rho$ (Lemma 1).

The anchoring choice drops the self-term $\partial_{Y_A}\lVert B\,P_B\,Y_A\rVert_F^2$ from the per-block linear cost, isolating the cross-coupling term.

**Lemma 4 (cross-coupling correction).** *Under Definition 9, the $A$-block anchored partial gradient at iteration $n+1$, written in factor coordinates, is $P_B\,\tilde u_A^{(n)}$ where*
$$
\boxed{\quad
\tilde u_A^{(n)} \;=\; u_A + \tfrac{1}{\eta}\,B^\top\,\mathrm dB^{(n)}\,A,
\qquad
\tilde u_B^{(n)} \;=\; u_B + \tfrac{1}{\eta}\,B\,\mathrm dA^{(n)}\,A^\top.
\quad}
$$
*The $B$-side is symmetric.*

*Proof.* Direct computation: $\partial_{Y_A}\Phi\bigl|_{(0, Y_B)} = P_B\,u_A + \tfrac{1}{\eta}\,P_B\,B^\top\,Y_B\,P_A\,A = P_B\bigl[u_A + \tfrac{1}{\eta}\,B^\top\,(Y_B\,P_A)\,A\bigr]$, and $Y_B\,P_A = \mathrm dB$. ∎

At $n = 0$, $\mathrm dA^{(0)} = \mathrm dB^{(0)} = 0$, so $\tilde u_A^{(0)} = u_A$ and $\tilde u_B^{(0)} = u_B$.

### Pseudocode

**Hyperparameters.**

- Adam: $\beta_1, \beta_2, \varepsilon$.
- Spectral step size: $\eta$.
- FW iteration count: $k$.
- Preconditioner damping: $\delta$.
- Stable-rank target: $\kappa$.

**Persistent state.**

- Adam moments for $A, B$ and step counter.
- Warm-started top singular vectors of $A, B$ (for power iteration on $\sigma_{\max}$).
- Cached SSC scale $c$ per layer, refreshed on the preconditioner schedule.

**One step on layer pair $(A, B)$.**

---

**Step 1 — Adam.** Form bias-corrected directions $u_A, u_B$.

**Step 2 — Preconditioners** (Definition 2).
$$
P_A \;=\; \bigl(A A^\top + \delta I\bigr)^{-1/2},
\qquad
P_B \;=\; \bigl(B^\top B + \delta I\bigr)^{-1/2}.
$$

**Step 3 — Tight-tangent radius** (Definition 1).
$$
\rho \;=\; \frac{\eta}{\sigma_{\max}(A) + \sigma_{\max}(B)}.
$$

**Step 4 — Initialize.** $\mathrm dA \gets 0$, $\mathrm dB \gets 0$.

---

**Step 5 — Anchored FW loop** (Definition 9). Repeat $k$ times:

*(a) Cross-coupling correction* (Lemma 4).
$$
\tilde u_A \;=\; u_A + \frac{1}{\eta}\, B^\top\, \mathrm dB\, A,
\qquad
\tilde u_B \;=\; u_B + \frac{1}{\eta}\, B\, \mathrm dA\, A^\top.
$$

*(b) Whiten.*
$$
Z_A \;=\; P_B\,\tilde u_A,
\qquad
Z_B \;=\; \tilde u_B\,P_A.
$$

*(c) Operator-norm rescale* (needed so $s_1 = 1$ for the oracle of Definition 8).
$$
\widehat Z_A \;=\; \frac{Z_A}{\lVert Z_A\rVert_2},
\qquad
\widehat Z_B \;=\; \frac{Z_B}{\lVert Z_B\rVert_2}.
$$

*(d) Per-block oracle* (Definition 8).
$$
R_A \;=\; R_\kappa^{\mathrm{SSC}}(\widehat Z_A),
\qquad
R_B \;=\; R_\kappa^{\mathrm{SSC}}(\widehat Z_B).
$$

*(e) Unwhiten* (Definition 3).
$$
D_A \;=\; P_B\,R_A,
\qquad
D_B \;=\; R_B\,P_A.
$$

*(f) Tight-tangent rescale to per-factor radius $\rho$* (Lemma 1).
$$
\mathrm dA \;=\; -\rho\,\frac{D_A}{\lVert D_A\rVert_2},
\qquad
\mathrm dB \;=\; -\rho\,\frac{D_B}{\lVert D_B\rVert_2}.
$$

---

**Step 6 — Apply.** $A \gets A + \mathrm dA$, $B \gets B + \mathrm dB$.

---

At $\kappa = 1$, Corollary 1 collapses step 5d to the polar map; the algorithm reduces to the standard FW–polar LoRA update. At $\kappa < 1$, the oracle concentrates Frobenius energy on the strong directions of $\widehat Z$.

## 8. Modeling choices and algorithmic approximations

The gap between the loss and the run algorithm has two distinct kinds of step. **Modeling choices** change *what we optimize* (substituting a different objective, regularization, or constraint set). **Algorithmic approximations** approximate the *solution* of the chosen object. Listing them separately keeps the two kinds of error from being confused.

**Modeling choices.**

**M1. Local quadratic with operator-norm trust region.** The loss is approximated by a local quadratic in $J$, capped at $\lVert J\rVert_2 \le \eta$. This defines (P). The trust-region semantics are tangent-based; the chord gap is bounded separately (item A1).

**M2. Adam surrogate.** The merged-weight gradient $g$ in (P) is replaced by Adam-preconditioned factor-space directions $u_A, u_B$. This is a per-block surrogate for $B^\top g$ and $g\,A^\top$.

**M3. Stable-rank regularization ($\kappa$).** The per-block LMO is restricted from the operator-norm ball (Definition 4) to its $\kappa$-rank-normalized-Frobenius subset (Definition 6). Per Corollary 1 + the Remark in §5, this is not an approximation of polar but a *restriction* of the feasible set — a regularization motivated by the wastefulness observation. The value $\kappa \in (1/r, 1]$ is a fixed hyperparameter applied uniformly across layers and FW iterations; choosing $\kappa$ globally (rather than per-layer or per-step) is a further simplification.

**Algorithmic approximations.**

**A1. Tangent versus chord.** The program caps $J$; the loss sees $\Delta W = J + \mathrm dB\,\mathrm dA$. Lemma 1 bounds the gap by $\rho^2$ — negligible when $\eta \ll \sigma_{\max}(A) + \sigma_{\max}(B)$.

**A2. Anchored FW with finite $k$.** The coupled $(\mathrm dA, \mathrm dB)$ optimum of (P') under the $\kappa$ regularization is replaced by $k$ iterations of the anchored FW iteration (Definition 9). At $k = 1$ the cross-coupling correction (Lemma 4) is zero. At $k \ge 2$ the iterates pick it up. The anchored linearization choice (self-terms dropped, $\gamma_n = 1$) is itself a non-standard FW variant.

**A3. SSC smooth surrogate.** Proposition 2's hard $\min$ is replaced by $h_c$ (Definition 7). $R_\kappa^{\mathrm{SSC}}$ matches $R_\kappa$ in op-norm exactly and in normalized stable rank by construction (Lemma 3); the two differ near the saturation knee.

**A4. One-spike-plus-flat-tail ansatz for $c$ (§6.1).** The matching equation in Definition 8 is solved under the ansatz that the spectrum of $\widehat Z$ is one spike at $1$ plus a flat tail; this gives the closed form in Proposition 3. Inputs: only $\lVert\widehat Z\rVert_F^2$ and $r$. The estimator is exact when the spectrum truly has this shape; otherwise it matches $\kappa_{\text{tail}}$ at the ansatz tail value $\sqrt m$ rather than at the true tail average of $h_c(\cdot)^2$ (a Jensen gap, vanishing in the flat-tail limit). Stateless — no warm-start or cache.

## 9. Three-line summary

- **What we optimize.** The whitened residual program (P') under the fixed-$\kappa$ stable-rank regularization (M3).
- **Exact.** Anchored block Frank-Wolfe (Definition 9) with the fixed-$\kappa$ LMO (Definition 6), whose closed form is hard spectral water-filling (Proposition 2).
- **Production.** Same FW iteration; the fixed-$\kappa$ LMO is replaced by its smooth SVD-free surrogate $R_\kappa^{\mathrm{SSC}}$ (Definition 8, Lemma 3); $c$ chosen to match $\kappa$ on a refresh schedule.
