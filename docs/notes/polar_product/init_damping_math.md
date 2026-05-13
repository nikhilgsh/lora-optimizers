# A μA-style note on chord-tight LoRA: initialization schemes and damping

**Author:** internal note, intended as input to a sanity check.
**Goal:** derive, in the style of Hayou–Vyas–Yang (μP) and Chen–Villar–Hayou (μA), how the *chord-tight polar-product* LoRA optimizer interacts with three initialization schemes (Init[A], Init[B], Init[AB] with PiSSA-style residual) and two damping schemes (absolute, σ_max-relative). The note is self-contained: it restates the algorithm, derives per-step output increments under each (init, damping) combination, and gives predictions for what should be observed empirically across rank `r`.

The reader is assumed to know Adam and the basic LoRA factorization. No prior knowledge of μA, μP, or the chord-tight algorithm is required.

---

## 1. Setup

### 1.1 LoRA layer

Fine-tune a frozen weight $W^\star \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$ by adding a low-rank correction:

$$
\bar W \;=\; W^\star \;+\; \frac{\alpha}{r}\, B A,
\qquad
A \in \mathbb{R}^{r \times n_{\text{in}}},\
B \in \mathbb{R}^{n_{\text{out}} \times r},
\qquad r \ll \min(n_{\text{in}}, n_{\text{out}}).
$$

For simplicity take $n_{\text{in}} = n_{\text{out}} = n$ (the typical attention QKV/O setting at hidden size $n$) and the convention $\alpha = r$, so $\bar W = W^\star + BA$. Only $A, B$ are trained; $W^\star$ is frozen.

On an input vector $Z \in \mathbb{R}^n$ to this layer:

$$
\bar Z \;=\; W^\star Z + BAZ,
\qquad
Z_A := AZ \in \mathbb{R}^r,
\qquad
Z_B := B Z_A = BAZ \in \mathbb{R}^n.
$$

$Z_B$ is the *LoRA contribution* to the layer output. The objective is to understand how each fine-tuning step changes $Z_B$.

### 1.2 The γ-operator notation

We analyze the regime $n \to \infty$ with rank $r$ either fixed or scaling jointly. For a scalar / vector / matrix quantity $v$ that depends on $n$, write $v = \Theta(n^\gamma)$ to mean its size (absolute value for scalars, entry-wise magnitude for vectors, operator norm for matrices) is asymptotically of order $n^\gamma$. Define the **γ-operator**:

$$
\gamma[v] \;:=\; \gamma \quad\text{such that}\quad v = \Theta(n^\gamma).
$$

Rules:

- $\gamma[u + v] = \max(\gamma[u], \gamma[v])$ when $u, v$ do not cancel.
- $\gamma[u \cdot v] = \gamma[u] + \gamma[v]$ (scalar product, or matrix product evaluated entry-wise).
- For a random matrix $M$ with i.i.d. entries of size $\Theta(n^{\gamma[M_{ij}]})$, the operator norm scales as $\gamma[\|M\|_2] = \gamma[M_{ij}] + \frac{1}{2}\max(\gamma[\text{rows}], \gamma[\text{cols}])$.

This is the same machinery used in μP and μA. We will read off $\gamma$ exponents and identify the regime where $\eta$ (a learning rate) can be chosen so that $Z_B$ remains $\Theta(1)$ entry-wise across $n$ and across $r$.

### 1.3 SignSGD as a proxy for Adam

Adam with bias correction, on a gradient $g$ with stable backward signal $g_{ij} = \Theta(1)$, produces an update direction whose entries asymptote to $\text{sign}(g)$. Following μA, replace the Adam update by

$$
u \;=\; \text{sign}(g) \;\in\; \{-1, 0, +1\}^{\text{shape}(g)},
\qquad
\gamma[u_{ij}] = 0.
$$

Assumption (standard for μP/μA, Li et al. Assumption 4.1):
- Layer input: $\gamma[Z_j] = 0$, so $\|Z\|_2 = \Theta(\sqrt{n})$.
- Backpropagated output gradient: $\gamma[(d\bar Z)_j] = 0$, $\|d\bar Z\|_2 = \Theta(\sqrt{n})$.

Then the factor gradients are

$$
g_A = B^\top d\bar Z\, Z^\top \;\in\; \mathbb{R}^{r \times n},
\qquad
g_B = d\bar Z\,(AZ)^\top \;\in\; \mathbb{R}^{n \times r}.
$$

For $u_A = \text{sign}(g_A)$ to have $\Theta(1)$ entries, $g_A$ needs $\Theta(1)$ entries. This holds whenever $\gamma[B_{ij}] = -1/2$: each entry of $B^\top d\bar Z$ is a length-$n$ random-sign sum of $\Theta(n^{-1/2}) \cdot \Theta(1)$ terms, hence $\Theta(1)$ by CLT. Same for the $A$-side gradient.

### 1.4 Reduction to a single layer

Following Hayou et al. and Li et al., we track a single LoRA module by freezing all other layers. Unlike those works we *cannot* reduce to a single data point: with one token, $g_A = (B^\top d\bar Z) Z^\top$ is rank one and $\operatorname{sign}(g_A)$ is the outer product $\operatorname{sign}(B^\top d\bar Z)\operatorname{sign}(Z)^\top$, with its right singular direction aligned with $\operatorname{sign}(Z)$. Whitening and polar preserve rank, so $\Delta A$ would be rank one with right singular vector aligned with $Z$ — exactly the case ruled out by the incoherence assumption below. Li et al.'s single-token reduction works for plain Adam on $(A, B)$ because Adam's preconditioner is diagonal in factor coordinates; it breaks for chord-tight because whitening + polar inherits the rank of $u_A$.

We therefore require a batch of $B \ge r$ tokens with sufficiently incoherent activations $\{Z^{(b)}\}$ that $g_A$ has rank $r$ and no single token dominates. The γ-analysis below is conditional on this batched-incoherence assumption.

---

## 2. The chord-tight update

The chord-tight optimizer derives its update from a variational program with two pieces: a whitening preconditioner and a chord trust region. For asymptotic analysis only the resulting update matters; the derivation is recorded separately in `algorithm_tight_chord.md`. We restate the update here.

### 2.1 Algorithm

One step on layer pair $(A, B)$ at step $t$:

1. **Adam preconditioning.** Compute Adam-corrected directions $u_A, u_B$ from factor gradients $g_A, g_B$. Under the SignSGD abstraction (§1.3):
   $u_A = \text{sign}(g_A) \in \{\pm 1\}^{r \times n}$,
   $u_B = \text{sign}(g_B) \in \{\pm 1\}^{n \times r}$.

2. **Top singular values.** Compute
   $\sigma_A := \sigma_{\max}(A)$, $\sigma_B := \sigma_{\max}(B)$, $s := \sigma_A + \sigma_B$.

3. **Tight-chord radius.**
   $$
   \rho \;=\; \frac{-s + \sqrt{s^2 + 4\eta}}{2}.
   \tag{2.1}
   $$
   The free hyperparameter $\eta$ is the *spectral step size* — the per-step cap on $\|\Delta W\|_2$. The radius $\rho$ is chosen so that this cap holds with equality under submultiplicativity (Proposition 3 of `algorithm_tight_chord.md`).

4. **Whitening.** Compute
   $$
   S_A := A A^\top,\quad S_B := B^\top B
   \quad (\text{both } r \times r,\ \text{PSD}),
   $$
   and form damped inverse square roots $(S_A + \delta_A I)^{-1/2}$, $(S_B + \delta_B I)^{-1/2}$. The damping rule $\delta_A, \delta_B$ is the second axis of this note (§5).

5. **Whitened pre-image.** Compute
   $$
   c_A := (S_B + \delta_B I)^{-1/2}\, u_A,\quad
   c_B := u_B\, (S_A + \delta_A I)^{-1/2}.
   $$

6. **Directions.**
   $$
   D_A := (S_B + \delta_B I)^{-1/2}\, \text{polar}(c_A),\quad
   D_B := \text{polar}(c_B)\, (S_A + \delta_A I)^{-1/2},
   $$
   where $\text{polar}(M) := UV^\top$ for $M = U\Sigma V^\top$. Both $D_A, D_B$ have rank $r$.

7. **Apply.**
   $$
   \Delta A = -\rho \frac{D_A}{\|D_A\|_2},\quad
   \Delta B = -\rho \frac{D_B}{\|D_B\|_2}.
   \tag{2.2}
   $$
   By construction $\|\Delta A\|_2 = \|\Delta B\|_2 = \rho$, hence $\|\Delta W\|_2 \le \eta$ (Prop 3).

### 2.2 Structural facts and assumptions

**(F1)** $\|\Delta A\|_2 = \|\Delta B\|_2 = \rho$ exactly, by (2.2). This is forced by the algorithm.

**(A2)** *Assumption:* $\Delta A$ has rank $r$ with all $r$ singular values close to $\rho$. Holds when $u_A$ and $S_B$ are both well-conditioned; fails in rank-deficient regimes (step 0 of Init[A]/Init[B], and the single-token setup discussed in §1.4).

**(F3)** Under (A2), $\|\Delta A\|_F = \sqrt{r}\,\rho$.

**(A4)** *Assumption:* under (A2) with incoherent right singular vectors, $\gamma[(\Delta A)_{ij}] = \gamma[\rho] - \tfrac{1}{2}$. Same for $\Delta B$. The rest of the note assumes (A4).

We will track $\gamma[\rho]$ as a function of $\gamma[\eta]$ via (2.1).

### 2.3 Two regimes of $\rho$

From (2.1):

- **Linear regime** $\eta \ll s^2$: $\rho \approx \eta / s$, so $\gamma[\rho] = \gamma[\eta] - \gamma[s]$.
- **Sqrt regime** $\eta \gg s^2$: $\rho \approx \sqrt{\eta} - s/2$, so $\gamma[\rho] = \gamma[\eta]/2$.

Which regime holds depends on $\gamma[\eta]$ vs $2\gamma[s]$.

---

## 3. Per-step output increment

Define

$$
\Delta Z_B^{(t)} := B_t\, \Delta A^{(t)}\, Z + \Delta B^{(t)}\, A_t\, Z + \Delta B^{(t)}\, \Delta A^{(t)}\, Z
= \delta_t^1 + \delta_t^2 + \delta_t^3.
\tag{3.1}
$$

Stability requires $Z_B^{(t)} = \Theta(1)$ entry-wise across $t$. By telescoping (Li et al. Lemma 3.6), it suffices to control each increment: $\gamma[\delta_t^i] \le 0$ for $i = 1, 2, 3$. **Efficient** feature learning (Li et al. Definition 4) further requires $\gamma[\delta_t^1] = \gamma[\delta_t^2] = 0$ — both $A$ and $B$ updates contribute non-vanishingly to the increment.

We compute $\gamma[\delta_t^i]$ for chord-tight.

### 3.1 The first term $\delta_t^1 = B_t \Delta A Z$

This is the contribution of the $A$-update at step $t$, passed through the current $B_t$.

- $\Delta A$: rank-$r$, operator norm $\rho$, entries $\Theta(\rho/\sqrt{n})$ (A4).
- $Z$: vector with entries $\Theta(1)$, $\|Z\|_2 = \Theta(\sqrt n)$.

Step 1 — compute $\Delta A \cdot Z$. The $k$-th entry: $\sum_j (\Delta A)_{kj} Z_j$, a sum of $n$ products of $\Theta(\rho/\sqrt n)$ and $\Theta(1)$. By random-sign CLT (which uses (A4)), the result has size $\Theta(\rho)$, so $\|\Delta A Z\|_2 = \Theta(\rho \sqrt r)$.

Step 2 — compute $B_t \cdot (\Delta A Z)$. Let $\gamma[B_{ij}] := \beta_t$ (entry-wise size of $B$ at step $t$; we will fill in $\beta_t$ per init scheme in §4). The $i$-th entry of the result is $\sum_k B_{ik} \cdot (\Delta A Z)_k$, a sum of $r$ products of $\Theta(n^{\beta_t})$ and $\Theta(\rho)$. With random signs:

$$
\gamma[\delta_t^1] \;=\; \beta_t + \gamma[\rho] + \frac{\gamma[r]}{2}.
\tag{3.2}
$$

Under the convention $\gamma[r] = 0$ (fixed rank as $n \to \infty$):

$$
\boxed{\quad \gamma[\delta_t^1] = \beta_t + \gamma[\rho].\quad}
\tag{3.3}
$$

### 3.2 The second term $\delta_t^2 = \Delta B A_t Z$

By symmetry of (3.1) under $A \leftrightarrow B$, with $\gamma[A_{kj}] := \alpha_t$:

$$
\boxed{\quad \gamma[\delta_t^2] = \alpha_t + \gamma[\rho].\quad}
\tag{3.4}
$$

### 3.3 The bilinear term $\delta_t^3 = \Delta B \Delta A Z$

Step 1 above gave $\Delta A Z$ with entries $\Theta(\rho)$. Step 2 replaces $B_t$ with $\Delta B$, which has entries $\Theta(\rho/\sqrt n)$:

$$
\gamma[\delta_t^3] \;=\; (\gamma[\rho] - \tfrac{1}{2}) + \gamma[\rho] + \tfrac{\gamma[r]}{2}
\;=\; 2\gamma[\rho] + \tfrac{\gamma[r] - 1}{2}.
$$

Under $\gamma[r] = 0$:

$$
\boxed{\quad \gamma[\delta_t^3] = 2\gamma[\rho] - \tfrac{1}{2}.\quad}
\tag{3.5}
$$

### 3.4 The combinatorial constant: $\sqrt r$ vs $r$ (conditional)

Equations (3.2)–(3.5) take the $r$-fold inner sums to cancel by random-sign CLT, contributing a constant factor of $\sqrt{r}$. This is correct when the per-row sign patterns in $\Delta A$, $\Delta B$, $A_t$, $B_t$ are decorrelated. Under the μA derivation for plain SignSGD (Chen–Villar–Hayou Thm 4.3) the factor gradient is an *outer product* $g_A = \alpha B^\top d\bar Z\,Z^\top$, so $\operatorname{sign}(g_A)$ factorizes as $\operatorname{sign}(B^\top d\bar Z)_k \cdot \operatorname{sign}(Z)_j$ — a rank-1 sign pattern. The same row-direction reappears in $\Delta A$ and (via accumulation) in $A_t$ at the next step, breaking sign-decorrelation in the $\sum_k$ contraction; the resulting bound is $\Theta(r)$ rather than $\Theta(\sqrt r)$. This is the mechanism by which μA produces $\eta \propto r^{-1/2}$ for plain SignSGD at Init[A] $\alpha = 1$ (Cor 4.4).

For chord-tight, $u_A = \operatorname{sign}(g_A)$ inherits the same rank-1 factorization pre-polar. The question is whether the chord-tight $\Delta A$ — obtained from $u_A$ via $(S_B + \delta_B I)^{-1/2}$ premultiplication and then the polar map — preserves that structure or destroys it.

**Conditional A (polar decorrelates).** $\operatorname{polar}(c_A) = U V^\top$ has orthonormal rows (when $r \le n$). For a rank-1 sign factorization $u_A = u\,v^\top$, polar collapses to a normalized rank-1 frame $\hat u \hat v^\top$ with $\hat u = u / \sqrt r,\ \hat v = v / \sqrt n$, suppressing rather than propagating the structure. For higher-rank generic sign matrices, polar maps to an orthonormal frame in the row span of $u_A$; if this frame is approximately Haar-random in that subspace, $(\operatorname{polar}(c_A) Z)_k$ has entries $\Theta(1)$ by random projection rather than $\Theta(n)$ by sign correlation. The $\sqrt r$-CLT bounds (3.2)–(3.5) hold as stated, and the equilibrium increment scales as $\rho^2 \sqrt r / \sqrt n$.

**Conditional B (polar preserves sign-correlation).** If polar does not decorrelate — e.g. because the rank-1 part of $u_A$ dominates the polar output after whitening, or because the row span of $\operatorname{polar}(c_A)$ inherits Z-correlation — then chord-tight reproduces μA's $\Theta(r)$ combinatorics, and the equilibrium increment scales as $\rho^2 r / \sqrt n$.

Resolving this requires either a direct small-$r$ calculation on $\operatorname{polar}$ of a sign-structured input, or a free-probability / Haar-approximation argument for the singular vectors of $c_A$. Neither is undertaken here. The two conditionals give different $r$-prescriptions for $\eta$ (§6.3), which is the testable consequence.

---

## 4. Initialization schemes

Three schemes. Throughout, "Gaussian init" means entries i.i.d. $\mathcal{N}(0, \sigma^2)$ with $\sigma^2 = 1/n$ (Kaiming-equivalent scale for an $n$-dimensional fan-in).

### 4.1 Init[A] (standard PEFT)

$A_0$: $r \times n$ Gaussian, entries $\Theta(1/\sqrt n)$, $\sigma_{\max}(A_0) = \Theta(1)$, $\alpha_0 = -1/2$.
$B_0 = 0$: zero matrix, $\sigma_{\max}(B_0) = 0$, $\beta_0 = -\infty$.

At step 0, $g_A = B_0^\top d\bar Z\, Z^\top = 0$ since $B_0 = 0$, so $\Delta A^{(0)} = 0$ and **only $\Delta B^{(0)}$ is active**. The $\Delta B$ update uses the $S_A$-side whitening (step 5/6 of §2.1: $c_B = u_B (S_A + \delta_A I)^{-1/2}$, $D_B = \operatorname{polar}(c_B)(S_A + \delta_A I)^{-1/2}$), and $S_A = A_0 A_0^\top$ is a healthy Wishart with $\sigma_{\max}(S_A) = \Theta(1)$. The $S_B = 0$ singularity sits on the inactive $A$-side and does not enter $\Delta B^{(0)}$. Then:

- $s = \Theta(1)$, sqrt regime, $\gamma[\rho] = \gamma[\eta]/2$.
- $\gamma[\delta_0^1] = -\infty$ ($B_0 = 0$ and $\Delta A^{(0)} = 0$).
- $\gamma[\delta_0^2] = -1/2 + \gamma[\rho]$.
- $\gamma[\delta_0^3] = -\infty$ ($\Delta A^{(0)} = 0$).

At step 1, $B_1 = \Delta B^{(0)}$ has entries $\Theta(\rho_0/\sqrt n)$, so $\beta_1 = \gamma[\rho_0] - 1/2$, and from step 1 onward the analysis equilibrates to $\alpha_t = \beta_t = \gamma[\rho_t] - 1/2$. The whitening of $S_B$ first appears in $\Delta A^{(1)}$; $\sigma_{\max}(S_B) = \rho_0^2$ at step 1, and grows with accumulated update mass thereafter. **The damping question first becomes operative at step 1**, not step 0.

Efficiency fails at step 0 — only $\delta_0^2$ contributes.

### 4.2 Init[B] (μA-style for one-sided)

$A_0 = 0$: $\sigma_{\max}(A_0) = 0$, $\alpha_0 = -\infty$.
$B_0$: $n \times r$ Gaussian, entries $\Theta(1/\sqrt n)$, $\sigma_{\max}(B_0) = \Theta(1)$, $\beta_0 = -1/2$.

By symmetry of (3.1) under $A \leftrightarrow B$, all γ-exponents are obtained by swapping $\alpha_t \leftrightarrow \beta_t$ from §4.1. At step 0, $g_B = d\bar Z\,(A_0 Z)^\top = 0$, so $\Delta B^{(0)} = 0$ and only $\Delta A^{(0)}$ is active; its whitening uses healthy $S_B$. Same one-step inefficiency, transposed.

### 4.3 Init[AB] (proposed: symmetric, both Gaussian)

$A_0, B_0$ both Gaussian with the same variance $1/n$. Operator norms: $\sigma_{\max}(A_0), \sigma_{\max}(B_0) = \Theta(1)$. Entry exponents: $\alpha_0 = \beta_0 = -1/2$.

At step 0, $s = \Theta(1)$ (same exponent as Init[A], constant doubled) and:

- $\gamma[\delta_0^1] = \gamma[\delta_0^2] = -1/2 + \gamma[\rho]$.
- $\gamma[\delta_0^3] = 2\gamma[\rho] - 1/2$.

Both linear terms are present from step 0, unlike Init[A]/Init[B] where one is identically zero. Under the stable scaling $\gamma[\rho] = 1/4$ (§6), both linear terms are only $\Theta(n^{-1/4})$ while $\delta_0^3$ is $\Theta(1)$ — so step-0 output movement is dominated by the bilinear $\Delta B\,\Delta A\,Z$ term, not by the linear ones.

### 4.4 The W₀ perturbation under Init[AB]

Under Init[AB], at step 0,

$$
\bar W_0 - W^\star = B_0 A_0 = \text{rank-}r,\ \ \|B_0 A_0\|_2 = \Theta(1),\ \ \gamma[(B_0 A_0)_{ij}] = \tfrac{\gamma[r]}{2} - 1.
$$

The merged weight differs from pretrained $W^\star$ by a rank-$r$ matrix that is $\Theta(1)$ in operator norm and entry-wise $\Theta(\sqrt r / n)$.

**PiSSA-style residual** removes this: snapshot $\Delta W_0 := B_0 A_0$ at init, set $W^\star_{\text{frozen}} \leftarrow W^\star - \Delta W_0$. Then $\bar W_0 = W^\star_{\text{frozen}} + B_0 A_0 = W^\star$ exactly. Optimization proceeds in $(A, B)$ space; the *effective* delta from pretrained is $B A - B_0 A_0$, which vanishes at step 0 and evolves freely thereafter.

After subtract-init, the merged weight at $t = 0$ is **identical** to the standard Init[A] case. The asymptotic analysis of §3 continues to hold (it depends on $\sigma_A, \sigma_B, \alpha_t, \beta_t$ — properties of the **factors**, not of $W^\star_{\text{frozen}}$).

---

## 5. The damping question

### 5.1 What damping does

The chord-tight whitening computes $(S_X + \delta I)^{-1/2}$ for $X \in \{A, B\}$. The damping $\delta$ bounds the spectral norm of the whitening operator: $\sigma_{\max}\bigl((S_X + \delta I)^{-1/2}\bigr) = (\sigma_{\min}(S_X) + \delta)^{-1/2} \le \delta^{-1/2}$. This regularizes against the rank-deficient case (e.g. Init[A] gives $S_B = 0$ at step 0, where the un-damped inverse square root is singular). Note: $S_B = 0$ at Init[A] step 0 is moot because $\Delta A^{(0)} = 0$ anyway (§4.1); the damping rule first matters at step 1.

Framing is in terms of $\sigma_{\max}(\cdot)$, *not* the condition number $\sigma_{\max}/\sigma_{\min}$. Tail eigenvalues over a stochastic state are noisy step to step; quantities built on them inherit that noise. The asymptotic statements below rest on $\sigma_{\max}$ — a stable bulk quantity — and on the stable rank $\operatorname{srank}(M) := \|M\|_F^2 / \sigma_{\max}(M)^2 \in [1, r]$ when we want a noise-robust "filled-ness" measure.

### 5.2 Absolute damping (current code)

$\delta = $ const, e.g. $10^{-6}$. As $n \to \infty$ with $r$ fixed:

- Under Init[AB]: $S_A, S_B$ are $r \times r$ Wishart matrices, $\sigma_{\max}(S_X) = \Theta(1)$. Absolute $\delta = 10^{-6}$ is six orders below the bulk and inert: $\delta / \sigma_{\max}(S_X) \approx 10^{-6}$.

- Under Init[A]: at step 1+, $\sigma_{\max}(S_B) = \rho^2 \cdot (\text{accumulated update mass})$ on the $B$-side. While $B$ has small accumulated mass, $\sigma_{\max}(S_B) \ll 1$.

The whitening operator's amplification is $(\sigma_{\min}(S_X) + \delta)^{-1/2}$, capped above by $\delta^{-1/2}$. Absolute damping therefore caps amplification at a *fixed* value $\delta^{-1/2}$, independent of $\sigma_{\max}(S_X)$. The cap is the same $10^3$ whether $\sigma_{\max}(S_X) \approx 1$ (bulk amplification $\Theta(1)$, cap three orders above bulk, inactive) or $\sigma_{\max}(S_X) \approx 10^{-2}$ (bulk amplification $\Theta(10)$, cap two orders above bulk, still inactive). The mismatch with relative damping is not that the cap fails but that it is disconnected from the natural spectral scale of the matrix being whitened.

### 5.3 σ_max-relative damping (proposed)

$$
\delta_X \;=\; \max\bigl(\delta_{\min},\ \varepsilon_{\text{rel}} \cdot \sigma_{\max}(S_X)\bigr)
\qquad \text{with } \varepsilon_{\text{rel}} \approx 10^{-4},\ \delta_{\min} \approx 10^{-12}.
\tag{5.1}
$$

The floor $\delta_{\min}$ is required because at Init[A] step 0, $\sigma_{\max}(S_B) = 0$ exactly and a pure relative rule gives $\delta_B = 0$ and a singular inverse square root — even though that singularity is harmless ($\Delta A^{(0)} = 0$). The floor is far below any non-degenerate spectral scale, so it does not affect the regularized regime.

Properties:

- $\sigma_{\max}(S_X)$ is the quantity already computed in the Higham coupled Newton–Schulz iteration for the scaling step $Y_0 = (S_X + \delta I)/s$. No new computation; one extra `max` and `mul`.
- Damping scales with the matrix's natural magnitude. Under Init[AB] at step 0, $\sigma_{\max}(S_X) = \Theta(1)$, so $\delta_X = \Theta(\varepsilon_{\text{rel}})$. Under Init[A] at step 1, $\sigma_{\max}(S_B) = \Theta(\rho^2)$, so $\delta_B = \Theta(\varepsilon_{\text{rel}} \rho^2)$ — adapted to the same scale.
- Whitening amplification is bounded: $\sigma_{\max}\bigl((S_X + \delta_X I)^{-1/2}\bigr) \le \delta_X^{-1/2} = (\varepsilon_{\text{rel}}\sigma_{\max}(S_X))^{-1/2}$. The *ratio* cap-to-bulk amplification is $(\sigma_{\max}(S_X)/\delta_X)^{1/2} = \varepsilon_{\text{rel}}^{-1/2}$ — a fixed multiplicative factor, scale-invariant.

### 5.4 What absolute and relative damping disagree about

The two schemes give the same whitening operator up to numerical floor when $\sigma_{\max}(S_X) = \Theta(1)$ — equilibrated training. They diverge only when one Gram is far from its equilibrium scale: Init[A] in the early steps before $B$ has accumulated mass, Init[B] symmetrically. There:

- Relative damping keeps cap-to-bulk amplification at the fixed ratio $\varepsilon_{\text{rel}}^{-1/2}$.
- Absolute damping caps amplification at the fixed value $\delta^{-1/2}$; the *ratio* to bulk is $(\sigma_{\max}(S_X)/\delta)^{1/2}$, growing as the spectrum shrinks.

Both schemes bound the amplification itself. What relative damping fixes is the scale-invariance of the cap. Under Init[AB], $\sigma_{\max}(S_A) = \sigma_{\max}(S_B) = \Theta(1)$ from step 0, so **Init[AB] makes the damping question moot at the γ-exponent level**.

---

## 6. The η-scaling claim

### 6.1 Stability and efficiency for chord-tight

Once updates accumulate, the factor entry exponents track the update scale rather than the initialization: $\Delta A$ has entries $\Theta(\rho/\sqrt n)$ (A4), so

$$
\alpha_t = \beta_t = \gamma[\rho] - \tfrac{1}{2} \qquad (\text{post-equilibration}).
\tag{6.0}
$$

Plugging (6.0) into (3.3)–(3.5):

$$
\gamma[\delta_t^1] = \gamma[\delta_t^2] = \gamma[\delta_t^3] = 2\gamma[\rho] - \tfrac{1}{2}.
$$

All three increments share one γ-exponent at equilibrium, so stability ($\gamma[\delta_t^i] \le 0$) and efficiency ($\gamma[\delta_t^1] = \gamma[\delta_t^2] = 0$) collapse to

$$
\gamma[\rho] \;=\; \tfrac{1}{4}.
$$

At step 0 the equilibrium does not yet hold: under Init[AB], $\alpha_0 = \beta_0 = -1/2$ literally, giving $\gamma[\delta_0^1] = \gamma[\delta_0^2] = -1/4$ at $\gamma[\rho] = 1/4$. Both linear terms are present but suppressed by $n^{-1/4}$; equilibrium kicks in within $\sim 1$ step.

### 6.2 What sets $\gamma[\rho]$

From (2.1), $\rho$ is determined by $\eta$ and $s = \sigma_A + \sigma_B$. At equilibrium, $\sigma_{\max}(B_t)$ is dominated by accumulated updates of operator norm $\rho$, so $\gamma[s] = \gamma[\rho]$. Plugging into the linear regime: $\gamma[\rho] = \gamma[\eta] - \gamma[\rho]$, i.e. $\gamma[\rho] = \gamma[\eta]/2$. The sqrt regime gives the same. The two coincide at the fixed point.

To achieve $\gamma[\rho] = 1/4$: **$\gamma[\eta] = 1/2$**.

This is the canonical μA-style $n$-scaling for chord-tight at fixed $r$:

$$
\boxed{\quad
\eta \;=\; \Theta(\sqrt{n})
\quad}
\tag{6.1}
$$

The interpretation: $\eta = \Theta(\sqrt n)$ is the largest spectral cap on $\|\Delta W\|_2$ whose $n$-exponent keeps the output increment stable under chord-tight's incoherent rank-$r$ updates. It is not the operator-norm scale of $W^\star$ (which is $\Theta(1)$ under fan-in-$1/n$ scaling), nor is it derived from a μP correspondence — it falls out of the fixed-point of (6.0) and the chord equation.

### 6.3 Rank dependence: $n$-exponent vs $r$-constant (conditional)

**$n$-exponent.** If $r = \Theta(n^c)$ with $c \in [0, 1)$, the bilinear bound at the $\sqrt r$-CLT level becomes $\gamma[\delta_t^3] = 2\gamma[\rho] + (c - 1)/2$. Setting to zero: $\gamma[\rho] = (1-c)/4$, $\gamma[\eta] = (1-c)/2$. For fixed $r$ ($c = 0$): $\gamma[\eta] = 1/2$ unconditionally. **The $n$-exponent of $\eta$ is independent of $r$.** This is the genuinely $r$-invariant part of the prediction.

**$r$-constant.** The constant-level prescription depends on which conditional from §3.4 holds:

- **Conditional A (polar decorrelates).** $\sqrt r$-CLT applies in (3.2)–(3.5). At equilibrium, $\delta_t^3 \sim \rho^2 \sqrt r / \sqrt n$. Stability $\delta_t^3 = \Theta(1)$ gives $\rho^2 \sim \sqrt n / \sqrt r$. Solving back through (2.1) at the equilibrium fixed point $\gamma[s] = \gamma[\rho]$, $\eta \asymp \rho^2 = n^{1/2}\, r^{-1/2}$.

- **Conditional B (polar preserves μA sign-correlation).** The inner $\sum_k$ contractions give $\Theta(r)$ rather than $\Theta(\sqrt r)$. Equilibrium $\delta_t^3 \sim \rho^2 r / \sqrt n$; stability gives $\rho^2 \sim \sqrt n / r$, hence $\eta \asymp \rho^2 = n^{1/2}\, r^{-1}$.

In both conditionals the $n$-exponent of $\eta$ is $1/2$. The $r$-exponent is $-1/2$ under conditional A or $-1$ under conditional B.

**Comparison with plain SignSGD.** μA Cor 4.4 for Init[A] $\alpha = 1$ gives the per-factor learning rate $\eta_{\text{Adam}} \asymp n^{-1/2}\, r^{-1/2}$. The $n$-exponent flip ($+1/2$ for chord-tight's spectral cap vs $-1/2$ for plain Adam's per-factor lr) is structural: $\eta$ here is a cap on $\|\Delta W\|_2$, not on factor entries. The $r$-exponents are directly comparable as predictions of how aggressively to shrink the corresponding step-size with rank. Conditional A matches μA on the $r$-axis; Conditional B is twice as aggressive.

---

## 7. Summary table

| Quantity | Init[A] | Init[B] | Init[AB] + subtract |
|---|---|---|---|
| $\sigma_{\max}(A_0)$ | $\Theta(1)$ | $0$ | $\Theta(1)$ |
| $\sigma_{\max}(B_0)$ | $0$ | $\Theta(1)$ | $\Theta(1)$ |
| $\sigma_{\max}(S_A)_{t=0}$ | $\Theta(1)$ | $0$ | $\Theta(1)$ |
| $\sigma_{\max}(S_B)_{t=0}$ | $0$ | $\Theta(1)$ | $\Theta(1)$ |
| Active step-0 update | $\Delta B$ only | $\Delta A$ only | both |
| Whitening used at step 0 | $S_A$ (healthy) | $S_B$ (healthy) | $S_A, S_B$ both healthy |
| $\delta_0^1$ exponent | $-\infty$ | $-1/2 + \gamma[\rho]$ | $-1/2 + \gamma[\rho]$ |
| $\delta_0^2$ exponent | $-1/2 + \gamma[\rho]$ | $-\infty$ | $-1/2 + \gamma[\rho]$ |
| $\delta_0^3$ exponent | $-\infty$ | $-\infty$ | $2\gamma[\rho] - 1/2$ |
| Step-0 efficiency | ✗ (only $\delta^2$) | ✗ (only $\delta^1$) | both linear $\Theta(n^{-1/4})$; $\delta^3$ dominates |
| Initial $W$ matches $W^\star$ | yes ($BA = 0$) | yes ($BA = 0$) | yes (after subtract-init) |
| Damping question first matters at | step 1 ($\Delta A$ uses $S_B$ small) | step 1 ($\Delta B$ uses $S_A$ small) | inert from step 0 |
| Optimal $\eta$, $n$-exponent | $1/2$ | $1/2$ | $1/2$ |
| Optimal $\eta$, $r$-exponent | $-1/2$ (cond. A) / $-1$ (cond. B) | same | same |

**Claims.**

- **(C1)** $n$-exponent of optimal $\eta$ is $1/2$ for chord-tight under all three init schemes, independent of rank. This is unconditional (rests only on §6 fixed-point and $\gamma[r] = 0$).
- **(C2)** $r$-exponent of optimal $\eta$ is conditional on whether the polar map breaks μA-style rank-1 sign factorization of $u_A$: $-1/2$ if it does (Conditional A, §3.4), $-1$ if it doesn't (Conditional B). Either way, $\eta$ shrinks with $r$ at the constant level — the spectral cap is *not* $r$-invariant.
- **(C3)** *Post-step-0* trajectories of Init[A] + σ_max-relative damping and Init[AB] + (either damping) reach the same equilibrium $\gamma$-exponents from §6.1. The damping rule first matters at step 1 of Init[A], when $\Delta A^{(1)}$ uses the whitening of $S_B$ with $\sigma_{\max}(S_B) = \rho_0^2 \ll 1$. Relative damping keeps cap-to-bulk amplification at the fixed ratio $\varepsilon_{\text{rel}}^{-1/2}$ throughout the $\sigma_{\max}(S_B) \ll 1$ transient; absolute damping has cap-to-bulk ratio $(\sigma_{\max}(S_B)/\delta)^{1/2}$, which falls below $1$ when the spectrum shrinks below $\delta$, leaving the whitening operator dominated by the damping floor rather than the matrix structure. **C3 does not predict that step-0 *movement* matches** — at step 0, Init[A] has $\Delta A^{(0)} = 0$ and no bilinear term, while Init[AB] has $\Theta(1)$ bilinear movement; damping cannot recreate the missing terms.

---

## 8. Open

- **Conditional A vs B.** Resolving which combinatorial regime applies to chord-tight requires either a direct small-$r$ calculation on $\operatorname{polar}$ of a sign-structured input, or a free-probability / Haar-approximation argument for the singular vectors of $c_A$. Not undertaken here.
- **Adam vs SignSGD.** SignSGD drops second-moment corrections. Li et al. show this is asymptotically tight under their setup; we adopt the same.

---

## 9. Glossary

- $W^\star$: frozen pretrained weight, $n \times n$.
- $A, B$: LoRA factors, $r \times n$ and $n \times r$ respectively.
- $\bar W = W^\star + BA$: merged weight applied to layer input.
- $Z, Z_B, Z_A$: layer input, LoRA output, intermediate ($AZ$).
- $u_A, u_B$: Adam-corrected gradient directions (under SignSGD abstraction: entry-wise signs of $g_A, g_B$).
- $\sigma_A = \sigma_{\max}(A), \sigma_B = \sigma_{\max}(B)$, $s = \sigma_A + \sigma_B$.
- $\eta$: chord-tight spectral step size — the per-step cap on $\|\Delta W\|_2$. Independent hyperparameter.
- $\rho$: tight-chord per-factor radius, defined by (2.1). Adapts to $\sigma_A, \sigma_B$.
- $S_A = AA^\top, S_B = B^\top B$: $r \times r$ PSD Gram matrices.
- $\delta_A, \delta_B$: damping in the whitened-inverse computation.
- $\delta_t^i$ ($i = 1, 2, 3$): three terms of $\Delta Z_B^{(t)}$, see (3.1).
- $\alpha_t, \beta_t$: entry-wise γ-exponents of $A_t, B_t$.
- $\gamma[\cdot]$: γ-operator, see §1.2.

## References

- Chen, Villar, Hayou. *Learning Rate Scaling across LoRA Ranks and Transfer to Full Finetuning* (μA). arXiv:2602.06204.
- Li et al. *Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics.* arXiv:2505.23194.
- Hayou, Ghosh, Yu. *LoRA+: Efficient Low Rank Adaptation of Large Models.* arXiv:2402.12354.
- Yang. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* (μP). arXiv:2203.03466.
- Chord-tight derivation: `docs/notes/polar_product/algorithm_tight_chord.md`.
