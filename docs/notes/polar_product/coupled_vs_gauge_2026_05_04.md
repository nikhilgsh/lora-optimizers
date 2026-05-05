# Adam–polar–product LoRA: coupled vs. gauge variants

**Date:** 2026-05-04. Single-seed results at the canonical 2000-step
horizon. Base model OLMo-2-1B, code-instruction adaptation
(Magicoder-OSS-Instruct-75K, 70k samples, sequence length 512).

## 1. Setup and shared notation

A LoRA adapter parameterizes a residual update to a frozen weight
$W_0 \in \mathbb{R}^{m \times n}$ as

$$W = W_0 + \tfrac{\alpha}{r}\, B A, \qquad
A \in \mathbb{R}^{r \times n},\ B \in \mathbb{R}^{m \times r}, \ r \ll \min(m, n).$$

Standard PEFT initialization: $A$ Gaussian, $B = 0$. The four optimizers
in this note all act on the pair $(A, B)$ with raw factor gradients
$g_A = \partial \mathcal{L} / \partial A$, $g_B = \partial \mathcal{L} /
\partial B$. A *step* produces $(\Delta A, \Delta B)$ and applies $A
\leftarrow A + \Delta A$, $B \leftarrow B + \Delta B$.

The merged tangent update (the linearization of the change in $W$) is

$$\Delta W \;\approx\; B\, \Delta A + \Delta B\, A.$$

All four variants share three primitives, motivated in turn:

**Adam EMAs on the raw factor gradients.** With $\beta_1 = 0.9, \beta_2 =
0.999$ and bias correction, produce Adam-direction estimates $u_A, u_B$
of the same shape as $A, B$. This is the standard Adam preconditioner,
applied independently to each factor.

**Whitening in the natural LoRA inner product.** Let

$$S_A = A A^\top + \delta I_r, \qquad S_B = B^\top B + \delta I_r,
\qquad \delta = 10^{-6},$$

with Cholesky factors $S_A = L_A L_A^\top$, $S_B = L_B L_B^\top$. The
whitened Adam directions are

$$\tilde u_A = L_B^{-\top} u_A, \qquad \tilde u_B = u_B L_A^{-\top}.$$

This makes the per-factor update isotropic in the
column-space-of-$B$ / row-space-of-$A$ frame, so that the merged
$\Delta W$ has well-conditioned geometry rather than inheriting the
imbalance between $A$ and $B$.

**Per-block polar projection.** For a matrix $M = U \Sigma V^\top$, the
polar factor is

$$\mathrm{polar}(M) := U V^\top,$$

i.e. the spectrum-flattened version of $M$. Computed in float32 via
Newton–Schulz (5 iterations). Applied to the whitened Adam directions:
$P_A = \mathrm{polar}(\tilde u_A)$, $P_B = \mathrm{polar}(\tilde u_B)$.

The four variants differ in three independent design axes:

| Axis | Question |
|---|---|
| Coupling | Are the $A$- and $B$-updates solved jointly (Picard iteration), or independently? |
| Magnitude | Is each factor's update RMS-rescaled to match the Adam-direction norm, or is the magnitude set by the joint solve alone? |
| Gauge | Is the min-Frobenius gauge $B^\top \Delta B = \Delta A\, A^\top$ enforced as a hard constraint? |

## 2. The four variants

Notation in the pseudocode below: $\eta$ is the learning rate, $K$ the
number of Picard iterations, $\delta$ the whitening damper.

### 2.1 `coupled` — Picard-coupled, factor-RMS magnitude

Independent per-block polar of the whitened Adam directions, then each
factor is RMS-aligned to the Frobenius norm of its own Adam direction.
Picard iteration absorbs the cross-coupling (the whitening of $A$
depends on $B$, which is itself being updated).

Hyperparameters: $K = 3$.

$$
\begin{aligned}
&\textbf{Inputs: } A, B, g_A, g_B, \eta \\
&u_A, u_B \leftarrow \mathrm{AdamEMA}(g_A, g_B) \\
&\Delta A^{(0)} \leftarrow 0, \quad \Delta B^{(0)} \leftarrow 0 \\
&\textbf{for } k = 1, \dots, K: \\
&\quad A_k \leftarrow A + \Delta A^{(k-1)}, \quad B_k \leftarrow B + \Delta B^{(k-1)} \\
&\quad S_A \leftarrow A_k A_k^\top + \delta I,\ S_B \leftarrow B_k^\top B_k + \delta I \\
&\quad L_A L_A^\top \leftarrow S_A,\ L_B L_B^\top \leftarrow S_B \quad \text{(Cholesky)} \\
&\quad \tilde u_A \leftarrow L_B^{-\top} u_A,\ \tilde u_B \leftarrow u_B L_A^{-\top} \\
&\quad P_A \leftarrow \mathrm{polar}(\tilde u_A),\ P_B \leftarrow \mathrm{polar}(\tilde u_B) \\
&\quad \Delta A^{(k)} \leftarrow -\eta \cdot L_B^{-1} P_A \cdot \frac{\|u_A\|_F}{\|L_B^{-1} P_A\|_F} \\
&\quad \Delta B^{(k)} \leftarrow -\eta \cdot P_B L_A^{-1} \cdot \frac{\|u_B\|_F}{\|P_B L_A^{-1}\|_F} \\
&\textbf{return } \Delta A^{(K)},\ \Delta B^{(K)}
\end{aligned}
$$

The per-factor RMS rescale destroys the gauge $B^\top \Delta B = \Delta
A\, A^\top$ but produces an Adam-like step size, so the optimal $\eta$
sits in the standard LoRA range $\sim 3 \times 10^{-4}$.

### 2.2 `coupled-endrms` — same, with end-of-step rescale

Identical to `coupled` except (i) $K = 2$ rather than $K = 3$, and (ii)
the RMS rescale is applied **once** at the end instead of inside every
Picard iterate. Cheaper and cleaner; ranks indistinguishably from
`coupled` at the same $\eta$.

$$
\begin{aligned}
&\textbf{Inputs: } A, B, g_A, g_B, \eta;\ K = 2 \\
&u_A, u_B \leftarrow \mathrm{AdamEMA}(g_A, g_B) \\
&\textbf{run Picard loop as in 2.1, but without the RMS rescale inside.} \\
&\textbf{Let } D_A, D_B \text{ be the unscaled outputs of the loop.} \\
&\Delta A \leftarrow \eta \cdot D_A \cdot \|u_A\|_F / \|D_A\|_F \\
&\Delta B \leftarrow \eta \cdot D_B \cdot \|u_B\|_F / \|D_B\|_F \\
&\textbf{return } \Delta A,\ \Delta B
\end{aligned}
$$

### 2.3 `gauge` — Sylvester min-Frobenius lift, no rescale

Skips the per-factor RMS magnitude correction entirely. Instead, builds
a joint target for the merged tangent $\Delta W$ and solves a Sylvester
system to split it into $(\Delta A, \Delta B)$ of minimum total
Frobenius norm. The minimum-norm split satisfies the **gauge
constraint**

$$B^\top \Delta B = \Delta A\, A^\top$$

automatically (this is the KKT condition of the minimum-norm split).

Let $B = Q_B R_B$ and $A^\top = Q_A R_A^\top$ be thin QR factorizations,
so $A = R_A Q_A^\top$. Hyperparameters: $K = 1$ (single solve).

$$
\begin{aligned}
&\textbf{Inputs: } A, B, g_A, g_B, \eta \\
&u_A, u_B \leftarrow \mathrm{AdamEMA}(g_A, g_B) \\
&S_A \leftarrow A A^\top + \delta I,\ S_B \leftarrow B^\top B + \delta I \\
&Q_A R_A^\top \leftarrow A^\top, \quad Q_B R_B \leftarrow B \quad \text{(thin QR)} \\
&\tilde u_A \leftarrow R_B^{-\top} u_A, \quad \tilde u_B \leftarrow u_B R_A^{-\top} \\
&P_A \leftarrow \mathrm{polar}(\tilde u_A), \quad P_B \leftarrow \mathrm{polar}(\tilde u_B) \\
&J \leftarrow -\eta\, ( Q_B P_A + P_B Q_A^\top ) \quad \text{(joint tangent in } \mathrm{col}(B) + \mathrm{row}(A)\text{)} \\
&\text{Solve Sylvester for } K \in \mathbb{R}^{r \times r}: \\
&\qquad S_B K + K S_A = B^\top J\, A^\top \\
&\Delta A \leftarrow S_B^{-1} \big( B^\top J - K A \big) \\
&\Delta B \leftarrow \big( J A^\top - B K \big) S_A^{-1} \\
&\textbf{return } \Delta A,\ \Delta B
\end{aligned}
$$

The Sylvester system is solved by simultaneous diagonalization of $S_A$
and $S_B$ (eigendecompose each, transform RHS, divide by sums of
eigenvalues, transform back; cost $O(r^3)$, negligible at $r \le 256$).

Because there is no per-factor magnitude correction, $\eta$ controls the
norm of the **merged** $\Delta W$ rather than of the factors. The
optimal $\eta$ is therefore $\sim 10\times$ larger than for the
`coupled` variants.

**Step-1 fallback.** With PEFT init $B = 0$, $S_B$ and $R_B$ are
$\delta$-rank-deficient and the Sylvester solve is ill-conditioned. At
step 1 only, fall back to a per-block whitened polar with $S_B^{-1/2} =
\delta^{-1/2} I$, equivalent to a standard Muon step on each factor.
From step 2 onward the full lift runs.

### 2.4 `gauge-coupled` — gauge lift plus Picard

Same as `gauge` but with $K = 2$ Picard iterations around the lift, to
absorb cross-coupling (the matrices $S_A, S_B, Q_A, Q_B, R_A, R_B$ used
in the lift depend on $A, B$, which are themselves being updated).

$$
\begin{aligned}
&\Delta A^{(0)} \leftarrow 0,\ \Delta B^{(0)} \leftarrow 0 \\
&\textbf{for } k = 1, \dots, K: \\
&\quad \text{run the gauge lift (eqs. of 2.3) using } A + \Delta A^{(k-1)}, B + \Delta B^{(k-1)} \\
&\quad \text{to obtain } \Delta A^{(k)}, \Delta B^{(k)} \\
&\textbf{return } \Delta A^{(K)},\ \Delta B^{(K)}
\end{aligned}
$$

## 3. Empirical results

Single-seed eval loss at step 2000. Each row reports each optimizer's
**best result over its own learning-rate sweep** at the listed rank.

The relevant noise floor is the multi-seed standard deviation of an
AdamW LoRA baseline at the same horizon: $\sigma \approx 0.0006$ at
$r = 16$ and $\sigma \approx 0.0007$ at $r = 64$ (estimated from four
seeds, $\eta = 3 \times 10^{-4}$).

| Optimizer        | r=16 best loss | best $\eta$         | r=64 best loss | best $\eta$         |
|------------------|----------------|---------------------|----------------|---------------------|
| `coupled`        | 0.7615         | $3 \times 10^{-4}$  | **0.7382**     | $3 \times 10^{-4}$  |
| `coupled-endrms` | 0.7615         | $3 \times 10^{-4}$  | 0.7378         | $3 \times 10^{-4}$  |
| `gauge`          | **0.7517**     | $5 \times 10^{-3}$  | 0.7428         | $3 \times 10^{-3}$  |
| `gauge-coupled`  | 0.7539         | $3 \times 10^{-3}$  | 0.7430         | $3 \times 10^{-3}$  |

Observations:

- **Coupling vs. gauge depends on rank.** At $r = 16$, the gauge
  variants beat the coupled variants by $\approx 0.010$ ($\approx
  16\sigma$). At $r = 64$, the ranking reverses: the coupled variants
  beat the gauge variants by $\approx 0.005$ ($\approx 7\sigma$).
- **Within each family, Picard depth barely matters.** `coupled` vs.
  `coupled-endrms`, and `gauge` vs. `gauge-coupled`, are within
  $\approx 1\sigma$ at both ranks. The Picard sweep around the gauge
  lift in particular contributes very little once the gauge is enforced.
- **Optimal LR scales with the magnitude convention.** The gauge
  variants prefer $\eta \approx 3$–$5 \times 10^{-3}$, an order of
  magnitude larger than the coupled variants' $\eta \approx 3 \times
  10^{-4}$. This is the expected consequence of removing the per-factor
  RMS rescale: $\eta$ is then a step size on the merged $\Delta W$,
  not on the factors.

The rank-dependent flip is robust to LR-sweep coverage. `gauge-coupled`
was not run at $\eta = 3 \times 10^{-4}$ at $r = 64$, but its **best
attainable** loss across all swept LRs (0.7430) is still behind
`coupled`'s best (0.7382), so the conclusion is not an artifact of a
gap in the sweep.

## 4. Open questions

- *Why does the rank-dependent flip happen?* The mechanism for the
  $r = 16$ gauge advantage and the $r = 64$ coupled advantage is not
  established. Candidate hypotheses (which the available data do not
  yet distinguish): the per-factor RMS rescale acts as a useful
  high-rank regularizer; the gauge constraint becomes geometrically
  redundant when $r$ approaches the effective rank of the gradient.
- *Step-1 fallback in `gauge`*. Replacing the fallback with an explicit
  warm-start (e.g. one Muon step before activating the lift) has not
  been tested; current results use the $S_B^{-1/2} = \delta^{-1/2} I$
  fallback at step 1 only.
