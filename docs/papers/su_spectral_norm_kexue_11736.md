# Su Jianlin — How to Estimate the Spectral Norm of a Matrix More Scientifically

> **Source:** [kexue.fm/archives/11736](https://kexue.fm/archives/11736), 2026-05-04, by 苏剑林 (Su Jianlin), 科学空间 / Scientific Spaces blog.
> **Local copy:** `docs/papers/spec_norm_estimation.pdf` (read 2026-05-11).
> **Original title:** 如何更科学地估计矩阵的谱范数？

## Motivation

Spectral norms appear in WGAN Lipschitz constraints, LLM training stability, and Muon-style optimizers. Power iteration is the standard tool but leaves room for both convergence speed and rigorous upper bounds.

## The four ideas, in order

### 1. Power iteration baseline + stop-gradient trick

Standard power iteration on the smaller-side Gram:

$$
v^{(t)} = \frac{W^\top W\,v^{(t-1)}}{\lVert W^\top W\,v^{(t-1)}\rVert_2}, \qquad \sigma_1 \approx \lVert W v^{(t)}\rVert_2.
$$

Converges at rate $(\sigma_2/\sigma_1)^{2t}$.

For applications that need the spectral norm to be **differentiable** (spectral normalization, spectral regularization) without backpropagating through the power-iter loop, use the gradient identity $\nabla_W \sigma_1 = u_1 v_1^\top$:

$$
\sigma_1 \;=\; [u_1^\top]_{sg}\,W\,[v_1]_{sg} \tag{5}
$$

where $[\cdot]_{sg}$ is the stop-gradient operator. The power iteration is internal; autograd sees only the final scalar's gradient as $u_1 v_1^\top$.

### 2. Krylov subspace acceleration

Instead of using only the final iterate $v^{(T)}$, **keep the last few power-iter vectors** $v^{(T-2)}, v^{(T-1)}, v^{(T)}$ and form a Krylov subspace. QR-orthonormalize into $Q \in \mathbb{R}^{m \times k}$, then solve the small eigenvalue problem on the projected matrix:

$$
W^\top W\,Q\,x \;\approx\; \sigma_1^2\,Q\,x
\quad\Longleftrightarrow\quad
(W Q)^\top (W Q)\,x \;=\; \sigma_1^2\,x.
\tag{6}
$$

The reduced problem is $k \times k$, where $k$ is small (e.g., 3). Eigendecomposition is trivial at that size.

This is a simplified Lanczos. Empirical results in the article:

| `T` (power iters) | Variant | Accuracy / Speed note |
|---|---|---|
| 5 | Krylov | matches T=10 vanilla power iter |
| 10 | Krylov | strictly higher accuracy and faster |
| 10 | Krylov | ~2/3 of time in power-iter matmuls, ~1/3 in QR, very little in the small eig |

**Caveat:** the last-three-iters trick relies on those iterates being already near-collinear with $v_1$. Cholesky-QR isn't usable in this regime (ill-conditioned). Use Householder/MGS QR.

### 3. Strict upper bound via Schatten norms

When you need $\sigma_1$ to be **strictly upper-bounded** (e.g., to certify $\lVert W\rVert_2 < 1$ for Lipschitz constraints), power iteration is one-sided wrong — it converges from below. Use the Schatten-$p$ norm instead:

$$
\lVert W\rVert_{S,p} \;=\; \Bigl(\sum_i \sigma_i^p\Bigr)^{1/p} \;\ge\; \sigma_1. \tag{7}
$$

Tightens monotonically as $p \to \infty$. For even $p = 2k$, computable purely from traces with no SVD:

$$
\lVert W\rVert_{S,2k}^{2k} \;=\; \mathrm{tr}\!\bigl((W^\top W)^k\bigr).
$$

#### Numerical stability for high $p$

Computing $(W^\top W)^k$ for large $k$ overflows or underflows quickly. The article gives an algorithm that maintains the log of an accumulated normalization factor and the unit-norm matrix:

```
Init:  log_S ← log tr(W^T W),  M ← (W^T W) / tr(W^T W)
For t = 1, ..., T:
    log_S ← 2·log_S + log tr(M^2)
    M ← M^2 / tr(M^2)
Output:  exp((log_S + log ‖M‖_F) / 2^(T+1))
```

Each step squares $M$, so after $T$ steps you have effectively computed $\lVert W\rVert_{S, 2^{T+2}}$ — exponential reach in $p$ per iteration.

### 4. Multi-moment bound (the most interesting result)

Plain Schatten bounds use a single moment. The article shows you can do better with **two** moments using a constrained-optimization argument. Let $S_p := \sum_i \sigma_i^p$ and assume $W \in \mathbb{R}^{m \times n}$ with $m \le n$. From the inequality

$$
\frac{S_2 - \sigma_1^2}{m - 1} \;\le\; \sqrt{\frac{S_4 - \sigma_1^4}{m - 1}}
\tag{11}
$$

(an Cauchy–Schwarz-style relation among the non-top singular values), solving the quadratic in $\sigma_1^2$ gives the **closed-form upper bound**:

$$
\boxed{\quad
\sigma_1 \;\le\; \sqrt{\frac{S_2 + \sqrt{(m-1)\,(m\,S_4 - S_2^2)}}{m}}
\quad} \tag{12}
$$

Both $S_2 = \lVert W\rVert_F^2$ and $S_4 = \lVert W^\top W\rVert_F^2 = \mathrm{tr}((W^\top W)^2)$ are computable with simple matmuls — no eig, no SVD, no iteration.

This is **tighter than $S_4^{1/4}$** alone because it uses the additional information in $S_2$ to constrain how concentrated the spectrum is. The generalization (eq 13) uses $S_{2k}$ and $S_{4k}$; further extension (eq 14, 15) is theoretically possible but practically clunky.

The article cites *Fast Tight Spectral-Norm Bounds* for the extension to combining $S_2, S_4, S_6, S_8$ via nonlinear programming.

## Relevance to this project — the chord-tight LoRA optimizer

The Stage-0 diagnostic readout (`docs/notes/polar_product/tight_chord_diagnostics_stage0.md`) found the existing chord-tight optimizer breaching its $\lVert\Delta W\rVert_2 \le \eta$ guarantee by up to $2.4\times$ at $r=64$ because three of the four $\sigma_{\max}$ estimates feeding the magnitude rule were power-iter under-estimates. We fixed those by switching to exact eigh on the $r \times r$ Gram (commits `57a932b` and `54311ba`).

Two ideas from this article are interesting beyond what we already did:

### A. The multi-moment bound (eq 12) would be a stronger safety property than eigh

Eigh on the $r \times r$ Gram is exact up to float32 precision (~$10^{-7}$ relative), which is essentially the right answer. But the multi-moment bound (eq 12) is a **strict upper bound** with no iteration, no eigendecomposition, and only two `r × r` matmul-like ops:

```python
W_Gram = W @ W.T               # (r, r) or (r, n) → S_2 = trace = ||W||_F^2
S2 = (W * W).sum()
S4 = (W_Gram * W_Gram).sum()   # = ||W^T W||_F^2
m = min(W.shape)
sigma_max_upper = sqrt((S2 + sqrt((m-1) * (m*S4 - S2**2))) / m)
```

Two trade-offs vs eigh:
- **Trade**: small over-estimate of $\sigma_{\max}$ when the spectrum is concentrated (eq 12 is loose unless all $\sigma_i$ for $i \ge 2$ are equal). Over-estimate translates to smaller $\rho$, smaller step, slightly slower training. For near-flat polar-map spectra (our chord-tight case at $r=64$) the bound is fairly tight.
- **Gain**: provably-held safety guarantee. Eigh has float-precision residual error; in principle a `chord_slack > 1` could still occur from $\sim 10^{-7}$ float-precision slop accumulating into a tiny breach. Multi-moment guarantees never.

For us, eigh is probably fine — the breach we observed (1.5–2.4×) was structural under-estimate, not float slop. But if the chord-tight optimizer becomes a published baseline, the multi-moment formula is the right "safe by construction" version.

### B. Krylov acceleration becomes relevant at very high LoRA rank

At our usual $r \in \{16, 64, 128, 256\}$, eigh on the $r \times r$ Gram is sub-millisecond per pair and trivial. At $r \ge 512$ (rare in LoRA but possible for full-rank-like settings), eigh's $O(r^3)$ catches up to the matmul costs, and the Krylov-accelerated power iter (last-few-iters + small eig) would be faster. Not relevant right now; flag for future scale-up work.

### C. Stop-gradient trick is not relevant

We use $\sigma_{\max}$ as a magnitude scalar; we don't backprop through it. The gradient identity (5) is for spectral normalization / regularization use cases that need $\nabla_W \sigma_1$ during forward. Skip.

## Cite as

```bibtex
@online{kexuefm-11736,
    title  = {How to Estimate the Spectral Norm of a Matrix More Scientifically?},
    author = {苏剑林 (Su Jianlin)},
    year   = {2026},
    month  = {May},
    url    = {https://kexue.fm/archives/11736},
}
```
