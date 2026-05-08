# Su Jianlin — How to More Scientifically Estimate the Spectral Norm of a Matrix

> **Source:** [kexue.fm/archives/11736](https://kexue.fm/archives/11736), 2026-05-04, by 苏剑林 (Su Jianlin), 科学空间 / Scientific Spaces blog.
> **Original title:** 如何更科学地估计矩阵的谱范数？

> **⚠️ Reading note (2026-05-08):** kexue.fm blocks direct fetches from this environment (HTTP 403 on Mozilla / curl / WebFetch). This summary is based on (a) the RSS feed lead paragraph, (b) two web-search results that paraphrased the article's content, and (c) my background knowledge of the techniques the article cites. **It has not been read in full.** Anything below the line marked "verified from RSS / search excerpts" is inferred and should be checked against the original before relying on it.

## Why the article exists

> *(verified from RSS / search excerpts)*

Spectral norm estimation matters in three modern contexts:

1. **WGAN-era Lipschitz constraints** — bounding $\lVert W\rVert_2$ in discriminators.
2. **LLM training stability** — preventing parameter / activation magnitude blow-up.
3. **Muon-style optimizers** — every Newton–Schulz polar map is gated by $\lVert M\rVert_2$ for the Frobenius normalization step that puts the matrix into the convergence basin $(0, \sqrt{3})$.

Power iteration is the standard tool. It converges at rate $(\sigma_2/\sigma_1)^{2t}$. The article asks how to do better — both in **convergence speed** and in **rigorous upper bounds**.

## Methods discussed

> *(inferred from search summaries; verify against the original)*

The article appears to cover at least three improvements over vanilla power iteration:

### A. Chebyshev-accelerated power iteration

Replace $v \leftarrow M^\top M v$ with a degree-$k$ Chebyshev polynomial in $M^\top M$ that flattens the spectrum away from the top eigenvalue. Convergence rate improves from $(\sigma_2/\sigma_1)^{2t}$ (geometric) to roughly $\exp(-2t \sqrt{1 - \sigma_2/\sigma_1})$ (Chebyshev) — essentially trading one factor of $\sqrt{\text{gap}}$ in the exponent. Asymptotically optimal for the worst-case spectrum.

Practical cost: same number of matrix–vector products per iteration, but with a polynomial-in-iterates schedule that needs the spectral interval $[\sigma_n^2, \sigma_1^2]$ pre-bracketed (or estimated on the fly).

### B. Lanczos / Krylov-subspace methods

Build the Krylov subspace $\{v, Mv, M^2v, \ldots, M^{k-1}v\}$, project $M^\top M$ onto it, take the top eigenvalue of the resulting tridiagonal $k \times k$ matrix as the estimate. For positive-definite matrices Lanczos achieves "good approximations to the extreme spectrum after a small number of iterations" (per cited Lanczos-algorithm references).

In practice: 5–10 Lanczos iters often match 30+ power iters, at the same per-iteration matrix–vector cost. Caveat: numerical orthogonalization issues at higher iteration counts (well-known Lanczos breakdown).

### C. Moment-based rigorous upper bounds

The Schatten norms $S_p = \big(\sum_i \sigma_i^p\big)^{1/p}$ satisfy $S_p \to \sigma_1 = \lVert M\rVert_2$ as $p \to \infty$. Even-$p$ Schatten norms are computable exactly via traces:

$$
S_2^2 = \lVert M\rVert_F^2 = \mathrm{tr}(M M^\top), \qquad
S_4^4 = \mathrm{tr}\!\big((M M^\top)^2\big), \qquad
S_{2k}^{2k} = \mathrm{tr}\!\big((M M^\top)^k\big).
$$

Standard bound: $\sigma_1 \le S_4^{1/4}$, with relative error $\le n^{1/4}$ if all $\sigma_i$ are equal. Su's improvement appears to combine $S_2$ and $S_4$ into a tighter inequality. The web-search excerpt rendered (with some uncertainty) as

$$
\sigma_1 \;\le\; \sqrt{\frac{S_2 + \sqrt{(m-1)(m S_4 - S_2^2)}}{m}}
$$

where $m$ is some matrix-shape constant — the exact form needs verification against the original, since I'm reconstructing from a paraphrase. The intuition: $S_2$ pins down the *total* spectral mass; the Cauchy–Schwarz-style residual $\sqrt{m S_4 - S_2^2}$ measures concentration; together they bracket $\sigma_1$ tighter than $S_4^{1/4}$ alone. This is a *strict upper bound*, in contrast to power iteration which converges from below.

## Recommendations and ties to optimizer work

> *(inferred; verify against the original)*

The post is positioned as a practical companion to the Muon-streaming-power-iteration series (kexue.fm posts 11697 / 11710 / 11719 / 11673 visible in the same RSS feed). The likely takeaway pattern:

- For **online / per-step** spectral-norm estimation in optimizer hot paths (where each step changes $M$ by ~$\eta$): warm-started power iteration is cheap and adequate when the spectral gap is wide, but loses ground when $\sigma_2 / \sigma_1 \to 1$ (flat spectra). Either Chebyshev acceleration (same cost, faster convergence) or a moment-based upper bound (deterministic, no convergence concerns) is a better alternative there.
- For **safety-critical bounds** where an under-estimate breaks a guarantee (e.g. spectral trust regions), prefer the moment-based upper bound — it cannot fall short, only over-estimate.

## Relevance to the chord-tight LoRA optimizer (this repo)

> *(my analysis, written 2026-05-08)*

This is directly relevant to the **F2 finding** in `docs/notes/polar_product/tight_chord_diagnostics_stage0.md`: at $r=64$, the existing chord-tight optimizer's $\sigma_{\max}(\text{geo}_A)$ estimate (8-iter cold-start power iteration) under-estimates by up to 28–63%, breaching the `‖ΔW‖_2 ≤ η` safety bound by up to 63%. The mechanism is exactly Su's flat-spectrum failure mode for power iteration: at higher LoRA rank, $S_B^{-1/2}$'s spectrum flattens, which flattens $\text{geo}_A$'s spectrum, which slows power-iter geometric convergence.

Three options the article suggests, mapped to the repo's situation:

| Option | Cost in our setting | Robustness |
|---|---|---|
| Warm-started power iter (3 iters) | ~3 small matmul launches per call; warm-start cached in `pair_state` | Works if `geo_A` direction stable across steps; could still under-estimate at flat spectra |
| Chebyshev-accelerated power iter | Same as above + slight scheduling complexity; needs $[\sigma_{\min}^2, \sigma_{\max}^2]$ bracket | Significantly better at flat spectra; still under-estimates (one-sided) |
| Moment-based upper bound (Su's $S_2/S_4$ formula) | One $\mathrm{tr}(M M^\top) = \lVert M\rVert_F^2$ + one $\mathrm{tr}((M M^\top)^2) = \lVert M M^\top\rVert_F^2$, both O($r^2 \cdot \max(d_{\text{in}}, d_{\text{out}})$); no iteration | **Strict upper bound** — never breaches the $\eta$ guarantee. May leave step size on the table when the spectrum is concentrated |
| Exact eigh on $r \times r$ Gram | One `eigh(M M^\top)` for $r \times r$; ~150 μs/pair on A100 | Exact; deterministic; no convergence concerns |

For the chord-tight context the **upper-bound option (Su's moment formula) is structurally superior** to power iteration in one sense: an under-estimate of $\sigma_{\max}(\text{geo}_A)$ means we apply $\lVert dA\rVert_2 > \rho$ and breach the bound — whereas an *over*-estimate means we apply $\lVert dA\rVert_2 < \rho$, so the bound holds but we under-step. Trading a soft convergence guarantee for a hard safety guarantee is the right direction when the failure mode is bound-breach.

That said, at $r \le 64$ exact eigh on the $r \times r$ Gram is also cheap and entirely deterministic. Worth profiling both on the chord-tight optimizer once the warm-start variant is in place.

## What to verify against the original

When you next have access to the post:

- The exact form of the moment-based bound (the formula above is a paraphrase reconstruction, not a verified quote).
- Whether the Chebyshev acceleration scheme requires pre-bracketed spectrum or self-adapts.
- Whether Su discusses the warm-start regime explicitly (likely yes given his Muon-streaming-power-iteration series).
- Concrete iteration counts / accuracy targets recommended for online optimizer use.
