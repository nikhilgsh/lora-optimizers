# Gram Newton-Schulz — Tri Dao (2026)

**Source:** https://tridao.me/blog/2026/gram-newton-schulz/
**Author:** Tri Dao
**Date saved:** 2026-05-02

This file is a summary + project-relevant takeaways. For the full content,
read the original blog post at the URL above.

## Summary of the technique

**Problem.** Muon's Newton–Schulz orthogonalization is the per-step bottleneck.
For an `m × n` input X with `m > n` (the typical "tall" case), standard NS
performs O(mn²) matrix ops per iteration, multiple iterations per step.
At trillion-parameter scale this is 2–17% of total training time.

**Mathematical reformulation: Theorem 1.** The composition of Newton–Schulz
polynomial steps applied to X can be decomposed so the iteration runs on
the small `n × n` Gram matrix `G = X^T X` instead of on the rectangular
input X. The mathematical equivalence is exact; only the work shifts to
the smaller symmetric object. This reduces FLOP cost by up to 68% compared
to standard implementations.

**Numerical stability.** Operating in half precision on the Gram matrix
introduces failure modes:
- spurious negative eigenvalues
- eigenvector drift across many iterations

The authors propose a **restarting** strategy: partway through the iteration,
re-form the Gram matrix from the current X iterate, which corrects accumulated
drift.

**Hardware-aware kernels.** Custom symmetric GEMM kernels implemented in
CuTeDSL for Hopper / Blackwell exploit `G = G^T` symmetry to halve the work
of standard GEMM, achieving state-of-the-art performance.

**Results.**
- 40–50% reduction in Newton-Schulz runtime on trillion-parameter MoE models.
- Training quality preserved within 0.01 validation perplexity.
- Up to 2× speedup on rectangular weight matrices.
- Drop-in replacement for existing Muon variants.

## Why this matters for this project

We use Newton–Schulz heavily in the polar-product LoRA family
(`AdamPolarProductLoRA`, gauge variants, etc.) inside the picard inner loop.
Per the cost analysis in the optimizer plan:

- k=4 picard adds ~1.7× wall time over k=1 baseline.
- The dominant cost is repeated NS calls — each one 5 NS steps × 3 matmul
  kernel launches. At LoRA rank `r ≪ d`, kernel launch overhead dominates
  over compute.

Gram-NS would directly attack our bottleneck:
1. **Smaller matrix per iteration.** For LoRA's polar pipeline with `X = R_B^{-T} u_A`
   of shape `(r, n)` (r ≪ n), the Gram is `r × r`. NS iterating on the Gram
   instead of the full `(r, n)` matrix is much cheaper. (Note: our case has
   `r < n` so the Gram-on-rows form should apply directly.)
2. **Fewer kernel launches.** Symmetric GEMM kernels and small-matrix
   iteration both reduce launch overhead — the actual binding constraint
   at our scale.
3. **Restart strategy.** Useful if we adopt warm-starting NS across picard
   inner iters: the Gram-form restart aligns with our "warm-start NS from
   previous iter's polar output" idea (the previous polar output gives an
   excellent restart point).

## Takeaways for our codebase

If picard `k=4` lands as the shipping rule, follow-up optimization could:

1. Replace `_newton_schulz` with a Gram-form NS that iterates on `(r, r)`
   instead of `(r, n)` / `(m, r)`.
2. Add a restart at iter 2-3 to correct drift in float32 (we don't run
   half precision at the polar step yet, but might in production).
3. Integrate with picard warm-starting: at iter k+1, restart NS from the
   Gram of the previous iter's polar output.

Estimated impact stacking:
- current k=4: 1.7× k=1
- + warm-start NS: ~1.3× k=1 (rough)
- + Gram-form NS: probably ~1.15–1.20× k=1 (rough; based on Dao's 40–50%
  reduction in NS runtime for typical Muon).

That puts picard k=4 at near-baseline overhead — picard becomes effectively
free, removing the cost objection if k=4 is the same-rule shipping winner.
