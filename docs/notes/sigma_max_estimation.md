# σ_max estimation in `lora_playground`

Multiple optimizers compute σ_max of matrices or matrix products in their hot
path. This doc names the primitives, lays out cost / fragility / accuracy
trade-offs, and pins the rules that prevent regressions.

Companion reading: `docs/papers/su_spectral_norm_kexue_11736.md` (Su's 2026
article on the four ideas: power-iter + stop-gradient, Krylov acceleration,
Schatten-p upper bounds, multi-moment bound).

## Two computational shapes

### Shape A — plain σ_max(M)

Largest singular value of a single matrix `M`. Examples in this codebase:
- `σ_max(A)`, `σ_max(B)`: spectral chord-tight magnitude rule
  (`optim.py:3540-3541`).
- `σ_max(geoA)`, `σ_max(geoB)`: chord-direction operator norms
  (`optim.py:3635-3636`).
- `σ_max(XA_pre)`, `σ_max(XB_pre)`: unit-polar normalization
  (`optim.py:3568-3569`).

All currently use `_sigma_max_power_iter` / `_sigma_max_power_iter_batched` —
warm-started power iter on the smaller-side Gram. Fast (one matvec per iter
in min(m,n)-space), robust, no damping.

### Shape B — σ_max of a product XY without forming XY

`X` is `(m, r)`, `Y` is `(r, n)`, `XY` is `(m, n)` and we want its spectral
norm. Three places need this in chord-direction:

  σ_BP = σ_max(B · P)
  σ_QA = σ_max(Q · A)
  σ_QP = σ_max(Q · P)

These feed `a, b` of the chord-direction step-size quadratic
`a·λ + b·λ² = lr`. The output is load-bearing for the optimizer step (NOT
diagnostic).

Currently computed via `_sigma_max_chol_eigvalsh(G_outer, G_inner)` —
Krylov-accelerated power iter on `M = L^T G_outer L` where
`L = chol(G_inner + ε · (trace/r) · I)`. **This is the fragile primitive.**
Cholesky fails on h100 bf16 when the relative damping `ε = 1e-12` is
overwhelmed by accumulated bf16 noise pushing `G_inner` slightly indefinite.

## Primitives available

| Name | Shape | Cost (per call, batched) | Failure mode | Accuracy |
|---|---|---|---|---|
| `sigma_max_power_iter` | A | n_iters × 1 matvec in min(m,n)-space | none (numerical) | ~5% p95 rel-err at n_iters=8 cold; ~0.1% at n_iters=3 warm |
| `sigma_max_power_iter_batched` | A | same, batched | none | same |
| `sigma_max_krylov_chol` | B (via Gram) | 1 chol + 1 setup matmul + 16 matvecs in r-space + QR + small eig | **chol fails when `G_inner + ε·I` numerically indefinite** | <1e-4 rel-err |
| `sigma_max_warm_power_iter_unfactored` | B (via factors) | n_iters × 4 matvecs in (m,n)-space | none | converges from below; rate `(σ₂/σ₁)^{2T}` |
| `sigma_max_multimoment_upper` | B (via Gram) | 2 matmuls in r-space + scalar | none | **strict upper bound**; loose for concentrated spectrum |

## Decision rules

### Rule 1: never propose eigh on batched `(..., r, r)` at r ≥ 64

cuSOLVER's batched `eigvalsh`/`eigh` falls back to per-element syevd calls at
r ≥ 64, giving ~10-20 ms per matrix × N pairs. At chord-direction's r=256
N=112 that's 1-2 s per σ_max call, 3 calls per step, 4-6 s/step. This is
the regression the current Krylov-chol path exists to fix.

**Same for `torch.linalg.solve`, `torch.linalg.lu_factor` on batched
`(..., r, r)`** — same kernel-storm class. Don't propose these as
replacements for the Cholesky path without batched benchmark evidence.

### Rule 2: when in shape B and choosing a primitive, the trade is real

- `sigma_max_krylov_chol`: cheapest per-iter (r-space matvecs) but fragile
  (Cholesky); needs `ε ≳ 1e-6` to be reliable on bf16 hardware.
- `sigma_max_warm_power_iter_unfactored`: numerically unconditional but
  matvecs are in (m,n)-space — at r=256, m=n=2048, each matvec is ~16× the
  flops of the Krylov-chol equivalent. Warm-start reduces iters but doesn't
  close the gap.
- `sigma_max_multimoment_upper`: chol-free, 2 matmuls in r-space, **strict
  upper bound** so optimizer step becomes slightly conservative. Tightness
  depends on spectrum concentration; loose when the top singular value
  dominates.

**No primitive is uniformly best. The choice depends on the call site's
tolerance for the trade-offs above.** This is what the bench measures.

### Rule 3: bench before refactoring

Flop counts and microbench numbers are suggestive but not load-bearing.
The historical Krylov-chol fix replaced eigvalsh after the kernel-storm
issue was seen only in integration. Decision rule: an integration bench at
production shapes is required before swapping primitives in optim.py.

## Call site → primitive mapping (current)

| Call site | Function | Primitive | Notes |
|---|---|---|---|
| chord-tight magnitude rule | `_step_per_pair` | `_sigma_max_power_iter` | shape A, plain σ_max(A)/σ_max(B) |
| chord-direction op norms | `_step_batched`, `_step_per_pair` | `_sigma_max_power_iter_batched` | shape A |
| chord-direction σ_BP/QA/QP | `_step_batched` (batched), `_step_per_pair` (loop) | `_sigma_max_chol_eigvalsh` | **shape B; the fragile one** |
| heavy-diagnostics `lambda_dir` | `_log_diagnostics` | `_sigma_max_via_chol_eigh` | shape B; diagnostic only |

The h100 chord-direction crash is the shape-B primitive failing under bf16
noise. The shape-A power-iter calls are unaffected.

## What's next (this work)

1. Move all four functions into `lora_playground/spectral.py` (no logic
   change). chord-tight / chord-direction call sites unchanged in behavior.
2. Add `sigma_max_warm_power_iter_unfactored` and `sigma_max_multimoment_upper`
   as new primitives (no call site uses them yet).
3. Correctness tests: each primitive vs exact `eigvalsh` on small synthetic;
   multi-moment strictly ≥ true σ_max.
4. Microbench + integration bench at production shapes. Data decides which
   primitive chord-direction should use.
5. If bench says swap: refactor chord-direction call sites. Otherwise:
   bump `eps` in the existing primitive (cheapest fix for h100 fragility),
   keep the rest of the library available for future use.
