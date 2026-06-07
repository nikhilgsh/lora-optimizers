# Making kl-shampoo-polar less lr-sensitive

**Goal: widen kl-shampoo-polar's useful step-size band while keeping its best-lr loss.** It wins
at its optimal η but degrades quickly off the optimum, so it needs more lr tuning than a "one
generally useful optimizer" should. We want the wide basin *and* the peak.

The optimizer's definition and the full cross-coupling derivation live in
[`kl_shampoo_polar_derivation.md`](kl_shampoo_polar_derivation.md). This note is the empirical
motivation (the lr-sensitivity measurements), the lever (Picard cross-coupling), and the
implementation plan; it points there for the derivation.

## The problem

opc r64, δ=1e-4, the factor-3 lr band (opt ±1 grid step), all at the 9000-step horizon
(`load_runs`, this session):

| η | step 250 | final (9000) | upticks |
|---|---|---|---|
| 0.01 | 0.8716 | 0.7708 | 0/35 |
| **0.03 (opt)** | 0.8635 | **0.7566** | 0/35 |
| 0.1 | 0.9028 | 0.7970 | 3/35 |

Band-spread 0.040, driven by the high side (η=0.1, +0.040). The sensitivity also appears at
openmath r256 and opc r256, so it is not low-rank-specific (refresh via `load_runs`).

**The high-η failure is monotone, not overshoot.** Every η is monotone and still decreasing at
step 9000 (min = final). η=0.1 does not diverge or bounce — it is simply worse at *every*
checkpoint (0.9028 vs the optimum's 0.8635 already at step 250) and lands on a worse asymptote.
This is a convergence-*rate* problem, not a stability one. It rules out a step-truncation lever
(Momo/NGN): at η=0.1 the loss falls every step, so a truncation that fires on overshoot never
engages.

A monotone-worse-as-η-rises failure is the signature of an uncorrected, η-growing systematic
error in the joint two-factor step — which the Picard cross-coupling corrects.

## The lever: Picard cross-coupling

When both factors move, the merged weight changes by
$\Delta W = B\,\mathrm dA + \mathrm dB\,A + \mathrm dB\,\mathrm dA$. The $k{=}1$ step solves each
factor as if the other were frozen, so the two first-order contributions $B\,\mathrm dA$ and
$\mathrm dB\,A$ double-count their overlap in merged-weight space (each aims to supply the whole
step). That joint-fit error grows with step size — the monotone high-η degradation above. A
Picard (block-coordinate) iteration re-solves each block against the residual the other leaves.
(The bilinear $\mathrm dB\,\mathrm dA$ is a separate $O(\eta^2)$ term, not what the correction
fixes.)

**Full derivation:** `kl_shampoo_polar_derivation.md` §"Cross-coupling: the Picard correction".
The corrected on-block input is
$$\boxed{\;\tilde g_A^{(n)} = \hat m_A + \tfrac1\eta\,B^\top D_{\text{out}}\,\mathrm dB^{(n)}\,A\,D_{\text{in}},
\qquad
\tilde g_B^{(n)} = \hat m_B + \tfrac1\eta\,D_{\text{out}}\,B\,\mathrm dA^{(n)}\,D_{\text{in}}\,A^\top\;}$$
fed through the existing $k{=}1$ pipeline each iter (iter 0 is the current step; every product
stays in the skinny $r\times d$ factors). The derivation settles three things this note's earlier
draft got wrong or left open:

- **Mixed-metric, not a single program.** The on-block self-solve uses kl's independently-fit
  $S_{\text{curv}}$; the cross term uses the full-space diagonals $D_{\text{out}},D_{\text{in}}$
  (the only curvature defined off $\operatorname{range}B$, where $\mathrm dB$ lives). No single
  global metric gives both — forcing one needs $S_{\text{curv},A}=B^\top D_{\text{out}}B$, which
  the KL fit does not satisfy. The cross-term *form* mirrors related_work's Algorithm 10.1 /
  AdaPreLoRA, but kl's use of it is an approximation, not a coherent block-coordinate solve.
- **Exponent pinned at 1.** kl whitens by $D^{-1/2}$, so its metric is $D^1$ and the cross carries
  $D$ linearly. (AdaPreLoRA's power $1/2$ comes from its $R^{-1/4}$ whitening; the cap-off
  AdaPreLoRA cross-check therefore validates a power-$1/2$ variant, not kl.)
- **Corrects the first-order overlap,** not the $O(\eta^2)$ bilinear term.

**Premise check (chord, verified this session).** `…-coupled-spectral-chord-tight`, hard-polar
ns, r64 opc, loss at matched step 3500 (k=2 runs only reached 3500):

| k (Picard) | opt η | factor-3 band-spread |
|---|---|---|
| k=1 | 0.001 | 0.0815 |
| k=2 | 0.1 | 0.0030 |

Picard k=2 flattens chord's band ${\sim}27\times$ and lifts the optimum $100\times$ (η=0.001→0.1):
the high-η region catastrophic at k=1 *becomes* the optimum at k=2. Peak-preservation is not yet
shown — the k=2 runs stop at 3500, so the converged-peak comparison against k=1@9000 is
confounded. **The kl test must run to the full 9000 horizon.**

## Plan

1. **Implement** a `cw_picard_iters` loop in `CurvatureWhitenLoRA` (`_cw_apply_grouped` and
   `_cw_apply_per_pair`, kept equivalent), forming $\tilde g_A^{(n)},\tilde g_B^{(n)}$ as above.
   New CLI flag `--cw_picard_iters` (default 1 ⇒ existing runs unchanged). Two implementation
   points: (a) use the **coupled** $D$ already in `pair_state`
   ($\operatorname{diag}\mathrm{EMA}(g_B S_{\text{curv},B}^{-1}g_B^\top)$), not a raw
   $\operatorname{diag}(gg^\top)$, so the cross and main step share the metric; (b) add the cross
   term to the momentum'd covector at solve-time, recomputed each iter from the physical-scale
   $\mathrm dB^{(n)}$ (σ_max=ρ), **not** folded into the EMA — kl re-pins magnitude only at the
   end, so there is no cross-iter normalization drift.
2. **Validate before any sweep.** (a) k=1 bit-identical to the current optimizer. (b) Numerical
   fixed-point check on a tiny model with deterministic grads: do $(\mathrm dA,\mathrm dB)$
   stabilize as `cw_picard_iters` grows 1→2→3→4? Non-convergence means the coefficient or the
   metric choice is wrong. (c) Structural cap-off cross-check: with the polar off, large
   `cw_picard_iters` should approach the cap-off joint least-squares solution of the
   (mixed-metric) program; against AdaPreLoRA's closed form (Thm 3.2,
   $L\to D_{\text{out}},R\to D_{\text{in}}$) expect agreement only up to the exponent (AdaPreLoRA
   power $1/2$ vs kl power 1).
3. **Test** at opc r64, δ=1e-4 (the most sensitive cell), full η band, diagnostics on, **9000-step
   horizon** (`workloads.find_workload(...).horizon`). Success: band-spread $0.040\to{\sim}0.020$
   via the high side (η=0.1) recovering, with best-η loss preserved ($\approx 0.757$).

## Not the lever

- **Momo / NGN** — fires on overshoot; the high-η failure is monotone convergence, so it never engages.
- **`soap_v=True`** — restores Adam $\sqrt v$ robustness but changes the optimizer to SOAP-curvature; an ablation, not a fix to kl-shampoo-polar.
- **Balance projection** (BaLoRA) — hurts at every η in a 220-step r16 pilot; dead lever, code removed.
- **More damping** (δ 1e-4→1e-3) — ≈ baseline.
