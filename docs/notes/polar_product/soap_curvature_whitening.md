# SOAP-curvature whitening and the chord-tight polar

`CurvatureWhitenLoRA` runs **SOAP-on-momentum in an affordable Kronecker
curvature basis, then the chord-tight spectral sandwich**. Registered as
`curvature-whiten-lora` and `curvature-whiten-polar-lora` (the arm that adds the
polar). This note: states the update, derives what it *is* in the two-sided
curvature-whitening language, separates the one load-bearing identity from the
heuristics, and lists the experiments that decide what to keep.

## Notation

One PEFT LoRA pair $A\in\mathbb{R}^{r\times d_{\mathrm{in}}}$,
$B\in\mathbb{R}^{d_{\mathrm{out}}\times r}$, $\Delta W=BA$, $r\ll d$. Per-factor
gradient $g_A=B^\top G$; first moment $m_A$; bias-corrected $\hat m_A$. $\varphi$
is the (soft) polar map, $\varphi(U\Sigma V^\top)=UV^\top$. Curvature factors
(EMAs, decay $\beta_{\mathrm{curv}}=0.99$):
$$
S_{\mathrm{curv},A}=\mathrm{EMA}(g_A g_A^\top)\in\mathbb{R}^{r\times r},\qquad
D_{\mathrm{in}}=\mathrm{diag}\,\mathrm{EMA}(g_A^\top g_A)\in\mathbb{R}^{d_{\mathrm{in}}}.
$$
Below everything is A-side; the B-side is symmetric ($S_{\mathrm{curv},B}=\mathrm{EMA}(g_B^\top g_B)$, $D_{\mathrm{out}}=\mathrm{diag}\,\mathrm{EMA}(g_B g_B^\top)$).

## The implemented update

SOAP step — Adam on momentum in the eigenbasis of $S_{\mathrm{curv},A}$ (the
large/$d_{\mathrm{in}}$ side stays in the coordinate basis, since its factor
$D_{\mathrm{in}}$ is diagonal):
$$
z_A=Q_A\!\left[\frac{Q_A^\top \hat m_A}{\sqrt{\hat v_A}+\epsilon}\right],\quad
Q_A=\mathrm{eigvecs}(S_{\mathrm{curv},A}),\quad
\hat v_A=\mathrm{EMA}\big((Q_A^\top g_A)^{\odot 2}\big)\in\mathbb{R}^{r\times d_{\mathrm{in}}}.
$$
Outer curvature sandwich and spectral budget:
$$
Y_A=S_{\mathrm{curv},A}^{-1/2}\,z_A\,D_{\mathrm{in}}^{-1/2},\qquad
\rho=\frac{\eta}{\sigma_{\max}(A)+\sigma_{\max}(B)},\qquad
\Delta A=-\rho\,\frac{Y_A}{\sigma_{\max}(Y_A)}.
$$
`curvature-whiten-polar-lora` replaces $z_A$ by $\varphi(z_A)$ before the
sandwich. $S^{-1/2}$ uses warm-started QR + Rayleigh eigenvalues (eigh only to
seed); relative damping $(\lambda/\lambda_{\max}+\delta)^{-1/2}$. Verified:
`tests/test_curvature_whiten_lora.py` pins the update (incl. the polar arm and
the rescale); the r256 packed smoke runs clean (~29 GB, no non-finite grads).

## What the update *is*: a two-sided sandwich with a forced $Q'$, plus a residual

This is the canonical statement of the update and the one to reason from.

**The SOAP second moment carries both curvature factors.** $\hat v_A$ is a
*full* $r\times d_{\mathrm{in}}$ array — one variance per (eigendirection,
input-column) pair. Its two marginals are exactly the small- and large-side
factors (using $Q_A$ orthogonal, so column norms are preserved):
$$
\textstyle\sum_j (\hat v_A)_{ij}=\lambda_i(S_{\mathrm{curv},A}),\qquad
\sum_i (\hat v_A)_{ij}=(D_{\mathrm{in}})_j.
$$

**Separable case $\Rightarrow$ the symmetric sandwich.** If $\hat v_A$ is rank-1,
$(\hat v_A)_{ij}=a_i b_j$, the marginals force $a_i\propto\lambda_i$,
$b_j\propto(D_{\mathrm{in}})_j$, so the entrywise division splits left/right and
the rotation collapses:
$$
\boxed{\,z_A\approx S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}\,}\qquad(\text{rank-1 }\hat v_A),
$$
since $Q_A\,\mathrm{diag}(\lambda)^{-1/2}Q_A^\top=S_{\mathrm{curv},A}^{-1/2}$ and
$\varphi$ is scale-invariant. Sandwiching gives
$$
\Delta A\propto S_{\mathrm{curv},A}^{-1/2}\,
\varphi\!\big(S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}\big)\,
D_{\mathrm{in}}^{-1/2},
$$
which is exactly the generalized two-sided program
$$
\Delta A\propto (B^\top P B)^{-1/2}\,
\varphi\!\big((B^\top P B)^{-1/2}\,m\,Q'^{-1/2}\big)\,Q'^{-1/2}
$$
with small side $B^\top P B=S_{\mathrm{curv},A}$ and large side $Q'=D_{\mathrm{in}}$.

- **$Q'$ is forced, not chosen.** It is the column-marginal of the same
  $\hat v_A$ SOAP already maintains — the gradient's per-input-feature energy. So
  "SOAP plus the outer $D_{\mathrm{in}}^{-1/2}$" is *one* symmetric sandwich, not
  two stacked conditioners; the input axis is conditioned inside $\varphi$ (via
  $\hat v_A$'s columns) **and** outside (the explicit $D_{\mathrm{in}}^{-1/2}$).

**What SOAP adds over the sandwich: the non-separable residual.** When
$\hat v_A$ is *not* rank-1, SOAP divides by the full $\sqrt{(\hat v_A)_{ij}}$
rather than $\sqrt{\lambda_i (D_{\mathrm{in}})_j}$ — curvature where an input
feature's variance depends on the latent eigendirection, structure no Kronecker
$P\otimes Q'$ can represent. Size is a cheap a-priori diagnostic:
$$
\rho_{\mathrm{nonsep}}=\frac{\lVert \hat v_A-\hat a\,\hat b^\top\rVert_F}{\lVert \hat v_A\rVert_F},\qquad \hat a,\hat b=\text{row / column marginals.}
$$
$\rho_{\mathrm{nonsep}}\!\approx\!0$: SOAP collapses to the $Q'=D_{\mathrm{in}}$
sandwich, and any explicit-$Q'$ choice can at best match it. $\rho_{\mathrm{nonsep}}$
large: SOAP's extra expressiveness is real and no Kronecker $Q'$ recovers it.

## The identity it rests on, and where it breaks

**Anchor (exact).** Whitening by the *instantaneous* Gram is the polar: for
$g_A=U\Sigma V^\top$,
$$
(g_A g_A^\top)^{-1/2}\,g_A=U\Sigma^{-1}U^\top\,U\Sigma V^\top=UV^\top=\varphi(g_A).
$$
So curvature whitening is an EMA generalization of "polar the gradient." In the
$\beta_{\mathrm{curv}}\!\to\!0$ limit the EMA curvature *is* the current sample's
Gram, so SOAP-whitening and the polar **coincide** and $\varphi(\mathrm{SOAP})=\varphi$
is redundant. They separate only as the EMA averages over samples.

**Saturation, and r256.** The "$\varphi(S^{-1/2}m)\approx S^{-1}m$" reading (a
full inverse-curvature Newton step) holds only when $S^{-1/2}m$ is already
near-orthogonal — the preconditioning-saturation regime. That holds at
$r{=}16/64$ but **fails at r256** ($B$ at ${\sim}14\%$ rank,
$S^{-1/2}$ spread ${\sim}340$, $\mathrm dA$ rotated ${\sim}60°$). So **all
discriminating comparisons run at r256**; $r{=}16/64$ collapse the variants.

## The polar is a separate ingredient — the open question

The polar is *not* part of the curvature whitening; it flattens the
SOAP-whitened direction's singular values to 1. Two readings of what it buys, and
they make opposite predictions — this is the question to settle:

- **Polar removes the per-sample spread (additive on top of curvature).** Even
  after perfect *statistical* whitening, a single minibatch gradient has a
  random spread of singular values (some directions large by sampling luck, not
  curvature). The EMA curvature cannot remove this; $\varphi$ does, *deterministically*,
  for the current step. Under this reading $\varphi$'s value floors at the
  sampling-noise level — nonzero even with ideal curvature, and growing with $r$
  (more directions → more spread). External support: SOAP+Muon (the polar
  cleans the residual a stale/one-sided curvature leaves, which is why they
  refresh SOAP only every ~40 steps) and Newton–Muon (the polar/`msgn` is an
  *implicit output-side curvature preconditioner*, $\approx H^{-1}$).
- **Polar is a crutch for bad curvature (redundant once curvature is right).**
  KL-Shampoo: solve the two-sided statistical whitening properly and you need
  neither the polar nor Adam-grafting. Under this reading $\varphi$'s value
  $\to 0$ as the curvature estimate becomes fresh and well-conditioned.

The Anchor says these agree at $\beta_{\mathrm{curv}}\!\to\!0$; they diverge with
EMA staleness. So **$\varphi$'s marginal value is governed by the EMA gap** —
how far the smoothed curvature is from the current gradient — and the question is
whether that gap floors above zero (sampling noise) or vanishes with good
curvature.

*Preliminary signal (to confirm from logs):* at r256, `curvature-whiten-polar`
appears to beat `curvature-whiten` (polar-on > polar-off) — i.e. the polar looks
additive here. Not yet a clean logged comparison.

## KL-coupled curvature estimation (the principled $Q'$)

Our factor $S_{\mathrm{curv},A}=\mathrm{EMA}(g_A g_A^\top)$ is an *ad-hoc
one-sided* estimate. The principled one treats curvature estimation as
**covariance estimation**: pick the Kronecker preconditioner $S=S_a\otimes S_b$
that minimizes the KL divergence to the gradient second moment $M=\mathbb E[gg^\top]$,
$g=\mathrm{vec}(G)$ (Lin et al., `kl_shampoo_2509.03378.pdf`).

**Objective — KL, not Frobenius.**
$$
\mathrm{KL}(M\,\|\,S)=\tfrac12\bigl(\log\det S+\mathrm{Tr}(M\,S^{-1})\bigr)+\text{const.}
$$
KL is a divergence *on the SPD cone* — it keeps $S$ SPD and weights errors
multiplicatively (the geometry a preconditioner lives in); the Frobenius fit
Shampoo/SOAP implicitly use ignores the SPD constraint and so needs Adam
step-size grafting to fix the scale. (KL is also the log-det / von-Neumann
divergence classical quasi-Newton minimizes.)

**Stationarity gives the coupling.** With $S=S_a\otimes S_b$ (sizes
$d_a,d_b$), $\log\det(S_a\otimes S_b)=d_b\log\det S_a+d_a\log\det S_b$ and
$\mathrm{Tr}(MS^{-1})=\mathbb E\,\mathrm{Tr}(G^\top S_a^{-1}G\,S_b^{-1})$, so
$$
J=d_b\log\det S_a+d_a\log\det S_b+\mathbb E\,\mathrm{Tr}\!\bigl(G^\top S_a^{-1}G\,S_b^{-1}\bigr).
$$
The trace term is $\mathrm{Tr}(S_a^{-1}\cdot G S_b^{-1}G^\top)$, so
$$
\frac{\partial J}{\partial S_a}=d_b S_a^{-1}-S_a^{-1}\,\mathbb E[G S_b^{-1}G^\top]\,S_a^{-1}=0
\;\Longrightarrow\;
\boxed{\,S_a=\tfrac1{d_b}\mathbb E[G S_b^{-1}G^\top],\quad S_b=\tfrac1{d_a}\mathbb E[G^\top S_a^{-1}G]\,}.
$$
The coupling is the Euler–Lagrange condition, not a recipe: the cross-term ties
the factors, so the optimal $S_a$ whitens $G$ by the *other* factor's inverse
before forming its Gram. The EMA $S_a\leftarrow(1-\beta)S_a+(\beta/d_b)\,G S_b^{-1}G^\top$
is a stochastic step toward this fixed point.

**Shampoo is the one-sided corner.** $S_a=\mathbb E[GG^\top]$ is stationary only
if $S_b=I$ — Shampoo (and our $\mathrm{EMA}(g_A g_A^\top)$) solve the *one-sided*
KL fit, double-counting directions the other factor already handles.

**LoRA form.** Fitting the covariance of $g_A\in\mathbb R^{r\times d_{\mathrm{in}}}$
($S_a=S_{\mathrm{curv},A}$ latent $r\times r$; $S_b=D_{\mathrm{in}}$ constrained
diagonal $\Rightarrow$ take $\mathrm{diag}$):
$$
S_{\mathrm{curv},A}=\tfrac1{d_{\mathrm{in}}}\mathbb E[g_A D_{\mathrm{in}}^{-1}g_A^\top],
\qquad
D_{\mathrm{in}}=\tfrac1{r}\,\mathrm{diag}\,\mathbb E[g_A^\top S_{\mathrm{curv},A}^{-1}g_A].
$$
Both inverses are cheap ($r\times r$, diagonal). This **derives** the consistent
$(S_{\mathrm{curv},A},D_{\mathrm{in}})$ jointly, so $Q'=D_{\mathrm{in}}$ is not a
swept knob but pinned by the joint fit — the coherent two-sided $(P,Q)$.

**Bearing on the polar fork.** In full-weight pretraining KL-Shampoo (this
coupled estimate, *no polar, no Adam*) beats SOAP/Shampoo, and beats KL-SOAP
(adding the elementwise $\hat v$ *hurts*). So **KL-Shampoo-LoRA** (dense $r\times r$
latent, coupled with $D_{\mathrm{in}}$, KL eigenvalue EMA, no $\hat v$) is the
clean "solve the curvature properly" baseline: if it matches/beats polar-on
SOAP, the polar and $\hat v$ were crutches; if polar-on still wins, the polar is
genuinely additive. *Caveat: their evidence is full-weight, not LoRA, and the
polar is absent from their setting.*

## $m$ vs $u$ as input

Raw $m$ is SOAP-native (SOAP normalizes in its own eigenbasis; the Adam direction
$u=m/(\sqrt v+\epsilon)$ would double-normalize the columns that $\hat v_A$
already scales). Since $\varphi$ is scale-invariant, $m$ vs $u$ is a pure
*direction* ablation, not a magnitude one.

## Historical `--curvature_whitening` flag

`--curvature_whitening` (commit `67cfea4`) is a *different, older* path: it swaps
the geometric Gram $B^\top B$ for $S_{\mathrm{curv},A}$ inside the existing
chord-tight pipeline, on the **Adam direction** $u_A$, with the matrix $S^{-1/2}$
sandwich and polar — $\Delta A\propto S_{\mathrm{curv},A}^{-1/2}\varphi(S_{\mathrm{curv},A}^{-1/2}u_A)$.
It is one-sided (no $D_{\mathrm{in}}$) and not the SOAP-on-$m$ variant above.

## Empirical status (r256, preliminary)

Older curvature A/B (OLMo r256, full polar $k{=}1$, step 9000, single seed,
$\sigma=0.0007$): curvature-ON best $0.7394$ (lr 3e-3), curvature-OFF best
$0.7414$ (lr 1e-2), AdamW $0.7524$ (lr 1e-4). ON edges OFF by ${\sim}2.9\sigma$ at
its best lr, **but confounded**: curvature shifts the optimal lr down and worsens
high-lr robustness (lr 1e-2: ON $0.7521$ vs OFF $0.7414$). A narrow-basin edge,
single seed, off the 4k horizon — the geometric-vs-curvature axis is not settled.

## What to do next (all at r256, one change at a time)

1. **`ρ_nonsep` diagnostic** (cheapest, a-priori). Log
   $\lVert \hat v_A-\hat a\hat b^\top\rVert_F/\lVert\hat v_A\rVert_F$ on a run. If
   small, SOAP's non-Kronecker richness has nothing to capture and the clean
   Kronecker sandwich should match it; if large, SOAP is doing real work no $Q'$
   can.
2. **Polar on/off** = `curvature-whiten-polar` vs `curvature-whiten`. Confirm the
   preliminary signal and quantify the polar's additive value.
3. **Staleness sweep** (`precond_refresh_every`, $\beta_{\mathrm{curv}}$). If
   $\varphi$'s benefit (run 2) *grows* with staleness and floors above zero with
   fresh curvature → polar removes sampling spread (keep it); if it $\to 0$ with
   fresh curvature → polar was a crutch (solve the curvature instead).
4. **Explicit $Q'$ sweep** $Q'\in\{I, D_{\mathrm{in}}, \tilde G^\top\tilde G\}$
   ($\tilde G=g_B A+B g_A$) in the clean Kronecker sandwich (input $m$, no
   $\hat v_A$). $I$→$D_{\mathrm{in}}$: does the input axis matter? best-of vs
   current SOAP: does the non-separable $\hat v_A$ beat any Kronecker $Q'$?
5. **KL-Shampoo-LoRA** — the principled version of (4): instead of choosing
   $Q'$, estimate $(S_{\mathrm{curv},A}, D_{\mathrm{in}})$ by the coupled KL fit
   (derived above), no $\hat v_A$, polar on/off. Tests whether properly-solved
   curvature makes the polar (and $\hat v_A$) redundant.

## Grounding

- Implementation: `CurvatureWhitenLoRA` in `lora_playground/optim.py`; tests `tests/test_curvature_whiten_lora.py`.
- Older one-sided path: `optim.py` chord-tight `curvature_whitening` flag.
- SOAP algorithm (real vs idealized): `docs/papers/soap_2409.11321.pdf`, Algs 1–3, Claim 1.
- Generalized two-sided program (the $(B^\top P B)^{-1/2}\varphi(\cdots)Q'^{-1/2}$ sandwich): `related_work_2026_05.md` §10.6.
- SOAP+Muon as iterative whitening (polar cleans the stale/one-sided residual): `docs/papers/soap_muon_vyas.pdf`.
- Polar/`msgn` as an implicit output-side curvature preconditioner ($\approx H^{-1}$): `docs/papers/newton_muon_2604.01472.pdf`.
- KL view of Shampoo/SOAP curvature, two-sided equilibrium (no polar needed if solved right): `docs/papers/kl_shampoo_2509.03378.pdf`.
- Saturation: `preconditioning_saturation_2026_05_03.md`. r256 whiten lag: `chord_tight_whiten_lag_r256.md`. Factor-conditioning: `factor_conditioning_hypothesis.md`. KFAC-LoRA plan: `docs/plans/optimizer_ideas.md`.
