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
the rescale); the high-rank packed smoke runs clean (no non-finite grads).

## What the update *is*: a two-sided sandwich with a forced $Q'$, plus a residual

This is the canonical statement of the update and the one to reason from.

**Background — the generalized two-sided program** (self-contained from
`related_work_2026_05.md` §10.6). Fix a Kronecker SPD metric on weight space,
$\langle X,Y\rangle_{\mathcal F}=\operatorname{tr}(X^\top P\,Y\,Q')$, with $P\succ0$
($d_{\mathrm{out}}\times d_{\mathrm{out}}$) and $Q'\succ0$
($d_{\mathrm{in}}\times d_{\mathrm{in}}$; written $Q'$ to avoid clashing with the
eigenvector matrix $Q_A$ below). The LoRA update is the projection of the
$\mathcal F$-preconditioned gradient step $T=P^{-1}G\,Q'^{-1}$ onto the reachable
tangent $J=B\,\mathrm dA+\mathrm dB\,A$, measured in the $\mathcal F$-norm, under a
per-block spectral cap $\lVert Y_A\rVert_2\le\tau$. Each per-block norm is the
Frobenius norm of a whitened variable,
$\lVert B\,\mathrm dA\rVert_{\mathcal F}=\lVert Y_A\rVert_F$ with
$Y_A=(B^\top P B)^{1/2}\,\mathrm dA\,Q'^{1/2}$; the linear term collapses to the raw
factor gradient $g_A=B^\top G$; and the cap is the spectral LMO
$Y_A=\tau\,\varphi(H_A)$ on the whitened gradient
$H_A=(B^\top P B)^{-1/2}\,g_A\,Q'^{-1/2}$. Unwhitening gives the per-block update
$$
\mathrm dA=\tau\,(B^\top P B)^{-1/2}\,\varphi\!\big((B^\top P B)^{-1/2}\,g_A\,Q'^{-1/2}\big)\,Q'^{-1/2}.
$$
Two named methods are corners of this family: Frobenius $P=Q'=I$ with the polar and
the per-factor Adam input is **chord-tight**; curvature $P=L^{1/2}$, $Q'=R^{1/2}$
with the cap off is **AdaPreLoRA**. This section's claim is that **SOAP-on-momentum
with the chord-tight sandwich is a third instance**, and the $(P,Q')$ it picks is
*forced* by the SOAP second moment rather than chosen.

**The SOAP second moment carries both curvature factors.** $\hat v_A$ is a
*full* $r\times d_{\mathrm{in}}$ array — one variance per (eigendirection,
input-column) pair — and its two marginals are exactly the small- and large-side
curvature factors.

**Lemma 1 (SOAP marginals).** With $Q_A$ orthogonal and held fixed across the EMA,
$$
\textstyle\sum_j (\hat v_A)_{ij}=\lambda_i(S_{\mathrm{curv},A}),\qquad
\sum_i (\hat v_A)_{ij}=(D_{\mathrm{in}})_j.
$$
*Proof:* Appendix A.

**Proposition 2 (separable collapse).** If $\hat v_A$ is rank-1,
$(\hat v_A)_{ij}=a_i b_j$, then by Lemma 1 the marginals force $a_i\propto\lambda_i$
and $b_j\propto(D_{\mathrm{in}})_j$, the entrywise division splits left/right, and
the rotation collapses:
$$
\boxed{\,z_A\approx S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}\,}\qquad(\text{rank-1 }\hat v_A),
$$
since $Q_A\,\mathrm{diag}(\lambda)^{-1/2}Q_A^\top=S_{\mathrm{curv},A}^{-1/2}$ and
$\varphi$ is scale-invariant (*Proof:* Appendix B). Sandwiching gives
$$
\Delta A\propto S_{\mathrm{curv},A}^{-1/2}\,
\varphi\!\big(S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}\big)\,
D_{\mathrm{in}}^{-1/2},
$$
which is exactly the generalized program above (input $m=\hat m_A$) with small side
$B^\top P B=S_{\mathrm{curv},A}$ and large side $Q'=D_{\mathrm{in}}$.

- **$Q'$ is forced, not chosen.** It is the column-marginal of the same
  $\hat v_A$ SOAP already maintains — the gradient's per-input-feature energy. So
  "SOAP plus the outer $D_{\mathrm{in}}^{-1/2}$" is *one* symmetric sandwich, not
  two stacked conditioners; the input axis is conditioned inside $\varphi$ (via
  $\hat v_A$'s columns) **and** outside (the explicit $D_{\mathrm{in}}^{-1/2}$).

**Remark (the update is a native $\varphi(\text{Shampoo})$ — SOAP Claim 1).** Once
$\hat v_A$ is rank-1, the polar arm computes the polar of a Shampoo step, assembled
from two pieces that were never designed for it:

- **The core is Shampoo.** $z_A=S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}$
  is the SOAP paper's Claim 1 — Adafactor (best rank-1 second moment) in the
  curvature eigenbasis equals idealized power-$1/2$ Shampoo; the proof's step
  $A_i=\lambda_i$ is exactly Lemma 1.
- **The polar arm makes it $\varphi(\text{Shampoo})$.**
  $\Delta A\propto S_{\mathrm{curv},A}^{-1/2}\,\varphi(z_A)\,D_{\mathrm{in}}^{-1/2}$,
  so $\varphi(z_A)$ is literally the polar of that Shampoo step. The identity
  $\varphi(\text{SOAP})\equiv\varphi(\text{power-}1/2\text{ Shampoo})$ emerges from
  composing SOAP's core with chord-tight's polar sandwich — $\varphi$ the sole
  addition.
- **Qualifiers.** Power $1/2$, not Shampoo's textbook $1/4$; and *one-sided* — we
  keep only $\mathrm{diag}(R)=D_{\mathrm{in}}$ and never rotate the large side
  ($Q_R=I$), i.e. Shampoo on the small $r$-side, Adafactor/diagonal on the
  $d_{\mathrm{in}}$ side. (Claim 1 omits momentum but notes the equivalence carries
  over.)

Grounding: `soap_2409.11321.pdf`, Alg 1–2, Claim 1.

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

## The exact limit it rests on, and where it breaks

The construction is anchored by one exact identity: whitening a single sample's
gradient by its own Gram returns the polar factor.

**Lemma 3 (zero-decay whitening is the polar).** For $g_A=U\Sigma V^\top$,
$$
(g_A g_A^\top)^{-1/2}\,g_A=U\Sigma^{-1}U^\top\,U\Sigma V^\top=UV^\top=\varphi(g_A).
$$

So curvature whitening is the EMA generalization of "polar the gradient." In the
$\beta_{\mathrm{curv}}\!\to\!0$ limit the EMA curvature *is* the current sample's
Gram, so SOAP-whitening and the polar **coincide** and $\varphi(\mathrm{SOAP})=\varphi$
is redundant; they separate only as the EMA averages over samples — that gap is
what the polar can still act on.

**The Newton-step reading holds only under saturation.** The
"$\varphi(S^{-1/2}m)\approx S^{-1}m$" reading — that the polar of the whitened
momentum is a full inverse-curvature Newton step — requires $S^{-1/2}m$ to be
already near-orthogonal, so that $\varphi$ only flattens its singular values. Once
the whitened momentum is far from orthogonal (low effective rank, a wide-spectrum
$S^{-1/2}$, a large reorientation angle), $\varphi$ *reorients* the direction
rather than merely rescaling it and the Newton-step reading fails — that is the
regime in which the polar does work the curvature alone cannot.

## The polar is a separate ingredient — the open question

Both arms apply the curvature **twice**: the SOAP core $z_A$ whitens once (Adam in
the curvature eigenbasis), then the outer sandwich
$S_{\mathrm{curv},A}^{-1/2}(\cdot)D_{\mathrm{in}}^{-1/2}$ whitens again. The only
difference between `curvature-whiten` and `curvature-whiten-polar` is what sits
*between* the two whitenings — and that is exactly the cap on/off axis of the
generalized program (Background above, §10.6).

**No polar = cap off = the full inverse-curvature step.** With nothing between them
the two half-power whitenings compound. In the rank-1 limit
($z_A=S_{\mathrm{curv},A}^{-1/2}\hat m_A D_{\mathrm{in}}^{-1/2}$, Prop 2),
$$
\Delta A\propto S_{\mathrm{curv},A}^{-1/2}\,z_A\,D_{\mathrm{in}}^{-1/2}
=S_{\mathrm{curv},A}^{-1}\,\hat m_A\,D_{\mathrm{in}}^{-1},
$$
the half-powers merging into a full inverse — exactly §10.6's cap-off corner
$\Delta A\propto(B^\top P B)^{-1}g_A\,Q'^{-1}$ (the AdaPreLoRA, +gauge row). Its
motivation is the unconstrained second-order step: precondition by the full inverse
curvature. (Away from rank-1 the compounding still holds, but the inverse is the
non-Kronecker one SOAP's full $\hat v_A$ implies, not a clean factored one.)

**Polar = cap on = the op-norm-capped spectral step.** $\varphi$ flattens $z_A$'s
singular values to 1 before the second whitening, breaking the compounding: the net
update is one curvature-whitening plus a spectral (op-norm) cap — the Muon/LMO step
$\Delta A\propto S_{\mathrm{curv},A}^{-1/2}\varphi(z_A)D_{\mathrm{in}}^{-1/2}$. Here
$\varphi$ *is* the cap.

**The two arms coincide under saturation and separate at high rank.** When $z_A$ is
already near-orthogonal, $\varphi(z_A)\approx z_A$, so cap-on and cap-off agree and
both reduce to the same Newton step — the saturation regime of the previous section,
which is why $r{=}16/64$ do not discriminate the arms. They separate exactly when
$z_A$ is far from orthogonal (stale EMA, or high rank), where $\varphi$ reorients it
substantially.

So both arms have clean motivations — an uncapped Newton step (no polar) vs an
op-norm-capped spectral step (polar) — and **the open question is whether the cap
earns its keep.** Write $v(\varphi)$ for the gap $\varphi$ opens between the two
(the value of the cap); it is nonzero exactly in the non-saturation regime above.
Two readings predict opposite behavior of $v(\varphi)$ as the curvature estimate
improves:

- **Reading A — polar removes per-sample spread (additive on top of curvature).**
  - *Mechanism:* even under perfect *statistical* whitening a single minibatch
    gradient has random singular-value spread (directions large by sampling luck,
    not curvature); the EMA cannot remove it, $\varphi$ does, deterministically,
    for the current step.
  - *Prediction:* $v(\varphi)$ floors at the sampling-noise level — nonzero even
    with ideal curvature, growing with $r$.
  - *External support:* SOAP+Muon (the polar cleans the residual a stale/one-sided
    curvature leaves; they refresh SOAP only every ${\sim}40$ steps); Newton–Muon
    ($\varphi$/`msgn` as an implicit output-side curvature preconditioner $\approx H^{-1}$).
- **Reading B — polar is a crutch for bad curvature (redundant once curvature is right).**
  - *Mechanism:* solve the two-sided statistical whitening properly (KL-Shampoo)
    and neither the polar nor Adam-grafting is needed.
  - *Prediction:* $v(\varphi)\to0$ as the curvature estimate becomes fresh and
    well-conditioned.

**The deciding quantity is the EMA gap.** Lemma 3 makes the two readings agree at
$\beta_{\mathrm{curv}}\!\to\!0$ — there curvature whitening already equals the polar,
$z_A$ is orthogonal, and $v(\varphi)=0$; they diverge with EMA staleness. The open
question is whether $v(\varphi)$ floors above zero (Reading A) or vanishes with good
curvature (Reading B).

*Preliminary signal (to confirm from logs):* in the high-rank regime
`curvature-whiten-polar` appears to beat `curvature-whiten` (polar-on > polar-off),
consistent with Reading A. Not yet a clean logged comparison.

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

**Proposition 4 (KL stationarity coupling).** The Kronecker factors minimizing
$\mathrm{KL}(M\,\|\,S_a\otimes S_b)$ satisfy the coupled fixed point
$$
\boxed{\,S_a=\tfrac1{d_b}\mathbb E[G S_b^{-1}G^\top],\quad S_b=\tfrac1{d_a}\mathbb E[G^\top S_a^{-1}G]\,}.
$$

*Proof.* With $S=S_a\otimes S_b$ (sizes $d_a,d_b$),
$\log\det(S_a\otimes S_b)=d_b\log\det S_a+d_a\log\det S_b$ and
$\mathrm{Tr}(MS^{-1})=\mathbb E\,\mathrm{Tr}(G^\top S_a^{-1}G\,S_b^{-1})$, so
$$
J=d_b\log\det S_a+d_a\log\det S_b+\mathbb E\,\mathrm{Tr}\!\bigl(G^\top S_a^{-1}G\,S_b^{-1}\bigr).
$$
The trace term is $\mathrm{Tr}(S_a^{-1}\,G S_b^{-1}G^\top)$, so
$$
\frac{\partial J}{\partial S_a}=d_b S_a^{-1}-S_a^{-1}\,\mathbb E[G S_b^{-1}G^\top]\,S_a^{-1}=0,
$$
and symmetrically in $S_b$; rearranging gives the boxed pair. ∎

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

**Remark ($m$ vs $u$ as the SOAP input).** Raw $m$ is SOAP-native — SOAP
normalizes in its own eigenbasis, so the Adam direction $u=m/(\sqrt v+\epsilon)$
would double-normalize the columns $\hat v_A$ already scales. Since $\varphi$ is
scale-invariant, $m$ vs $u$ is a pure *direction* ablation, not a magnitude one.

## The older one-sided curvature path

A distinct, earlier path swaps the geometric Gram $B^\top B$ for
$S_{\mathrm{curv},A}$ inside the existing chord-tight pipeline, acting on the
**Adam direction** $u_A$ with the matrix $S^{-1/2}$ sandwich and polar —
$\Delta A\propto S_{\mathrm{curv},A}^{-1/2}\varphi(S_{\mathrm{curv},A}^{-1/2}u_A)$.
It is one-sided (no $D_{\mathrm{in}}$) and not the SOAP-on-$m$ variant above (code
pointer in Grounding).

## Empirical status (high-rank regime, preliminary)

An older curvature A/B in the high-rank regime (full polar $k{=}1$, single seed,
$\sigma=0.0007$): curvature-ON edges curvature-OFF by ${\sim}2.9\sigma$ at its best
lr, **but confounded** — curvature shifts the optimal lr down and worsens high-lr
robustness, so the two arms' best-lr cells sit in different basins. A narrow-basin
edge, single seed, off the canonical horizon — the geometric-vs-curvature axis is
not settled.

## What to do next (in the discriminating regime, one change at a time)

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
- Older one-sided path (`--curvature_whitening`, commit `67cfea4`): `optim.py` chord-tight `curvature_whitening` flag.
- SOAP algorithm (real vs idealized): `docs/papers/soap_2409.11321.pdf`, Algs 1–3, Claim 1.
- Generalized two-sided program (the $(B^\top P B)^{-1/2}\varphi(\cdots)Q'^{-1/2}$ sandwich): `related_work_2026_05.md` §10.6.
- SOAP+Muon as iterative whitening (polar cleans the stale/one-sided residual): `docs/papers/soap_muon_vyas.pdf`.
- Polar/`msgn` as an implicit output-side curvature preconditioner ($\approx H^{-1}$): `docs/papers/newton_muon_2604.01472.pdf`.
- KL view of Shampoo/SOAP curvature, two-sided equilibrium (no polar needed if solved right): `docs/papers/kl_shampoo_2509.03378.pdf`.
- Saturation: `preconditioning_saturation_2026_05_03.md`. r256 whiten lag: `chord_tight_whiten_lag_r256.md`. Factor-conditioning: `factor_conditioning_hypothesis.md`. KFAC-LoRA plan: `docs/plans/optimizer_ideas.md`.

## Appendix A: proof of Lemma 1 (SOAP marginals)

Recall the objects (all A-side, $g_A\in\mathbb R^{r\times d_{\mathrm{in}}}$ the
factor gradient): $S_{\mathrm{curv},A}=\mathrm{EMA}(g_Ag_A^\top)$ with eigenpairs
$(\lambda_i,Q_A)$; $D_{\mathrm{in}}=\mathrm{diag}\,\mathrm{EMA}(g_A^\top g_A)$; and
the SOAP second moment $\hat v_A=\mathrm{EMA}\big((Q_A^\top g_A)^{\odot2}\big)$, with
$Q_A$ held fixed across the EMA. $\odot$ is the entrywise (Hadamard) square.

**Lemma 1 (SOAP marginals, restated).**
$\sum_j(\hat v_A)_{ij}=\lambda_i$ and $\sum_i(\hat v_A)_{ij}=(D_{\mathrm{in}})_j$.

The identity is one instance of a fact about Hadamard squares.

**Fact A.** For any matrix $M$, the Hadamard square $M^{\odot2}$ has row-marginals
$\mathrm{diag}(MM^\top)$ and column-marginals $\mathrm{diag}(M^\top M)$ (the squared
row/column norms).

*Proof of Lemma 1.* Apply Fact A under the EMA — legitimate since the EMA is
linear and $Q_A$ is fixed — to $M=Q_A^\top g_A$. Orthogonality gives
$MM^\top=Q_A^\top g_Ag_A^\top Q_A$ and $M^\top M=g_A^\top g_A$, so the marginals are
$\mathrm{diag}\,(Q_A^\top S_{\mathrm{curv},A}Q_A)=\lambda$ (as $Q_A$ diagonalizes
the same EMA) and $\mathrm{diag}\,\mathrm{EMA}(g_A^\top g_A)=D_{\mathrm{in}}$. $\blacksquare$

**Corollary A.1 (shared total).** Both marginals sum to
$\mathrm{tr}\,S_{\mathrm{curv},A}=\mathrm{EMA}\lVert g_A\rVert_F^2$, so
$\hat v_A/\mathrm{tr}\,S_{\mathrm{curv},A}$ is a coupling whose marginals are the
normalized spectrum and input energies; $\rho_{\mathrm{nonsep}}$ is its distance
from the product of those marginals.

## Appendix B: proof of Proposition 2 (separable collapse)

Recall the SOAP step is the Adam direction divided in the curvature eigenbasis,
$z_A=Q_A\big[(Q_A^\top\hat m_A)/(\sqrt{\hat v_A}+\epsilon)\big]$ (the division
elementwise).

**Proposition 2 (separable collapse, restated).** If $\hat v_A=ab^\top$ is rank-1
and nonzero, then, dropping $\epsilon$, $z_A\propto S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}$,
and the proportionality constant is immaterial because the sandwich feeds $z_A$
through the scale-invariant $\varphi$.

Two facts about a nonnegative rank-1 array reduce the elementwise Adam division to
a matrix sandwich.

**Fact B1.** A nonzero entrywise-nonnegative rank-1 matrix factors as $ab^\top$
with $a,b\ge0$ (mixed signs in $a$ would force $b\equiv0$).

**Fact B2.** Elementwise division by $ab^\top$ is the two-sided diagonal scaling
$X\mapsto\mathrm{diag}(a)^{-1/2}X\,\mathrm{diag}(b)^{-1/2}$, since
$\sqrt{a_ib_j}=\sqrt{a_i}\sqrt{b_j}$.

*Proof of Proposition 2.* $\hat v_A\ge0$ entrywise, so Fact B1 applies and the
square roots are real. Lemma 1 fixes the marginals,
$a_i\lVert b\rVert_1=\lambda_i$ and $b_j\lVert a\rVert_1=(D_{\mathrm{in}})_j$, i.e.
$a\propto\lambda$ and $b\propto D_{\mathrm{in}}$. By Fact B2 and $z_A=Q_A[\,\cdot\,]$,
$$
z_A=Q_A\,\mathrm{diag}(a)^{-1/2}Q_A^\top\,\hat m_A\,\mathrm{diag}(b)^{-1/2}
\;\propto\;S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2},
$$
using $Q_A\,\mathrm{diag}(\lambda)^{-1/2}Q_A^\top=S_{\mathrm{curv},A}^{-1/2}$ and
$D_{\mathrm{in}}$ diagonal; the positive scalar is removed by $\varphi(cX)=\varphi(X)$.
(Strict positivity $a_i>0$ holds wherever $\lambda_i>0$, which the relative damping
guarantees.) $\blacksquare$
