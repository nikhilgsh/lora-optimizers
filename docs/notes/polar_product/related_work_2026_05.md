# Low-rank optimizer landscape: updates, derivations, and how ours fits

**Scope.** A unified read of four external methods against our `spectral_chord_tight_clean`
optimizer (`AdamPolarProductLoRA` in `lora_playground/optim.py:3861`; pipeline
`._chord_tight_clean_polar_pipeline` at `:5090`). Goal: write down every update *concretely*,
show how each is *derived*, and compare carefully. All external numbers are the papers'
own reported values (single-seed unless noted) — they are not our sweep data and have not
been reproduced here. PDFs in `docs/papers/`:

| short | paper | id |
|---|---|---|
| Riemannion | Bogachev et al., *LoRA meets Riemannion* | `riemannion_2507.12142.pdf` |
| iMuon | Li et al., *Intrinsic Muon* | `imuon_2605.09238.pdf` |
| AdaPreLoRA | Liu, Bian, Cai, *AdaPreLoRA* | `adaprelora_2605.08734.pdf` |
| μA / LR-scaling | Chen, Villar, Hayou, *Learning Rate Scaling across LoRA Ranks* | `mua_2602.06204.pdf` |

## 1. Shared setup and notation

LoRA adds $BA$ to the frozen weight: $A\in\mathbb{R}^{r\times d_{\text{in}}}$,
$B\in\mathbb{R}^{d_{\text{out}}\times r}$, product $\Delta W = BA$.

- $G=\partial L/\partial(\Delta W)\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ — gradient w.r.t. the **product**.
- Factor gradients (chain rule): $g_A = B^\top G$, $g_B = G A^\top$.
- Gram matrices (small, $r\times r$): $S_A = AA^\top$, $S_B = B^\top B$.
- Polar map: $\phi(M)=UV^\top$ for the SVD $M=U\Sigma V^\top$ (sets all singular values to $1$).

**The one fact behind every design.** An $A$-step $\mathrm dA$ reaches the product only as
$B\,\mathrm dA$ — $B$ *filters* the step. Where $B$ is small or ill-conditioned, an $A$-step
in those directions barely moves the product. Every "whitening" below undoes that filtering,
and each method whitens the $A$-side by the **opposite** factor's Gram $S_B$ (and the $B$-side
by $S_A$). All updates are written $A$-side; $B$-side is the mirror image.

## 2. The unifying split: LMO vs projection

Two recipes generate all of these methods, and the recipe — not the details — predicts
whether the update flattens the spectrum.

**LMO (linear maximization oracle).** Pick the update maximizing alignment with the gradient,
subject to a size cap in some norm:
$$
Z^\star = \arg\max_{\|Z\|\le\tau}\ \langle Z, M\rangle .
$$
The norm decides how the budget is spent across singular values of $M=U\Sigma V^\top$:

| norm on $Z$ | $Z^\star$ | effect on spectrum |
|---|---|---|
| Frobenius $\|Z\|_F\le\tau$ | $\tau\,M/\|M\|_F$ | keep (proportional) |
| spectral $\|Z\|_2\le\tau$ | $\tau\,\phi(M)=\tau UV^\top$ | **flatten** (all $\sigma\to\tau$) |
| nuclear $\|Z\|_*\le\tau$ | $\tau\,u_1v_1^\top$ | collapse to top direction |

The spectral case *is* the polar — that is the entire content of Muon. Derivations: Frobenius
by Cauchy–Schwarz; spectral by von Neumann's trace inequality
$\langle Z,M\rangle\le\sum_i\sigma_i(Z)\sigma_i(M)$, maximized at $\sigma_i(Z)\equiv\tau$;
nuclear by spectral/nuclear duality.

**Projection.** Build a preconditioned ideal step $T$ in weight space, then find the closest
*reachable* (rank-$\le 2r$) change, measured in some norm:
$$
\min_{\mathrm dA,\mathrm dB}\ \bigl\|\,(B\,\mathrm dA + \mathrm dB\,A) - T\,\bigr\|^2 .
$$
A least-squares projection is linear, so it **keeps** the gradient's singular values — no polar.

**This is the deep fork.** LMO + spectral norm $\Rightarrow$ polar (flatten). Projection
$\Rightarrow$ keep spectrum. iMuon, Riemannion, and ours are LMO/spectral; AdaPreLoRA, SOAP,
ScaledGD are projection-style. So "does the polar help?" is really "is the spectral-norm
trust region the right way to spend the step budget, or should we keep the singular values?"

## 3. The updates, derived ($A$-side)

### 3.1 Baselines

**SGD on factors:** $\mathrm dA = -\eta\,g_A = -\eta\,B^\top G$. Product moves by
$B\,\mathrm dA = -\eta\,(BB^\top)G$ — squashed in $B$'s weak directions. No correction.

**Adam on factors (our AdamW baseline):**
$\mathrm dA = -\eta\,m_A/(\sqrt{v_A}+\varepsilon)$ with $m_A,v_A$ the running mean / mean-square
of $g_A$. Fixes per-entry scale; the filtering by $B$ is untouched. The μA paper shows that with
the PEFT-default init ($B{=}0$, $A$ random) and multiplier $1$ — our convention, since
`lora_alpha = lora_r` gives scale $\alpha/r = 1$ — the optimal LR drifts as
$\eta^\star\propto r^{-1/2}$ (`mua_2602.06204.pdf`, Cor. 4.4, Fig. 2a). It also shows rank-invariance
does not require spectral machinery: two routes reach it — the multiplier $\alpha=r^{-1}$
(Cor. 4.4, $\gamma{=}1$), or init $B$ with $\alpha=1$ (Cor. 4.6). So rank-invariance alone is not a
spectral selling point.

### 3.2 ScaledGD / Riemannian-preconditioned LoRA (= iMuon's Frobenius case)

*Derivation:* treating $B$ fixed, the natural-gradient step whitens by the opposite Gram:
$$
\mathrm dA = -\eta\,S_B^{-1}\,g_A = -\eta\,(B^\top B)^{-1}B^\top G .
$$
Product moves by $B\,\mathrm dA = -\eta\,B(B^\top B)^{-1}B^\top G = -\eta\,P\,G$, with $P$ the
projector onto $B$'s column space: $B$'s scale is divided out, singular values **kept**, no polar.
With Adam moments instead of $g_A$ this is "Scaled AdamW."

### 3.3 iMuon (`imuon_2605.09238.pdf`, Cor. 4.1)

*Driver:* parametrization invariance is the axiom. The factorization has a gauge freedom
$(B,A)\sim(BN^{-1},NA)$; iMuon demands the product update be identical across gauges and
*derives* the unique metric that enforces it — the $\mathrm{GL}(r)$-invariant metric.

*Derivation.* The recipe is: whiten the step by $S_B^{-1/2}$, run the §2 Euclidean LMO, un-whiten.
Concretely, the intrinsic-norm LMO measures the $A$-step's size *after* the metric,
$\|S_B^{1/2}\mathrm dA\|\le\tau$. Substituting $Z_A = S_B^{1/2}\mathrm dA$ (so
$\mathrm dA = S_B^{-1/2}Z_A$) turns the problem into a plain Euclidean LMO:
$$
\max_{\|Z_A\|\le\tau}\ \langle Z_A, H_A\rangle,
\qquad H_A := S_B^{-1/2}g_A\ \text{(the scaled gradient)} .
$$
Solve for $Z_A^\star$, then map back with $\mathrm dA = S_B^{-1/2}Z_A^\star$. The two norm choices
of §2 then diverge for one reason — the Frobenius solution is *linear* in $H_A$, the spectral one
is *nonlinear*:

- **Frobenius:** $Z_A^\star = \tau H_A/\|H_A\|_F$ is linear, so the two half-powers merge into a
  full inverse: $\mathrm dA = \tau\,S_B^{-1}g_A/\|H_A\|_F$ — spectrum kept (this is ScaledGD).
- **Spectral:** $Z_A^\star = \tau\,\phi(H_A)$ is nonlinear, so the inner half-power stays trapped
  inside the polar — the **sandwich**, spectrum flattened:
  $$
  \boxed{\ \mathrm dA = -\eta\,S_B^{-1/2}\,\phi\!\bigl(S_B^{-1/2}\,g_A\bigr)\ } \qquad\text{(iMuon)}
  $$

That linear-merge-vs-nonlinear-split is the *entire* difference between the no-polar full-inverse
(Frobenius) and the polar half-power sandwich (spectral); both run the identical "scale by
$S_B^{-1/2}$, solve Euclidean LMO, scale back." Factors update directly
($A\leftarrow A-\eta\,\mathrm dA$); no product SVD.

*Norm bound.* $\|B\,\mathrm dA\|_2=\eta$ exactly, independent of $B$'s conditioning, because
$BS_B^{-1/2}=U_BV_B^\top$ is a partial isometry (orthonormal columns times orthonormal rows; iMuon
App. H.1 proves this for the mirror block $\|\dot B A\|_2\le\tau$, and the $A$-side is identical by
symmetry). This is why iMuon needs **no** $\sigma_{\max}$ rescaling. Its *convergence* guarantee
comes instead from the intrinsic-norm bound $\|\xi^\star\|_x^2\le 2r\tau^2$ (Lemma D.3(ii);
$C_\varphi=2r$ in D.6). See §7.

### 3.4 Riemannion (`riemannion_2507.12142.pdf`, Eq. 12 + retraction)

*Driver:* delete the factors. Optimize $X=\Delta W$ on the rank-$r$ manifold directly, with the
plain metric $G_x=I$; the gauge problem cannot arise because there are no factors mid-step.

*Update:*
$$
\text{grad}=P_T(G),\quad M=\beta\,(\text{transported }M)+\text{grad},\quad
\tilde M = P_T\bigl(\phi(M)\bigr),\quad X_{\text{new}}=\mathrm{SVD}_r\!\bigl(X-\eta\tilde M\bigr).
$$
$P_T$ projects onto the rank-$2r$ tangent at $X$ (needs only $X$'s maintained singular subspaces).
One polar, on the **combined** tangent. **Does it SVD each step?** Yes, but never the full
$d_{\text{out}}\times d_{\text{in}}$ matrix: $X-\eta\tilde M$ is in thin factored form (factors
$\sim m\times 2r$ and $n\times 2r$), so the retraction is two thin QRs + one **$2r\times 2r$**
SVD ($O((m{+}n)r^2 + r^3)$); the orthogonalization step is likewise QR + a small SVD. No
Newton–Schulz — exact small SVD/QR (contrast our gram-NS, chosen for tensor-core / launch
behavior). The tangent projection only *approximately*
satisfies the spectral constraint — reported singular values land in $(0.9,1.1)$ (their §4).

### 3.5 AdaPreLoRA (`adaprelora_2605.08734.pdf`, Thm. 3.2)

*Driver:* preserve loss curvature. It already trusts a curvature preconditioner for the full
weight; the problem is purely how to push it onto $A,B$ given the rank-deficient factor map.

*Derivation:* Adafactor tracks per-row / per-column gradient energy $L$ (length $d_{\text{out}}$),
$R$ (length $d_{\text{in}}$). The ideal full-weight step is the curvature-divided gradient
$\tilde G = L^{-1/2}G\,R^{-1/2}$. You can only realize changes $B\,\mathrm dA+\mathrm dB\,A$, so
**project** $\tilde G$ onto that set in the curvature-weighted norm
$\|Y\|_H^2=\langle Y, L^{1/2}YR^{1/2}\rangle$ (so a miss in a steep direction costs more). The
solution family is non-unique (gauge $(\mathrm dA-XA,\ \mathrm dB+BX)$); pick the member balancing
the two factors in the $H$-norm. Closed form:
$$
\mathrm dA = -\eta\,(B^\top L^{1/2}B)^{-1}\,g_A\,R^{-1/2}\,\bigl(I-\tfrac12 Q\bigr),\quad
Q=R^{1/2}A^\top(AR^{1/2}A^\top)^{-1}A .
$$
Reading it: $(B^\top L^{1/2}B)^{-1}$ is a **curvature-weighted** opposite-Gram inverse (set
$L=R=I$ and it becomes ScaledGD's $(B^\top B)^{-1}$); $R^{-1/2}$ whitens the big side by column
curvature; $(I-\tfrac12 Q)$ is the gauge balance. **No polar** — a linear solve, spectrum kept.
Net: **AdaPreLoRA ≈ ScaledGD + curvature-weighting + gauge balance.** Memory
$O((d_{\text{in}}{+}d_{\text{out}})r)$ — but per-step *compute* is $O(d_{\text{in}}d_{\text{out}})$:
forming the Adafactor energies $L,R$ requires the full product gradient $G$ (you cannot recover its
row/col energies from $g_A,g_B$), which the factor-only LoRA path never materializes.

### 3.6 Ours — chord-tight

$$
\mathrm dA = -\rho\,\frac{S_B^{-1/2}\,\phi\!\bigl(S_B^{-1/2}\,u_A\bigr)}{\sigma_{\max}\!\bigl(S_B^{-1/2}\,\phi(S_B^{-1/2}\,u_A)\bigr)},
\qquad \rho=\frac{\eta}{\sigma_{\max}(A)+\sigma_{\max}(B)} .
$$
This is iMuon's spectral sandwich (§3.3) with three additions:

1. **Adam input.** The polar input is the Adam direction $u_A=m_A/(\sqrt{v_A}+\varepsilon)$, not the
   raw gradient.
2. **$\eta$-pinning.** The $\rho$ prefactor and the $\sigma_{\max}(\cdot)$ denominator pin the factor
   step to $\sigma_{\max}(\mathrm dA)=\rho$, and hence the product tangent
   $J=B\,\mathrm dA+\mathrm dB\,A$ to $\|J\|_2\le\eta$. It
   also unit-op-norm-normalizes the polar *input* first.
3. **Cross-coupling correction.** An optional second-order term at $k{\ge}2$ (§10).

The soft-polar SSC variant (stable-rank target $\kappa$) is the Ky-Fan / Schatten interpolation
iMuon also lists (their App. C).

## 4. The whole space on one grid

Each method: take a raw direction, optionally whiten the small ($r\times r$) side, optionally
whiten the big side, optionally polar.

| method | raw direction | small-side whiten | big-side whiten | polar (flatten $\sigma$)? | recipe |
|---|---|---|---|---|---|
| SGD | $g_A$ | — | — | no | — |
| Adam | $g_A$ | — (per-entry $\sqrt v$) | (per-entry $\sqrt v$) | no | — |
| ScaledGD / Scaled-AdamW | $g_A$ or Adam | $S_B^{-1}$ (geometric) | — | **no** | projection (Frob LMO) |
| AdaPreLoRA | $g_A$ | $(B^\top L^{1/2}B)^{-1}$ (curvature) | $R^{-1/2}$ (curvature) | no | projection |
| iMuon | $g_A$ or momentum | $S_B^{-1/2}$ sandwich | — | **yes** | LMO (spectral) |
| Riemannion | — (on product $X$) | — ($G_x{=}I$) | — | yes (combined tangent) | LMO (spectral) |
| **ours (chord-tight)** | **Adam $u_A$** | **$S_B^{-1/2}$ sandwich** | — | **yes** | LMO (spectral) + $\eta$-pin |

Three knobs vary, and that is the entire design space:

1. **Raw direction** — gradient / momentum / Adam. (Ours and Scaled-AdamW are the only rows
   feeding full Adam into the geometry; iMuon and Riemannion use raw gradient or momentum only.)
2. **Where the whitening comes from** — geometric Gram $S_B$ (ours, iMuon, ScaledGD) vs observed
   curvature $L,R$ (AdaPreLoRA, SOAP).
3. **Polar or not** — flatten singular values (ours, iMuon, Riemannion) vs keep them
   (ScaledGD, AdaPreLoRA, SOAP).

Our chord-tight is the only row saying yes to all three at once. Each of the three ingredients
has a strong nearby method that *omits* it — which is exactly why isolating ours matters.

## 5. Update-norm checks ($\|\mathrm dA\|_2$, $\|B\,\mathrm dA\|_2$, $\|J\|_2$)

These are the operator norms each method actually produces — the most concrete way to see the
differences, and verifiable on any snapshot. Several are already unit-tested for ours. Here $\eta$ is the
scalar LR; for ours $\rho=\eta/(\sigma_{\max}(A)+\sigma_{\max}(B))$; $\phi$ has $\sigma_{\max}=1$;
"cond-free" = independent of factor conditioning. $J=B\,\mathrm dA+\mathrm dB\,A$ is the
first-order product change.

| method | factor step $\|\mathrm dA\|_2$ | product contribution $\|B\,\mathrm dA\|_2$ | total product step $\|J\|_2$ |
|---|---|---|---|
| SGD | $\le\eta\,\sigma_{\max}(B)\,\|G\|_2$ | $\le \eta\,\sigma_{\max}(B)^2\,\|G\|_2$ | $\le\eta(\sigma_{\max}(A)^2{+}\sigma_{\max}(B)^2)\|G\|_2$ |
| Adam | ${\sim}\,\eta$ per entry | $\propto\sigma_{\max}(B)$ | uncontrolled |
| ScaledGD | $\le\eta\,\|G\|_2/\sigma_{\min}(B)$ | $\le \eta\,\|G\|_2$ | $\le 2\eta\,\|G\|_2$ |
| iMuon | $\le \eta/\sigma_{\min}(B)$ | $=\eta$ | $\le 2\eta$ |
| Riemannion | — | — | $\|\Delta X\|_2\in[0.9,1.1]\,\eta$ |
| AdaPreLoRA | cond-dependent | — | $\le \|L^{-1/2}G R^{-1/2}\|_H$ |
| **ours** | $=\rho$ | $\le\sigma_{\max}(B)\,\rho$ | $\le\eta$ |

Derivations (each one line):

- *iMuon:* $B S_B^{-1/2}=U_BV_B^\top$ is a partial isometry and $\phi$ has unit singular values, so
  $\|B\,\mathrm dA\|_2=\eta$.
- *ScaledGD:* $B\,\mathrm dA=-\eta P_B G$ (projection onto $B$'s column space), so
  $\|B\,\mathrm dA\|_2\le\eta\|G\|_2$.
- *ours:* $\sigma_{\max}(\mathrm dA)=\rho$ by construction (we divide by $\sigma_{\max}$), and Lemma 1
  gives $\|J\|_2\le(\sigma_{\max}(A)+\sigma_{\max}(B))\rho=\eta$.

(Riemannion has no factor steps — $\mathrm dA,\mathrm dB$ are read off the retraction's SVD.)

What the table says:

1. **Factor step $\|\mathrm dA\|_2$: only ours pins it** ($=\rho$). iMuon and ScaledGD let it
   blow up as $1/\sigma_{\min}(B)$ — they bound the *product*, not the factor (the §7 trade).
2. **Product step $\|J\|_2$:** ours $\le\eta$ (tight), iMuon $\le 2\eta$, both cond-free;
   ScaledGD $\le 2\eta\|G\|_2$. Note ScaledGD/AdaPreLoRA carry an extra $\|G\|$ — **no polar
   means the step magnitude scales with the gradient size.** iMuon / ours / Riemannion magnitudes
   are set by $\eta$ alone (the polar flattens magnitude). That is "flatten vs keep" showing up
   directly in the norm.
3. **Cheap diagnostic:** on any snapshot, measure $\sigma_{\max}(\mathrm dA)$,
   $\sigma_{\max}(B\,\mathrm dA)$, $\sigma_{\max}(J)$ and check against the column.
   an existing unit test already asserts $\sigma_{\max}(\mathrm dA)\approx\rho$ for ours;
   the cross-method version is a ${\sim}10$-line probe on a stored LoRA pair.

## 6. Performance — carefully

No controlled three-way comparison exists: three groups, three benchmarks, none runs the others.

**iMuon vs Riemannion** — only head-to-head is in the iMuon paper (Riemannion as a baseline),
on iMuon's benchmarks and reimplementation (home-field caveat):

| benchmark | Riemannion | iMuon | gap |
|---|---|---|---|
| E2E, GPT-2 Med, $r{=}4$, no momentum (BLEU; Tbl 5) | 70.02 | 70.74 | $+0.72$ |
| E2E, with momentum (BLEU; Tbl 5) | 70.21 | 70.36 | $+0.15$ |
| GLUE, Mistral-7B, $r{=}16$, no momentum (avg; Tbl 9) | 82.65 | 86.25 | $+3.60$ |
| GLUE, with momentum (avg; Tbl 9) | 82.84 | 88.33 | $+5.49$ |

iMuon wins everywhere; Riemannion-with-Adam *collapses* on GLUE (82.84, below plain AdamW 88.28).
Riemannion's own paper reports it beating Adam/DoRA on commonsense (Llama-3-8B, $r{=}16$: 88.1
avg vs Adam 87.1; `riemannion_2507.12142.pdf` Tbl 1) — but iMuon is absent there, so there is no
comparison on Riemannion's home turf.

**The polar in the adaptive setting (iMuon's cleanest internal contrast).** Compare a no-polar
whitened method (keep spectrum) against iMuon (whiten + polar) on GLUE (Tbl 9). No momentum:
**Scaled GD** $80.36\to$ iMuon $86.25$ (polar $+5.89$). With momentum: **Scaled AdamW** $89.01\to$
iMuon $88.33$ (polar $-0.68$). So on Mistral/GLUE the polar's value shrinks/reverses once Adam is added. **But this
does not transfer to our setting:** ScaledGD and Scaled-Adam are independently weak here
(measured, our OLMo runs — single source, this project), which says the no-polar family is weak
in our regime and the polar is doing real work for us. Treat the iMuon GLUE-with-momentum
"drop the polar" hint as benchmark-specific, not a verdict for us.

**Stitched cross-paper E2E** (GPT-2-Medium, $r{=}4$), anchored on the shared AdamW $=68.9$ BLEU
(both papers follow the standard LoRA-E2E protocol, which is why the anchor matches exactly):

| method (E2E, GPT-2 Med, $r{=}4$, BLEU) | source | BLEU |
|---|---|---|
| AdamW | both | 68.9 |
| Scaled AdamW (ScaledGD + Adam) | iMuon Tbl 5 | 69.6 |
| LoRA-Pro AdamW | AdaPreLoRA Tbl 2 | 69.8 |
| AdaPreLoRA-AdamW | AdaPreLoRA Tbl 2 | 70.3 |
| iMuon | iMuon Tbl 5 | 70.74 |

Ordering: **iMuon (polar) ≳ AdaPreLoRA (curvature, no polar) ≳ ScaledGD
≳ AdamW.** Consistent with everything else — polar on top, no-polar projection methods
below. **Heavy caveat:** stitched across two papers; same task/model/rank and matching AdamW
anchor, but different seeds, LR grids, and decoding, so the ${\sim}0.4$ BLEU iMuon–AdaPreLoRA gap
is inside cross-paper noise. Directional, not controlled.

## 7. Two sharp technical contrasts with our optimizer

**Our $\sigma_{\max}$ rescaling is the runtime factor-rescaling iMuon proves unnecessary — but it
does double duty.** The A-side update is, with $\phi$ the polar map and $u_A$ the Adam direction,
$$
\mathrm dA=\rho\,\frac{S_B^{-1/2}\,\phi\big(S_B^{-1/2}u_A\big)}{\sigma_{\max}\!\big(S_B^{-1/2}\phi(S_B^{-1/2}u_A)\big)},
\qquad \rho=\frac{\eta}{\sigma_{\max}(A)+\sigma_{\max}(B)} .
$$
The denominator $\sigma_{\max}(S_B^{-1/2}\phi)$ does three separable jobs:

1. **Bound the product step — redundant.** $BS_B^{-1/2}=U_BV_B^\top$ is a partial isometry, so
   $B\,\mathrm dA$ has unit singular values *before* the rescale:
   $\|B\,\mathrm dA\|_2=\rho$ regardless of $\operatorname{cond}(B)$. iMuon proves exactly this
   (App. H.1, mirror block $\|\dot B A\|_2\le\tau$; A-side by symmetry) and so uses a fixed $\tau$
   with no $\sigma_{\max}$. Its *convergence* guarantee is the intrinsic-norm bound
   $\|\xi^\star\|_x^2\le 2r\tau^2$ (Lemma D.3(ii); $C_\varphi=2r$, D.6).
2. **Crude conditioning control — not redundant.** When $\phi$ lands in $B$'s weak directions,
   $\sigma_{\max}(S_B^{-1/2}\phi)\approx 1/\sigma_{\min}(B)$, so the division shrinks the step there.
   this plausibly matters most at $r{=}256$ (the factor-conditioning hypothesis). A targeted version
   of the same instinct is relative damping $S_B+\epsilon\lambda_{\max}I$.
3. **LR-transfer across $r$ (§9) — not redundant.** Via $\rho$: dividing by
   $\sigma_{\max}(A)+\sigma_{\max}(B)$, which grows with $r$, pins the product spectral norm to
   $\eta$ independent of $r$.

So dropping the denominator is a real ablation (it removes jobs 2 and 3), not a free
simplification — and it must be checked against rank transfer, not just final loss.

**iMuon's factor-conditioning robustness is real but conditional.** Rescale a trained head's
factors by $(\,\alpha B_0,\ \alpha^{-1}A_0)$, $\alpha=10^3$ (same product, imbalanced): intrinsic
methods hold at $33.9$–$34.3\%$, Euclidean factor methods collapse to $6.8$–$10.0\%$ (iMuon
Tbl 11). But with *balanced* init, all methods are within $0.11$ (Tbl 12), and at high condition
number Euclidean Muon actually *beats* iMuon (Fig 2c, $\kappa{=}100$, where observation noise dominates).
So the Gram-root buys robustness to **imbalance**, not a free lift — relevant because LoRA drifts
into imbalance on its own ($B{=}0$ init; asymmetric factor-gradient dynamics; $r{=}256$
ill-conditioning, cond$(S_B)\approx120$–$900$).

## 8. What it means for our line

- Our chord-tight is, structurally, **iMuon's spectral fixed-rank update plus an Adam input,
  plus $\eta$-pinning, plus an optional second-order correction.** We independently rebuilt
  iMuon's core; the additions are the Adam direction and the $\rho/\sigma_{\max}$ machinery.
- **The polar looks load-bearing in our setting** (no-polar ScaledGD/Scaled-Adam are weak here),
  which narrows — does not close — the long-standing curvature-whitening question.
- **The genuinely untested cell is polar + curvature whitening** (iMuon's geometry with
  AdaPreLoRA's $L,R$ curvature in place of, or alongside, the geometric Gram). Every external
  paper does exactly one of {polar, curvature}; none does both. Our `--curvature_whitening` flag
  gropes toward it (curvature on the small side, polar kept) but is not the two-sided object.
  §10 works out the projection framing that turns this into a concrete construction.
- **LR-transfer-across-$r$ is not a clear spectral win on paper** (μA gives default Adam
  $\eta^\star\propto r^{-1/2}$; iMuon's rate-optimal step is also $\propto r^{-1/2}$ via
  $C_\varphi=2r$). The project has measured this empirically; defer to that, not to the idealized
  theory.

## 9. Operating learning-rate scale

The operating LR differs by 1–3 orders of magnitude across methods, because the **output
normalization** (the §5 norms) sets what $\eta$ *means*. Practically: LR grids do not transfer
across methods, and every paper re-tunes per method.

| method | what $\eta$ multiplies | operating $\eta$ (source) |
|---|---|---|
| SGD | raw gradient (scales with $\|g\|$) | ${\sim}10^{-1}$ — GPT-2 E2E (AdaPreLoRA Tbl 9) |
| AdamW | per-entry unit-RMS direction | $1\text{–}3{\times}10^{-4}$ OLMo-opc (leaderboard); $2{\times}10^{-4}$ GPT-2 E2E (AdaPreLoRA Tbl 9) |
| ScaledGD / Scaled-AdamW | Gram-whitened gradient, $\sigma$ kept | $8{\times}10^{-4}\text{–}4{\times}10^{-3}$ GPT-2 E2E (AdaPreLoRA Tbl 9) |
| AdaPreLoRA | curvature-whitened, $\sigma$ kept | $1\text{–}8{\times}10^{-4}$ GPT-2 E2E (Tbl 9) |
| Muon / iMuon / Riemannion (polar, $\tau{=}1$) | unit-spectral-norm direction | iMuon $5{\times}10^{-3}$, Riemannion $5{\times}10^{-2}$, Muon $10^{-3}$ — GPT-2 E2E (iMuon Tbl 7); $5{\times}10^{-5}\text{–}10^{-4}$ Mistral GLUE (Tbl 10) |
| **ours (chord-tight)** | unit-spectral product step (polar + $\rho$) | $10^{-2}\text{–}10^{-1}$ OLMo-opc (leaderboard) |

Readings:

1. **The output normalization sets the scale, not the input.** We feed an Adam direction yet
   operate at $10^{-2}\text{–}10^{-1}$ (spectral scale), not Adam's $10^{-4}$ — the polar and
   $\rho$ re-normalize the output to a spectral step. You can read a method's output
   normalization off its operating $\eta$: ${\sim}10^{-4}\Rightarrow$ per-entry/Adam;
   ${\sim}10^{-2}\Rightarrow$ operator-norm/polar; ${\sim}10^{-1}\Rightarrow$ raw-gradient.
2. **Grids don't transfer across methods** (3+ orders apart) — every paper tunes per method
   (iMuon Tbl 10, AdaPreLoRA Tbl 9, Riemannion Tbl 4/5). A fair head-to-head re-sweeps $\eta$.
3. **$\eta$ is also strongly model/task-dependent.** iMuon is $5{\times}10^{-3}$ on GPT-2-E2E but
   $5{\times}10^{-5}\text{–}10^{-4}$ on Mistral-7B-GLUE (${\sim}50\times$). "Method X operates at
   $\eta$" is only meaningful within a fixed (model, dataset).
4. **Across rank, ours transfers; AdamW does not** (in-house, leaderboard). Best $\eta$ at fixed
   (OLMo, opc): chord-tight $k{=}1$ is $10^{-2}$ at *both* $r{=}64$ and $r{=}256$; AdamW drops
   $3{\times}10^{-4}\to10^{-4}$. Mechanism: $\rho=\eta/(\sigma_{\max}(A)+\sigma_{\max}(B))$ divides
   by factor spectral norms, which grow with $r$, so the per-step product spectral norm stays
   pinned to $\eta$ independent of $r$. So the $\sigma_{\max}$ denominator — §7 notes it is
   *redundant for the product-norm bound* (iMuon's point) — is plausibly **what buys
   LR-transfer-across-rank**, a role on top of conditioning control. Drop-it ablations must be
   checked against rank transfer, not just final loss. (Measured result; defer to it over
   idealized-rate arguments, which give $\eta^\star\propto r^{-1/2}$ for both Adam and spectral and
   thus predict a wash the data does not show.)

## 10. Our step is a clipped projection — and how AdaPreLoRA plugs in

In words, chord-tight does this each step: **find the closest reachable product-change to the
gradient step, then clip its spectrum.** "Reachable" means $\mathcal R=\{B\,\mathrm dA+\mathrm dB\,A\}$,
the product-changes one LoRA step can make. The two halves — projection (closest reachable) and clip
(spectral cap) — are separable to explain but coupled to solve, and that coupling is the Picard loop.
Math below verified numerically to machine precision. ($g_A=B^\top(-\eta g)$, $g_B=(-\eta g)A^\top$;
$A$-side shown, $B$-side mirrors.) All §10 formulas use the gradient $g_A$;
chord-tight substitutes the per-factor Adam direction $u_A$ (§3.6) — the incoherent-Adam choice (§10.5).

### 10.1 The projection half

Without the clip, the program is the Frobenius projection of the gradient step $-\eta g$ onto
$\mathcal R$:
$$
\min_{\mathrm dA,\mathrm dB}\ \bigl\|\,B\,\mathrm dA+\mathrm dB\,A-(-\eta g)\,\bigr\|_F^2,
\qquad P_{\mathcal R}(-\eta g)=P_B(-\eta g)+(-\eta g)P_A-P_B(-\eta g)P_A,
$$
with $P_B=B S_B^{-1}B^\top$, $P_A=A^\top S_A^{-1}A$. One factor split realizing it (the split is
gauge-free — many $(\mathrm dA,\mathrm dB)$ give the same product change; we pin it implicitly by
equal-$\sigma_{\max}$ steps, AdaPreLoRA by factor-balance):
$$
\boxed{\ \mathrm dA=S_B^{-1}g_A,\qquad \mathrm dB=g_B S_A^{-1}-B S_B^{-1}(g_A A^\top)S_A^{-1}\ }
$$
*Check.* $B\,\mathrm dA=P_B(-\eta g)$ and $\mathrm dB\,A=(I-P_B)(-\eta g)P_A$, so their sum is
$P_{\mathcal R}(-\eta g)$, the projection we wanted.

*Cost.* The two $r\times r$ inverses $S_A^{-1},S_B^{-1}$ (the square of the Higham $S^{-1/2}$ we
already hold), plus $(d_{\text{in}}{+}d_{\text{out}})\times r$ matmuls — no eigh, no Sylvester, no
iteration.

*Relation to other methods.* $\mathrm dA=S_B^{-1}g_A$ is the ScaledGD update (§3.2); the $\mathrm dB$
formula adds one cross term. This is exactly **AdaPreLoRA's program with $\mathcal F=I$** (its
LoRA-Pro / Frobenius case), and our LinLoRA optimizer already solves a system like it in closed form.

### 10.2 The inner loop: block-coordinate descent, under saturation

The normal equations of the uncapped projection are
$$
S_B\,\mathrm dA+B^\top\mathrm dB\,A=g_A,\qquad \mathrm dB\,S_A+B\,\mathrm dA\,A^\top=g_B .
$$
The optimizer does **block-coordinate descent (BCD)** on these — fix one block, update the other
(the loop we call "Picard"). The off-diagonal term $B^\top\mathrm dB\,A$ is the cross-coupling — it
enters the $A$-block as $\tilde u_A=u_A+(1/\eta)B^\top\mathrm dB\,A$. The iteration depth
$k$ controls whether it is included: $k{=}1$ ignores it ($\mathrm dB^0=0$), $k{\ge}2$ picks it up —
this is the "second-order correction."

Chord-tight makes this BCD cheap **under a saturation assumption**, via two simplifications that are
exact only when the clip is active (saturating):

- **Polar instead of clip.** Each block is solved with the polar map (Muon), which equals the exact
  clip-prox block-solve only when the clip binds.
- **Self-term dropped** (the *anchored* linearization), exact when the per-block contribution norm
  depends on state only.

So chord-tight is "BCD + saturation" (also called "anchored Frank–Wolfe"). §10.5 gives the
exact-vs-approximate variants.

### 10.3 The clip is the op-norm cap, and it does not separate from the projection

This half is what makes us not-AdaPreLoRA. The full program keeps the per-block cap
$\|B\,\mathrm dA\|_2\le\tau$. Whiten $\mathrm dA=S_B^{-1/2}Y_A$; since $B S_B^{-1/2}=U_BV_B^\top$ is a
partial isometry, the cap becomes the clean $\|Y_A\|_2\le\tau$, and the per-block solution is the
**projection onto the spectral-norm ball**, $Y_A=\mathrm{clip}_\tau(-\eta\,c_A)$. The three spectral
treatments differ only in how hard they squash the singular values:

- $\operatorname{clip}_\tau$ keeps singular values $\le\tau$ and caps the rest;
- the full polar (chord-tight) flattens all of them to $\tau$;
- SSC's $\kappa$ sits between the two.

The cap couples back into the projection: $c_A$ carries the cross term $(1/\eta)B^\top\mathrm dB\,A$,
and $\mathrm dB$ was itself clipped. Clip and coupling feed each other — that is *why* there is an
iteration at all. Drop the cap and you get AdaPreLoRA/LoRA-Pro (closed form, no spectral control);
add it and the projection loses its one-shot solution.

### 10.4 Damping, variationally: a ridge in the projection metric

Damping is not a separate mechanism bolted onto the step — it is a Tikhonov ridge added to the
projection objective, and it folds entirely into the metric.

The $A$-block of the projection (§10.1, §10.3) carries the quadratic
$\tfrac12\|B\,\mathrm dA\|_F^2=\tfrac12\operatorname{tr}(\mathrm dA^\top S_B\,\mathrm dA)$, so the
natural metric on the factor step is the opposite-factor Gram $S_B$. A ridge
$\tfrac{\delta}{2}\|\mathrm dA\|_F^2$ on the size of the step adds $\delta I$ to that metric:
$$
\tfrac12\|B\,\mathrm dA\|_F^2+\tfrac{\delta}{2}\|\mathrm dA\|_F^2
=\tfrac12\operatorname{tr}\!\big(\mathrm dA^\top M\,\mathrm dA\big),
\qquad M:=S_B+\delta I .
$$
That is the whole story: the whitener $S_B^{-1/2}$ becomes $(S_B+\delta I)^{-1/2}$, and nothing else
in the step changes.

**What the shift does.** $(S_B+\delta I)^{-1/2}$ acts on each singular direction of $B$ as
$$
\frac{1}{\sqrt{\sigma_i^2+\delta}},\qquad\text{so}\qquad
\frac{\sigma_i}{\sqrt{\sigma_i^2+\delta}}\approx
\begin{cases}1, & \sigma_i^2\gg\delta\\[2pt]\sigma_i/\sqrt\delta, & \sigma_i^2\ll\delta .\end{cases}
$$
The reweighting is $\approx1$ on $B$'s strong directions and only shrinks its tail — precisely the
directions where the undamped inverse Gram would blow up. So the ridge steers the step *direction*
off $B$'s ill-conditioned modes while leaving its well-conditioned modes untouched. The $\sigma_{\max}$
normalization of §3.6 still pins the magnitude $\sigma_{\max}(\mathrm dA)=\rho$ on top, so damping
changes only where the step points, not how large it is.

**Setting $\delta$.** Pin it relative to $B$'s spectrum rather than as an absolute constant, so it
scales with the problem:
$$
\delta=\varepsilon\,\sigma_{\max}(B)^2\quad\Rightarrow\quad\operatorname{cond}(M)\le1+\tfrac1\varepsilon .
$$
The one dimensionless knob is $\varepsilon$; $\delta$ follows from the current spectrum. Chord-tight
already builds the damped whitener $(S_B+\delta I)^{-1/2}$, so enabling damping is a choice of
$\varepsilon$, not new machinery.

### 10.5 The capped program (P), and two ways to attack it

Full program — the §10.1 projection *with* the §10.3 caps:
$$
\textbf{(P)}\qquad \min_{\mathrm dA,\mathrm dB}\ \|B\,\mathrm dA+\mathrm dB\,A+\eta g\|_F^2
\quad\text{s.t.}\quad \|B\,\mathrm dA\|_2\le\tau,\ \ \|\mathrm dB\,A\|_2\le\tau .
$$
(P) has **no** closed form — the caps couple the two blocks. Two ways to attack it, all whitening
$\mathrm dA=S_B^{-1/2}Y_A$, $\mathrm dB=Y_B S_A^{-1/2}$ (caps become $\|Y_A\|_2,\|Y_B\|_2\le\tau$):

**(i) Exact clip-prox BCD** (a construct on paper, not the chord-tight implementation).
Block-coordinate descent solving each block *exactly* by clip-prox. Init
$\mathrm dA^{0}=\mathrm dB^{0}=0$; for $n=0,\dots,k-1$:
$$
Y_A^{\,n+1}=\operatorname{clip}_\tau\!\Big(-\eta\,S_B^{-1/2}\big(g_A+\tfrac1\eta\,B^\top \mathrm dB^{\,n} A\big)\Big),
\qquad \mathrm dA^{\,n+1}=S_B^{-1/2}\,Y_A^{\,n+1}
$$
(mirror for $\mathrm dB$). The self-term is not in the linear cost — it *is* the prox:
$\operatorname{clip}_\tau$ is the Frobenius projection onto the spectral ball, i.e.
$\min_{Y_A}\langle c_A,Y_A\rangle+\tfrac{1}{2\eta}\|Y_A\|_F^2$ s.t. $\|Y_A\|_2\le\tau$, and that
$\|Y_A\|_F^2$ is the self-term. Run to convergence this *is* (P)'s capped minimum.

**(ii) Chord-tight = BCD + saturation** ("anchored FW", "Picard"). Two changes from (i), harmless
only under saturation: **polar** instead of clip (equal when the clip is active); self-term dropped
(*anchored*), exact when the per-block norm is state-only. So it matches (i) under saturation, else
it is approximate. (A "full" linearization adds the self-term back to the polar input — still polar,
not clip.) $k{=}1$ coupling off, $k{\ge}2$ on.

**Solves (P)?** (i) solves it exactly; (ii) solves it under saturation, else approximately.

**Incoherent Adam.** (P) uses one $g$; chord-tight feeds per-factor Adam $u_A,u_B$ (not $B^\top(-\eta g)$,
$(-\eta g)A^\top$ for any common $g$), so the $k{\ge}2$ fixed point is the KKT of an *incoherent*
objective, not literally (P). The "(P)" claims hold for the coherent-gradient program.

**Tests at r=256** (where $k{=}2$ currently hurts), one change at a time:

1. Compare anchored $k{=}2$ against anchored $k{=}1$, with plain Gram whitening. (Does the
   cross-coupling correction help at all?)
2. Add damping $\delta$ to whichever won step 1. (Does §10.4's ridge help?)
3. Swap plain Gram whitening for curvature whitening. (Does loss-curvature help on top?)

### 10.6 Generalized program: formal derivation

**Metric.** Take an SPD metric $\mathcal F$ on $\mathbb R^{d_{\text{out}}\times d_{\text{in}}}$ of Kronecker form
$$
\langle X,Y\rangle_{\mathcal F}=\operatorname{tr}(X^\top P\,Y\,Q),\qquad P\succ0\ (d_{\text{out}}{\times}d_{\text{out}}),\quad Q\succ0\ (d_{\text{in}}{\times}d_{\text{in}}).
$$
Frobenius: $P=Q=I$. Curvature: $P=L^{1/2},\ Q=R^{1/2}$, with $L\in\mathbb R^{d_{\text{out}}}$,
$R\in\mathbb R^{d_{\text{in}}}$ the Adafactor row/column energies of $G$ (§3.5). (Damping is a ridge on the
block Gram, handled in §10.4, not a $P,Q$ choice.)

**Program.** Project the $\mathcal F$-preconditioned gradient step $T=\mathcal F^{-1}G=P^{-1}G\,Q^{-1}$ onto
the tangent $J=B\,\mathrm dA+\mathrm dB\,A$, in the $\mathcal F$-norm, under a per-block spectral cap:
$$
\min_{\mathrm dA,\mathrm dB}\ \tfrac12\|J-T\|_{\mathcal F}^2\quad\text{s.t.}\quad \|Y_A\|_2\le\tau,\ \|Y_B\|_2\le\tau .
$$

**Whitening (defines both $Y_A$ and $Y_B$).** Each per-block $\mathcal F$-norm is a plain Frobenius norm of
a whitened variable:
$$
\|B\,\mathrm dA\|_{\mathcal F}^2=\|Y_A\|_F^2,\quad Y_A:=(B^\top P B)^{1/2}\mathrm dA\,Q^{1/2};\qquad
\|\mathrm dB\,A\|_{\mathcal F}^2=\|Y_B\|_F^2,\quad Y_B:=P^{1/2}\mathrm dB\,(AQA^\top)^{1/2}.
$$
Hence $\mathrm dA=(B^\top P B)^{-1/2}Y_A\,Q^{-1/2}$, $\mathrm dB=P^{-1/2}Y_B(AQA^\top)^{-1/2}$, and the caps
read $\|Y_A\|_2,\|Y_B\|_2\le\tau$.

**Linear cost collapses to the raw gradient.** Because $T=\mathcal F^{-1}G$, the metric cancels:
$$
\langle B\,\mathrm dA,\ \mathcal F(-T)\rangle=\langle B\,\mathrm dA,\ -G\rangle=\langle\mathrm dA,\ -g_A\rangle,\qquad g_A=B^\top G,
$$
with whitened form $H_A:=(B^\top P B)^{-1/2}\,g_A\,Q^{-1/2}$. The input is therefore the *raw* factor
gradient $g_A$ — coherent. (Chord-tight substitutes the per-factor Adam $u_A$ for $g_A$: the
incoherent-Adam caveat, §10.5.)

**Update.** Spectral LMO $Y_A=\tau\,\phi(H_A)$, then unwhiten:
$$
\boxed{\ \mathrm dA=\tau\,(B^\top P B)^{-1/2}\,\phi\!\big((B^\top P B)^{-1/2}\,g_A\,Q^{-1/2}\big)\,Q^{-1/2}\ }
$$
($\tau$ routed through the $-\rho/\sigma_{\max}$ scaling of §3.6). Cap off ($\phi\to$ identity, $Y_A=H_A$):
the two half-powers merge, $\mathrm dA=(B^\top P B)^{-1}g_A\,Q^{-1}$.

**Instances.**

| $\mathcal F$ | $(P,Q)$ | $\mathrm dA$, cap on (omit $-\rho/\sigma_{\max}$) | $\mathrm dA$, cap off |
|---|---|---|---|
| Frobenius | $(I,I)$ | $S_B^{-1/2}\phi(S_B^{-1/2}g_A)$ | $S_B^{-1}g_A$ (ScaledGD) |
| curvature | $(L^{1/2},R^{1/2})$ | $(B^\top L^{1/2}B)^{-1/2}\phi\big((B^\top L^{1/2}B)^{-1/2}g_A R^{-1/4}\big)R^{-1/4}$ | $(B^\top L^{1/2}B)^{-1}g_A R^{-1/2}$ (AdaPreLoRA, +gauge) |

The $R^{-1/4}$ (cap on) vs $R^{-1/2}$ (cap off) is exactly $Q^{-1/2}$ vs $Q^{-1}$ for $Q=R^{1/2}$; the
cap-off curvature row reproducing AdaPreLoRA's §3.5 form is the consistency check.

**The design space.** Three independent choices generate this family:

- **Metric $\mathcal F$** — Frobenius (plain Gram $S_B$) or curvature (Adafactor $L,R$).
- **Cap** — on (the polar, which flattens the spectrum) or off (which keeps it).
- **Input** — the raw factor gradient $g_A$ or the per-factor Adam direction $u_A$.

The two named methods sit at opposite corners:

| method | metric $\mathcal F$ | cap | input |
|---|---|---|---|
| chord-tight | Frobenius | on (polar) | $u_A$ |
| AdaPreLoRA | curvature | off | $g_A$ |

**Cost.** Curvature is the expensive corner: forming $L,R$ needs the full product gradient $G$, so
$O(d_{\text{in}}d_{\text{out}})$ per step — versus $O((d_{\text{in}}{+}d_{\text{out}})r)$ for the
Frobenius (factor-only) path.

**Pick two: cheap, coherent, adaptive.** No single input/metric gets all three. Here *coherent*
means the update is the projection of one weight-space target (§10.5); *adaptive* means it tracks
per-coordinate gradient scale.

| input / metric | cheap | coherent | adaptive |
|---|---|---|---|
| raw $g_A$, $\mathcal F{=}I$ | yes | yes | no |
| per-factor Adam $u_A$ | yes | no | yes |
| curvature $L,R$ | no | yes | yes |

Chord-tight takes the cheap-and-adaptive corner; in our runs the lost coherence is interpretive
tidiness, not a measured cost.

## Sources

External: the four PDFs in `docs/papers/` (tables cited inline). Ours: `optim.py` `AdamPolarProductLoRA._chord_tight_clean_polar_pipeline`,
`algorithm_ssc_kappa.md` (residual-program / LMO derivation), `algorithm_clean_implementation.md`
(gram-NS, $\sigma$-sites), `factor_conditioning_hypothesis.md`, `soap_curvature_whitening.md`.
All external numbers are the papers' reported values, single-seed unless noted, not reproduced here.
