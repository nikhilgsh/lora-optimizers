# Deriving kl-shampoo-polar

`kl-shampoo-polar` is a LoRA optimizer that, per factor, takes a momentum step,
**whitens it by a two-sided curvature estimate, projects it to the spectral cap
with a polar map, unwhitens through the same curvature, and rescales to a spectral
trust region.** The curvature is not an ad-hoc Gram EMA: it is the Kronecker
preconditioner that minimizes the KL divergence to the factor gradient's second
moment, which couples the two sides of the sandwich.

It sits in a small family of related updates. Dropping the coupling between the two
curvature sides gives one-sided Shampoo; replacing the curvature-eigenbasis
whitening with a SOAP-style Adam step gives SOAP-on-momentum; dropping the polar map
drops the spectral cap. This note treats the coupled-KL-plus-polar corner.

This is a pedagogical derivation. It states the full per-step update first, then
derives its two ingredients — the curvature estimate and the whitened-polar step it
feeds — and finally the refinements that sit on top.

**Roadmap.**

- *Part I — the method.* §Notation fixes symbols; §"The update, per step" states the
  whole single-step ($k{=}1$) algorithm; §"The curvature" derives where the curvature
  factors come from (the KL fit); §"The whitened-polar step" shows what the
  whiten–polar–unwhiten sandwich computes; §"The magnitude rule" sizes the step.
- *Part II — refinements.* §"Cross-coupling" is the correction for both factors
  moving at once (a Picard iteration, $k{\ge}2$); §"Regularization" is the numerical
  flooring that keeps every curvature invertible (it applies to the base method too).

## Part I — the method

*The sections through "The magnitude rule" specify the whole single-step ($k{=}1$)
optimizer; there is no Picard / cross-coupling inner loop.*

## Notation

One PEFT LoRA pair, with $r\ll d_{\mathrm{in}},d_{\mathrm{out}}$:

- $A\in\mathbb{R}^{r\times d_{\mathrm{in}}}$, $B\in\mathbb{R}^{d_{\mathrm{out}}\times r}$, adapter weight $\Delta W=BA$.
- $G=\nabla_{\Delta W}\mathcal L$ — loss gradient w.r.t. the merged weight (upstream of the factors).
- $g_A=B^\top G\in\mathbb{R}^{r\times d_{\mathrm{in}}}$, $g_B=GA^\top\in\mathbb{R}^{d_{\mathrm{out}}\times r}$ — the per-factor gradients.
- $m_A,m_B$ — $\beta_1$ EMAs of $g_A,g_B$; $\hat m_A,\hat m_B$ — their bias-corrected forms.
- $\varphi$ — the polar map, $\varphi(U\Sigma V^\top)=UV^\top$ (sets all nonzero singular values to 1).

Everything below is stated A-side; the B-side is the symmetric construction with
$A\leftrightarrow B$, $d_{\mathrm{in}}\leftrightarrow d_{\mathrm{out}}$.

The curvature state, per pair, is **four objects** — a dense $r\times r$ factor and
a diagonal full-dimension factor on each side: $S_{\mathrm{curv},A}\in\mathbb{R}^{r\times r}$
with $D_{\mathrm{in}}\in\mathbb{R}^{d_{\mathrm{in}}}$ on the A-side, and
$S_{\mathrm{curv},B}\in\mathbb{R}^{r\times r}$ with $D_{\mathrm{out}}\in\mathbb{R}^{d_{\mathrm{out}}}$
on the B-side. They are the Kronecker factors of a single KL fit to the
factor-gradient second moment, **coupled** so each side is whitened by its conjugate
factor; §"The curvature" derives them and states the fixed-point equations they
satisfy. In practice the factors are maintained as damped streaming EMAs of that
fixed point (one alternating update per step), not solved exactly.

## The update, per step

The five steps below are the whole optimizer ($k{=}1$; there is no Picard /
cross-coupling inner loop).

1. **Momentum.**
$$
\hat m_A=\frac{m_A}{1-\beta_1^{\,t}},\qquad m_A\leftarrow\beta_1 m_A+(1-\beta_1)g_A.
$$

2. **Two-sided curvature whiten** (the inner Shampoo core):
$$
z_A=S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}\in\mathbb{R}^{r\times d_{\mathrm{in}}}.
$$

3. **Polar cap.** $z_A\leftarrow\varphi(z_A)$.

4. **Unwhiten through the same sandwich:**
$$
W_A=S_{\mathrm{curv},A}^{-1/2}\,\varphi(z_A)\,D_{\mathrm{in}}^{-1/2}.
$$

5. **Spectral trust-region rescale:**
$$
\rho=\frac{\eta}{\sigma_{\max}(A)+\sigma_{\max}(B)},\qquad
\mathrm dA=-\rho\,\frac{W_A}{\sigma_{\max}(W_A)}
$$
($\mathrm dB$ symmetric). So $\sigma_{\max}(\mathrm dA)=\rho$ by construction of the rescale —
exact up to the $\sigma_{\max}(W_A)$ estimate (warm power iteration) and the finite-step polar.

Composing steps 2–4, the direction the rescale acts on is
$$
\boxed{\,W_A\;\propto\;S_{\mathrm{curv},A}^{-1/2}\,
\varphi\!\bigl(S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}\bigr)\,
D_{\mathrm{in}}^{-1/2}\,}
$$
— the **same curvature sandwich applied twice with $\varphi$ in between.** The rest
of the doc derives the two ingredients: where $(S_{\mathrm{curv},A},D_{\mathrm{in}})$
come from (the KL fit), and what $\varphi$ does (the spectral cap).

The inverse-square-roots are **relative-damped**: a nonnegative spectrum $x$ is
mapped to $(x/x_{\max}+\delta)^{-1/2}$ rather than $x^{-1/2}$, and an uninitialized
(all-zero) factor maps to the identity — so before the EMAs warm up the step is a
plain momentum step. The dense $S_{\mathrm{curv},A}^{-1/2}$ is applied in the
eigenbasis of $S_{\mathrm{curv},A}$; $D_{\mathrm{in}}^{-1/2}$ is a diagonal scaling.
(§"Regularization" treats the damping $\delta$ in full.)

### The B-side update

The $B$ factor runs the identical pipeline, but the sandwich is **mirrored**: $B$
is $d_{\mathrm{out}}\times r$, so its small ($r$) axis is the *columns* and its
large ($d_{\mathrm{out}}$) axis is the *rows*. The dense $r\times r$ curvature
$S_{\mathrm{curv},B}$ therefore multiplies on the **right** and the diagonal
$D_{\mathrm{out}}^{-1/2}$ on the **left** — the transpose of the A-side placement.

1. **Momentum.** $\hat m_B=m_B/(1-\beta_1^{\,t})$, with $m_B\leftarrow\beta_1 m_B+(1-\beta_1)g_B$ and $g_B=GA^\top\in\mathbb R^{d_{\mathrm{out}}\times r}$.
2. **Whiten.**
$$
z_B=D_{\mathrm{out}}^{-1/2}\,\hat m_B\,S_{\mathrm{curv},B}^{-1/2}\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
$$
3. **Polar cap.** $z_B\leftarrow\varphi(z_B)$.
4. **Unwhiten.** $W_B=D_{\mathrm{out}}^{-1/2}\,\varphi(z_B)\,S_{\mathrm{curv},B}^{-1/2}$.
5. **Rescale,** sharing the *same* $\rho$ as the A-side (it is computed once from $\sigma_{\max}(A)+\sigma_{\max}(B)$):
$$
\mathrm dB=-\rho\,\frac{W_B}{\sigma_{\max}(W_B)}.
$$

Composing 2–4, the boxed B-direction is the mirror of the A-side:
$$
\boxed{\,W_B\;\propto\;D_{\mathrm{out}}^{-1/2}\,
\varphi\!\bigl(D_{\mathrm{out}}^{-1/2}\,\hat m_B\,S_{\mathrm{curv},B}^{-1/2}\bigr)\,
S_{\mathrm{curv},B}^{-1/2}\,.}
$$
The only asymmetries between the two factors are this orientation flip, the shared
$\rho$, and the extra multiplier $c$ on $\mathrm dB$ ($c=1$ recovers full symmetry).

## The curvature: a KL fit, not a Gram EMA

The defining choice of `kl-shampoo` over plain Shampoo is **how the two curvature
factors are estimated**. Shampoo sets each factor to its own gradient Gram EMA
independently. KL-Shampoo instead treats curvature estimation as covariance
estimation: fit the Kronecker preconditioner $S=S_a\otimes S_b$ closest to the
gradient second moment $M=\mathbb E[gg^\top]$ (with $g=\mathrm{vec}(G)$) in KL
divergence on the SPD cone.

**Why KL, not Frobenius.**
$$
\mathrm{KL}(M\,\|\,S)=\tfrac12\bigl(\log\det S+\operatorname{Tr}(M S^{-1})\bigr)+\text{const.}
$$

- KL is a divergence *on the SPD cone*: it keeps $S$ positive-definite and weights
  errors multiplicatively — the geometry a preconditioner actually lives in.
- The Frobenius fit Shampoo/SOAP implicitly minimize ignores the SPD constraint,
  which is why those methods need an Adam step-size graft to fix the overall scale.
  KL needs no such graft.
- KL is the log-det / von-Neumann divergence that classical quasi-Newton already
  minimizes.

**Proposition 1 (KL stationarity coupling).** *The Kronecker factors minimizing
$\mathrm{KL}(M\,\|\,S_a\otimes S_b)$, with $S_a$ of size $d_a$ and $S_b$ of size
$d_b$, satisfy the coupled fixed point*
$$
S_a=\frac1{d_b}\,\mathbb E\!\bigl[G\,S_b^{-1}G^\top\bigr],
\qquad
S_b=\frac1{d_a}\,\mathbb E\!\bigl[G^\top S_a^{-1}G\bigr].
$$

*Proof.* With $S=S_a\otimes S_b$,
$\log\det(S_a\otimes S_b)=d_b\log\det S_a+d_a\log\det S_b$ and
$\operatorname{Tr}(MS^{-1})=\mathbb E\operatorname{Tr}(G^\top S_a^{-1}G\,S_b^{-1})$,
so the objective is
$$
J=d_b\log\det S_a+d_a\log\det S_b+\mathbb E\operatorname{Tr}\!\bigl(G^\top S_a^{-1}G\,S_b^{-1}\bigr).
$$
The trace term equals $\operatorname{Tr}\bigl(S_a^{-1}\,G S_b^{-1}G^\top\bigr)$, so
$$
\frac{\partial J}{\partial S_a}=d_b S_a^{-1}-S_a^{-1}\,\mathbb E[G S_b^{-1}G^\top]\,S_a^{-1}=0,
$$
and symmetrically in $S_b$; rearranging each gives the boxed pair. $\blacksquare$

The coupling is the content of the result: the optimal $S_a$ whitens $G$ by the
*other* factor's inverse before forming its Gram, so the two factors do not
double-count the structure they share. Setting $S_b=I$ recovers
$S_a=\mathbb E[GG^\top]$ — plain Shampoo is exactly the one-sided corner where the
conjugate factor is ignored.

**Remark (it is a fixed-point iteration, not a one-shot solve).** The coupled
condition is the maximum-likelihood estimate of a zero-mean matrix-normal
(Kronecker-factored) Gaussian, which has no closed form. The classical "flip-flop"
solver alternates $S_a\leftarrow f(S_b)$, $S_b\leftarrow g(S_a)$ to convergence.
The streaming form used here runs one alternation per optimizer step as an EMA
(decay $\beta_{\mathrm{curv}}$): each step nudges the factors toward the fixed
point while the target $M$ itself drifts (since $g_A=B^\top G$ moves as $B$
updates), so "solving more exactly" buys little.

### LoRA instantiation

Apply Proposition 1 to each LoRA factor as its own weight — small ($r$) side dense,
large side constrained diagonal. The substitution dictionary:

| | weight $\Theta$ | gradient $G$ | $d_a$ | $d_b$ | $S_a$ (size $d_a$) | $S_b$ (size $d_b$) |
|---|---|---|---|---|---|---|
| **A-side** | $A$ | $g_A=B^\top G$ | $r$ | $d_{\mathrm{in}}$ | $S_{\mathrm{curv},A}$ (dense) | $D_{\mathrm{in}}$ (diag) |
| **B-side** | $B$ | $g_B=GA^\top$ | $d_{\mathrm{out}}$ | $r$ | $D_{\mathrm{out}}$ (diag) | $S_{\mathrm{curv},B}$ (dense) |

Substituting each row into Proposition 1's boxed pair
$\bigl(S_a=\tfrac1{d_b}\mathbb E[GS_b^{-1}G^\top],\ S_b=\tfrac1{d_a}\mathbb E[G^\top S_a^{-1}G]\bigr)$
yields the four coupled curvature equations the rest of the note uses. The A-side row
($\Theta=A$, $S_a\to S_{\mathrm{curv},A}$, $S_b\to D_{\mathrm{in}}$) gives
$$
S_{\mathrm{curv},A}=\tfrac1{d_{\mathrm{in}}}\,\mathbb E\!\bigl[g_A\,D_{\mathrm{in}}^{-1}\,g_A^\top\bigr],
\qquad
D_{\mathrm{in}}=\tfrac1{r}\,\operatorname{diag}\mathbb E\!\bigl[g_A^\top\,S_{\mathrm{curv},A}^{-1}\,g_A\bigr],
$$
and the B-side row ($\Theta=B$, $S_a\to D_{\mathrm{out}}$, $S_b\to S_{\mathrm{curv},B}$) gives
$$
S_{\mathrm{curv},B}=\tfrac1{d_{\mathrm{out}}}\,\mathbb E\!\bigl[g_B^\top\,D_{\mathrm{out}}^{-1}\,g_B\bigr],
\qquad
D_{\mathrm{out}}=\tfrac1{r}\,\operatorname{diag}\mathbb E\!\bigl[g_B\,S_{\mathrm{curv},B}^{-1}\,g_B^\top\bigr].
$$
Both inverses are cheap ($r\times r$ dense and length-$d$ diagonal), and each factor
is maintained by the streaming EMA of the Remark above (one alternating update per
step).

This is what makes the A-factor's two metric sides *consistent*: $D_{\mathrm{in}}$
is not a separately-chosen right preconditioner but is pinned jointly with
$S_{\mathrm{curv},A}$ by the same fit, with no swept knob between them. (The pairing
is within each factor; the A- and B-factors are still fit independently — see §"The
whitened-polar step".)

**Caveat — the factor gradient is not a free weight's gradient.** Proposition 1
assumes $G$ is the gradient of a free weight. Here $g_A=B^\top G$ already carries
the other factor $B$, so the fit estimates the covariance of the *factor* gradient
(filtered through $B$), not of a free $r\times d_{\mathrm{in}}$ weight. The
construction is the faithful LoRA analogue, not the identical object.

## The whitened-polar step

Each factor's update (steps 2–4) is a **single-block spectral-cap LMO**: maximize
alignment with the momentum'd gradient inside a spectral ball measured in a
Kronecker metric on that factor. For a factor $X$ with SPD left metric $M_L$ and
right metric $M_R$, the solution is the **whitened polar** (the spectral-ball LMO,
solved by the polar map $\varphi$; related_work §10, Thm 10.1, the same von Neumann
argument for any SPD $M_L,M_R$):
$$
\mathrm dX\;\propto\;M_L^{-1/2}\,\varphi\!\bigl(M_L^{-1/2}\,g_X\,M_R^{-1/2}\bigr)\,M_R^{-1/2}.
$$
The metric only reshapes the cap; $\varphi$ flattens the whitened spectrum to a
constant.

**kl supplies each factor's own two-sided metric from its KL fit:**

- A-factor ($r\times d_{\mathrm{in}}$): $(M_L,M_R)=(S_{\mathrm{curv},A},\,D_{\mathrm{in}})$.
- B-factor ($d_{\mathrm{out}}\times r$): $(M_L,M_R)=(D_{\mathrm{out}},\,S_{\mathrm{curv},B})$.

Substituting, with $g_X\to\hat m_X$ (momentum), reproduces the boxed $W_A,W_B$ of
§"The update, per step" exactly. Reading the A-factor left to right: the inner
$S_{\mathrm{curv},A}^{-1/2}(\cdot)\,D_{\mathrm{in}}^{-1/2}$ whitens the momentum into
the metric where the cap is isotropic; $\varphi$ caps the whitened step at spectral
norm 1; the outer sandwich unwhitens back to parameter space.

**Why this is *not* the single-metric "two-sided program."** related_work §10
derives *both* factors' metrics from **one** weight-space metric
$\langle X,Y\rangle=\operatorname{tr}(X^\top P\,Y\,Q)$. Its two block updates are
$$
\mathrm dA\propto(B^\top P B)^{-1/2}\varphi\!\bigl((B^\top P B)^{-1/2}g_A\,Q^{-1/2}\bigr)Q^{-1/2},
$$
$$
\mathrm dB\propto P^{-1/2}\varphi\!\bigl(P^{-1/2}g_B\,(AQA^\top)^{-1/2}\bigr)(AQA^\top)^{-1/2},
$$
so the A-factor's metric is $(B^\top P B,\,Q)$ and the B-factor's is $(P,\,AQA^\top)$
— **tied** through the shared $P,Q$. To read kl as such an instance, the diagonal
slots fix $Q=D_{\mathrm{in}}$ and $P=D_{\mathrm{out}}$, which then *force*
$$
S_{\mathrm{curv},A}=B^\top P B=B^\top D_{\mathrm{out}}B,\qquad
S_{\mathrm{curv},B}=AQA^\top=A\,D_{\mathrm{in}}A^\top.
$$
The KL fit does **not** satisfy these: it estimates $S_{\mathrm{curv},A}$ and
$S_{\mathrm{curv},B}$ *independently*, each as its own factor's curvature, never as a
contraction of the conjugate factor's diagonal. So kl-shampoo-polar is the
**per-factor** spectral-cap LMO under two **independently fit** Kronecker metrics; it
shares the block update *formula* with the §10 program but is **not** an instance of
the coupled single-weight-metric version.

**What the cap does, and its exact limit.** The role of $\varphi$ is clearest at
zero EMA decay, where the curvature is a single sample's Gram.

**Lemma 2 (zero-decay whitening is the polar).** *For a single sample
$g_A=U\Sigma V^\top$ with curvature $S_{\mathrm{curv},A}=g_Ag_A^\top$,*
$$
S_{\mathrm{curv},A}^{-1/2}\,g_A=(g_Ag_A^\top)^{-1/2}g_A=U\Sigma^{-1}U^\top\,U\Sigma V^\top=UV^\top=\varphi(g_A).
$$

*Proof.* Direct from the SVD, as displayed. $\blacksquare$

So curvature whitening *is* the EMA generalization of "polar the gradient." In the
$\beta_{\mathrm{curv}}\to0$ limit the inner whitening already returns an orthogonal
$z_A$, and $\varphi(z_A)=z_A$ is redundant. The two separate only as the EMA
averages over samples: then $z_A$ is no longer orthogonal, and $\varphi$ contributes
real work — flattening the per-step singular-value spread that statistical whitening
leaves behind.

## The magnitude rule

Steps 2–4 fix only the *direction* $W_A$. Step 5 sets its size. The trust radius
$$
\rho=\frac{\eta}{\sigma_{\max}(A)+\sigma_{\max}(B)}
$$
scales the learning rate by the current factor norms so that the induced change in
$\Delta W=BA$ is controlled rather than the raw factor change; dividing by
$\sigma_{\max}(W_A)$ then pins $\sigma_{\max}(\mathrm dA)=\rho$. This is the same
spectral magnitude rule the chord-tight polar family uses, inherited unchanged.

**Guarding the $\sigma_{\max}$ estimate.** Both the polar pre-norm and the final
rescale divide by an estimated $\sigma_{\max}$ obtained by warm-started power
iteration. A stale or cold start vector can *under*-estimate $\sigma_{\max}$, which
over-scales the update — pushing the polar (Newton–Schulz) iteration into its
divergent region and, eventually, to all-parameter blow-up. Two cheap safeguards
prevent this: the estimator is floored at
$\max(\text{max row }L_2,\ \text{max col }L_2)$ — both valid lower bounds on
$\sigma_{\max}$ — and any non-finite polar output is recomputed from the
Frobenius-normalized input (for which $\sigma_{\max}\le1$ is guaranteed). The floor
binds only when the warm estimate is pathological; otherwise it leaves the
denominator unchanged.

## Part II — refinements

*Two layers sit on top of the method above: a correction for both factors moving at
once (this section), and the numerical flooring that keeps the curvatures invertible
(§"Regularization"). Both reduce to the Part I update at their trivial settings.*

## Cross-coupling: the Picard correction

The single-step update solves each factor's subproblem as if the other were frozen.
When both move together that leaves a first-order coupling unaddressed; closing it is
a block-coordinate (Picard) iteration, run for $k\ge2$ inner steps. The $k{=}1$ case
is exactly the Part I update.

**The gap the $k{=}1$ step leaves.** When both factors move, the merged weight
changes by
$$
\Delta W = B\,\mathrm dA + \mathrm dB\,A + \mathrm dB\,\mathrm dA .
$$
The $k{=}1$ step solves each factor's LMO as if the other were frozen. The two
first-order contributions $B\,\mathrm dA$ and $\mathrm dB\,A$ then overlap in
merged-weight space: solved independently against the same target, they double-count
that overlap (each tries to supply the whole step, so the sum over- or undershoots).
Correcting this is a block-coordinate (Picard) iteration. The genuine bilinear
$\mathrm dB\,\mathrm dA$ is a separate $O(\eta^2)$ term that the first-order program
below drops; it is *not* what this correction addresses.

**The generalized program it descends from.** related_work §10 fits the joint
first-order step by a spectral-capped residual over the shared tangent
$J=B\,\mathrm dA+\mathrm dB\,A$, in one global Kronecker metric
$\langle X,Y\rangle=\operatorname{tr}(X^\top P\,Y\,Q)$ — the first-order loss model
with a metric trust region:
$$
\min_{\mathrm dA,\mathrm dB}\ \langle G,J\rangle+\tfrac1{2\eta}\lVert J\rVert_{(P,Q)}^2
\quad\text{s.t. per-block spectral caps.}
$$
Completing the square, this is $\tfrac1{2\eta}\lVert J-T\rVert_{(P,Q)}^2$ up to a
constant, with $T=-\eta\,P^{-1}GQ^{-1}$ the metric-preconditioned step; the
$\tfrac1{2\eta}$ weight is what produces the $\tfrac1\eta$ prefactor in Proposition 3.
Its block-coordinate solver (Algorithm 10.1) fixes the off-block and solves the
on-block LMO against a target reduced by the off-block's contribution.

**Proposition 3 (cross-coupling correction).** *In factor coordinates, the on-block
input of Algorithm 10.1 gains the correction (B-side symmetric):*
$$
\tilde g_A^{(n)} = g_A + \tfrac1\eta\,B^\top P\,\mathrm dB^{(n)}\,A\,Q,
\qquad
\tilde g_B^{(n)} = g_B + \tfrac1\eta\,P\,B\,\mathrm dA^{(n)}\,Q\,A^\top .
$$

*Proof.* The only $\mathrm dA$–$\mathrm dB$ coupling in the quadratic is the cross
term $\langle B\,\mathrm dA,\,\mathrm dB\,A\rangle_{(P,Q)}=\operatorname{tr}(\mathrm dA^\top B^\top P\,\mathrm dB\,A\,Q)$.
Its gradient in $\mathrm dA$ is $B^\top P\,\mathrm dB\,A\,Q$; its gradient in
$\mathrm dB$ is $P\,B\,\mathrm dA\,Q\,A^\top$ (both $P,Q$ symmetric). Each adds to the
respective on-block linear cost, the $\tfrac1\eta$ coming from the $\tfrac1{2\eta}$
proximal weight. $\blacksquare$

Chord is the $(P,Q)=(I,I)$ instance, $\tilde g_A=g_A+\tfrac1\eta B^\top \mathrm dB\,A$;
AdaPreLoRA is the cap-off curvature instance $(P,Q)=(L^{1/2},R^{1/2})$.

**kl's instantiation is mixed-metric, by necessity.** kl is *not* a single-$(P,Q)$
instance of Proposition 3, for two compounding reasons:

- **Self $\ne$ cross metric.** A true instance uses one $(P,Q)$ for both the on-block
  self-whitening $(B^\top P B,\,Q)$ *and* the cross term. kl's self-whitening is the
  independent KL fit $(S_{\mathrm{curv},A},D_{\mathrm{in}})$, and
  $S_{\mathrm{curv},A}\ne B^\top D_{\mathrm{out}}B$ (§"The whitened-polar step"). No
  single $(P,Q)$ gives both.
- **Only the diagonals reach the cross.** The cross term acts on $\mathrm dB\,A$,
  which spans all of $d_{\mathrm{out}}$; the dense $S_{\mathrm{curv}}$ are latent
  ($r\times r$, on $\operatorname{range}B/\operatorname{range}A$) and cannot weight
  it. kl's only full-space curvatures are $D_{\mathrm{out}},D_{\mathrm{in}}$.

So kl keeps its $k{=}1$ self-solve and fills the cross term's unavailable full-space
metric with the diagonals, $P\rightsquigarrow D_{\mathrm{out}}$, $Q\rightsquigarrow D_{\mathrm{in}}$:
$$
\boxed{\;\tilde g_A^{(n)} = \hat m_A + \tfrac1\eta\,B^\top D_{\mathrm{out}}\,\mathrm dB^{(n)}\,A\,D_{\mathrm{in}},
\qquad
\tilde g_B^{(n)} = \hat m_B + \tfrac1\eta\,D_{\mathrm{out}}\,B\,\mathrm dA^{(n)}\,D_{\mathrm{in}}\,A^\top\;}
$$
(momentum $\hat m$ for the raw $g$, as in the $k{=}1$ step). This is a **mixed-metric**
step — the self-solve uses $S_{\mathrm{curv}}$, the cross uses the diagonals — not a
coherent single-program BCD. It is the Algorithm 10.1 cross term with the only
full-space metric kl has, and reduces to chord at $D_{\mathrm{out}}=D_{\mathrm{in}}=I$.

**Remark (option (b): commit to one diagonal metric).** The cross term needs a
*full-space* output metric, but kl's dense $S_{\mathrm{curv}}$ lives only on
$\operatorname{range}B$ and cannot weight it; kl's only full-space curvature is the
diagonal $D_{\mathrm{out}}$. Two resolutions:

- **Option (a)** (the mixed metric above) — keep the dense $S_{\mathrm{curv}}$ small
  side; the cross uses the diagonals instead, a *different* metric, so it is approximate.
- **Option (b)** — use one metric $(P,Q)$ in *every* block, making the program a single
  consistent Algorithm 10.1 instance with an **exact** cross:
  $$
  M_A=B^\top P\,B,\qquad M_B=A\,Q\,A^\top,\qquad C=B^\top P\,\mathrm dB\,A\,Q .
  $$
  The dense small side $S_{\mathrm{curv},A}$ is replaced by $M_A=B^\top P B$. This is
  legitimate because, at the KL fixed point, $S_{\mathrm{curv},A}=B^\top M_{\mathrm{out}}B$
  with $M_{\mathrm{out}}=\tfrac1{d_{\mathrm{in}}}\mathbb E[G\,D_{\mathrm{in}}^{-1}G^\top]$
  — so $S_{\mathrm{curv}}$ is just the dense output curvature seen on
  $\operatorname{range}B$, and (b) uses its diagonal proxy everywhere.

The metric $(P,Q)$ is the pair of **relative-damped diagonals**
$(\bar D_{\mathrm{out}},\bar D_{\mathrm{in}})$,
$$
\bar D_{\mathrm{out}}=\operatorname{diag}\!\bigl(D_{\mathrm{out}}/D_{\mathrm{out},\max}+\delta\bigr),
\qquad
\bar D_{\mathrm{in}}=\operatorname{diag}\!\bigl(D_{\mathrm{in}}/D_{\mathrm{in},\max}+\delta\bigr),
$$
where $D_{\mathrm{out},\max}=\max_i (D_{\mathrm{out}})_i$ is the largest diagonal entry
and $\delta>0$ floors the inverse root. The single-metric property is what makes (b)
exact, so the *same* $\bar D$ must appear in $M_A$, $M_B$, and $C$ — using raw $D$ in one
block and $\bar D$ in another breaks it. And $\bar D$ vs $D$ is not a global scalar: the
$\delta\,B^\top B$ floor tilts $M_A$ and shifts even the $k{=}1$ step.

**Corollary (the diagonal power is pinned at 1).** *The cross carries each metric
diagonal to the power 1: $C=B^\top \bar D_{\mathrm{out}}\,\mathrm dB\,A\,\bar D_{\mathrm{in}}$,
linear in $\bar D$.*

*Proof.* The on-block large-side whitening is $Q^{-1/2}$, and kl whitens by $D^{-1/2}$,
so $Q=D$ — power 1. $\blacksquare$

The power is therefore fixed by the whitening convention, not free; only the *metric
choice* $P\rightsquigarrow\bar D_{\mathrm{out}}$ is an approximation. AdaPreLoRA whitens
instead by $R^{-1/4}$, so its $Q=R^{1/2}$ and its cross carries the diagonal at power
$1/2$ — the exponents differ because the conventions differ, not because kl's is
ambiguous. (So the cap-off check against AdaPreLoRA's closed form, `adaprelora` Thm 3.2
with $L\to D_{\mathrm{out}},R\to D_{\mathrm{in}}$, validates a power-$1/2$ variant, not
kl's power-1 step.)

**The loop.** Initialize $\mathrm dA^{(0)}=\mathrm dB^{(0)}=0$, so iter 0 is the
$k{=}1$ step. For $n=0,\dots,k-1$: recompute $\tilde g_A^{(n)},\tilde g_B^{(n)}$ from
the current $\mathrm dB^{(n)},\mathrm dA^{(n)}$ (at physical scale $\sigma_{\max}=\rho$),
run each through the $k{=}1$ pipeline (whiten → polar → unwhiten → $\rho$-rescale), and
update. The cross term is added to the momentum at solve time, not folded into the EMA.
Every product stays in the skinny $r\times d$ factors — the dense
$d_{\mathrm{out}}\times d_{\mathrm{in}}$ weight is never formed.

### Normalization: how large is the cross-correction?

This section answers one question: how large is the cross term relative to the momentum
it corrects? The metric in it is option (b)'s single $(\bar D_{\mathrm{out}},\bar D_{\mathrm{in}})$
(Remark above). Name the objects:

- $\hat m_A$ — kl's input to the polar step: the raw momentum (running average of the
  A-gradient $g_A=B^\top G$, with $G=\nabla_{\Delta W}\mathcal L$ the merged-weight
  gradient, §Notation).
- $u_A=\hat m_A/(\sqrt{\hat v_A}+\varepsilon)$ — chord's input instead: the Adam
  (RMS-normalized) direction.
- $C=B^\top\bar D_{\mathrm{out}}\,\mathrm dB\,A\,\bar D_{\mathrm{in}}$ — the cross term
  (Proposition 3), added to the input with coefficient $\tfrac1\eta$.
- $z_A=M_A^{-1/2}\,(\text{input})\,\bar D_{\mathrm{in}}^{-1/2}$ — the whitened input the
  polar step $\varphi$ consumes; $s_A:=\sigma_{\max}(z_A)$ its largest singular value.
- $r$ — the **correction-to-input ratio**, the size of the whitened cross over the size
  of the whitened input:
  $$
  r:=\frac{\bigl\lVert M_A^{-1/2}\,(\tfrac1\eta C)\,\bar D_{\mathrm{in}}^{-1/2}\bigr\rVert}
          {\bigl\lVert M_A^{-1/2}\,(\text{input})\,\bar D_{\mathrm{in}}^{-1/2}\bigr\rVert}.
  $$

**Only $r$ matters.** $\varphi$ normalizes — it keeps the direction of $z_A$ and discards
its magnitude ($\varphi(c\,z_A)=\varphi(z_A)$); the update size is restored afterward by
$\rho=\eta/(\sigma_{\max}(A)+\sigma_{\max}(B))$. So at $k{=}1$ the input's scale is
irrelevant. At $k\ge2$ the input is $\hat m_A+\tfrac1\eta C$ and $\varphi$'s output
depends only on $r$. What remains is to ask which input scaling sets $r$.

**Proposition 4 (the raw momentum self-sizes $r$).** *With input $\hat m_A$,*
$$
r \sim \frac{\sigma_{\max}(A)}{(\sigma_{\max}(A)+\sigma_{\max}(B))\,\lVert G\rVert},
$$
*independent of the learning rate $\eta$.*

*Proof.* $\bar D$ is $O(1)$ and $\mathrm dB$ is $\rho$-scaled, so
$\lVert C\rVert\sim\sigma_{\max}(A)\sigma_{\max}(B)\rho$; and $\hat m_A$ is the running
mean of $g_A=B^\top G$, so $\lVert\hat m_A\rVert\sim\sigma_{\max}(B)\lVert G\rVert$. With
$\rho=\eta/(\sigma_{\max}(A)+\sigma_{\max}(B))$,
$$
r\sim\frac{\tfrac1\eta\,\sigma_{\max}(A)\sigma_{\max}(B)\rho}{\sigma_{\max}(B)\lVert G\rVert}
=\frac{\sigma_{\max}(A)}{(\sigma_{\max}(A)+\sigma_{\max}(B))\,\lVert G\rVert};
$$
the two $\eta$'s cancel. $\blacksquare$

So with $\hat m_A$ the ratio neither grows with the learning rate nor needs tuning, and
the $1/\lVert G\rVert$ makes the correction small early (large gradients) and larger
toward convergence — a modest, self-sizing correction. (Read this as a leading-order
operator-norm *scaling*, not an identity: it tracks the $\sigma_{\max}$ factors, measures
the ratio in the whitened norm, and ignores metric-conditioning effects.)

**kl vs chord, through $s_A$.** Chord's input is $u_A$, not $\hat m_A$, and chord
rescales its whitened input to unit singular value — which multiplies the ratio by
$s_A$, i.e. $r\mapsto s_A\,r$. The effect flips on $s_A\gtrless 1$:

- chord ($u_A$): $s_A$ is large (e.g. ${\approx}70$) — the rescale *lifts* $r$.
- kl ($\hat m_A$): $s_A$ is small (e.g. ${\approx}0.05$) — the same rescale *shrinks* $r$ (${\sim}20\times$).

So chord's rescale does not transfer to kl. And chord's correction is large not because
of that rescale but because $u_A$ has $\lVert G\rVert$ divided out (the $\sqrt{\hat v_A}$),
removing the $1/\lVert G\rVert$ that damps $r$ for $\hat m_A$.

**The open choice.**

- Keep $\hat m_A$ — the exact-gradient covector; modest self-sizing $r$ (Proposition 4,
  to leading order). This is what `CurvatureWhitenLoRA` does now.
- Switch to $u_A$ — removes the $1/\lVert G\rVert$ damping, giving a larger,
  $\lVert G\rVert$-independent $r$ and flatter behavior across learning rates. A
  deliberate change of what kl optimizes ($u_A$ in place of the gradient $B^\top G$),
  not a bug fix.

## Regularization

The $(P,Q)$ program above is undamped — the ideal method. Two quantities are singular at
initialization, so the running algorithm regularizes them. This is a numerical layer, not
part of the derivation: the literature's variational derivations are likewise undamped
($\mathbb E[GG^\top]^{-1/2}G$, $S^*=\mathbb E[GG^\top]$), with damping added afterward.

**The two singularities.** The A-update applies $M_A^{-1/2}(\cdot)\,Q^{-1/2}$, the B-update
$P^{-1/2}(\cdot)\,M_B^{-1/2}$. Two distinct things vanish at init:

- *The metric* — $P=D_{\mathrm{out}}$, $Q=D_{\mathrm{in}}$ are EMAs that start at $0$, so
  $D^{-1/2}$ is undefined at step 0.
- *The factor* — LoRA sets $B=0$, so $M_A=B^\top P B=0$ exactly ($M_B=A Q A^\top$ is milder).

**The floor — additive form.** Keep each curvature away from zero by **adding a multiple of
the identity** before inverting:
$$
X^{-1/2}\ \longrightarrow\ \bigl(X+\delta\,\lambda_{\max}(X)\,I\bigr)^{-1/2}
$$
— add $\delta$ times the top eigenvalue $\lambda_{\max}(X)$ (for the diagonal $P,Q$, the largest
entry), the same $\delta$ at all four sites ($P$, $Q$, $M_A$, $M_B$). The floor on $P,Q$ handles
$D=0$; the floor on $M_A,M_B$ handles $B=0$. This is distributed Shampoo's *scaled damping*
(Anil et al. 2020, App. D).

**Relative form.** Equivalently, divide inside the root by the top eigenvalue:
$$
\bigl(X/\lambda_{\max}(X)+\delta\bigr)^{-1/2}=\lambda_{\max}(X)^{1/2}\,\bigl(X+\delta\,\lambda_{\max}(X)\,I\bigr)^{-1/2}
$$
— the additive form times the scalar $\lambda_{\max}(X)^{1/2}$. This is what the code computes.

**Identical at $k=1$.** That scalar washes out: each inverse-root multiplies the whitened
direction $z$ by an overall constant, and both the polar map ($\varphi(c\,z)=\varphi(z)$) and
the $\rho$-rescale ($\sigma_{\max}(\mathrm dA)=\rho$) discard overall constants. So with no
cross ($k=1$) the additive and relative forms give the **identical** update.

**Relative ensures proper scaling at $k\ge2$.** The two forms diverge only at the cross, which
enters the covector at power $+1$ beside a momentum that carries no curvature factor. Rescale
the curvature $P\to sP,\ Q\to tQ$ (a stiffer layer, or drift over training). Under the
**additive** form,
$$
z_A\ \longrightarrow\ (st)^{-1/2}\,[\text{momentum part}]\ +\ (st)^{+1/2}\,[\text{cross part}],
$$
so the cross-to-momentum balance moves by $st$ — the cross rides the curvature magnitude. The
**relative** form divides $\lambda_{\max}$ back out, so $z_A$ (and the balance) is unchanged.
Since the rest of the optimizer is curvature-scale-invariant (the polar discards magnitude),
the relative form extends that invariance to the cross — so we use it.

**Proposition 5 (the floored update is a coherent program).** *Floor every curvature by adding
a multiple of $I$ — $X\mapsto X+c_X I$ at the four sites $P,Q,M_A,M_B$. The floored update
(any Picard depth) is the exact block-coordinate LMO of*
$$
\min_{\mathrm dA,\mathrm dB}\ \langle G,\,B\,\mathrm dA+\mathrm dB\,A\rangle
+\tfrac1{2\eta}\Bigl[\lVert B\,\mathrm dA+\mathrm dB\,A\rVert^2_{(P+c_P I,\,Q+c_Q I)}
+c_{M_A}\lVert\mathrm dA\rVert^2_{(I,\,Q+c_Q I)}
+c_{M_B}\lVert\mathrm dB\rVert^2_{(P+c_P I,\,I)}\Bigr].
$$
*Here $c_X=\delta\,\lambda_{\max}(X)$ is the floor of the previous subsection (its additive and
relative forms are the same program — identical at $k=1$, the relative one scale-invariant at
$k\ge2$).*

Write $\tilde P=P+c_P I$, $\tilde Q=Q+c_Q I$ for the floored metrics and
$\tilde M_A=B^\top\tilde P B+c_{M_A}I$ for the floored A small-side. The proposition gives this
explicit **A-block update** (the B-block is identical with $A\!\leftrightarrow\!B$,
$\tilde P\!\leftrightarrow\!\tilde Q$):
$$
\begin{aligned}
\tilde g_A &= \hat m_A + \tfrac1\eta\,B^\top\tilde P\,\mathrm dB\,A\,\tilde Q
   &&\text{(covector; cross term only at }k\ge2),\\
z_A &= \tilde M_A^{-1/2}\,\tilde g_A\,\tilde Q^{-1/2},\qquad
W_A = \tilde M_A^{-1/2}\,\varphi(z_A)\,\tilde Q^{-1/2}
   &&\text{(whiten, polar, unwhiten)},\\
\mathrm dA &= -\,\rho\,\frac{W_A}{\sigma_{\max}(W_A)},\qquad
   \rho=\frac{\eta}{\sigma_{\max}(A)+\sigma_{\max}(B)}
   &&\text{(spectral rescale)}.
\end{aligned}
$$

*Proof.* Fix $\mathrm dB$. The $\mathrm dA$-dependent part of the objective is
$$
\langle B^\top G,\,\mathrm dA\rangle
+\tfrac1\eta\,\langle B^\top\tilde P\,\mathrm dB\,A\,\tilde Q,\ \mathrm dA\rangle
+\tfrac1{2\eta}\,\operatorname{tr}\!\bigl(\mathrm dA^\top\,\tilde M_A\,\mathrm dA\,\tilde Q\bigr),
$$
using the three identities
$$
\begin{aligned}
\langle G,\,B\,\mathrm dA\rangle &= \langle B^\top G,\,\mathrm dA\rangle,\\
\langle B\,\mathrm dA,\ \mathrm dB\,A\rangle_{(\tilde P,\tilde Q)} &= \langle B^\top\tilde P\,\mathrm dB\,A\,\tilde Q,\ \mathrm dA\rangle,\\
\lVert B\,\mathrm dA\rVert^2_{(\tilde P,\tilde Q)}+c_{M_A}\lVert\mathrm dA\rVert^2_{(I,\tilde Q)}
&= \operatorname{tr}\!\bigl(\mathrm dA^\top\,\tilde M_A\,\mathrm dA\,\tilde Q\bigr).
\end{aligned}
$$
So the on-block linear cost is $\tilde g_A$ (momentum $\hat m_A$ in place of $B^\top G$, as
elsewhere) and the self-metric is $(\tilde M_A,\tilde Q)$. The spectral-cap LMO of a linear
cost under a Kronecker metric is the whitened polar (§"The whitened-polar step"), which is the
A-block update above. The floors keep it well-posed: $\tilde M_A\!\to\!c_{M_A}I$ at $B{=}0$,
and $\tilde P,\tilde Q$ stay positive-definite at $D{=}0$. The B-block is symmetric.
$\blacksquare$

**Recompute vs seed.** $M_A=B^\top\tilde P B$ is recomputed from the current $B$ every step, so
it equals the curvature of the current factor — but is zero at $B=0$ and needs the floor. The
alternative is to *seed* it: hold $M_A$ as its own running average over steps, initialized to
a nonzero constant so it is never zero (dense kl; the `KL-Methods` code starts it at $0.1$).
Seeding removes the floor but (i) makes that constant a knob and (ii) replaces the
current-step curvature with an average of past steps' curvatures.

**Knobs.** One tuning knob: $\delta$ (relative, so one fixed value, not swept). Everything
else is universal ($\eta$, momentum $\beta_1$, the curvature-EMA rate) or a fixed
implementation constant (polar iterations, refresh cadence, $\sigma_{\max}$-guard iterations).
Picard depth defaults to $k=1$.

**Cross-check (how others damp).**

- *Distributed Shampoo (Anil et al.):* $G+\varepsilon\,\lambda_{\max}I$ — relative, our form.
- *Clarifying Shampoo:* $(L+\varepsilon I)^{-p}$, absolute, tuned over $10^{-23}$–$10^{-10}$.
- *KL-Shampoo:* $\kappa I$ on the target moment plus a caption-level "damping or clipping";
  curvature eigenvalues seeded at $0.1$.
- *AdaPreLoRA:* one $\varepsilon=10^{-6}$, conditional "add $\varepsilon I$ to
  $B^\top L^{1/2}B$ if not invertible" — the ad-hoc version of our floor, same $B=0$ singularity.

## Sources

- Implementation: `CurvatureWhitenLoRA` in `lora_playground/optim.py` — KL curvature update at the `kl_coupled` branch of `_cw_apply_grouped` / `_cw_apply_per_pair`; the inner core, polar, and unwhiten sandwich in the same methods; guarded polar in `_polar_ns_guarded`; relative damping in `_rdinv`. Tests: `tests/test_kl_shampoo_lora.py`, `tests/test_curvature_whiten_lora.py`.
- KL covariance fit and the two-sided stationarity coupling (Proposition 1): `docs/papers/kl_shampoo_2509.03378.pdf`; matrix-normal MLE reading (Dutilleul, 1999).
- The two-sided spectral-cap program (the $(B^\top P B)^{-1/2}\varphi(\cdots)Q^{-1/2}$ sandwich, metric factors $(P,Q)$): `related_work_2026_05.md` §10.1–10.2 (Thm 10.1). AdaPreLoRA is the curvature instance $(P,Q)=(L^{1/2},R^{1/2})$ with the Frobenius cap (Cor 10.1; `adaprelora_2605.08734.pdf` Thm 3.2); chord-tight is the Frobenius instance $(P,Q)=(I,I)$ with the spectral cap (`algorithm_tight_chord.md`).
- The generalized Picard block-coordinate solver and its cross-coupling term (Proposition 3): `related_work_2026_05.md` §10 (Algorithm 10.1); chord's factor-coordinate form (Lemma 1): `algorithm_tight_chord.md` §5.
- Relative ("scaled") damping — $G+\varepsilon\,\lambda_{\max}(G)\,I$, scaled to the matrix's spectral norm to match the curvature scale and address rank-deficiency: Anil, Gupta, Koren, Regan, Singer, *Scalable Second Order Optimization for Deep Learning* (arXiv 2002.09018), App. D ("Scaled damping") and Alg. I.
- The polar-on/off open question and the SOAP / non-separable-residual variants of the same class: `soap_curvature_whitening.md`.
- Empirical lr-sensitivity motivation, the lr-band measurements, and the implementation plan for the Picard extension: `lr_robustness.md` (the derivation itself is §"Cross-coupling" above).
