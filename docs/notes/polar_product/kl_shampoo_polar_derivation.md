# Deriving kl-shampoo-polar

`kl-shampoo-polar` is the LoRA optimizer that, per factor, takes a momentum step,
**whitens it by a two-sided curvature estimate, projects it to the spectral cap
with a polar map, unwhitens through the same curvature, and rescales to a spectral
trust region.** The curvature is not an ad-hoc Gram EMA: it is the Kronecker
preconditioner that minimizes the KL divergence to the factor gradient's second
moment, which couples the two sides of the sandwich. This note derives the update
end to end — the curvature estimate first, then the whitened-polar step it feeds.

Registered as `kl-shampoo-polar-lora` =
`CurvatureWhitenLoRA(kl_coupled=True, soap_v=False, use_polar=True)`. The same
class with `kl_coupled=False` is one-sided Shampoo, with `soap_v=True` is
SOAP-on-momentum, and with `use_polar=False` drops the cap; this note covers only
the KL + polar corner.

## Notation

One PEFT LoRA pair, with $r\ll d_{\mathrm{in}},d_{\mathrm{out}}$:

- $A\in\mathbb{R}^{r\times d_{\mathrm{in}}}$, $B\in\mathbb{R}^{d_{\mathrm{out}}\times r}$, adapter weight $\Delta W=BA$.
- $G=\nabla_{\Delta W}\mathcal L$ — loss gradient w.r.t. the merged weight (upstream of the factors).
- $g_A=B^\top G\in\mathbb{R}^{r\times d_{\mathrm{in}}}$, $g_B=GA^\top\in\mathbb{R}^{d_{\mathrm{out}}\times r}$ — the per-factor gradients.
- $m_A,m_B$ — $\beta_1$ EMAs of $g_A,g_B$; $\hat m_A,\hat m_B$ — their bias-corrected forms.
- $\varphi$ — the polar map, $\varphi(U\Sigma V^\top)=UV^\top$ (sets all nonzero singular values to 1).

Everything below is stated A-side; the B-side is the symmetric construction with
$A\leftrightarrow B$, $d_{\mathrm{in}}\leftrightarrow d_{\mathrm{out}}$.

The curvature state, per pair, is **four objects** — one dense and one diagonal
factor per side — the Kronecker factors of the KL fit to the factor-gradient second
moment, **coupled** so each is whitened by its conjugate factor ($\mathbb E[\cdot]$
are EMAs; §"The curvature" derives *why* this is the KL-optimal fit). For the $A$
factor, $S_{\mathrm{curv},A}\in\mathbb{R}^{r\times r}$ (dense) and
$D_{\mathrm{in}}\in\mathbb{R}^{d_{\mathrm{in}}}$ (diagonal):
$$
S_{\mathrm{curv},A}=\tfrac1{d_{\mathrm{in}}}\,\mathbb E\!\bigl[g_A\,D_{\mathrm{in}}^{-1}\,g_A^\top\bigr],
\qquad
D_{\mathrm{in}}=\tfrac1{r}\,\operatorname{diag}\mathbb E\!\bigl[g_A^\top\,S_{\mathrm{curv},A}^{-1}\,g_A\bigr].
$$
For the $B$ factor, $S_{\mathrm{curv},B}\in\mathbb{R}^{r\times r}$ (dense) and
$D_{\mathrm{out}}\in\mathbb{R}^{d_{\mathrm{out}}}$ (diagonal):
$$
S_{\mathrm{curv},B}=\tfrac1{d_{\mathrm{out}}}\,\mathbb E\!\bigl[g_B^\top\,D_{\mathrm{out}}^{-1}\,g_B\bigr],
\qquad
D_{\mathrm{out}}=\tfrac1{r}\,\operatorname{diag}\mathbb E\!\bigl[g_B\,S_{\mathrm{curv},B}^{-1}\,g_B^\top\bigr].
$$

These equations define the *ideal* KL fixed point. The implementation maintains a
**damped streaming approximation** to it: the factors are running EMAs (one
flip-flop alternation per step), and the conjugate-factor inverses
($D_{\mathrm{in}}^{-1}$, $S_{\mathrm{curv},A}^{-1}$, …) are the relative-damped
$(x/x_{\max}+\delta)^{-1}$ form used throughout (§"The update, per step"), not exact
matrix inverses. So the running factors approximate, rather than exactly satisfy,
the fixed point above.

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
($\mathrm dB$ symmetric, times the LoRA+ B-multiplier). So $\sigma_{\max}(\mathrm dA)=\rho$ exactly.

Composing steps 2–4, the direction the rescale acts on is
$$
\boxed{\,W_A\;\propto\;S_{\mathrm{curv},A}^{-1/2}\,
\varphi\!\bigl(S_{\mathrm{curv},A}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}\bigr)\,
D_{\mathrm{in}}^{-1/2}\,}
$$
— the **same curvature sandwich applied twice with $\varphi$ in between.** The rest
of the doc derives the two ingredients: where $(S_{\mathrm{curv},A},D_{\mathrm{in}})$
come from (the KL fit), and what $\varphi$ does (the spectral cap).

The inverse-square-roots in practice are **relative-damped**: a nonnegative
spectrum $x$ is mapped to $(x/x_{\max}+\delta)^{-1/2}$ rather than $x^{-1/2}$, and
an uninitialized (all-zero) factor maps to the identity, so before the EMAs warm
up the step is a plain momentum step. $S_{\mathrm{curv},A}^{-1/2}$ is formed in the
eigenbasis $U_A$ of $S_{\mathrm{curv},A}$ (periodic `eigh` seed + QR refresh;
eigenvalues by Rayleigh quotient $\lambda_i=u_i^\top S_{\mathrm{curv},A}u_i$);
$D_{\mathrm{in}}^{-1/2}$ is a diagonal scaling.

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
\mathrm dB=-c\,\rho\,\frac{W_B}{\sigma_{\max}(W_B)},\qquad c=\text{LoRA+ B-multiplier}.
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
reproduces exactly the four coupled curvature equations stated in the Notation — the
A-side row yields $(S_{\mathrm{curv},A},D_{\mathrm{in}})$, the B-side row yields
$(S_{\mathrm{curv},B},D_{\mathrm{out}})$. Both inverses are cheap ($r\times r$ dense
and length-$d$ diagonal), and each factor is maintained by the streaming EMA of the
Remark above (one flip-flop alternation per optimizer step).

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

**Guarding the $\sigma_{\max}$ estimate (load-bearing).** Both the polar pre-norm
and the final rescale divide by an estimated $\sigma_{\max}$ obtained by warm-started
power iteration. A stale or cold start vector can *under*-estimate $\sigma_{\max}$,
which over-scales the update into the Newton–Schulz iteration's divergent region and
produces all-parameter NaN — a slow-onset failure that can appear hundreds of steps
into an otherwise-healthy run. The estimator is floored at
$\max(\text{max row }L_2,\ \text{max col }L_2)$ — both valid lower bounds on
$\sigma_{\max}$ — and any non-finite polar output is recomputed from the
Frobenius-normalized input (for which $\sigma_{\max}\le1$ is guaranteed). The floor
binds only when the warm estimate is pathological; in the healthy case it leaves the
denominator unchanged.

## Cross-coupling: the Picard correction (proposed extension)

> **Status.** `CurvatureWhitenLoRA` has no Picard loop — the $k{=}1$ step above is
> the whole of the shipped optimizer. This section is the canonical derivation of the
> *proposed* coupled step; `lr_robustness.md` carries its empirical (lr-sensitivity)
> motivation and the implementation plan.

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

**Remark (a consistent alternative — commit to the diagonals).** The mixed metric is
a *fidelity-vs-consistency* choice, not a forced compromise. Committing to one global
metric $(P,Q)=(D_{\mathrm{out}},D_{\mathrm{in}})$ makes every block read off the
*same* metric — A-block self $(B^\top D_{\mathrm{out}}B,\,D_{\mathrm{in}})$, B-block
self $(D_{\mathrm{out}},\,A D_{\mathrm{in}}A^\top)$, cross
$B^\top D_{\mathrm{out}}\,\mathrm dB\,A\,D_{\mathrm{in}}$ — a fully consistent
Algorithm 10.1 instance whose cross term is **exact** (valid when
$D_{\mathrm{out}}\succ0$ and $B$ has full column rank, else the small-side inverse
root is damped as elsewhere). The price is that it replaces kl's dense small side
$S_{\mathrm{curv},A}$ by $B^\top D_{\mathrm{out}}B$. The identity
that explains why is, at the KL fixed point (treating $B$ fixed in the EMA),
$$
S_{\mathrm{curv},A}=B^\top M_{\mathrm{out}}B,
\qquad
M_{\mathrm{out}}:=\tfrac1{d_{\mathrm{in}}}\,\mathbb E\!\bigl[G\,D_{\mathrm{in}}^{-1}G^\top\bigr]\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{out}}} :
$$
$S_{\mathrm{curv},A}$ is the *dense* output curvature $M_{\mathrm{out}}$ seen only on
$\operatorname{range}B$, whereas the cross term needs an output metric *off*
$\operatorname{range}B$ — where kl maintains only the diagonal $D_{\mathrm{out}}$. So
the "inconsistency" is a resolution mismatch: dense on $\operatorname{range}B$ (kept
as $S_{\mathrm{curv}}$), diagonal full-space (kept as $D$). **Option (a)** (current
kl) keeps the dense small side and approximates the cross; **option (b)** commits to
the diagonals everywhere for an exact program — an AdaPreLoRA-like global
diagonal-metric polar variant, but with power-1 KL-coupled diagonals rather than
AdaPreLoRA's power-$1/2$ Adafactor diagonal. Which trades better is empirical.

**Corollary (the diagonal exponent is pinned at 1).** The cross carries the metric
factors at power 1, and that power is fixed by kl's whitening, not free: the on-block
large-side whitening is $Q^{-1/2}$, and kl whitens by $D^{-1/2}$, so $Q=D$ (power 1)
— hence $B^\top D_{\mathrm{out}}\,\mathrm dB\,A\,D_{\mathrm{in}}$, linear in each
diagonal. AdaPreLoRA whitens by $R^{-1/4}$, so its $Q=R^{1/2}$ and its cross carries
$R^{1/2}$ (power $1/2$); the exponents differ because the whitening conventions
differ, not because kl's is ambiguous. The *metric choice*
$P\rightsquigarrow D_{\mathrm{out}}$ is the approximation; the *exponent given that
choice* is pinned. (So the cap-off cross-check against AdaPreLoRA's closed form —
`adaprelora` Thm 3.2 under $L\to D_{\mathrm{out}},R\to D_{\mathrm{in}}$ — validates a
power-$1/2$ variant, not kl's power-1 step.)

**The loop.** Initialize $\mathrm dA^{(0)}=\mathrm dB^{(0)}=0$, so iter 0 is the
$k{=}1$ step. For $n=0,\dots,k-1$: recompute $\tilde g_A^{(n)},\tilde g_B^{(n)}$ from
the current $\mathrm dB^{(n)},\mathrm dA^{(n)}$ (at physical scale $\sigma_{\max}=\rho$),
run each through the $k{=}1$ pipeline (whiten → polar → unwhiten → $\rho$-rescale),
and update. The cross term is added to the momentum at solve time, not folded into
the EMA; the $\tfrac1\eta$ makes it $O(1)$ relative to $\hat m$ (since
$\mathrm dB\propto\rho\propto\eta$). Every product stays in the skinny $r\times d$
factors — the dense $d_{\mathrm{out}}\times d_{\mathrm{in}}$ weight is never formed.

## Sources

- Implementation: `CurvatureWhitenLoRA` in `lora_playground/optim.py` — KL curvature update at the `kl_coupled` branch of `_cw_apply_grouped` / `_cw_apply_per_pair`; the inner core, polar, and unwhiten sandwich in the same methods; guarded polar in `_polar_ns_guarded`; relative damping in `_rdinv`. Tests: `tests/test_kl_shampoo_lora.py`, `tests/test_curvature_whiten_lora.py`.
- KL covariance fit and the two-sided stationarity coupling (Proposition 1): `docs/papers/kl_shampoo_2509.03378.pdf`; matrix-normal MLE reading (Dutilleul, 1999).
- The two-sided spectral-cap program (the $(B^\top P B)^{-1/2}\varphi(\cdots)Q^{-1/2}$ sandwich, metric factors $(P,Q)$): `related_work_2026_05.md` §10.1–10.2 (Thm 10.1). AdaPreLoRA is the curvature instance $(P,Q)=(L^{1/2},R^{1/2})$ with the Frobenius cap (Cor 10.1; `adaprelora_2605.08734.pdf` Thm 3.2); chord-tight is the Frobenius instance $(P,Q)=(I,I)$ with the spectral cap (`algorithm_tight_chord.md`).
- The generalized Picard block-coordinate solver and its cross-coupling term (Proposition 3): `related_work_2026_05.md` §10 (Algorithm 10.1); chord's factor-coordinate form (Lemma 1): `algorithm_tight_chord.md` §5.
- The polar-on/off open question and the SOAP / non-separable-residual variants of the same class: `soap_curvature_whitening.md`.
- Empirical lr-sensitivity motivation, the lr-band measurements, and the implementation plan for the Picard extension: `lr_robustness.md` (the derivation itself is §"Cross-coupling" above).
