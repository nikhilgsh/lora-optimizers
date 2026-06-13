# Per-factor step scaling: should $\dot A$ and $\dot B$ get equal operator-norm budget?

## Setup

LoRA factors and their updates, in the PEFT convention used throughout the paper:

- $A\in\mathbb{R}^{r\times d_{in}}$, $B\in\mathbb{R}^{d_{out}\times r}$; adapter $\Delta W=(\alpha/r)\,BA$.
- $\dot A,\dot B$: the updates applied this step ($A\leftarrow A+\dot A$).
- $\varphi(M)=UV^\top$ for the SVD $M=U\Sigma V^\top$ — the **polar map** (keeps the
  direction, sets every nonzero singular value to $1$).
- $\sigma_{\max}(\cdot)$: top singular value (operator norm).
- $\eta$: learning rate. $W_A,W_B$: the curvature-whitened polar directions before the
  final rescale.

Algorithm 1 (`\methodname`) rescales both factor updates to the **same** operator-norm
budget $\rho$:

$$\dot A=-\rho\,\frac{W_A}{\sigma_{\max}(W_A)},\quad
  \dot B=-\rho\,\frac{W_B}{\sigma_{\max}(W_B)},\qquad
  \rho=\frac{\eta}{\sigma_{\max}(A)+\sigma_{\max}(B)},$$

so $\sigma_{\max}(\dot A)=\sigma_{\max}(\dot B)=\rho$.

**Question.** $A$ maps $d_{in}\!\to\!r$ and $B$ maps $r\!\to\!d_{out}$ — very different
shapes. Should the two updates instead be scaled *asymmetrically*, by a $\mu$P / Keller
shape factor? This note argues **no**, and says what the paper should claim instead.

---

## 1. The per-factor feature-RMS frame, and why it does not apply

**Lemma 1 (Su feature-RMS scaling).** For a dense layer $y=xW$,
$W:\mathbb{R}^{d_{in}}\!\to\mathbb{R}^{d_{out}}$, scaling a polar update $\varphi(\cdot)$ by

$$\alpha=\sqrt{\max(1,\,d_{out}/d_{in})}\ \text{(Keller)}\quad\text{or}\quad
  \alpha=\sqrt{d_{out}/d_{in}}\ \text{(MuP)}$$

makes the feature increment uniform across layer shapes, $\|\Delta y\|_{RMS}=\eta\|x\|_{RMS}$.
*Source: Su, kexue.fm/archives/11772; this is the origin of the $\max(1,\cdot)$ in Keller
Jordan's Muon.*

Applying Lemma 1 to each factor as if it were a standalone Muon layer ($A:d_{in}\!\to\!r$,
$B:r\!\to\!d_{out}$) gives $\alpha_A=\sqrt{\max(1,r/d_{in})}=1$ (since $r<d_{in}$) and
$\alpha_B=\sqrt{d_{out}/r}$, i.e. a target ratio

$$\frac{\sigma_{\max}(\dot B)}{\sigma_{\max}(\dot A)}=\frac{\alpha_B}{\alpha_A}
  \in\{5.66\ (\text{Keller}),\ 32\ (\text{MuP})\}\quad\text{at }d=2048,\ r=64,$$

far from Algorithm 1's $1{:}1$.

**Why Lemma 1 does not apply here.** Lemma 1's uniformity protects the *input to the next
nonlinearity* — it keeps each layer's output RMS controlled so the following activation
sees a well-scaled input. In LoRA, $A$ and $B$ have **no nonlinearity between them**. This
does not mean only the product matters: we optimize $A$ and $B$, not $\Delta W$, and the
feature increment is *bilinear* in $\dot A,\dot B$ (§2). The standalone-layer picture
mis-models the intermediate $z=Az$, which is not a protected feature but is coupled to $B$
through the product. The correct object is the bilinear increment of §2.

## 2. The bilinear frame: $\mu$A feature increment

**Proposition 2 ($\mu$A decomposition).** With layer input $Z$, intermediate feature
$Z_A=AZ$, output feature $Z_B=(\alpha/r)BAZ$, the per-step output-feature increment splits as

$$\Delta Z_B=\underbrace{\tfrac{\alpha}{r}B\,\dot A Z}_{\delta^1}
  +\underbrace{\tfrac{\alpha}{r}\dot B\,Z_A}_{\delta^2}
  +\underbrace{\tfrac{\alpha}{r}\dot B\,\dot A Z}_{\delta^3}.$$

*Source: `\cite{mua}` (arXiv 2602.06204), Eq. 1.* The three terms are the three pieces of
the exact weight increment $\Delta W\propto(B+\dot B)(A+\dot A)-BA=B\dot A+\dot B A+\dot B\dot A$
applied to $Z$:

- $\delta^1$ — the update to $A$, seen through the current $B$.
- $\delta^2$ — the update to $B$, acting on the current $A$.
- $\delta^3$ — the cross term (second order in the step).

**Lemma 3 (operator-norm bounds on the three terms).** Under Algorithm 1
($\sigma_{\max}(\dot A)=\sigma_{\max}(\dot B)=\rho$),

$$\|\delta^1\|\le\sigma_{\max}(B)\,\rho,\qquad
  \|\delta^2\|\le\sigma_{\max}(A)\,\rho,\qquad
  \|\delta^3\|\le\rho^2,$$

and $\|\delta^1\|+\|\delta^2\|\le\rho(\sigma_{\max}(A)+\sigma_{\max}(B))=\eta$ (the merged
operator-norm cap, paper Eq. ours-prodcap). *Proof: submultiplicativity of $\sigma_{\max}$,
then substitute $\rho$.* $\blacksquare$

**Definition (maximal feature learning, $\mu$A).** LoRA learns *maximally* if
$\Delta Z_B=\Theta(1)$ as $n,r\to\infty$ ($r\le n$), with each $\delta^i=O(1)$ and at least
one $\Omega(1)$. The ideal is all three $\Theta(1)$ — **both factors contribute** (`\cite{mua}`
§3.2).

**The standard-LoRA failure mode.** With Init[A] ($A\sim\mathcal N(0,1/n)$, $B=0$, $\alpha=1$
— our exact initialization), $\mu$A gives (`\cite{mua}` Cor. 4.4)

$$\delta^1=\Theta(r^{-1/2}),\qquad \delta^2=\Theta(1),\qquad \delta^3=\Theta(r^{-1/2}).$$

For $r\ll n$ only $\delta^2$ survives: learning runs through $B$, while $A$ behaves as a
fixed random projection $Z_A\approx A_0Z$. Updating $A$ ($\delta^1$) and the cross term
($\delta^3$) are asymptotically inert.

## 3. LoRA+ targets the same balance; the per-factor split is its opposite

LoRA+ (`\cite{loraplus}`, arXiv 2402.12354 — the precursor to `\cite{mua}` with the
identical $\delta^1,\delta^2,\delta^3$ split) *cures* the frozen-$A$ regime with asymmetric
learning rates $\eta_A=\Theta(n^{-1})$, $\eta_B=\Theta(1)$ (ratio $\Theta(n)$, empirically
$2$–$16\times$). The mechanism:

- $\eta_B\gg\eta_A$ **grows $B$ to $\Theta(1)$**, escaping the $B=0$ init, so $A$'s updates
  have a non-zero $B$ to act through ($\delta^1$ becomes live).
- small $\eta_A$ keeps $\delta^1$ bounded rather than exploding.
- end state: $\|\delta^1\|\approx\|\delta^2\|$ — balanced contributions.

So LoRA+ (asymmetric lr on *raw Adam* updates) and Algorithm 1 (equal $\rho$ on
$\sigma_{\max}$-normalized updates) **target the same thing**, $\|\delta^1\|\approx\|\delta^2\|$;
they differ only in mechanism. The $\sigma_{\max}$-normalization removes up front the raw
gradient imbalance that LoRA+ has to correct with $\eta_B/\eta_A$.

**Remark (the Keller split is the opposite of LoRA+, despite both enlarging $B$'s step).**
Keller scales $\dot B$ up by $\alpha_B=\sqrt{d_{out}/r}$ over $\dot A$ ($\alpha_A=1$), which
directly inflates $\|\delta^2\|=\|\dot B A\|$ over $\|\delta^1\|=\|B\dot A\|$ — pushing
*toward* $\delta^2$-dominance, the standard-LoRA failure. LoRA+'s $\eta_B\gg\eta_A$ is a
grow-$B$-then-bound-$\delta^1$ maneuver, not a make-$\delta^2$-bigger one. The surface
direction agrees; the effect on the balance is opposite.

## 4. What Algorithm 1 already does (measured)

The radius $\rho=\eta/(\sigma_{\max}(A)+\sigma_{\max}(B))$ does two $\mu$A jobs at once.

**(i) It lifts $A$ off the frozen floor — at the $\sigma_{\max}$ level.** Equal budget
$\rho$ plus the measured $\sigma_{\max}(A)\approx\sigma_{\max}(B)$ makes the two bounds of
Lemma 3 equal, so $A$ gets a $\Theta(\eta)$ step rather than $\delta^1=\Theta(r^{-1/2})\!\to\!0$.

*Scope (honest limit):* Lemma 3 bounds $\|\delta^1\|,\|\delta^2\|$ via the top singular
value; equal bounds are not a proof that the *actual* $\|\delta^1\|\approx\|\delta^2\|$
(that depends on alignment, not just $\sigma_{\max}$). The direct check is the
discriminating measurement under Sources.

| quantity (step 9000) | $\sigma_{\max}(A)$ | $\sigma_{\max}(B)$ | ratio $B/A$ |
|---|---|---|---|
| rank $r=64$  | 2.74 | 2.71 | 0.99 |
| rank $r=256$ | 3.22 | 2.43 | 0.76 |

**(ii) It absorbs the rank dependence of $\eta$.** The denominator
$\sigma_{\max}(A)+\sigma_{\max}(B)$ is measured to be $O(1)$ and ~rank-invariant ($5.44$ at
$r=64$, $5.65$ at $r=256$), so $\rho\approx\eta/5.5$ at both ranks and the merged cap
$\|\delta^1+\delta^2\|\le\eta$ holds at fixed $\eta$ for all $r$. This is the operator-norm
analog of $\mu$A's $\alpha=r^{-1}$ rank absorption — dividing by a rank-stable denominator
instead of scaling $\alpha$ by $r^{-1}$ — and is the mechanism behind the paper's empirical
$\eta$-transfer-across-rank result.

> **Paper constraint.** The lr-transfer claim stays *empirical* — do not assert $r^{-1/2}$
> or "theory predicts." $\mu$A is cited as related work on rank-aware lr; the
> $\sigma_{\max}$-rank-stability above is reported as a *measured* mechanism, not a theorem.

**(iii) The cross term is negligible — consistent with $\mu$A.** Lemma 3 gives
$\|\delta^3\|\le\rho^2=O(\eta^2)$; $\mu$A predicts $\delta^3=\Theta(r^{-1/2})\!\to\!0$ under
Init[A]. The Picard iteration (`cw_picard_iters`$\ge2$) forms $\delta^3$ explicitly and the
project found $k\ge2\approx k=1$ — both say the cross is inert in this regime.

**(iv) $A$'s subspace still contracts under our optimizer.** Stable rank of $A$ falls
monotonically ($48\to13$ of $64$ at $r=64$; $148\to17$ of $256$ at $r=256$) while $B$
settles higher ($\to27$, $\to60$). So (i) buys $A$ a $\Theta(\eta)$ step, not a fully
active high-rank $A$ — the frozen-$A$ picture is softened, not erased.

## 5. Recommendation

**Keep the equal-$\rho$ update; do not add a per-factor split.** A Keller/MuP split forces
$\sigma_{\max}(\dot B)>\sigma_{\max}(\dot A)$, i.e. $\delta^2>\delta^1$ — it re-introduces
the $\mu$A imbalance (§2) and is the opposite of LoRA+ (§3). Predicted ordering of the
ablation arms: $\texttt{none}\ge\texttt{keller}>\texttt{mup}$.

**Strengthen the paper's exposition.** The paper motivates $\rho$ only as the operator-norm
budget giving the merged cap. Add a remark: the same $\rho$ (i) balances the two factor
feature-increment contributions $\delta^1,\delta^2$ — so both factors learn, unlike standard
Adam-LoRA where $\delta^1$ vanishes — and (ii) absorbs the rank dependence of $\eta$. This
upgrades the lr-transfer bullet from a bare empirical claim to a measured mechanism (within
the no-$r^{-1/2}$ constraint) and frames the symmetric split as a deliberate design choice.

## 6. Proposals

**P1 — Exposition (no new runs).** Add the §5 remark to the method section; cite the
$\sigma_{\max}$-balance and rank-stability figures.

**P2 — Confirming ablation.** A flag `cw_factor_mup_split ∈ {none, keller, mup}` folding
$\alpha_A,\alpha_B$ into the radius,

$$\rho=\frac{\eta}{\alpha_A\sigma_{\max}(B)+\alpha_B\sigma_{\max}(A)},\quad
  \dot A\propto\alpha_A,\ \dot B\propto\alpha_B,$$

with `none` ($\alpha_A=\alpha_B=1$) bit-identical to the protagonist. The split changes
*where* the budget goes but preserves the merged cap, so $\eta$ still transfers. Each arm
**lr-swept independently** — changing the split moves the optimal $\eta$, so comparing at a
shared best-$\eta$ confounds split with lr. Predicted: $\texttt{none}\ge\texttt{keller}>\texttt{mup}$.
Value: a `none`-wins result shows the symmetric split is load-bearing, not incidental.

**P3 — Init sensitivity (optional).** $\mu$A says any residual imbalance is
Init-dependent: Init[A] freezes $A$ via $\delta^1$, Init[B] freezes $B$ via $\delta^2$.
Equal-$\rho$ may make the optimizer Init-robust *because* it balances $\sigma_{\max}$. Arm:
Init[A] vs Init[B] under `\methodname`; measure whether $\sigma_{\max}(A)\approx\sigma_{\max}(B)$
and the final loss are Init-invariant (they are not for Adam-LoRA).

## Sources

- Backing measurements and figures: `notebooks/snapshot_analysis/06_factor_step_scaling.ipynb`
  (chord-tight-k1, OLMo-2-1B all-linear, $r\in\{64,256\}$, steps $0$–$9000$).
- Cross term $k\ge2\approx k=1$: `docs/notes/polar_product/algorithm_tight_chord.md`.
- Papers: Su (kexue.fm/archives/11772); LoRA+ (2402.12354); $\mu$A (2602.06204).
- **Discriminating measurement** for claim 4(i) — reconstruct $\|\delta^1\|=\|B\dot A\|$,
  $\|\delta^2\|=\|\dot B A\|$ directly via one optimizer step on snapshot state (replaces
  the $\sigma_{\max}$ proxy). Run if P2 surprises.
