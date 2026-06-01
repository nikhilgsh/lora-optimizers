# SOAP, Curvature Whitening, and the Chord-Tight Sandwich

## Current status

`CurvatureWhitenLoRA` implements the requested SOAP-curvature/chord-tight
variant. It is registered as `curvature-whiten-lora` and
`curvature-whiten-polar-lora`.

Per PEFT LoRA pair $A\in\mathbb{R}^{r\times d_{\mathrm{in}}}$,
$B\in\mathbb{R}^{d_{\mathrm{out}}\times r}$, it keeps an affordable Kronecker
curvature estimate:

$$
S_A=\mathrm{EMA}(g_A g_A^\top),\qquad
D_A=\mathrm{EMA}(\mathrm{diag}(g_A^\top g_A)),
$$

and symmetrically $S_B=\mathrm{EMA}(g_B^\top g_B)$,
$D_B=\mathrm{EMA}(\mathrm{diag}(g_B g_B^\top))$.

The SOAP step is Adam on momentum in the eigenbasis of
$S\otimes D$. Since $D$ is diagonal, the large-side eigenbasis is the coordinate
basis:

$$
z_A
= Q_A\left[
\frac{Q_A^\top \hat m_A}{\sqrt{\hat v_A}+\epsilon}
\right],
\qquad
\hat v_A=\mathrm{EMA}\big((Q_A^\top g_A)^{\odot 2}\big),
$$

and

$$
z_B
= \left[
\frac{\hat m_B Q_B}{\sqrt{\hat v_B}+\epsilon}
\right]Q_B^\top,
\qquad
\hat v_B=\mathrm{EMA}\big((g_B Q_B)^{\odot 2}\big).
$$

The applied direction then uses the chord-tight outer curvature sandwich and
spectral budget:

$$
Y_A=S_A^{-1/2}z_A D_A^{-1/2},\qquad
Y_B=D_B^{-1/2}z_B S_B^{-1/2},
$$

$$
\rho=\eta/(\sigma_{\max}(A)+\sigma_{\max}(B)),\qquad
\Delta A=-\rho\,Y_A/\sigma_{\max}(Y_A),
$$

with the analogous $B$ update. `curvature-whiten-polar-lora` is the ablation
that replaces $z$ by $\phi(z)$ before the outer sandwich.

Verification:

- `tests/test_curvature_whiten_lora.py` pins the update against the equation
  above, including nontrivial $Q_A,Q_B$ rotations, the optional polar arm, and
  the chord-tight spectral rescale.
- `tests/test_optimizer_config_dict.py` covers optimizer config serialization.
- The current block-$\sigma_{\max}$ implementation has a 250-step r256
  packed-data smoke at $\eta=3\times10^{-4}$ on the deterministic 128-example
  eval subset:
  $1.008563\to0.987070\to0.968609\to0.955514\to0.946585$ at steps 50, 100,
  150, 200, 250. It had zero
  non-finite gradients and no runtime errors. Raw log:
  `logs/soap_curv_verify/curv_block_lr3e-4_steps250_eval128.log`; parsed
  summary:
  `logs/soap_curv_verify/curv_block_lr3e-4_steps250_eval128.summary.json`.
- The production wrapper path was smoked with compile enabled for
  `curvature-whiten-polar-lora` at r=256, packed seq2048, batch 4,
  grad-accum 4, diagnostics off. Eval loss was $1.026423\to1.019203$ at
  steps 10 and 20, peak memory was about 29.0 GB, and
  `n_non_finite_grads=0`. Raw log:
  `logs/soap_curv_verify/curv_block_polar_wrapper_lr3e-4_steps20.log`.

## Notation

For one LoRA pair $A \in \mathbb{R}^{r \times d_{\mathrm{in}}}$,
$B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$, $\Delta W = BA$. Per-factor
gradient $g_A$; first moment $m_A$; Adam direction
$u_A = m_A / (\sqrt{v_A} + \epsilon)$ with $v_A$ the elementwise second moment.
$\phi$ is the (soft) polar map, $\phi(U\Sigma V^\top) = UV^\top$. The A-side
curvature metric is the $r \times r$ EMA of factor-gradient outer products,
$$
S_{\mathrm{curv},A} = \mathrm{EMA}\big(g_A g_A^\top\big) = \mathrm{EMA}\big(B^\top H B\big),
\qquad \beta_{\mathrm{curv}} = 0.99,
$$
and symmetrically $S_{\mathrm{curv},B} = \mathrm{EMA}(g_B^\top g_B)$. Below $S$
and $m$ are A-side; B-side is symmetric.

## Historical curvature flag

`--curvature_whitening` (commit `67cfea4`) swaps the geometric factor Gram
$B^\top B$ for $S_{\mathrm{curv},A}$ inside the *existing* chord-tight whiten
pipeline (`optim.py:5469-5554`):
$$
\Delta A \;\propto\; S_{\mathrm{curv},A}^{-1/2}\,\phi\!\Big(S_{\mathrm{curv},A}^{-1/2}\, u_A\Big),
$$
magnitude set by the chord-tight $\rho$ and a $\sigma_{\max}$ renormalization.
The input is the Adam direction $u_A$ (`optim.py:5481`); the inverse-sqrt uses a
matrix Higham solve with $\lambda_{\max} = \sigma_{\max}(S_{\mathrm{curv}})$
(`optim.py:5546`). When off, the path is bit-identical to geometric-Gram
whitening. This older path is $S = S_\mathrm{curv}$, input $u$, polar
sandwiched between two matrix $S^{-1/2}$; it is not the implemented
SOAP-on-$m$ variant above.

## The implemented update

$$
\Delta A \;\propto\; S^{-1/2}\,\mathrm{SOAP}(m;S\otimes D)\,D^{-1/2},
$$
with $S = S_{\mathrm{curv},A}$, $D=D_A$, raw momentum $m=m_A$, and
$\mathrm{SOAP}(m;S\otimes D)$ the Adam-normalized update in the eigenbasis of
the Kronecker curvature estimate. The $B$ side is symmetric.

## What is and isn't true

**The exact fact (Anchor).** Whitening by the *instantaneous* Gram is exactly the
polar map: for $g_A = U\Sigma V^\top$,
$$
(g_A g_A^\top)^{-1/2}\,g_A = U\Sigma^{-1}U^\top \, U\Sigma V^\top = U V^\top = \phi(g_A).
$$
So curvature whitening is a momentum/EMA generalization of "polar the gradient,"
and the chord-tight sandwich is built on a real identity. This is the load-bearing
truth; everything below qualifies how far it extends.

**SOAP is not matrix whitening.** Reading Vyas et al. (`soap_2409.11321.pdf`,
Algs 1–3, Claim 1):

- *Idealized Adafactor in the eigenbasis (Alg 2) = idealized Shampoo (Alg 1)*
  (Claim 1) — the only regime where SOAP equals matrix whitening. It is
  **two-sided**: both eigenbases $Q_L$ of $\mathbb{E}[GG^\top]$ and $Q_R$ of
  $\mathbb{E}[G^\top G]$, with a rank-structured second moment
  $\widehat V_{ij}=A_iC_j/\sum_iA_i$ (row energy $A_i=\lambda_i$, column energy
  $C_j=\mu_j$), giving $L^{-1/2}GR^{-1/2}/\mathrm{Trace}(L)^{1/2}$.
- *Real SOAP (Alg 3) uses Adam*: a full elementwise EMA second moment of the
  rotated gradient ($V\leftarrow\beta_2V+(1-\beta_2)(G'\odot G')$), **not** the
  rank-1 reconstruction. This is strictly more expressive than the Kronecker
  structure and is the paper's improvement over Shampoo.

So real single-sided SOAP is $\mathrm{SOAP}(m;S)=Q\big[(Q^\top m)\oslash\sqrt{V}\big]$
with $Q=\mathrm{eigvecs}(S)$ and $V$ the elementwise rotated second moment — **Adam
in the curvature eigenbasis**, neither $S^{-1/2}m$ nor a polar.

**Consequences for the heuristic.**

| object | what it is | is it $S^{-1/2}m$? |
|---|---|---|
| repo `--curvature_whitening` | $S^{-1/2}\phi(S^{-1/2}u)$ (matrix whiten $+$ polar) | — (has the polar) |
| real 1-sided SOAP on $m$ | $Q[(Q^\top m)\oslash\sqrt V]$ | no (Adam-in-eigenbasis) |
| idealized 1-sided SOAP | $S^{-1/2}m$ | yes, but only two-sided + drop $C_j$ |

The polar in $\mathrm{SOAP}(m;S)\approx\phi(S^{-1/2}m)$ does not come from SOAP. It
comes from the Anchor identity and holds only when $S^{-1/2}m$ is already
near-orthogonal — the preconditioning-saturation regime
(`preconditioning_saturation_2026_05_03.md`), which holds at r=16/r=64 but **fails
at r256** (B at ${\sim}14\%$ rank, $S^{-1/2}$ spread ${\sim}340$, dA rotated
${\sim}60°$; `chord_tight_whiten_lag_r256.md`). The reading
"$S^{-1/2}\mathrm{SOAP}(m;S)=S^{-1}m$" (a full inverse-curvature Newton step on
momentum) is valid only in the idealized-Shampoo limit; for real Adam SOAP the
second whitening does not cleanly compound. r256 is therefore the cell where these
variants separate.

## $m$ vs $u$, and why the question is a symptom

Feeding raw $m$ is the SOAP-native input (SOAP normalizes in its own eigenbasis;
$u$ double-normalizes the $r$-index). But $S_\mathrm{curv}$ is $r\times r$ and
conditions only the row index — it is blind to the $d_{\mathrm{in}}$ column index
that Adam's $v$ also scales, so $m$ alone drops information $u$ carries. Since
$\phi$ is scale-invariant, $m$ vs $u$ is a pure *direction* ablation.

The fork exists only because the preconditioner is inhomogeneous: a full-matrix
whitener on the left index next to Adam's diagonal $v$ on both. The principled fix
is to normalize both sides from curvature, Kronecker-factored, big side diagonal.

## Harmonious two-sided normalization

$$
\Delta A \;\propto\; S_{\mathrm{curv},A}^{-1/2}\;
\mathrm{SOAP}(m_A;S_{\mathrm{curv},A}\otimes D_{\mathrm{in},A})\;
D_{\mathrm{in},A}^{-1/2},
\qquad
D_{\mathrm{in},A}=\mathrm{EMA}\!\big(\mathrm{diag}(g_A^\top g_A)\big)\in\mathbb{R}^{d_{\mathrm{in}}}.
$$

- **Left ($r$):** full $r\times r$ curvature matrix — cheap since $r\ll d$ (the
  object the flag already builds).
- **Right ($d_{\mathrm{in}}$):** diagonal curvature, $O(d_{\mathrm{in}})$ memory. A
  full right Gram would be $d_{\mathrm{in}}\times d_{\mathrm{in}}$ — the cost LoRA
  avoids. The diagonal is SOAP's own recipe for the huge side and AdaFactor's
  column factor.

Both indices use the same curvature signal: $S$ supplies the rank-side
eigenbasis for SOAP and the outer inverse square root, while $D$ supplies the
large-side coordinate basis and the outer diagonal inverse square root. Adam's
$v$ lives in that $S\otimes D$ basis; it is not an inverse eigenvalue.

**Keep or drop the polar.** The SOAP update above retains the SOAP-normalized
direction's singular values. The polar $\phi$ is a
*separate* ingredient that orthogonalizes — flattens those singular values to 1,
keeping only rotation. Keeping $\phi$ before the outer sandwich leaves
the update in the polar/Muon family with chord-rule magnitude; dropping it gives a
plain SOAP-curvature/chord-tight step. Whether $\phi$ helps on top of SOAP is a
separate ablation, so test both.

## Existing curvature-whitening A/B (already inconclusive)

OLMo r256, full polar k=1, packed_v1.1, step 9000, single seed, $\sigma=0.0007$.
Groups: `curvature_whitening_ns8_k1_r256_olmo_opc` (ON),
`chord_tight_polar_express_phase_L_lrsweep_r256_blackwell` (OFF $\equiv$ full
polar), `adamw_phase_L_lrsweep_r256_blackwell`.

| arm | best lr | final | $\Delta$ AdamW |
|---|---|---|---|
| curvature ON | 3e-3 | 0.7394 | $-18.6\sigma$ |
| curvature OFF | 1e-2 | 0.7414 | $-15.8\sigma$ |
| AdamW | 1e-4 | 0.7524 | — |

ON edges OFF by ${\sim}2.9\sigma$ at its best lr, but the win is confounded:
curvature whitening shifts the optimal lr down (3e-3 vs 1e-2) and worsens high-lr
robustness (lr=1e-2: ON 0.7521 vs OFF 0.7414, ${+}15\sigma$; far worse at 3e-2,
1e-1). At matched low lr it wins, at matched high lr it loses. A narrow-basin
${\sim}3\sigma$ edge, single seed, off the canonical 4k horizon — not a clean win,
and the geometric-Gram-vs-curvature axis is not settled by it.

## Built / remaining

1. `curvature-whiten-lora`: SOAP on $m$ in the $S\otimes D$ basis, followed by
   the outer curvature sandwich and chord-tight spectral rescale.
2. `curvature-whiten-polar-lora`: same, but replaces $z$ by $\phi(z)$ before the
   outer sandwich.
3. Long-horizon optimizer comparison remains open; early stability must be
   checked at r256 because r=16/64 collapse the variants together.

All discriminating comparisons run at **r256** (where saturation fails); r=16/64
collapse the variants together.

## Grounding

- SOAP-curvature implementation: `CurvatureWhitenLoRA` in
  `lora_playground/optim.py`.
- Equation-level tests: `tests/test_curvature_whiten_lora.py`.
- Historical curvature pipeline: `lora_playground/optim.py` chord-tight
  `curvature_whitening` path.
- SOAP algorithm: `docs/papers/soap_2409.11321.pdf`, Algs 1–3, Claim 1.
- Saturation: `preconditioning_saturation_2026_05_03.md` (commit `3ce7844`).
- r256 whiten lag: `chord_tight_whiten_lag_r256.md`.
- Factor-conditioning: `factor_conditioning_hypothesis.md`.
- KFAC-LoRA plan: `docs/plans/optimizer_ideas.md`.
