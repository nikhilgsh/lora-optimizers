# SOAP, curvature whitening, and the chord-tight sandwich

> **Status: design discussion, no new code.** Works out what the proposed
> $S^{-1/2}\,\mathrm{SOAP}(m;S)$ update is, how it relates to the implemented
> `--curvature_whitening` flag, and what would actually be new to build.

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

## What is implemented today

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
whitening. So today: $S = S_\mathrm{curv}$, input $u$, polar sandwiched between
two matrix $S^{-1/2}$.

## The proposed update

$$
\Delta A \;\propto\; S^{-1/2}\,\mathrm{SOAP}(m;S),
$$
with $S = S_{\mathrm{curv},A}$, raw momentum $m = m_A$, and $\mathrm{SOAP}(m;S)$
the SOAP update on $m$ in the eigenbasis of $S$. The motivating heuristic is
$\mathrm{SOAP}(m;S) \approx \phi(S^{-1/2}m)$, which would make the proposal
$\approx S^{-1/2}\phi(S^{-1/2}m)$ — the implemented sandwich, with $S_\mathrm{curv}$
for $B^\top B$ and $m$ for $u$. The rest of this note checks that heuristic.

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
\Delta A \;\propto\; S_{\mathrm{curv},A}^{-1/2}\; m_A\; D_{\mathrm{in},A}^{-1/2},
\qquad
D_{\mathrm{in},A}=\mathrm{EMA}\!\big(\mathrm{diag}(g_A^\top g_A)\big)\in\mathbb{R}^{d_{\mathrm{in}}}.
$$

- **Left ($r$):** full $r\times r$ curvature matrix — cheap since $r\ll d$ (the
  object the flag already builds).
- **Right ($d_{\mathrm{in}}$):** diagonal curvature, $O(d_{\mathrm{in}})$ memory. A
  full right Gram would be $d_{\mathrm{in}}\times d_{\mathrm{in}}$ — the cost LoRA
  avoids. The diagonal is SOAP's own recipe for the huge side and AdaFactor's
  column factor.

Both indices are whitened by the same curvature signal, so there is no competing
Adam $v$ and no $m$-vs-$u$ choice. This is the **KFAC-LoRA** sketch in
`docs/plans/optimizer_ideas.md` ($H_A$ full $r\times r$ $+$ diagonal $D_V$, power
$\gamma=1/2$); treat them as one line of work.

**Keep or drop the polar.** The whitened update above is a 2nd-order (Shampoo/SOAP)
step that retains the whitened gradient's singular values. The polar $\phi$ is a
*separate* ingredient that orthogonalizes — flattens those singular values to 1,
keeping only rotation. Keeping $\phi$ (sandwiching it between the whitens) leaves
the update in the polar/Muon family with chord-rule magnitude; dropping it gives a
plain LoRA-subspace Shampoo/SOAP. Whether $\phi$ helps on top of good whitening is
the central open question of the polar line, so test both.

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

## What is new to build

1. **Sandwich with $m$ instead of $u$** (`optim.py:5481`) — smallest delta from the
   live flag; isolates the $m$-vs-$u$ direction effect.
2. **Harmonious two-sided whiten** $S_\mathrm{curv}^{-1/2} m\, D_\mathrm{in}^{-1/2}$,
   tested with and without the polar — the principled object; removes the
   $m$-vs-$u$ fork and reuses the KFAC-LoRA plan. Recommended.
3. **Real one-sided SOAP on $m$** (Adam-in-curvature-eigenbasis) as a clean baseline
   distinct from matrix whitening, to measure how far real SOAP sits from the
   $S^{-1/2}m$ idealization at LoRA's near-rank-1 gradients.

All discriminating comparisons run at **r256** (where saturation fails); r=16/64
collapse the variants together.

## Grounding

- Live pipeline: `lora_playground/optim.py:5465-5554`.
- SOAP algorithm: `docs/papers/soap_2409.11321.pdf`, Algs 1–3, Claim 1.
- Saturation: `preconditioning_saturation_2026_05_03.md` (commit `3ce7844`).
- r256 whiten lag: `chord_tight_whiten_lag_r256.md`.
- Factor-conditioning: `factor_conditioning_hypothesis.md`.
- KFAC-LoRA plan: `docs/plans/optimizer_ideas.md`.
