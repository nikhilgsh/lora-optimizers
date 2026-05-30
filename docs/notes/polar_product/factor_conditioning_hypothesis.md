# Factor-conditioning hypothesis for OLMo vs Llama r256

This note summarizes the current non-temporal hypothesis for why OLMo r256
prefers fewer Newton-Schulz iterations or clipped polar-like updates, while
Llama-3.2 r256 appears to tolerate a more complete polar map.

## Setup

For one LoRA pair,

$$
A \in \mathbb{R}^{r \times d_{\mathrm{in}}}, \qquad
B \in \mathbb{R}^{d_{\mathrm{out}} \times r}, \qquad
\Delta W = B A .
$$

The small-side factor metrics are

$$
S_A := A A^\top, \qquad S_B := B^\top B .
$$

For the A-side update, the opposite factor is $B$: an update $\Delta A$
contributes to the product as $B\Delta A$. The chord-tight polar-product
update therefore whitens the A-side Adam direction by $S_B^{-1/2}$:

$$
X_A = S_B^{-1/2} u_A .
$$

Symmetrically, the B-side direction is whitened by $S_A^{-1/2}$:

$$
X_B = u_B S_A^{-1/2}.
$$

Thus $S_B$ controls which A-side directions are product-effective, and $S_A$
controls which B-side directions are product-effective.

## Hypothesis

At r256, OLMo develops a weaker opposite-factor spectrum than Llama, especially
on the A side through $S_B$. In this regime, a full polar map can allocate too
much update norm to directions where $B$ has little support. This is harmful
because the chord-tight update has a fixed product-step budget: norm spent in
directions with small $\lVert B\Delta A\rVert$ cannot also be spent in directions
that move the product effectively.

This does not require the tail directions to rotate rapidly over time. The
static condition is enough:

$$
\lambda_i(S_B) \ll \lambda_{\max}(S_B)
\quad\Longrightarrow\quad
S_B^{-1/2} \text{ strongly lifts the corresponding A-side coordinate.}
$$

If the lifted coordinate is not useful after being mapped back through $B$, then
full polar is a bad use of the step budget. Fewer NS iterations, SSC, or explicit
damping can all reduce this tail lift, but damping is the cleaner intervention
because it targets the factor metric directly.

## Evidence From Existing Logs

The strongest existing comparison is early r256 OPC, step 100, lr $=3\cdot
10^{-3}$, k=1, absolute damping, same optimizer family and packed data
pipeline. The relevant A-side metric differs sharply:

| run | $\mathrm{stable\_rank}(S_B)/r$ | $\mathrm{cond}(S_B)$ | $\lVert B\Delta A\rVert_F / \lVert \Delta A\rVert_F$ |
|---|---:|---:|---:|
| OLMo, ns=5 | 0.104 | 120.5 | 0.032 |
| Llama-3.2, ns=8 | 0.312 | 19.3 | 0.066 |

In the same comparison, $S_A$ is similar across the two models:
$\mathrm{stable\_rank}(S_A)/r \approx 0.55$ and
$\mathrm{cond}(S_A) \approx 4.3$ for both. That makes the early discrepancy
mostly an A-side-through-$B$ issue, not a symmetric factor issue.

The interpretation is simple: OLMo's $B$ factor spans fewer effective directions
at r256, so many A-side coordinates have weak product leverage. Llama's $B$
factor is broader and better conditioned, so lifting more directions with a more
complete polar map is less obviously wasteful.

## Snapshot Tail Check

Stored OLMo snapshots show the specific amplification mechanism. In bad high-lr
r256 snapshots, only a small fraction of raw $u_A$ energy lies in the bottom
quarter of $S_B$ eigen-directions, but whitening moves a much larger fraction of
energy there. Values below are medians over 24 sampled LoRA pairs:

| snapshot | $\mathrm{stable\_rank}(S_B)/r$ | $\mathrm{cond}(S_B)$ | raw bottom-25% $u_A$ energy | whitened bottom-25% energy |
|---|---:|---:|---:|---:|
| r256, lr $=3\cdot 10^{-2}$, ns=5, step 4000 | 0.211 | 138.7 | 0.086 | 0.401 |
| r256, lr $=10^{-1}$, ns=10, chord-tight-clean, step 4000 | 0.122 | 896.2 | 0.052 | 0.435 |

This does not prove that the amplified tail is useless. It does show that the
whitening operator can substantially move update energy into low-$S_B$
directions in exactly the regime where OLMo is brittle.

## What To Test

The direct fix is relative damping of the factor metrics:

$$
\widetilde S_B = S_B + \epsilon\,\lambda_{\max}(S_B) I,
\qquad
\widetilde S_A = S_A + \epsilon\,\lambda_{\max}(S_A) I .
$$

With $\epsilon=10^{-2}$, this caps the damped Gram condition number at about
$101$ before taking the inverse square root. This is better targeted than
choosing fewer NS iterations as an implicit clip, because it directly limits the
weak-factor amplification that the hypothesis identifies.

Predictions:

- Product-effectiveness should improve, especially
  $\lVert B\Delta A\rVert_F / \lVert\Delta A\rVert_F$ on OLMo r256.
- The gap between ns=5 and ns=8 should shrink if OLMo's preference for fewer NS
  mostly came from undamped factor-tail lift.
- If ns=8 remains worse after relative damping, then the factor-conditioning
  story is incomplete; the remaining issue is not just $S_A,S_B$ tail
  amplification.

## Limits

This is a conditioning hypothesis, not a claim that the tail modes are noisy,
temporally unstable, or intrinsically bad. The diagnostics above measure
factor support, whitening amplification, and product-through ratios. They do not
measure whether a low-$S_B$ direction carries useful task signal. Training loss
under the relative-damping sweep is the necessary outcome test.

## Reproducing The Numbers

The log values come from `lora_playground.loader.load_runs(...)` using
`_optim_steps` fields emitted by the optimizer diagnostics. The relevant filter
is:

```python
from lora_playground.loader import load_runs

runs = load_runs(where={
    "optimizer": "adam-polar-product-lora-coupled-spectral-chord-tight",
    "lora_r": 256,
    "data_pipeline_version": "packed_v1.1",
})
```

Read the step-100 `_optim_steps` records for the k=1 OPC rows at lr
`3e-3`, excluding relative-damping runs. The fields used are:

- `stable_rank_A_median`, `stable_rank_B_median`
- `cond_SA_median`, `cond_SB_median`
- `frac_dA_through_B_median`, `frac_dB_through_A_median`

The snapshot tail values come from:

```bash
python scripts/analysis/factor_tail_snapshot_probe.py \
  --steps 4000 \
  --max-pairs 24 \
  --out /tmp/factor_tail_rows.csv \
  --summary-out /tmp/factor_tail_summary.csv
```
