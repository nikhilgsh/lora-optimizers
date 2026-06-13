# LoRA factor-shape scaling for Muon-style updates

## Decision

The optimizer recommendation is to keep the equal-radius split at the measured
ranks:

$$
c_A = c_B = 1.
$$

This decision uses the MuA-style output-feature criterion as primary. The loss
sees the adapter output increment

$$
\Delta Z_B
= \delta^1 + \delta^2 + \delta^3,
\qquad
\delta^1 = B\dot A x,\quad
\delta^2 = \dot B A x,\quad
\delta^3 = \dot B\dot A x.
$$

For first-order factor scaling, the load-bearing diagnostic is therefore the
direct branch ratio $\|\delta^2\|/\|\delta^1\|$, not an isolated factor-RMS
argument for A or B. The available direct output probe says:

| rank | median $\|\delta^2\|/\|\delta^1\|$ under equal radii | implication |
|---:|---:|---|
| 64 | 0.535 | A's branch is larger; do not shrink A or boost B. |
| 256 | 1.048 | branches are already balanced. |

The A-side activation probe remains useful: it shows that the A-update rowspace
is not isotropic with respect to the inputs entering `lora_A`, and its fitted
A-only correction is close to $(r/d_{in})^{1/4}$. But that is not sufficient to
make $(r/d_{in})^{1/4}$ an optimizer default, because the relevant branch
$B\dot A x$ is already $\Theta(1)$ and is not too small in the measured ranks.

The full standalone factor-shape rule

$$
c_A = (r/d_{in})^{1/4}, \qquad c_B = \sqrt{d_{out}/r}
$$

is not recommended as a default. On square 2048-wide modules it gives
$c_A/c_B = (r/d)^{3/4}$: about $0.074$ at $r=64$ and $0.210$ at $r=256$. Applied
to the measured output ratios, this would make $\delta^2/\delta^1$ about $7.2$
at $r=64$ and $5.0$ at $r=256$, pushing the update toward B-branch dominance.

The only honest opening for a static $c_A/c_B \ne 1$ rule is a large-rank trend
where $\|\delta^1\|/\|\delta^2\|$ continues to fall and A's branch starts to
vanish. That would call for an A-side boost, not the B-heavy shape rule above.

## Setup

This repository uses the PEFT LoRA convention:

$$
A \in \mathbb{R}^{r \times d_{in}}, \qquad
B \in \mathbb{R}^{d_{out} \times r}, \qquad
\Delta W \propto BA.
$$

The current chord-tight update gives both factors the same operator-norm radius:

$$
\|\dot A\|_{op} = \|\dot B\|_{op} = \rho,
\qquad
\rho = \frac{\eta}{\|A\|_{op} + \|B\|_{op}}.
$$

The question is whether those two factor radii should instead receive
shape-dependent coefficients:

$$
\|\dot A\|_{op} = \lambda c_A, \qquad
\|\dot B\|_{op} = \lambda c_B.
$$

If preserving the same first-order merged-update cap, the natural normalization is:

$$
\lambda =
\frac{\eta}{c_A\|B\|_{op} + c_B\|A\|_{op}}.
$$

This keeps the bound on $\|B\dot A\|_{op} + \|\dot B A\|_{op}$ at $\eta$.

## What the Keller/MuP test must measure for A

For A, the factor map is a compression:

$$
A: \mathbb{R}^{d_{in}} \to \mathbb{R}^r, \qquad r < d_{in}.
$$

The Keller and MuP prescriptions differ only in this compression case:

| A-side rule | Corrective coefficient |
|---|---:|
| Keller isotropic assumption | $c_A = 1$ |
| MuP worst-case assumption | $c_A = \sqrt{r/d_{in}}$ |

The discriminating quantity is not $\|A\|_{op}$, $\|B\|_{op}$, or the current
optimizer state by itself. The blog-post assumption is about the actual input to
the map being updated. For A, that means the activations entering `lora_A`.

For an A update with right singular basis $V \in \mathbb{R}^{d_{in}\times r}$,
the measured quantity is:

$$
R_A =
\frac{\operatorname{RMS}(xV)}
{\operatorname{RMS}(x)}
\sqrt{\frac{d_{in}}{r}},
$$

implemented equivalently as:

$$
R_A =
\sqrt{
\frac{d_{in}}{r}
\frac{\|xV\|_F^2}{\|x\|_F^2}
}.
$$

Interpretation:

- $R_A \approx 1$ supports the Keller isotropic assumption.
- $R_A \approx \sqrt{d_{in}/r}$ supports the MuP worst-case endpoint.
- The best A-side correction for that rowspace is $c_A \approx 1/R_A$.

## Measurement

The probe in `scripts/analysis/muon_activation_isotropy_probe.py` does three
things:

1. Loads a saved optimizer snapshot.
2. Replays the current chord-tight A-update rowspace from the saved optimizer
   state.
3. Restores the checkpointed LoRA factors into the model, captures real inputs
   to `lora_A` on eval batches, and measures $R_A$.

The timeline probe was run on:

- Snapshot root:
  `/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell/task_0`
- Steps: 0, 50, 200, 500, 1000, 2000, 4000
- Eval capture: 4 batches, batch size 1
- Model/data: inherited from each checkpoint's `meta.json`

Raw outputs are in `notebooks/snapshot_analysis/_data/`:

- `muon_activation_isotropy_rows_step*.csv`
- `muon_activation_isotropy_summary_step*.csv`
- `muon_activation_isotropy_timeline_rows.csv`
- `muon_activation_isotropy_timeline_summary.csv`

The rendered notebook is
`notebooks/snapshot_analysis/07_muon_a_activation_isotropy.ipynb`.

## Results

Median measured A-side RMS ratio by module:

| step | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.791 | 0.791 | 0.791 | 0.778 | 0.797 | 0.797 | 0.920 |
| 50 | 2.071 | 2.152 | 2.057 | 2.668 | 2.263 | 2.261 | 3.423 |
| 200 | 2.127 | 2.219 | 2.120 | 2.668 | 2.262 | 2.239 | 3.427 |
| 500 | 2.153 | 2.217 | 2.136 | 2.729 | 2.245 | 2.247 | 3.416 |
| 1000 | 2.137 | 2.234 | 2.155 | 2.648 | 2.209 | 2.232 | 3.403 |
| 2000 | 2.103 | 2.178 | 2.161 | 2.665 | 2.206 | 2.215 | 3.391 |
| 4000 | 2.125 | 2.168 | 2.141 | 2.641 | 2.205 | 2.206 | 3.303 |

For the 2048-wide modules at r=64:

- Keller predicts $R_A = 1$.
- MuP worst-case predicts $R_A = \sqrt{2048/64} = 5.66$.
- The measured post-step-50 range is about 2.1 to 2.7.

For `down_proj`, where $d_{in}=8192$:

- Keller predicts $R_A = 1$.
- MuP worst-case predicts $R_A = \sqrt{8192/64} = 11.31$.
- The measured post-step-50 value is about 3.3 to 3.4.

The fitted exponent $p$ in $c_A = (r/d_{in})^p$ is stable after step 50:

| step | median fitted p |
|---:|---:|
| 0 | -0.064 |
| 50 | 0.232 |
| 200 | 0.234 |
| 500 | 0.236 |
| 1000 | 0.237 |
| 2000 | 0.232 |
| 4000 | 0.232 |

Thus $p=1/4$ is a simple fixed value consistent with the measured timeline.

## Interpretation

Keller is reasonable only at initialization in this run. After a small number
of optimizer steps, the A-update subspace is not isotropic with respect to the
captured `lora_A` inputs.

Full MuP is also not right for this snapshot family. It assumes the worst-case
compression endpoint, but the measured A-update rowspace captures only about
30% to 47% of the worst-case RMS inflation, depending on module kind.

The direct measurement supports an intermediate rule:

$$
c_A = (r/d_{in})^{1/4}.
$$

This is not a theorem. It is a measured single-run mechanism on the available
r=64 chord-tight snapshots. It is strong enough to motivate an ablation, not
strong enough to be a final architecture-independent rule.

## Why this disagrees with the equal-radius conclusion

The equal-radius argument is useful, but it answers a different question. It
starts from the bilinear LoRA product increment:

$$
\Delta W \propto B\dot A + \dot B A + \dot B\dot A,
$$

and observes that equal operator-norm radii give a clean product-cap bound:

$$
\|B\dot A\|_{op} + \|\dot B A\|_{op}
\le
\rho\left(\|B\|_{op} + \|A\|_{op}\right)
= \eta.
$$

That is a valid worst-case bound on the merged adapter update. It is not a test
of the Keller isotropy assumption for A.

The Keller/MuP distinction for A is about the **activation distribution entering
A**, projected onto the current A-update rowspace. A product-cap bound can be
true while the A input is still far from isotropic in the relevant subspace.
That is exactly what the activation probe found.

The disagreement is therefore:

- The equal-radius note treats $c_A=1$ as acceptable because the product-cap
  bound remains balanced at the operator-norm level.
- The activation probe measures the A-side feature RMS directly and finds
  $R_A \approx 2.1$ to $3.4$ after step 50.
- Therefore $c_A=1$ is too large for the A-side feature-RMS criterion during
  training, even though it still preserves a valid product-cap bound.

This also explains why the direct measurement supersedes the proxy. The proxy
uses $\|A\|_{op}$, $\|B\|_{op}$, and submultiplicative upper bounds. The probe
uses the actual captured `lora_A` inputs and the actual current A-update
rowspace. For the Keller-vs-MuP question, that is the load-bearing quantity.

## Why the quarter-power recommendation is better

The quarter-power recommendation is better calibrated to the measured A-side
effect.

For 2048-wide modules at r=64:

| rule | A correction | implied A RMS ratio |
|---|---:|---:|
| Keller | 1.000 | 1.00 |
| measured quarter-power | 0.420 | 2.38 |
| MuP | 0.177 | 5.66 |
| observed post-step-50 range | 0.37-0.49 | 2.06-2.73 |

For `down_proj` at r=64, where $d_{in}=8192$:

| rule | A correction | implied A RMS ratio |
|---|---:|---:|
| Keller | 1.000 | 1.00 |
| measured quarter-power | 0.297 | 3.36 |
| MuP | 0.088 | 11.31 |
| observed post-step-50 range | 0.29-0.30 | 3.30-3.43 |

Thus the proposed value is not an arbitrary extra hyperparameter. It is the
single fixed exponent that matches both the 2048-wide modules and the
8192-wide down projection:

$$
c_A = (r/d_{in})^{1/4}.
$$

It is also a safer ablation than jumping directly to a full Keller or MuP split:

- It changes only the A side first, which is the side measured by the probe.
- It keeps the merged product-cap normalization, so the learning-rate meaning is
  still comparable to the current optimizer.
- It avoids Keller's over-update of A under the measured activation RMS.
- It avoids MuP's 2x to 3.4x under-update relative to the measured rowspace.

In short, equal radius is a clean product-norm design, but it is not the best
answer to the activation-space scaling question. The quarter-power A correction
is the smallest change that follows from the direct measurement.

## Relation to νGPT mid alignment

Shigida, Hanin, and Gromov's νGPT paper, *Learning Rate Transfer in
Normalized Transformers* (arXiv:2604.27077), uses an alignment-exponent
argument that is structurally close to this measurement.

Their width-transfer rule interpolates between:

- no alignment, with exponent $1/2$;
- full alignment, with exponent $1$;
- mid alignment, with exponent $3/4$.

That $3/4$ exponent is not the same number as the $1/4$ exponent here because
it is attached to a different quantity: their exponent is a width learning-rate
exponent, while ours is the A-side corrective multiplier exponent. But the
logic is the same. Both choose the midpoint between an isotropic endpoint and a
fully aligned endpoint.

For A-side compression:

- Keller is the no-alignment endpoint: $c_A=(r/d_{in})^0$.
- MuP is the full worst-case endpoint: $c_A=(r/d_{in})^{1/2}$.
- The measured midpoint is $c_A=(r/d_{in})^{1/4}$.

Equivalently, the measured RMS inflation is $(d_{in}/r)^{1/4}$ and the measured
energy concentration is $(d_{in}/r)^{1/2}$, the geometric mean between isotropic
energy and worst-case energy.

This is a useful theoretical analogy, not a proof of universality. The νGPT
paper also treats the mid exponent as empirically supported rather than forced
by first principles: measured alignment exponents vary through training, and
the chosen midpoint is justified by transfer behavior and loss-weighted
measurements.

## Relation to LoRA+, MuA, and BaLoRA

### LoRA+

LoRA+ argues that equal learning rates for A and B are inefficient in the
large-width limit and motivates a larger B-side learning rate. That result is
about raw factor learning rates and the bilinear LoRA feature increment.

It does not decide the Keller-vs-MuP A compression question. The A compression
question requires measuring the activations entering A against the current
A-update rowspace, which is what the probe does.

### MuA

MuA analyzes how the optimal LoRA learning rate scales with width, rank,
initialization, and adapter multiplier. Under this repository's default PEFT
initialization, A is random and B is zero. With effective adapter multiplier
one, MuA predicts rank-dependent learning-rate behavior for standard LoRA.

That rank-scaling question is separate from the A-side rowspace isotropy
question. The activation probe says that, conditional on the current optimizer's
A-update rowspace, the A input is neither isotropic nor worst-case.

### BaLoRA

BaLoRA addresses the product-preserving factor gauge. It projects factors onto
a balanced manifold so that the same product has better conditioning.

That is complementary to A-side shape scaling. A BaLoRA-style projection can be
tested later, but it should not be used to replace the A-side activation
measurement. Balance changes the factor gauge; the activation probe measures
the physical effect of the proposed update rowspace.

## Recommendation for implementation

Add a factor-shape option with the first ablation arm:

$$
c_A = (r/d_{in})^{1/4}, \qquad c_B = 1.
$$

Apply it through the product-cap-preserving normalization:

$$
\lambda =
\frac{\eta}{c_A\|B\|_{op} + c_B\|A\|_{op}},
\qquad
\|\dot A\|_{op} = \lambda c_A,
\qquad
\|\dot B\|_{op} = \lambda c_B.
$$

This arm answers the direct question from the measurement: does correcting the
A compression by the measured activation factor improve the optimizer?

Only after that should we test the full factor-RMS variant with:

$$
c_A = (r/d_{in})^{1/4}, \qquad
c_B = \sqrt{d_{out}/r}.
$$

The full variant changes both sides at once. It may be valid, but it is not what
the A probe alone establishes.

## Limits

- The measurement is from one snapshot family, one optimizer family, one rank,
  and one model/data regime.
- The capture uses four eval batches. The result is stable over saved training
  steps, but the sample is still small.
- The probe tests A-side compression only. It does not test whether B-side
  expansion by $\sqrt{d_{out}/r}$ is beneficial inside the LoRA product update.
- Step 0 is special: the update rowspace has not yet been shaped by training.
  The relevant training-time behavior appears by step 50.

## Reproducibility

Run one checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/analysis/muon_activation_isotropy_probe.py \
  --ckpt-dir /mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell/task_0/step_4000 \
  --num-batches 4 \
  --batch-size 1 \
  --device cuda
```

The saved-step timeline was generated by running the same command over:

```text
step_0
step_50
step_200
step_500
step_1000
step_2000
step_4000
```

The notebook `notebooks/snapshot_analysis/07_muon_a_activation_isotropy.ipynb`
loads the generated CSVs and renders the decision plots.
