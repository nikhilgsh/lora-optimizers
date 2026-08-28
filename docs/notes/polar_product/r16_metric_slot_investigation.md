# r=16 learned metric-slot investigation

Last updated: 2026-08-27T16:03:51-04:00

Status: active. The late same-state gradient and finite-step probes are
complete. No new training should be launched before a local scale profile from
the same checkpoint separates the linear direction from finite-step curvature.

## Question

At Llama-3.2-1B, OpenMathInstruct-2, and LoRA rank $r=16$, why does the
factorwise branch underperform the one-sided branch? Both branches use the same
diagonal matrices $P,Q$, matrix-sign direction, spectral-norm magnitude rule,
training data, and horizon. They differ only in the two $r\times r$ slots:

| branch | $C_B$ | $C_A$ |
|---|---|---|
| factorwise | learned $P_A$ | learned $Q_B$ |
| one-sided | $I_r$ | $I_r$ |

The primary outcome is the final-loss gap between factorwise and one-sided,
with each branch evaluated at its own best learning rate on the same grid and
at 9000 steps. A positive gap means the learned slots are worse.

## Current result

The corrected post-fix implementation still loses to identity at $r=16$.
Increasing the curvature EMA decay $\beta_2$ from $0.99$ to $0.999$ slightly
reduces the deficit but leaves most of it intact:

| $\beta_2$ | factorwise best $\eta$ | factorwise loss | one-sided best $\eta$ | one-sided loss | gap |
|---:|---:|---:|---:|---:|---:|
| 0.99 | 0.03 | 0.418685 | 0.01 | 0.415820 | 0.002866 |
| 0.999 | 0.03 | 0.418115 | 0.01 | 0.415791 | 0.002325 |

The gap shrinks by 0.000541, or 18.9%. Factorwise improves by 0.000570,
whereas one-sided changes by only 0.000029. Thus the longer EMA window has a
small differential benefit for the learned slots, but 81.1% of the original
deficit remains. Curvature-estimation noise or lag is a secondary effect, not
the main explanation of the $r=16$ result.

The best learning rates do not move when $\beta_2$ changes, so this conclusion
is not caused by selecting different points on the learning-rate grid. Within
each $\beta_2$ comparison, the factorwise and one-sided endpoints also come
from the same execution source:

- $\beta_2=0.99$ uses source `801a80d2`: factorwise is `log_1.out` and
  one-sided is `log_3.out` under `e2_precond_r16_postfix_xl`.
- $\beta_2=0.999$ uses source `8487a947`: factorwise is `log_3.out` and
  one-sided is `log_7.out` under `e2_beta2_r16_full_xl`.

The matched-step trajectories show that the deficit develops late. At both
$\beta_2$ values, factorwise is slightly better through step 750. The gap is
near zero around steps 1000–1500, then becomes consistently positive. At step
5000 the gap is 0.002780 for $\beta_2=0.99$ and 0.002062 for
$\beta_2=0.999$. At step 7500 the corresponding gaps are 0.003063 and
0.002458. A larger $\beta_2$ delays and modestly reduces the separation but
does not change its qualitative trajectory.

## What Figure 30 establishes

Figure 30 tests whether increasing the EMA window helps factorwise more than
one-sided. It does, but the effect is too small to explain the learned-slot
deficit. The injected-isotropic diagnostic remains evidence that a finite EMA
can manufacture anisotropy. The weak validation-loss response to $\beta_2$
shows that this diagnostic is not sufficient evidence for the training
mechanism.

The experiment does not establish whether the remaining deficit comes from
the direction selected by $P_A,Q_B$, the interaction of that direction with
the final spectral rescale, or a late mismatch between the learned slots and
the current gradients.

## Where the optimizer states diverge

The existing post-fix diagnostics show that the two trajectories separate in
factor geometry long before their held-out losses separate. Stable rank is
$\|X\|_F^2/\|X\|_2^2$. The table reports medians across the 112 LoRA pairs;
the tangent norm is the fully applied first-order product update
$B\,dA+dB\,A$ at each branch's own best learning rate.

| step | factorwise sr$(A)$ / sr$(B)$ | one-sided sr$(A)$ / sr$(B)$ | factorwise / one-sided balance residual | factorwise / one-sided tangent norm |
|---:|---:|---:|---:|---:|
| 100 | 9.43 / 7.56 | 13.46 / 12.61 | 0.670 / 0.746 | 0.0351 / 0.0259 |
| 1000 | 6.24 / 7.32 | 13.00 / 13.01 | 0.861 / 0.503 | 0.0275 / 0.0242 |
| 7000 | 5.22 / 7.52 | 13.19 / 12.99 | 0.931 / 0.414 | 0.0251 / 0.0247 |
| 9000 | 5.21 / 7.49 | 13.18 / 12.93 | 0.932 / 0.406 | 0.0245 / 0.0246 |

The factorwise factors are already more spectrally concentrated at the first
logged diagnostic, step 100, while factorwise remains slightly better in
held-out loss through step 750. Therefore low factor stable rank is not by
itself harmful. The candidate mechanism is stage-dependent: concentration may
fit dominant modes efficiently early but become a bottleneck when the residual
gradient moves toward weak factor modes. The balance residual first becomes
worse for factorwise at step 200 and then increases monotonically in the rows
above. This is correlated path evidence, not yet a causal result.

## Late same-state measurement

The late probe resumes the $\beta_2=0.999$, $\eta=0.03$ factorwise checkpoint
at step 7000 and varies only the two $16\times16$ slots. `learned` consumes the
stored $P_A,Q_B$; `identity` substitutes $I_{16}$ while retaining the
factorwise-trained $P,Q$ history. Both directions include polar normalization,
the final per-factor spectral rescale, and the adaptive radius.

Define the weak modes as the bottom eight singular modes of the current factor.
The held-out gradient below is the token-weighted gradient of the same eight
batches used for exact loss. At step 7001 the median energy and gradient demand
are:

| quantity in weak modes | $A$ side | $B$ side |
|---|---:|---:|
| factor energy | 0.268 | 0.263 |
| training-gradient energy | 0.737 | 0.783 |
| held-out-gradient energy | 0.728 | 0.783 |
| learned-update energy | 0.233 | 0.269 |
| identity-update energy | 0.499 | 0.500 |

Thus the late gradient predominantly requests the weak factor modes, while the
learned slots allocate update energy approximately in proportion to the factor
spectrum rather than to that residual demand. Identity restores equal modal
allocation.

The local direction measurement is unambiguous. Summed over all 112 pairs, the
exact first-order change on the eight-batch objective is
$-4.92\times10^{-5}$ for learned slots and $-1.89\times10^{-4}$ for identity.
Identity therefore has a $3.83\times$ more favorable local derivative. Its
weak-mode contribution is $-1.25\times10^{-4}$, compared with only
$-2.10\times10^{-5}$ under learned slots.

The finite-step measurement shows why this local advantage is not yet an
algorithmic fix:

| late same-state quantity | learned | identity | identity at $0.01/0.03$ scale |
|---|---:|---:|---:|
| tangent norm relative to learned | 1.000 | 1.997 | 0.666 |
| tangent stable rank | 14.47 | 9.96 | 9.96 |
| first-order held-out loss change | $-4.92\times10^{-5}$ | $-1.89\times10^{-4}$ | $-6.29\times10^{-5}$ |
| eight-batch exact held-out loss change | $-6.98\times10^{-5}$ | $+7.10\times10^{-4}$ | $+1.64\times10^{-4}$ |

The tangent cosine between learned and identity is 0.918. Identity obtains a
much larger first-order decrease from weak modes, but the full identity step
worsens exact loss. Scaling identity by the observed best-learning-rate ratio
$0.01/0.03$ makes its tangent smaller than learned and still does not rescue
the exact loss. The nonlinear remainder at that scale is
$+2.27\times10^{-4}$: the exact change $+1.64\times10^{-4}$ minus the linear
prediction $-6.29\times10^{-5}$. Thus magnitude mismatch is real but not the
whole explanation. The factorwise-trained state has substantial finite-step
curvature along the identity counterfactual.

The per-factor spectral rescale does not equalize the product tangent because
its orientation relative to $A,B$ and cancellation between $B\,dA$ and
$dB\,A$ remain different. More importantly, matching or reducing that tangent
norm does not by itself match the finite-step loss response.

Metric staleness is negligible geometrically at this checkpoint. Incorporating
the current gradient changes the median normalized $P_A,Q_B$ slots by only
0.000484 and 0.000226; the fresh-versus-stored tangent cosine is greater than
0.999999 and its norm ratio is 0.999992. This agrees with Figure 30: increasing
$\beta_2$ is not the main remedy. The full-step exact loss is too curved to use
as an additional staleness metric: nearly identical stored and fresh directions
can have different finite-step loss.

The exact measurement is deterministic for a fixed state: repeating the
pre-step loss after all candidate evaluations gives an absolute difference of
zero. Its per-batch signs are nevertheless mixed. Learned improves five of
eight batches; full identity improves two; one-third-scale identity improves
three. Therefore the eight-batch exact changes characterize this checkpoint
probe, not a held-out-loss recommendation by themselves. The gradient-energy
and first-order measurements use the same eight-batch target and are the
load-bearing evidence about direction.

## Interpretation and next measurement

Small factor stable rank is not the causal statement. It appears by step 100,
while factorwise remains better through step 750. The more precise late-stage
statement is that learned slots allocate only 23--27% of factor-update energy
to weak modes while 73--78% of the held-out gradient energy lies there.
Identity removes that local directional mismatch, but the resulting finite
step interacts badly with the factorwise-trained $A,B,P,Q$ state even after the
observed learning-rate ratio is applied.

The next cheap measurement is a same-direction loss profile at several fixed
fractions near zero, evaluated in one pass on the same batches. Its purpose is
to locate the end of the linear regime and estimate curvature; it is not a new
optimizer hyperparameter or a training sweep. Only after that profile should
we decide whether the algorithm needs a magnitude rule shared across slot
choices or whether the late failure is primarily trajectory-dependent.

## Decisions

- Treat the pre-fix factorwise runs as invalid and exclude them. Commit
  `7792797` changed the factorwise initialization, so pre- and post-fix points
  are different implementations.
- Stop treating larger $\beta_2$ as the likely solution. It closes only 18.9%
  of the observed gap.
- Do not add damping, refresh cadence, or another learned-metric knob before
  the scaled-identity measurement separates direction from magnitude.
- Keep the factorwise method initialized at the identity-compatible positive
  definite state. The current result is about the subsequent learned metric,
  not the invalid zero initialization.

## Evidence and regeneration

- Notebook figures: `paper/paper_plots.ipynb`, Figure 17 and Figure 30.
- Arm definitions: `lora_playground/plotting/arms.py:525-551`.
- Post-fix source filter and Figure 30 loader:
  `lora_playground/plotting/paper_plots_lib.py:730-755,807-852`.
- Completed post-fix task map:
  `logs/e2_precond_r16_postfix_xl/run_info/tasks`.
- Completed $\beta_2=0.999$ task map:
  `logs/e2_beta2_r16_full_xl/run_info/tasks`.
- Late checkpoint: `scripts/results/cw_shadow_r16_b999_factorwise_lr003_step7000/ckpt_step7000`.
- Eight-batch late probe:
  `scripts/results/cw_shadow_r16_b999_step7001_8batch_aggregate_grad_repeat.log`.
- Probe wall time:
  `scripts/results/cw_shadow_r16_b999_step7001_8batch_aggregate_grad_repeat.time`
  (58.64 seconds; diagnostic exit before full eval).

Regenerate the endpoint table from the canonical plotting loader:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate ffcv-pl
MPLBACKEND=Agg python -c 'from lora_playground.plotting import paper_plots_lib as P; P.clear_runs_cache(); print(P.precond_beta2_panel(16).to_string(index=False))'
```
