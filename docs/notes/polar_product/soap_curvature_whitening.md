# SOAP, curvature whitening, and the chord-tight sandwich

> **Status (2026-05-31 handoff): implemented (`CurvatureWhitenLoRA`, commit
> `eaa86f9`), Codex-audited, unit-tested, but the r256 opc run DIVERGES early and
> the root cause is UNRESOLVED.** Read "Implementation status & handoff" below
> before continuing. The lower sections are the original design discussion.

## Implementation status & handoff (2026-05-31)

### What is implemented — `CurvatureWhitenLoRA` (`lora_playground/optim.py`)
Registered as `curvature-whiten-lora` (no polar) and `curvature-whiten-polar-lora`
(`use_polar=True`). Per LoRA factor (A: r×d_in, B: d_out×r):
- **Direction (harmonious Kronecker):** $\Delta A \propto S_{rA}^{-1/2}\,\hat m_A\,D_{\mathrm{in}}^{-1/2}$,
  $\Delta B \propto D_{\mathrm{out}}^{-1/2}\,\hat m_B\,S_{rB}^{-1/2}$. Small side = full
  $r\times r$ whitening $S_r^{-1/2}=Q\Lambda^{-1/2}Q^\top$ with $Q$ from **warm-started
  QR power-iteration** (eigh only to seed the first refresh — per
  [[feedback_no_eigh_in_production]], NO per-step eigh) and $\lambda$ = Rayleigh
  diagonal $\mathrm{diag}(Q^\top L Q)$. Large side = explicit diagonal
  $D=\mathrm{EMA}(\mathrm{diag}(g^\top g))$. Relative damping $\delta$ (default `1e-3`,
  `--precond_delta`). Input is bias-corrected momentum $\hat m$.
- **Magnitude (chord-tight-clean convention, matches the curvature baselines):**
  $\rho = lr/(\sigma_{\max}(A)+\sigma_{\max}(B))$, each update rescaled to
  $\sigma_{\max}(\Delta A)=\rho$ via warm-started power iteration. So `lr` is the
  same spectral-budget meaning as the baselines.

### Confirmed correct
- **Codex-audited** (two passes): orientation, ordering (no self-preconditioning —
  EMAs/Q updated AFTER the step), zero-init EMA, eigenbasis refresh, `use_polar`.
- **9 unit tests pass** (`tests/test_curvature_whiten_lora.py`).
- **Magnitude is NOT the bug.** Synthetic diagnostic (`/tmp/mag_diag.py`, r64 d512,
  random grads, B nonzero): $\sigma_{\max}(\Delta A)\approx\rho$ and
  $\sigma_{\max}(\Delta W)\approx 1.5\!\times\!10^{-3} \le lr=3\!\times\!10^{-3}$ —
  the chord bound holds with margin. Do NOT re-chase a magnitude bug.

### BROKEN — the open problem
Real r256 × OLMo-2-1B × opc smokes (50–60 step, train_loss, `δ=1e-3`):
- `curvature-whiten-lora` lr=1e-3: flat/noisy at base ~1.0 (no learning).
- lr=3e-3: train_loss **spikes to ~9** by step 10, then recovers to ~2 by step 50.
- lr=1e-2: spikes to ~20.
- `curvature-whiten-polar-lora` (use_polar): WORSE — spikes to ~22 and stays.

### Ruled out
- **δ / weak-direction amplification.** δ=1e-2 spikes **identically** to δ=1e-3
  (8.4/10.5/8.8/7.6) → δ-insensitive → the spike is NOT the whitening's tail
  amplification.
- **Magnitude / σ_max overshoot.** Verified bounded in synthetic (above).
- **Polar (orthogonalization).** Makes it worse, not better.

### Leading hypotheses (UNVERIFIED — for the next agent)
1. **Wrong-direction (anti-descent) whitened step.** Magnitude is bounded
   ($\sigma_{\max}(\Delta W)\le lr$) but the whitened DIRECTION points uphill in the
   real regime → loss rises → gradient grows → runaway (consistent with loss
   1→14 in 5 steps despite ≤3e-3 spectral steps). Prime suspect: the large-side
   $D_{\mathrm{in}}^{-1/2}=\mathrm{EMA}(\mathrm{diag}(g^\top g))^{-1/2}$ UP-weights
   low-gradient-energy columns (anti-signal), like a too-aggressive natural-grad.
2. **B=0 LoRA-init dynamics** + real (structured) gradients — the synthetic used
   B≠0 and random grads and did NOT spike, so the failure is specific to one/both.
3. **Possible red herring:** the 50-step *train_loss* (per-batch, noisy) spike may
   be a recoverable startup transient (no-polar lr=3e-3 already recovers 9→2 by
   step 50; the chord-tight baselines use the same ρ and are healthy by step 200 —
   their early steps were never inspected). Confirm with a longer **eval**-based
   run before concluding it's broken.

### Concrete next diagnostics
1. Instrument the real first ~10 steps: log $\sigma_A,\sigma_B,\rho,\sigma_{\max}(\Delta W)$
   and **`cos(applied ΔW, −∇W)`** — is each step descent or ascent? (Distinguishes
   "wrong direction" from "noisy transient".)
2. **Ablate $D$** (set the large side to identity) — does the spike vanish? Isolates
   the column-energy whitening (hypothesis 1).
3. Ablate $S_r$ (identity small side) — isolates the other factor.
4. Run one lr=3e-3 to ~300 steps with eval (not train_loss) to settle hypothesis 3.
5. Compare against `curvature_whitening_ns8_k1_r256_olmo_opc` (the chord-tight
   curvature baseline) early-step behavior at the same lr.

### Reproduce / infra
- Smoke: `bash` activate `ffcv-pl`; `PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u train_lora.py --data_dir data/opc_sft_stage2_all_packed_seq2048 --data_pipeline_version packed_v1.1 --max_seq_length 2048 --bf16 --batch_size 4 --grad_accum_steps 4 --lr <lr> --optimizer curvature-whiten-lora --lora_r 256 --lora_alpha 256 --curvature_beta 0.99 --precond_refresh_every 10 --precond_delta <δ> --max_steps N --train_loss_every 5`.
- Sweep wrapper `scripts/sweep/sweep_curvature_whiten_r256_opc.sh` (positional: lr,
  optimizer, seed, precond_delta); pending sbatch (gpuxl h100) staged earlier but
  NOT updated for lr×δ and NOT submitted.
- **Local GPU (shared A6000) caveats** (see [[feedback_parallelize_and_gpu_planning]]):
  use `~/ml_utils/bin/gpu-reap.sh` to clear orphaned 40 GB holders before/after each
  smoke (reaper kills fine but has a cosmetic exit-1 bug to fix); NEVER `conda run`
  (buffers — use activate + `python -u`); screen on train_loss not the 158s eval.

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
