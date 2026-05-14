# Handoff: r=256 chord-tight k=3 lr=1e-2 ε_rel=1e-2 NaN

Unresolved NaN failure in the chord-tight (whiten) k=3 optimizer at LoRA rank
r=256, learning rate 1e-2, with relative damping ε_rel=1e-2. Three mechanism
hypotheses have been tried and rejected against the data; root cause is not
yet identified.

This doc lists measurements (with sources), rejected hypotheses (with reasons),
and what's needed for verification.

## 1. Setup

- Repo: `/mnt/home/nghosh/lora` (commit `5f10887`, with the
  `precond_delta_relative` bug fix from `ef6b3bc`)
- Optimizer: `adam-polar-product-lora-coupled-spectral-chord-tight` (`AdamPolarProductLoRA` with
  `magnitude_rule="spectral_chord_tight"`, `picard_iters=3`)
- LoRA rank: 256, alpha: 256
- Learning rate: 1e-2
- Damping: `--precond_delta_relative --precond_delta 1e-2`
- Hardware: Blackwell GPU (workergpu174)
- Data pipeline: `packed_v1` (4k-step horizon, eval_every=200)
- Model: `allenai/OLMo-2-0425-1B`
- Log file: `logs/chord_tight_k3_eps_rel_1e-2_r256_lr_sweep_4k_blackwell_fixed/run_info/logs/log_1.out`
  (task index 1 in the disBatch 4-cell r=256 sweep)
- SLURM job: 6402761
- Code path: `_step_batched` in `lora_playground/optim.py` (LoRA r=256
  meets `_batched_path_eligible`).

The bug fix in `ef6b3bc` propagated `eps_relative=self.precond_delta_relative`
to `spd_inv_sqrt_higham_batched` at all call sites. Before that commit,
`--precond_delta_relative` was silently dropped and the code applied absolute
δ instead. The run analyzed here is post-fix; relative damping is verified to
flow into the kernel (covered by `tests/test_eps_relative_propagation.py`).

## 2. Measurements

All measurements are from `log_1.out` of the run above. Step numbers are
optimizer step counts.

### 2.1 Trajectory before NaN

```
step train_loss
1650 0.5364
1660 0.5231
1670 0.5127
1680 0.4945
1690 0.5613
1700 NaN (terminal)
```

Loss bumped up at step 1690 (0.494 → 0.561) but stayed finite. Then went NaN
from step 1700 onward.

### 2.2 First non_finite events

The optimizer emits two diagnostic event types when entries go non-finite
(controlled in `_step_batched`):

- `non_finite_detected` — fires at TOP of step, per-pair, on (A, B, grad_A, grad_B).
- `non_finite_intermediate` — fires at END of step, with all chain-of-intermediate
  tensors (u_A, u_B, SA_half_inv, SB_half_inv, X_A, X_B, P_A, P_B, geo_A,
  geo_B, op_geoA_b, op_geoB_b, u_A_eff, u_B_eff, dA, dB).

Both default-on in the current run because the gating flag postdates the
submission of 6402761. So whichever fires earlier truly is first.

Sequence in the log:

```
step 1690 (end of optimizer.step):
  non_finite_intermediate event
  pair_index=9, pair_name="base_model.model.model.layers.1.self_attn.v_proj[default]"
  affected intermediates: SA_half_inv, u_B, u_B_eff, u_A_eff,
                          X_A, X_B, P_A, P_B, geo_A, geo_B,
                          op_geoA_b, op_geoB_b, dA, dB
  NOT affected: u_A, sigma_A, sigma_B, rho, picard_coeff_s

step 1691 (start of optimizer.step):
  non_finite_detected events for pair indices 0, 1, 2, ... (all pairs)
  where = {A: False, B: False, grad_A: True, grad_B: True}
```

No `non_finite_detected` event at step 1690's start. No `non_finite_detected`
event at any step from 1500 through 1689.

### 2.3 Optim diagnostics at the failure window

`log_basic_diagnostics` events fire every 20 steps with per-pair-aggregate
statistics. Values at steps 1620, 1640, 1660, 1680 (the four cleanly logged
optim_step events before NaN):

```
step  cond_SA_max  cond_SB_max  SA_max_min  SA_min_min  sigma_A_exact_min
1620      125.8       8.13e3      0.969       0.0209      0.984
1640      126.6       8.23e3      0.977       0.0208      0.988
1660      127.7       8.25e3      0.978       0.0207      0.989
1680      128.6       8.21e3      0.985       0.0207      0.992
1700      NaN         NaN         NaN         NaN         NaN
```

`SA_max_min` is the MIN across pairs of σ_max(SA_pair) (= the smallest
σ_max(SA) any pair has). `cond_SA_max` is the MAX across pairs of cond(SA_pair).

Pair-9-specific values are not logged at this cadence; only per-pair-aggregate
min/max/median.

`norm_B_max` = 7.12 at step 1680. `norm_dB_max` = 0.032. No signal of magnitude
runaway.

### 2.4 Comparison cells: same sweep, different lrs

In the same sweep (group `chord_tight_k3_eps_rel_1e-2_r256_lr_sweep_4k_blackwell_fixed`):

- r=256, lr=3e-3, ε_rel=1e-2: running clean at step 1600, eval_loss 0.539.
- r=256, lr=1e-2, ε_rel=1e-2: NaN at step 1690 (this doc).
- r=256, lr=3e-2, ε_rel=1e-2: running clean at step 1800, eval_loss 0.531.
- r=256, lr=1e-1, ε_rel=1e-2: running clean at step 1600, eval_loss 0.554.

Same optimizer, same damping rule, same r, same Blackwell hardware. Only lr
differs. The middle two lrs (1e-2 and 3e-2) sit on different sides of the
failure: lr=1e-2 NaNs, lr=3e-2 doesn't.

### 2.5 Prior similar NaN runs (different code path)

The same (r=256, k=3, lr=1e-2) cell run earlier under the BUGGY code (commit
predating `ef6b3bc`, applied absolute δ=0.01 instead of relative ε_rel=0.01)
also NaN'd, at step 1800 (100 steps later than the post-fix version).
That run is in `chord_tight_k3_eps_rel_1e-2_r256_lr_sweep_4k_blackwell/run_info/logs/log_1.out`
(buggy version, now relabeled δ_abs=1e-2 in the analysis notebook). First
`non_finite_intermediate` there at step 1725, pair 104 = `layers.14.mlp.down_proj`.

So the NaN appears at this (r, lr) under both absolute δ=0.01 AND relative
ε_rel=0.01, on different layers (1.v_proj vs 14.mlp.down_proj), and at
different step numbers (1690 vs 1725).

The very first NaN we ever saw (r=256, k=3, lr=1e-2, default-δ=1e-6, h100
hardware, group `chord_k3_leaderboard_fills_4k_h100`) was at step 300, on pair
88 = `layers.12.mlp.gate_proj`.

## 3. Hypotheses rejected against the data

### 3.1 "Gradient overflow / bf16 forward-backward overflow"

Claim: backward at some step returned grad_B with non-finite or fp32-overflowing
entries, poisoning Adam state.

Rejected because:
- `non_finite_detected` at step 1690 START would have fired if grad_A or grad_B
  contained any non-finite entry. It did not. The first such event is at step
  1691 START (after the dA, dB applied at step 1690 had already corrupted
  A_9 and B_9, which then made step 1691's forward NaN and backward grads
  NaN-everywhere).
- For fp32 grad² → inf in `v_B`, the entry would need |grad_B| > sqrt(3.4e38)
  ≈ 1.84e19. That's astronomical and would also corrupt loss/train_step output.
  train_loss at step 1690 was 0.561, finite and modest.

### 3.2 "Higham basin failure: cond_SB → ∞ pushes NS-10 out of basin"

Claim: cond_SB grew so large that the damped Higham NS-10 still failed.

Rejected because:
- Effective cond after relative damping is bounded by 1/ε_rel = 100.
  Higham NS-10 in fp32 at κ=100 is well within its convergence basin.
- `cond_SA_max = 129` (raw, pre-damping) at step 1680. After damping
  with σ_max(SA_max) ≈ 1, effective cond ≤ 100. Higham at this cond is the
  textbook safe regime.
- The clean lr=3e-2 cell in the same sweep reached higher cond_SB values
  earlier in training without NaN.

### 3.3 "Missing δ_min floor in relative damping"

Claim: `δ_min` floor (per `init_damping_math.md` §5.3) isn't implemented in
`spd_inv_sqrt_higham_batched`; when σ_max(SA_pair) is near zero, δ_eff = 0 and
NS fails.

Rejected because:
- `SA_max_min = 0.985` at step 1680. The minimum-across-pairs σ_max(SA) is ~1,
  not near zero. Relative damping gives δ_eff = 0.01 × 0.985 ≈ 0.01, healthy.
- The doc-recommended floor δ_min ≈ 1e-12 only kicks in when σ_max(H) < 1e-10,
  which is not the regime here.

Confirmed implementation gap: `utils.py:spd_inv_sqrt_higham_batched` does not
have the floor `δ_X = max(δ_min, ε_rel × σ_max)`. That IS a bug per the doc
(unaddressed because it doesn't affect the regime we're in). It would matter
only at literal Init[A] step 0 with σ_max(SB) = 0, which is moot because at
that step Δ_A = 0 anyway (the doc says so in §5.1).

## 4. What's measurable and still UNKNOWN

The chain check at step 1690 shows `u_B` and `SA_half_inv` (and downstream) all
non-finite for pair 9. The mechanism that connects "step 1690 START: grads
finite, A finite, B finite" to "step 1690 END: SA_half_inv NaN, u_B NaN" is
not deducible from currently-logged diagnostics:

- `u_B = (m_B / bc1) / (sqrt(v_B / bc2) + eps)`. For this to be NaN at step
  1690, either `m_B[pair 9]` or `v_B[pair 9]` was NaN at the start of step
  1690's optimizer.step. Adam state magnitudes are NOT in any logged event;
  only the resulting `u_B` is computed and chain-checked.
- `SA_half_inv` for pair 9: comes from `spd_inv_sqrt_higham_batched(SA = A @
  A.T, ...)` with relative damping. A_9 is finite at step 1690 start (top-of-step
  check did not fire). The Higham routine is fp32 throughout. The
  per-pair σ_max(SA_9) is not logged (only aggregate min/max/median across
  pairs).
- The actual numerical state inside spd_inv_sqrt_higham_batched at the failing
  pair (intermediate Z values across NS iterations) is not logged unless
  `--debug_higham_residual` is set, which it wasn't.

So the chain "what was the value of m_B[9] / v_B[9] at step 1690 start? what
was σ_max(SA_9)?" cannot be reconstructed from this run's logs.

## 5. Suggested verification

To resolve which intermediate first becomes non-finite, a re-run is needed with
finer-grain instrumentation. Concretely, instrument the optimizer to emit at
EVERY step:

1. Per-pair Adam state magnitudes: `|m_B[pair]|_max`, `|v_B[pair]|_max` (forward
   from `_step_batched` AFTER the m/v update, BEFORE forming u_B). One float
   per pair per step.
2. Per-pair `σ_max(SA[pair])`, `σ_max(SB[pair])` from the actual power-iter
   used in `spd_inv_sqrt_higham_batched` (which already computes them in the
   `eps_relative` branch). Currently these values are computed and discarded.
3. Per-pair Higham residual `‖ZHZ − I‖_F` per call (already available via
   `--debug_higham_residual`).

Then re-run the failing cell. Look at pair 9 specifically across steps
1680-1700. Specifically:

- If `|v_B[9]|_max` is much larger than `|v_B[other_pairs]|_max` at some step
  before 1690, that's grad accumulation diverging on this pair → root cause
  is upstream (grad magnitudes).
- If `σ_max(SA_9)` differs sharply from the aggregate `SA_max_max` at step 1680,
  that pair is a spectral outlier and per-pair damping might be miscomputed.
- If Higham residual on pair 9's SA call spikes at step 1690 specifically, NS
  is the culprit.

The instrumentation cost is ~6 extra floats per pair per step. At r=256 with
~112 pairs, that's ~700 floats/step ≈ 5 KB/step extra log volume. Cheap.

## 6. What the codebase looks like

- `lora_playground/optim.py`: contains `AdamPolarProductLoRA` class.
  `_step_batched` is the production hot path at r ≥ 64. Search for
  `_emit_non_finite_chain` and `_emit_non_finite_event` for the existing
  diagnostic emit points.
- `lora_playground/utils.py`: `spd_inv_sqrt_higham_batched` is the actual
  damped inverse-sqrt kernel. The `eps_relative` branch is at lines 142-157
  (verified by `tests/test_eps_relative_propagation.py`).
- `lora_playground/training_kernel.py`: per-step training loop. Top-of-step
  `non_finite_detected` and end-of-step `non_finite_intermediate` are emitted
  from inside `_step_batched`; the `train_norms` event is emitted from
  training_kernel.
- `docs/notes/polar_product/init_damping_math.md`: theoretical analysis of the
  chord-tight + damping scheme. §5.3 has the relative damping formula
  including the `δ_min` floor; §6 has the η-scaling argument that predicts
  η ~ √n, r-exponent -1/2 or -1.
- `tests/test_eps_relative_propagation.py`: regression test for the
  precond_delta_relative bug fix.

## 7. Compute available

Same hardware (Blackwell) is available via `sbatch_blackwell.sh` (8h wall).
Single-cell r=256 4k-step run takes ~4 hr with current diagnostic stack
(~30% overhead from logging) or ~1-1.5 hr with `--log_non_finite` off and
`optim_diagnostics_every 80` (production defaults).
