# Algorithm 2′ — clean implementation, tensor shapes, FLOP budget

Companion to `algorithm_tight_chord.md` §10. Walks Algorithm 2′ (`magnitude_rule = "spectral_chord_tight_clean"`) through `lora_playground/optim.py`, names every tensor that crosses a step boundary, and counts the FLOPs of every operation. Final section audits the three power-iteration sites and identifies the redundancy the legacy `spectral_chord_tight` carries.

## 1. Shapes and notation

One LoRA pair: $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$. Within `_step_batched_group`, all $N$ pairs sharing the same shape are stacked:

$$
A: (N, r, d_{\text{in}}),
\quad
B: (N, d_{\text{out}}, r).
$$

Let $d := \max(d_{\text{in}}, d_{\text{out}})$ and $r$ the LoRA rank ($r \ll d$). All counts below are float-multiply-adds; we ignore additive elementwise costs of $O(N r d)$ since they are subdominant to matmul $O(N r^2 d)$ at $r \ge 16$.

Persistent state per group:
- $m_A, v_A$ shape $(N, r, d_{\text{in}})$, $m_B, v_B$ shape $(N, d_{\text{out}}, r)$ — Adam moments.
- $S_A^{-1/2}, S_B^{-1/2}$ shape $(N, r, r)$ — whitening matrices, refreshed every `precond_refresh_every` steps.
- Warm-start top-singular vectors for power-iter (the smaller-side Gram converges as $(\sigma_2/\sigma_1)^{2\,n_{\text{iter}}}$; warm-starting from the prior step's converged $v$ lets us cut $n_{\text{iter}}$ from 8 cold to 3 warm). $r$ is the smaller side of every matrix below, so all warm-start vectors have shape $(N, r)$:

  | key | matrix | norm computed |
  |---|---|---|
  | `v_sigma_A` | $A \in \mathbb{R}^{N \times r \times d_{\text{in}}}$ | $\sigma_{\max}(A)$ — feeds $\rho = \eta/s$ |
  | `v_sigma_B` | $B \in \mathbb{R}^{N \times d_{\text{out}} \times r}$ | $\sigma_{\max}(B)$ — feeds $\rho = \eta/s$ |
  | `v_sigma_XA` | $X_A = S_B^{-1/2}\,u_A \in \mathbb{R}^{N \times r \times d_{\text{in}}}$ | $\sigma_{\max}(X_A)$ — pre-rescale (Alg 2′) |
  | `v_sigma_XB` | $X_B = u_B\,S_A^{-1/2} \in \mathbb{R}^{N \times d_{\text{out}} \times r}$ | $\sigma_{\max}(X_B)$ — pre-rescale (Alg 2′) |
  | `v_op_geoA` | $\text{geo}_A = S_B^{-1/2}\,P_A \in \mathbb{R}^{N \times r \times d_{\text{in}}}$ | $\sigma_{\max}(\text{geo}_A)$ — magnitude rescale |
  | `v_op_geoB` | $\text{geo}_B = P_B\,S_A^{-1/2} \in \mathbb{R}^{N \times d_{\text{out}} \times r}$ | $\sigma_{\max}(\text{geo}_B)$ — magnitude rescale |

## 2. Step walkthrough — `spectral_chord_tight_clean`, single Picard iter ($k = 1$)

### 2.1. Adam direction

In-place EMA updates on `m_A, v_A, m_B, v_B`, then bias-correct:

$$
u_A = \frac{m_A/(1-\beta_1^t)}{\sqrt{v_A/(1-\beta_2^t)} + \varepsilon},
\qquad u_B = \frac{m_B/(1-\beta_1^t)}{\sqrt{v_B/(1-\beta_2^t)} + \varepsilon}.
$$

FLOPs: $\Theta(N r d)$ elementwise, subdominant.

### 2.2. Whitening refresh (every `precond_refresh_every` steps)

$$
S_A = A\,A^\top \in \mathbb{R}^{N \times r \times r},
\qquad
S_B = B^\top B \in \mathbb{R}^{N \times r \times r}.
$$

| op | shape | FLOPs |
|---|---|---|
| $A A^\top$ | $(N, r, d_{\text{in}}) \cdot (N, d_{\text{in}}, r) \to (N, r, r)$ | $2 N r^2 d_{\text{in}}$ |
| $B^\top B$ | $(N, r, d_{\text{out}}) \cdot (N, d_{\text{out}}, r) \to (N, r, r)$ | $2 N r^2 d_{\text{out}}$ |
| $S_A^{-1/2}$ via Higham (n_iters=10) | $(N, r, r)$ | $\approx 60 N r^3$ |
| $S_B^{-1/2}$ via Higham | $(N, r, r)$ | $\approx 60 N r^3$ |

Higham (`spd_inv_sqrt_higham_batched`): 10 coupled NS iters of 3 `(r,r)·(r,r)` bmms each (≈ 6 r³ per iter), plus one λ_max power-iter (≈ 8 mat-vec, 16 r² × 8) at start. Total $\approx 60 r^3$ per pair.

### 2.3. Pre-rescale (Algorithm 2′ vs 2)

This is the only step where Algorithm 2′ differs from Algorithm 2 at $k = 1$. Compute $\sigma_{\max}(X_A)$, $\sigma_{\max}(X_B)$ via power iteration (8 iters), then divide.

| op | shape | FLOPs |
|---|---|---|
| $X_A := S_B^{-1/2}\,u_A$ | $(N, r, r) \cdot (N, r, d_{\text{in}}) \to (N, r, d_{\text{in}})$ | $2 N r^2 d_{\text{in}}$ |
| $X_B := u_B\,S_A^{-1/2}$ | $(N, d_{\text{out}}, r) \cdot (N, r, r) \to (N, d_{\text{out}}, r)$ | $2 N r^2 d_{\text{out}}$ |
| $\sigma_{\max}(X_A)$, 8 iters | smaller side $r$ | $32 N r d_{\text{in}}$ |
| $\sigma_{\max}(X_B)$, 8 iters | smaller side $r$ | $32 N r d_{\text{out}}$ |
| $u_A \mathrel{{/}{=}} \sigma_{\max}(X_A)$ | elementwise | $N r d_{\text{in}}$ |
| $u_B \mathrel{{/}{=}} \sigma_{\max}(X_B)$ | elementwise | $N r d_{\text{out}}$ |

Power-iter cost per iter on a matrix with smaller side $r$ is $2 \cdot 2 N r d$ (two mat-vecs, each $2 N r d$); × 8 iters = $32 N r d$. **Note**: $X_A$ here is *computed* and immediately *discarded*. The power iter consumes it, but the matrix itself is not reused downstream. The legacy code recomputes the very same product $S_B^{-1/2}\,u_A$ in §2.5 below.

### 2.4. Tight-tangent radius

$$
\sigma_A := \sigma_{\max}(A),
\quad
\sigma_B := \sigma_{\max}(B),
\quad
s := \sigma_A + \sigma_B,
\quad
\rho := \eta / s.
$$

| op | shape | FLOPs |
|---|---|---|
| $\sigma_{\max}(A)$, 8 iters | smaller side $r$ | $32 N r d_{\text{in}}$ |
| $\sigma_{\max}(B)$, 8 iters | smaller side $r$ | $32 N r d_{\text{out}}$ |
| $\rho$ | scalar per pair | $\Theta(N)$ |

### 2.5. Polar pipeline ($k = 1$: one iteration)

The cross-coupling step is a no-op at $n = 1$ since $\mathrm dA^{(0)} = \mathrm dB^{(0)} = 0$, so we go directly from $u_A, u_B$ (post-rescale) to the polar map.

| op | shape | FLOPs |
|---|---|---|
| $X_A^{\text{eff}} := S_B^{-1/2}\,u_A$ | $(N, r, r) \cdot (N, r, d_{\text{in}})$ | $2 N r^2 d_{\text{in}}$ |
| $X_B^{\text{eff}} := u_B\,S_A^{-1/2}$ | $(N, d_{\text{out}}, r) \cdot (N, r, r)$ | $2 N r^2 d_{\text{out}}$ |
| $P_A := \mathrm{polar}_{\text{NS-}j}(X_A^{\text{eff}})$, $j = 5$ | bf16 NS on $(N, r, d_{\text{in}})$ | $\approx 20 N r^2 d_{\text{in}}$ |
| $P_B := \mathrm{polar}_{\text{NS-}j}(X_B^{\text{eff}})$, $j = 5$ | bf16 NS on $(N, d_{\text{out}}, r)$ | $\approx 20 N r^2 d_{\text{out}}$ |

Newton–Schulz per iter: $X X^\top$ ($(N, r, r)$, $2 N r^2 d$) plus $(X X^\top)\,X$ ($(N, r, d)$, $2 N r^2 d$). Total $4 N r^2 d$ per iter; × 5 iters = $20 N r^2 d$. Runs in bf16 on tensor cores; the FLOP count is dtype-invariant but wall-clock is $\sim 2{-}4\times$ faster than fp32.

### 2.6. Unwhiten and magnitude rescale

$$
\text{geo}_A := S_B^{-1/2}\,P_A,
\quad
\text{geo}_B := P_B\,S_A^{-1/2},
\quad
\mathrm dA = -\rho\,\frac{\text{geo}_A}{\sigma_{\max}(\text{geo}_A)},
\quad
\mathrm dB = -\rho\,\frac{\text{geo}_B}{\sigma_{\max}(\text{geo}_B)}.
$$

| op | shape | FLOPs |
|---|---|---|
| $S_B^{-1/2}\,P_A$ | $(N, r, r) \cdot (N, r, d_{\text{in}})$ | $2 N r^2 d_{\text{in}}$ |
| $P_B\,S_A^{-1/2}$ | $(N, d_{\text{out}}, r) \cdot (N, r, r)$ | $2 N r^2 d_{\text{out}}$ |
| $\sigma_{\max}(\text{geo}_A)$, 8 iters | smaller side $r$ | $32 N r d_{\text{in}}$ |
| $\sigma_{\max}(\text{geo}_B)$, 8 iters | smaller side $r$ | $32 N r d_{\text{out}}$ |
| $\mathrm dA$, $\mathrm dB$ scaled-add | elementwise | $\Theta(N r d)$ |

### 2.7. Apply

$A \mathrel{+}= \mathrm dA$, $B \mathrel{+}= \mathrm dB$. Elementwise, $\Theta(N r d)$.

## 3. FLOP budget — totals at $k = 1$

Let $D := d_{\text{in}} + d_{\text{out}}$. Per step, ignoring elementwise terms:

| block | FLOPs | refresh cadence |
|---|---|---|
| Whitening matmuls ($AA^\top, B^\top B$) | $2 N r^2 D$ | every refresh |
| Higham ($S_A^{-1/2}, S_B^{-1/2}$) | $120 N r^3$ | every refresh |
| Pre-rescale matmuls ($X_A, X_B$) | $2 N r^2 D$ | every step |
| Pre-rescale power iters | $32 N r D$ | every step |
| $\sigma_{\max}(A,B)$ power iters | $32 N r D$ | every step |
| Polar pipeline whiten ($X_A^{\text{eff}}, X_B^{\text{eff}}$) | $2 N r^2 D$ | every step |
| NS polar map ($P_A, P_B$, $j=5$) | $20 N r^2 D$ | every step |
| Unwhiten ($\text{geo}_A, \text{geo}_B$) | $2 N r^2 D$ | every step |
| $\sigma_{\max}(\text{geo})$ power iters | $32 N r D$ | every step |

**Numerical totals** at $r = 64$, $D = 2 \cdot 2048 = 4096$, refresh every step:

| block | FLOPs / pair | fraction of step |
|---|---|---|
| NS polar (dominant) | $3.4 \cdot 10^8$ | 65% |
| All matmuls combined (whiten + unwhiten + pre + polar-prep) | $1.4 \cdot 10^8$ | 27% |
| Higham (refresh) | $3.1 \cdot 10^7$ | 6% |
| 3 power-iter sites combined | $2.5 \cdot 10^7$ | 5% |
| **Total** | $5.3 \cdot 10^8$ | — |

At $r = 256$, $D = 4096$: NS dominates at 78%; power iters drop to 1.3%. At $r = 16$, $D = 4096$: power iters rise to ~15% because they scale as $N r D$ vs the $N r^2 D$ of everything else.

## 3.1. Gram-form Newton–Schulz (`--ns_form gram`)

The polar map at §2.5 dominates step cost; Tri Dao's Gram NS (2026, `docs/papers/gram_newton_schulz_dao_2026.md`) reformulates the same composition to iterate on the smaller-side $r \times r$ Gram instead of on the rectangular $X \in \mathbb{R}^{r \times d}$. Enabled via `--ns_form gram`; default is `rect` for trajectory continuity with existing sweeps.

### Math (Dao Theorem 1)

For NS polynomial $p_t(x) = a_t x + b_t x^3 + c_t x^5$, write $p_t(x) = x \cdot h_t(x^2)$ with $h_t(y) = a_t + b_t y + c_t y^2$. Standard NS acts on $X$; equivalent Gram form acts on $R = X X^\top \in \mathbb{R}^{r \times r}$ and accumulates a single $Q \in \mathbb{R}^{r \times r}$ that produces the final $X$ via $X_T = Q_T X_0$:

$$
R_0 = X_0 X_0^\top, \quad Q_0 = I, \qquad
\begin{cases}
Z_t = h_t(R_{t-1}) = a_t I + b_t R_{t-1} + c_t R_{t-1}^2 \\
Q_t = Q_{t-1} Z_t \\
R_t = Z_t R_{t-1} Z_t
\end{cases}
$$

All work between $R_0$ and the final reconstruction is on $(N, r, r)$.

### FLOP win at $K = 5$, cubic Muon ($c_t = 0$)

| op | shape | FLOPs |
|---|---|---|
| $R_0 = X X^\top$ | $(N, r, d) \cdot (N, d, r) \to (N, r, r)$ | $2 N r^2 d$ |
| Per-iter $R$ update ($Z \cdot R \cdot Z$ via direct $R^k$ poly) | $(N, r, r)$ | $\approx 4 N r^3$ |
| Per-iter $Q$ update ($Q \leftarrow M_k Q$) | $(N, r, r)$ | $\approx 2 N r^3$ |
| Reconstruction $X_T = Q_T X_0$ | $(N, r, r) \cdot (N, r, d) \to (N, r, d)$ | $2 N r^2 d$ |

Total $K = 5$: $4 N r^2 d + 30 N r^3$. Vs rect: $20 N r^2 d$. At $r = 64$, $d = 2048$: rect $1.7 \times 10^8 N$, gram $2.4 \times 10^7 N$ — **~7× FLOP reduction in the dominant block**. Measured wall impact is much smaller than the FLOP ratio suggests because the polar block itself is a small fraction of optimizer step on Blackwell at $r=64$: per-scope profile (`scripts/bench/profile_chord_tight_clean.py`, see `walltime_profile.md` §"Gram-NS + k=2 wall-time") puts the entire Picard loop at ~75% of opt_ms but opt_ms is only ~15% of step wall. Gram vs rect at fixed $k=3$ shaves ~10% off the Picard scope (35.3→31.8 ms) and < 2% off total step wall. Expected to show a larger gap at $r=256$ where rect's $r^2 d$ dominates and gram's $r^3$ amortizes; not measured yet.

### Precision strategy — fp16 with restart at $\tau = 2$ (Dao Algorithm 3)

Gram NS has two failure modes that rect NS does not, both arising because $R_t$ is *cumulative state* that re-enters every iteration:

1. **Spurious negative eigenvalues.** $R_0 = X X^\top$ in half precision develops eigenvalues $\approx -\varepsilon_{\text{half}}$ where the true eigenvalue is 0. The scalar recurrence $r_t \approx 1.5^2 \, r_{t-1}$ (cubic Muon $h(0) = 1.5$; Polar Express has $h(0) = 15/8$, faster blowup) drives any tiny negative toward $-\infty$. Once $|r| \sim 1$, the polynomial leaves the basin and $z_t = h_t(r)$ blows up super-exponentially.
2. **Eigenvector drift.** In exact arithmetic $R_t, Q_t, X_t$ all share $X_0$'s singular vectors. In finite precision $Q$ drifts away — the final $X_T = Q_T X_0$ no longer captures $X_0$'s dominant directions correctly.

**Fix (restart at $\tau = 2$):**
$$
X_0 \leftarrow Q_\tau X_0, \quad R_\tau \leftarrow X_0 X_0^\top, \quad Q_\tau \leftarrow I.
$$
Resets accumulated negative-eigenvalue magnitudes to the noise floor on the *current* iterate, and resets the $R \leftrightarrow Q$ eigenvector drift to zero. Cost: two extra matmuls.

**fp16 not bf16.** fp16's 10-bit mantissa shrinks the spurious-negative noise floor $\sim 10\times$ vs bf16's 7-bit. After Frobenius pre-norm (with safety factor 1.05) values sit near $[0, 1]$ — fp16's narrow range doesn't bind. Rect NS uses bf16 because it doesn't compound state; gram reverses that priority.

Implementation at `lora_playground/optim.py:_newton_schulz_gram_batched`; also exposes a safety-mode `dtype=torch.float32` path that disables TF32 (mirroring `spd_power_batched` at `utils.py:235`) for diagnostics — no restart needed because the fp32 noise floor never reaches the basin.

### Verification

- Equivalence: `tests/test_ns_gram.py` confirms gram-fp32 matches rect-fp32 to fp32 noise ($< 10^{-5}$) on random, real (Tier 1 — chord-tight r=64 snapshots), and synthetic cond=$10^4$ inputs.
- fp16+restart: matches rect-fp32 within $5 \times 10^{-2}$ rel-err on Tier 1 corpus (top-20 worst-conditioned X_eff from `/mnt/ceph/users/nghosh/lora_snapshots/chord_tight_r64_k3_snapshot_blackwell/`; max measured cond$(G)$ = $1.4 \times 10^5$).
- End-to-end smoke: 5-step train at r=64, k=2, OLMo-2-1B: gram eval_loss = 1.1759, rect = 1.1792, $|\Delta| < 5 \times 10^{-3}$.

## 4. Per-Picard-iter incremental cost ($k \ge 2$)

**Reminder.** Each Picard iter runs the **full polar pipeline**: whiten ($X_A^{\text{eff}}$ = $S_B^{-1/2}\,\tilde u_A$), one Newton–Schulz polar map ($P_A = \text{polar}_{\text{NS-}j}(X_A^{\text{eff}})$, $j$ inner NS quintic iterations), unwhiten, and σ-rescale. So $k$ Picard iters means $k$ NS polar maps and $k$ σ_max(geo) power iters; the only operations that run *once* per step are the Adam direction (§2.1), the whitening refresh (§2.2, schedule-gated), the pre-rescale (§2.3), and the ρ computation (§2.4). The cross-coupling correction (§2.5 with $n \ge 2$) is the only piece that's skipped at $n = 1$.

Each additional Picard iter $n \ge 2$ adds, inside the inner loop:

| op | shape | FLOPs |
|---|---|---|
| $B^\top\,\mathrm dB^{(n-1)}\,A$ | two bmms | $2 N r^2 d_{\text{out}} + 2 N r^2 d_{\text{in}}$ |
| $B\,\mathrm dA^{(n-1)}\,A^\top$ | two bmms | $2 N r^2 d_{\text{in}} + 2 N r^2 d_{\text{out}}$ |
| $\tilde u_A, \tilde u_B$ scaled-add | elementwise | $\Theta(N r D)$ |
| Polar pipeline (whiten + NS + unwhiten) | same as §2.5–2.6 | $24 N r^2 D + 64 N r D$ |

Cross-coupling adds $8 N r^2 D$ per iter; the polar pipeline repeats at $24 N r^2 D$. So each $n \ge 2$ Picard iter costs $\approx 32 N r^2 D$, roughly 60% of the $k = 1$ step.

## 5. Power-iter audit — the three sites

| site | bounds | redundancy under Alg 2′ |
|---|---|---|
| **A** $\sigma_{\max}(X_A), \sigma_{\max}(X_B)$ — pre-rescale | required: the pre-rescale that makes 2′ ≠ 2 | $X_A$ is computed here, used only for its operator norm, then discarded. The product $S_B^{-1/2}\,u_A$ is recomputed at §2.5 (after rescaling $u_A$). |
| **B** $\sigma_{\max}(A), \sigma_{\max}(B)$ — ρ formula | required: feeds $\rho = \eta/s$ | clean — no duplication. Could in principle re-use the λ_max estimates that Higham computed for $S_A = A A^\top$, since $\sigma_{\max}(A) = \sqrt{\lambda_{\max}(S_A)}$; legacy code does not exploit this. |
| **C** $\sigma_{\max}(\text{geo}_A), \sigma_{\max}(\text{geo}_B)$ — magnitude rescale | required: the §2.6 normalization | warm-start vector is cached but only by name (`v_op_geoA`), not keyed by Picard iter $n$. At $k \ge 2$ this means iter-$(n-1)$'s top singular vector warms iter-$n$'s start, even though $\text{geo}_A$ is recomputed from a different $u_A^{\text{eff}}$. Silently wrong if the largest singular vector rotates between Picard iters. |

### 5.1. Concrete redundancies in `spectral_chord_tight`

The current `_step_batched_group` body at `optim.py:~3370–3600` does the following extra work:

1. **Double-computed $S_B^{-1/2}\,u_A$.** Lines ~3390 (pre-rescale) and ~3554 (polar pipeline whiten) both compute $S_B^{-1/2}\,u_A$ — the second time on the rescaled $u_A$, so the matrix is different by a positive scalar but the matmul work is identical. Save the post-rescale matmul by computing $X_A := S_B^{-1/2}\,u_A$ once, taking $\sigma_{\max}(X_A)$, rescaling $X_A \mathrel{{/}{=}} \sigma_{\max}(X_A)$, and feeding the rescaled matrix into NS directly. Cost saved: $2 N r^2 D$, i.e. ~4% of step.

2. **Hoist $\lambda_{\max}(S_A)$ out of Higham; pass it in instead.** `spd_inv_sqrt_higham_batched(H, ...)` at `utils.py:146` currently calls `lambda_max_power_iter_psd_batched(H, ...)` internally to scale $Y_0 = H/s$ into Newton–Schulz's basin (and, with `eps_relative=True`, a *second* call on raw $H$ for the damping). But for $H = S_A = A A^\top$, we have the identity
   $$
   \lambda_{\max}(S_A) \;=\; \sigma_{\max}(A)^2,
   $$
   and $\sigma_{\max}(A)$ is *already* computed every step at site B for $\rho = \eta/s$. The current code computes $\lambda_{\max}(S_A)$ twice: once inside Higham, once at site B. Single-source it by:

   - Computing $\sigma_{\max}(A), \sigma_{\max}(B)$ at the top of the step (site B, as today).
   - Adding a `lam_max` keyword argument to `spd_inv_sqrt_higham_batched` that, when provided, skips both internal `lambda_max_power_iter_psd_batched` calls and uses the passed value.
   - Passing `lam_max=sigma_A.pow(2)` (and analogously for $S_B$) into the refresh.

   With `eps_relative=True` the same `lam_max` feeds both the damping ($\delta_A = \varepsilon_{\text{rel}} \cdot \lambda_{\max}(S_A)$) and the NS scaling on the damped matrix ($\lambda_{\max}(S_A + \delta_A I) = \lambda_{\max}(S_A) + \delta_A = \lambda_{\max}(S_A)(1 + \varepsilon_{\text{rel}})$, exact closed form, no second power iter needed).

   FLOP win is small (Higham's internal calls are on $(N, r, r)$, cost $32 N r^2$ each — dwarfed by the NS iteration's $60 N r^3$). The point is architectural: one canonical computation of $\sigma_{\max}(A)$ per step, used by both ρ and the preconditioner damping. Removes the "Higham silently runs its own power iter" footgun for anyone reading the refresh code.

3. **Cold vs warm-start asymmetry.** All three sites use `n_iters=8` unconditionally. The helper docstring says "n_iters=3 reaches the same accuracy with warm start." On non-refresh steps this is 5 wasted iters per call × 3 sites = up to 9% of step at $r = 16$ (where power-iter is non-negligible). Adoption is gated on: (a) does the warm-start key actually correspond to the same matrix the next step (yes for site B; no for sites A, C if recomputed each step on a different matrix); (b) is the convergence still adequate? Verify via a per-pair residual test before committing.

4. **Picard-keyed warm-start at site C.** As noted in the table above, `v_op_geoA` is cached by name only. Either re-key by Picard iter (`v_op_geoA_n0`, `v_op_geoA_n1`, …) or accept that warm-start is meaningless for $n \ge 2$ and force `n_iters=8` (cold) at $n \ge 1$. The first option is cleaner.

5. **Dead branches in the gated `_step_batched_group`.** Variant-A's clean rule is currently a set of `if magnitude_rule == "spectral_chord_tight_clean"` branches threaded through 250 lines that also handle `spectral_chord`, `spectral_chord_tight`, `spectral_chord_direction`, `spectral_chord_tight_no_rho`, and `adam_frobenius`. The clean rule does not need: the quadratic-ρ branch, the `exact_chord` per-iter Higham refresh, the `direction`'s $\lambda_{\text{dir}}$ quartic root, the `picard_alpha/lr` legacy coefficient, or the `disable_whitening` ablation path. A dedicated `_step_chord_tight_clean` method drops ~150 lines from the active code path and removes interleaving with other rules.

## 6. Proposed clean-method skeleton

```python
def _step_chord_tight_clean(self, group, gs, indices):
    """Algorithm 2′ — single dedicated entry. Algorithm 2 + pre-rescale +
    1/η cross-coupling + linear ρ = η/s. No exact-chord refresh, no
    direction-aware variant, no whitening ablation.
    """
    A_f, B_f = gs['A_stack'], gs['B_stack']
    N = A_f.shape[0]
    lr = group['lr']

    # 1. Adam direction (shared with other rules — keep in caller).
    #    EMAs on (m_A, v_A, m_B, v_B) + bias-correct → (u_A, u_B).
    u_A, u_B = self._adam_direction_batched(gs)

    # 2. σ_max(A), σ_max(B) FIRST. Two consumers:
    #    (a) ρ = η/s below;
    #    (b) the whitening refresh — λ_max(S_A) = σ_max(A)² is passed in,
    #        so Higham skips its internal lambda_max_power_iter calls.
    sigma_A, gs['v_sigma_A'] = _sigma_max_power_iter_batched(
        A_f, v_init=gs.get('v_sigma_A'), n_iters=8)
    sigma_B, gs['v_sigma_B'] = _sigma_max_power_iter_batched(
        B_f, v_init=gs.get('v_sigma_B'), n_iters=8)
    rho = lr / (sigma_A + sigma_B).clamp_min(1e-30)     # (N,)

    # 3. Whitening refresh, inlined. On schedule steps recompute SA, SB
    #    grams and run Higham with the precomputed λ_max. Otherwise the
    #    cached `gs['SA_half_inv']` / `gs['SB_half_inv']` carry forward.
    if (step_count - 1) % self.precond_refresh_every == 0:
        SA_grams = A_f @ A_f.transpose(-2, -1)          # (N, r, r)
        SB_grams = B_f.transpose(-2, -1) @ B_f          # (N, r, r)
        gs['SA_half_inv'].copy_(spd_inv_sqrt_higham_batched(
            SA_grams, n_iters=self.higham_iters,
            eps=self.delta, eps_relative=self.precond_delta_relative,
            lam_max=sigma_A.pow(2),                     # NEW kwarg in utils.py
        ))
        gs['SB_half_inv'].copy_(spd_inv_sqrt_higham_batched(
            SB_grams, n_iters=self.higham_iters,
            eps=self.delta, eps_relative=self.precond_delta_relative,
            lam_max=sigma_B.pow(2),
        ))
    SA_inv = gs['SA_half_inv']
    SB_inv = gs['SB_half_inv']

    # 4. Whiten Adam direction ONCE; reuse for pre-rescale and polar input.
    X_A = SB_inv @ u_A                                  # (N, r, d_in)
    X_B = u_B @ SA_inv                                  # (N, d_out, r)

    # 5. Pre-rescale: σ_max(X_A), σ_max(X_B) via power iter (warm-start).
    sigma_XA, gs['v_sigma_XA'] = _sigma_max_power_iter_batched(
        X_A, v_init=gs.get('v_sigma_XA'), n_iters=8)
    sigma_XB, gs['v_sigma_XB'] = _sigma_max_power_iter_batched(
        X_B, v_init=gs.get('v_sigma_XB'), n_iters=8)
    X_A = X_A / sigma_XA[..., None, None].clamp_min(1e-30)
    X_B = X_B / sigma_XB[..., None, None].clamp_min(1e-30)

    # 6. Picard loop (k=1: no cross-coupling; k≥2: re-key warm-starts by n).
    #    (Step 5 above produced X_A, X_B at unit operator norm.)
    dA = torch.zeros_like(u_A)
    dB = torch.zeros_like(u_B)
    for n in range(self.picard_iters):
        if n == 0:
            X_A_eff, X_B_eff = X_A, X_B
        else:
            # Cross-coupling — 1/η coefficient straight from §10 derivation.
            X_A_eff = X_A + (1.0 / lr) * (SB_inv @ B_f.transpose(-2, -1) @ dB @ A_f)
            X_B_eff = X_B + (1.0 / lr) * (B_f @ dA @ A_f.transpose(-2, -1) @ SA_inv)

        P_A = _newton_schulz_batched(X_A_eff, nsteps=self.ns_steps, dtype=torch.bfloat16).float()
        P_B = _newton_schulz_batched(X_B_eff, nsteps=self.ns_steps, dtype=torch.bfloat16).float()

        geo_A = SB_inv @ P_A
        geo_B = P_B @ SA_inv

        # σ_max(geo) — Picard-iter-keyed warm-start.
        key_A = f'v_op_geoA_n{n}'
        key_B = f'v_op_geoB_n{n}'
        sigma_geoA, gs[key_A] = _sigma_max_power_iter_batched(
            geo_A, v_init=gs.get(key_A), n_iters=8)
        sigma_geoB, gs[key_B] = _sigma_max_power_iter_batched(
            geo_B, v_init=gs.get(key_B), n_iters=8)

        rho_b = rho[..., None, None]
        dA = -rho_b / sigma_geoA[..., None, None].clamp_min(1e-30) * geo_A
        dB = -(self.lora_plus_multiplier) * rho_b / sigma_geoB[..., None, None].clamp_min(1e-30) * geo_B

    # 7. Apply.
    self._apply_updates_batched(gs, indices, dA, dB)

    # 8. Diagnostics — caller-side, unchanged.
```

Wins relative to the gated implementation:
- No legacy code paths in the active method.
- $S_B^{-1/2}\,u_A$ matmul once instead of twice (saves $2 N r^2 D$/step).
- Picard-iter-keyed warm-start at site C eliminates the silent staleness at $k \ge 2$.
- Cross-coupling reads as $(1/\eta) \cdot S_B^{-1/2} (B^\top \mathrm dB A)$, directly matching §10 notation — no $2/(\rho s)$ doubling or other empirical multipliers.
- λ_max(S_A) hoisted out of Higham (idea #2 in §5.1): single canonical source.

Companion change in `utils.py`: `spd_inv_sqrt_higham_batched` gains an optional `lam_max=None` kwarg. When `None` (default) the existing behavior is preserved — Higham runs its own `lambda_max_power_iter_psd_batched`. When a tensor is provided, both internal calls are skipped (the `eps_relative=True` damped λ_max is `lam_max·(1 + ε_rel)` in closed form). Non-breaking for every other caller of Higham in the project.

What this skeleton does **not** do (out of scope for this refactor):
- Switch power iter from 8 cold to 3 warm. Defer to a separate measurement-driven change.
- Swap Newton–Schulz for a Gram-form variant (see §8 below). Defer; significant kernel change.
- Add timing/diagnostic hooks. Keep the existing `maybe_time` and snapshot machinery in the caller-side wrapper.

## 7. Verification plan

The refactor preserves trajectory only if:

1. At $k = 1$, `spectral_chord_tight_clean` (refactored) produces bit-identical output to `spectral_chord_tight_clean` (current gated form). Test: tiny CPU model, same seed, $\lVert \mathrm dA_{\text{new}} - \mathrm dA_{\text{old}}\rVert_F / \lVert \mathrm dA_{\text{old}}\rVert_F < 10^{-6}$ in fp32.
2. At $k = 3$, allow $\le 5\%$ relative diff (Picard warm-start rekey changes the trajectory by exactly the staleness the legacy code was carrying).
3. End-to-end smoke through `train_lora.py` — 5 steps at lr=3e-3, r=64, picard_iters_override=3 — same final eval loss within 0.01.
4. Gram-NS equivalence and precision tests: `tests/test_ns_gram.py` (equivalence to rect-fp32, fp16+restart on Tier 1 real X_eff corpus, synthetic cond=$10^4$ stress). Fixtures built by `scripts/build_gram_ns_test_fixtures.py` from existing chord-tight r=64 snapshots — no extra sweep required.

After verification, the pending sbatch `chord_tight_clean_lrsweep_k3_r64_4k_blackwell.sbatch` re-runs at the new commit. The decision rule in `there-is-an-ablation-magical-stroustrup.md` §2.4 still applies (compare at lr=3e-3 overlap with existing chord-tight).

## 9. Tested and rejected: σ → σ^p (HTMuon)

HTMuon (Pang et al. 2026, `docs/papers/htmuon_2603.10067.pdf`) proposes σ → σ^p, p ∈ (0, 1), as a "graduated" alternative to the polar σ → 1. The argument is that finite-NS Muon (with its non-flat singular spectrum) outperforms exact-polar Muon by preserving information in noise-dominated directions; σ^p makes that preservation explicit and tunable. Implementation: `htmuon(X, p) = U Σ^p V^T = (X X^T)^(p/2) · polar(X)`, applied as a sub-mode of `spectral_chord_tight_clean` via the `htmuon_p` CLI flag (`lora_playground/utils.py:spd_power_batched`).

**Result (`chord_tight_clean_htmuon_p_lr_grid_r256_blackwell`, 3-p × 3-lr Cartesian at r=256, k=3, default-δ, step=4000):**

| | eval@4000 | Δ vs NS=5 (σ_AdamW=0.0017) |
|---|---|---|
| NS=5 baseline (clean default-δ, lr=3e-2) | 0.5025 | — |
| NS=10 reference (lr=3e-2) | 0.5054 | +1.7σ |
| **best htmuon: lr=3e-2, p=0.0625** | **0.5048** | **+1.34σ** |
| other htmuon cells | 0.5055 — 0.5143 | +1.78σ to +6.94σ |

**Every measured (p, lr) is ≥1.3σ worse than NS=5.** Smaller p is better within each lr row (0.0625 < 0.125 < 0.25), and best lr is 3e-2 (matches NS=5's lr-best — not lr-pinned). Best htmuon ≈ NS=10 (within 1σ) but neither matches NS=5.

**Reading.** NS=5's specific shape — moderate cutoff at σ_in ≈ 0.08·σ_max — does noise rejection that neither extreme replicates: NS=10 (σ→1, no cutoff) and σ^p (graduated preservation of small σ) both lose to it by ~1.3-1.7σ. The HTMuon "soft msign" framing doesn't transfer to the chord-tight whitened pipeline; the cutoff structure of NS=5 is doing real work the bench-time accuracy criterion (rel-err < 1e-4 on the polar output) didn't measure.

**Implementation note (numerical).** `spd_power_batched` requires fp32 matmul internally (forced via `torch.backends.cuda.matmul.allow_tf32 = False` inside the call). TF32's ~10-bit mantissa over the n_outer × n_iters ≥ 50 cascaded matmuls compounds to NaN output for Gram matrices with cond ≳ 1e4 (real LoRA training Grams at r=256). The first version of the sweep without this guard NaN'd 9/9 cells by step ≤ 400; with the guard, all 9 train to step 4000 cleanly. The bench at `scripts/bench/bench_htmuon_op.py` missed this because random Gaussian inputs at LoRA shape have cond ~ 100, well within TF32's stable range — a measurement-vs-production gap worth remembering for future numerical primitives.

**What would change the verdict.** σ^p applied to the raw Adam direction (not the whitened input) is the HTMuon paper's regime and untested here. The whitening preconditioner `S_B^{-1/2}` may itself be incompatible with σ^p in a way that the cutoff structure of NS=5 was hiding. If we ever turn off chord-tight whitening (`disable_whitening` exists as an ablation flag), σ^p would be worth re-running there.
