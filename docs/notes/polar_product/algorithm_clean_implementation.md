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
- Warm-start top-singular vectors `v_sigma_A`, `v_sigma_B`, `v_sigma_XA`, `v_sigma_XB`, `v_op_geoA`, `v_op_geoB`, each shape $(N, r)$ or $(N, d)$ depending on which side of its matrix is smaller.

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

## 4. Per-Picard-iter incremental cost ($k \ge 2$)

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

2. **Unused Higham byproduct.** `spd_inv_sqrt_higham_batched` calls `lambda_max_power_iter_psd_batched(H, n_iters=8)` on $H = S_A = A A^\top$ once per refresh. $\lambda_{\max}(S_A) = \sigma_{\max}(A)^2$. Site B then runs another 8-iter power iter on $A$ for $\sigma_{\max}(A)$. The Higham λ_max is on $S_A$ not $A$, so it is not directly returned by the helper; trivially exposed by returning it alongside $S_A^{-1/2}$. Saves the site-B power iter on refresh steps only — small (≤ 2.5% of step).

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

    # 1. Adam direction (shared with other rules — keep in caller)
    u_A, u_B = self._adam_direction_batched(gs)

    # 2. Whitening refresh (shared — keep in caller)
    SA_inv, SB_inv = self._refresh_whitening_batched(gs, A_f, B_f)

    # 3. Whiten Adam direction ONCE; reuse for pre-rescale and polar input.
    X_A = SB_inv @ u_A                                  # (N, r, d_in)
    X_B = u_B @ SA_inv                                  # (N, d_out, r)

    # 4. Pre-rescale: σ_max(X_A), σ_max(X_B) via power iter (warm-start).
    sigma_XA, gs['v_sigma_XA'] = _sigma_max_power_iter_batched(
        X_A, v_init=gs.get('v_sigma_XA'), n_iters=8)
    sigma_XB, gs['v_sigma_XB'] = _sigma_max_power_iter_batched(
        X_B, v_init=gs.get('v_sigma_XB'), n_iters=8)
    X_A = X_A / sigma_XA[..., None, None].clamp_min(1e-30)
    X_B = X_B / sigma_XB[..., None, None].clamp_min(1e-30)

    # 5. ρ = η / (σ_A + σ_B).
    sigma_A, gs['v_sigma_A'] = _sigma_max_power_iter_batched(
        A_f, v_init=gs.get('v_sigma_A'), n_iters=8)
    sigma_B, gs['v_sigma_B'] = _sigma_max_power_iter_batched(
        B_f, v_init=gs.get('v_sigma_B'), n_iters=8)
    rho = lr / (sigma_A + sigma_B).clamp_min(1e-30)     # (N,)

    # 6. Picard loop (k=1: no cross-coupling; k≥2: re-key warm-starts by n).
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

What this skeleton does **not** do (out of scope for this refactor):
- Switch power iter from 8 cold to 3 warm. Defer to a separate measurement-driven change.
- Re-use Higham's λ_max byproduct for site B. Defer.
- Add timing/diagnostic hooks. Keep the existing `maybe_time` and snapshot machinery in the caller-side wrapper.

## 7. Verification plan

The refactor preserves trajectory only if:

1. At $k = 1$, `spectral_chord_tight_clean` (refactored) produces bit-identical output to `spectral_chord_tight_clean` (current gated form). Test: tiny CPU model, same seed, $\lVert \mathrm dA_{\text{new}} - \mathrm dA_{\text{old}}\rVert_F / \lVert \mathrm dA_{\text{old}}\rVert_F < 10^{-6}$ in fp32.
2. At $k = 3$, allow $\le 5\%$ relative diff (Picard warm-start rekey changes the trajectory by exactly the staleness the legacy code was carrying).
3. End-to-end smoke through `train_lora.py` — 5 steps at lr=3e-3, r=64, picard_iters_override=3 — same final eval loss within 0.01.

After verification, the pending sbatch `chord_tight_clean_lrsweep_k3_r64_4k_blackwell.sbatch` re-runs at the new commit. The decision rule in `there-is-an-ablation-magical-stroustrup.md` §2.4 still applies (compare at lr=3e-3 overlap with existing chord-tight).
