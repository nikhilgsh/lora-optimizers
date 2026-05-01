# Stale-preconditioner speedup for Gram-preconditioned LoRA optimizers

## Problem

At r=256, the Gram-preconditioned LoRA optimizers run far slower per step than
AdamW. Measured on OLMo-2-1B + Magicoder, all-linear LoRA, single A100:

| optimizer                     | r=256 sec/step | × vs AdamW |
|-------------------------------|----------------|------------|
| `adamw`                       | 0.88           | 1.0×       |
| `adam-muon-lora`              | 0.97           | 1.1×       |
| `adam-scaled-lora`            | 2.54           | 2.9×       |
| `adam-polar-product-lora`     | 3.88           | 4.4×       |
| `adamuon-polar-product-lora`  | 3.91           | 4.4×       |

`adam-muon-lora` does Newton-Schulz on Adam's direction and is essentially
free, so NS is not the cost. The slow optimizers share two properties NS-only
optimizers don't:

1. They build a per-pair Gram matrix every step:
   `S_A = A @ A.T + δI` and/or `S_B = B.T @ B + δI`.
2. They invert (or invert-and-square-root) that Gram per step:
   - `adam-scaled-lora` does `solve_spd(S_B, ...)` via Cholesky.
   - `adam-polar-product-lora` does `spd_frac_power_inv(S_A, gamma=0.5)` via
     `torch.linalg.eigh`.

Cholesky launches one GPU kernel; `eigh` on a 256×256 matrix launches several
(tridiagonalize → divide-and-conquer → back-transform). With ~112 LoRA pairs
in OLMo-2-1B and 2 Gram-inversions per pair per step, kernel-launch overhead
is the dominant cost on small (r×r) matrices. That gap explains the 2.9× vs
4.4× difference between `adam-scaled-lora` and `adam-polar-product-lora`.

## Engineering claim

`S_A` and `S_B` change slowly across steps — they are quadratic functions of
the LoRA factors, which themselves drift slowly under bf16 Adam updates at
ηₛ ≈ 1e-4–1e-3. Recomputing `S_A^{-1/2}` and `S_B^{-1}` every step is overkill;
a cached value reused for K steps should give an indistinguishable trajectory
at K small relative to the timescale on which `S` actually changes.

This is the same engineering pattern as Newton-Muon
(`docs/papers/newton_muon_2604.01472.pdf`, Du & Su 2026):

> "we maintain a running estimate of the second moment matrix ZZᵀ and
> recompute both this estimate and its inverse only periodically, rather than
> at every optimization step. Between refreshes, the cached inverse is
> reused. We compute the damped inverse (ZZᵀ + γIₙ)⁻¹ via a Cholesky
> factorization followed by triangular solves."

Newton-Muon's preconditioner is the input-activation second moment, not a
LoRA factor Gram, so the math is unrelated to ours; only the engineering
pattern transfers.

## Plan

### Phase 1 — stale preconditioner only

Add a `precond_refresh_every: int = 1` argument to `AdamScaledLoRA`,
`AdamLinLoRA`, `AdamPolarProductLoRA`, `AdamuonPolarProductLoRA`. When
`step % refresh_every == 0`, recompute `S_A^{-1/...}` / `S_B^{-1/...}` and
cache it in `pair_state[i]`. Otherwise reuse the cached value. Default = 1
(current behavior). Production sweep override: K = 5.

Implementation sketch (per pair, per step):

```python
state = self.pair_state[i]
state['step'] += 1
if state['step'] % self.precond_refresh_every == 1 or 'SA_inv' not in state:
    state['SA_inv']  = spd_frac_power_inv(A.float() @ A.float().T,
                                          gamma=0.5, eps=self.delta)
    state['SB_inv']  = spd_frac_power_inv(B.float().T @ B.float(),
                                          gamma=0.5, eps=self.delta)
SA_half_inv = state['SA_inv']
SB_half_inv = state['SB_inv']
```

About 10 lines per optimizer class. Memory: one extra `(r, r)` float32 per
factor per pair (`r²` × 4 bytes × 2 × 112 ≈ 60 MB at r=256). Acceptable.

### Phase 2 — replace `eigh` with Cholesky-based S^{-1/2} (only if Phase 1 isn't enough)

`adam-polar-product-lora` needs `S^{-1/2}` (a non-trivial fractional power),
not `S^{-1}`. Two routes:
- **Higham iteration** for inverse square root: 5–7 matmuls of (r, r)
  matrices, fully GPU-friendly. Replaces `eigh + diag-pow + reconstruct`.
- **Cholesky `S = LLᵀ`** + use `L⁻ᵀ` in place of `S^{-1/2}` where the
  spectral-product geometry permits. Note: `L⁻ᵀ` is NOT `S^{-1/2}` (Cholesky
  factor ≠ matrix square root unless `S` is diagonal), so this changes the
  algorithm's geometry. Acceptable only if it stays in the same equivalence
  class for the polar-product update — needs separate analysis. Default
  recommendation: Higham iteration; preserves the math.

### Validation A/B

Pick a single (optimizer, r) cell with a clean reference final loss in the
existing record. Adam-polar-product-lora r=64 η=3e-4 → 0.7453 single-seed
at 2k steps from `polar_product_2k` is an obvious choice.

Rerun at the same (optimizer, r, η, seed) with `precond_refresh_every=5`.
Acceptance: final eval loss matches the cached 0.7453 to within the
single-seed jitter you'd see if you reran the unmodified optimizer with
the same seed (set this expectation by also rerunning the unmodified
optimizer once and noting the gap). Diagnostic: also log the per-pair
Frobenius distance between cached `SA_inv` and the would-be-fresh `SA_inv`
right before each refresh — if this distance is small at K=5, the staleness
is harmless; if large, drop to K=2 or 3.

If Phase 1 alone closes most of the gap (`adam-polar-product-lora` r=256
goes from 4.4× → ~1.5–2× AdamW), stop. Phase 2 is only worth doing if
Phase 1 leaves significant residual cost.

## Out of scope

- Newton-Muon as a *new optimizer* for LoRA (preconditioning by input
  activations rather than LoRA-factor Gram). That's a separate algorithm,
  not a speedup of the existing ones, and it's not what this plan is about.
- Batched ops across pairs (`stack 112 r×r matrices, one batched eigh`).
  Higher engineering cost, blocked on pair-shape grouping (q/k/v at d,
  gate/up/down at 4d). Worth revisiting after Phase 1 if r=512+ becomes
  the comparison budget.

## Validation results (Phase 1)

Logged in `docs/experiment_log.md` (2026-05-01 entry). One-line summary
(single-seed, lr=3e-4, `adam-polar-product-lora` non-coupled, 2k steps):

- r=64: K=20 is harmless (Δ=+0.0008 vs K=1).
- r=256: K=5 matches K=1 within incomplete-K=1 data; K=10 costs +0.0071,
  K=20 costs +0.0099.
- higham at r=256 K=5 matches eigh K=5 (Δ=+0.0007). Wall-time speedup not
  measured cleanly in this round.

Phase 1 closes the engineering case for K=5 at the ranks tested. Phase 2
(higham swap of `eigh`) preserves quality at this single comparison point.
Whether higham delivers the expected wall-time speedup hasn't been measured.

K=1 r=256 cells (eigh truncated at step 1820, higham probe-crashed at step
800) re-running as `polar_K1_r256_rerun_2k` to close the data point.
