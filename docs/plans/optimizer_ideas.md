# Optimizer Ideas

New optimizer candidates to implement and compare against AdamW LoRA and PSI-LoRA baselines.
Baselines to implement first: PSI-LoRA (from psi_lora_2602.16456.pdf) and GaLore (from galore_2403.03507.pdf).

---

## Muon-LoRA

Apply Newton-Schulz (NS) orthogonalization to the momentum buffer of each LoRA factor before
applying the update. Muon was proposed for pretraining and showed strong results; application
to LoRA fine-tuning is unexplored.

**Update rule** (per LoRA pair A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}):

    m_A ← β m_A + (1 − β) G_A
    m_B ← β m_B + (1 − β) G_B
    A   ← A − η · NS(m_A)
    B   ← B − η · NS(m_B)

where NS(X) = Newton-Schulz orthogonalization (5 iterations of X ← 1.5X − 0.5 X X^T X
for square; for rectangular X ∈ ℝ^{r×d}, gives matrix with orthonormal rows).

**Memory**: m_A + m_B = r(d_in + d_out) — half of AdamW (no second moment).

**Motivation**: orthogonalizing the update gives each of the r LoRA dimensions equal-magnitude
steps, preventing rank collapse where a few singular values dominate and the effective rank
drops below r. No hooks required.

**Questions to answer**:
- Does orthogonalization in factor space (A, B separately) meaningfully differ from AdamW?
- Is the update sensitive to the NS iteration count?
- Does it help more at larger r where rank collapse is a bigger risk?

---

## Full r×r KFAC-LoRA

PSI-LoRA's diagonal K-FAC (D_U ∈ ℝ^{d_out}, D_V ∈ ℝ^{d_in}) captures per-neuron activation
magnitudes but ignores correlations between the r LoRA dimensions. These r×r correlations are
cheap to track (r² ≪ r·d) and capture the curvature of the loss within the LoRA subspace.

**State** (beyond LoRA parameters):

    H_A = EMA(G_A G_A^T) ∈ ℝ^{r×r}   # left Kronecker factor for A; = EMA(B^T G_W G_W^T B)
    H_B = EMA(G_B^T G_B) ∈ ℝ^{r×r}   # right Kronecker factor for B; = EMA(A G_W^T G_W A^T)
    D_V = EMA(diag(X^T X / B)) ∈ ℝ^{d_in}   # from forward hook (layer inputs X)
    D_U = EMA(diag(S^T S / B)) ∈ ℝ^{d_out}  # from backward hook (output grads S)

**Update** (fractional power γ, following PSI-LoRA paper; γ = 0.5 as default):

    ΔA = H_A^{−γ} G_A D_V^{−γ}
    ΔB = D_U^{−γ} G_B H_B^{−γ}
    A ← A − η ΔA
    B ← B − η ΔB

**Memory**: 2r² + d_in + d_out. For r=16, d=2048: ≈ 4600 vs AdamW's ≈ 131K.
Memory is small because r ≪ d — not a deep insight, just follows from LoRA's design.

**Motivation**: PSI-LoRA baseline: diagonal only (misses r×r); AdamW: diagonal only (misses
r×r AND ignores layer activation magnitudes). This method has BOTH. Key question is whether
the r×r curvature information matters empirically beyond the diagonal.

**Requires**: forward/backward hooks (same infrastructure as PSI-LoRA).

**Ablations**:
1. D_U, D_V only (= PSI-LoRA diagonal baseline, no r×r)
2. H_A, H_B only (= r×r only, no layer-level scaling)
3. Both (= full KFAC-LoRA)
4. With and without momentum

**Questions to answer**:
- Does the r×r factor help beyond PSI-LoRA's diagonal?
- Best fractional power γ (try 0.25, 0.5, 1.0)?
- Is it stable without additional proximal regularization?

---

## Implementation Order

1. Add forward/backward hook infrastructure to `train.py` (needed by KFAC-LoRA and PSI-LoRA)
2. Implement Muon-LoRA (no hooks needed, simpler)
3. Implement PSI-LoRA K=1 diagonal (= Scaled PSI-LoRA from paper, baseline for KFAC-LoRA)
4. Implement KFAC-LoRA (adds r×r factors on top of PSI-LoRA infrastructure)

Sweep all against AdamW LoRA at η=3e-4, r=16, 2000 steps, OLMo-2-0425-1B + Magicoder.
