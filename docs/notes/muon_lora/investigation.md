# Muon-LoRA — beat AdamW investigation

Tracking the campaign to make a Muon variant **strictly beat** AdamW
(0.7579) and `adam-lin-lora` (0.7581) on the 2k-step r=16 LoRA fine-tune of
OLMo-2-1B on Magicoder. All numbers single-seed at the canonical 2k-step
horizon; multi-seed verification deferred.

## Result: adam-muon-lora wins at r=64; LoRA+ confounded the r=16 headline

Best-η eval loss at step 2000, by optimizer × rank × [LoRA+
multiplier](glossary.md#lora-setup) m. Sources: `lr_sweep_2k`,
`optim_compare_high_eta_2k`, `new_optimizers_high_eta_2k`,
`adam_muon_clean_2k`, `adam_muon_r64_m4_2k`.

| optimizer        | r  | best η | m  | eval loss @ 2000 |
|------------------|----|--------|----|------------------|
| adam-muon-lora       | 64 | 3e-3   | 1  | **0.7515**       |
| adam-muon-lora       | 16 | 3e-3   | 4  | 0.7557           |
| adam-lin-lora        | 16 | 1e-3   | 1  | 0.7581           |
| adam-scaled-lora     | 16 | 1e-3   | 1  | 0.7572           |
| adamw                | 64 | 3e-4   | 1  | 0.7550           |
| adamw                | 16 | 3e-4   | 1  | 0.7579           |
| adam-muon-lora       | 16 | 3e-3   | 1  | 0.7624           |
| muon-lora            | 16 | 3e-3   | 1  | 0.7675           |

Reading the table: at r=64 with m=1, `adam-muon-lora` lands at 0.7515,
below AdamW r=64 (0.7550) by 0.0035. At r=16 with m=1, it lands at
0.7624 — above AdamW r=16 (0.7579) by 0.0045. The original 0.7557 r=16
headline came mostly from LoRA+ (m=4), not from per-factor
Newton–Schulz orthogonalization of Adam's preconditioned direction. The
r=16 m=4 η-sweep is pinned at the upper end ({1e-3, 3e-3}), so the true
optimum at m=4 is unknown and the entry is reported with that caveat.

The mechanism for `adam-muon-lora`: Newton–Schulz orthogonalization
applied to Adam's per-element preconditioned direction
$\hat m / (\sqrt{\hat v} + \varepsilon)$, per-factor independently. The
cleanest single-seed below-AdamW result for this family is at r=64. The
lowest single-seed loss in the project overall is
`adam-polar-product-lora` at r=64 (0.7453), a [Hybrid Picard /
spectral-product](glossary.md#optimizer-concepts) variant; the
spectral-product geometry beats per-factor NS on the same
Adam-then-spectral-correction composition.

**Diagnosis of independent per-factor NS as a proximal geometry.** It
does not produce a spectrally-bounded product update
$\Delta W = (\alpha/r)(B\,\delta A + \delta B\,A)$, breaks gauge
invariance under $A \to RA,\, B \to BR^{-1}$, and at PEFT's $B = 0$ init
makes NS on $m_A$ wasted (since $B\,\delta A \approx 0$ contributes
nothing to $\Delta W$ early). The fix lives in the spectral-product
geometry (see `../polar_product/investigations.md`).

## H1 — B-init asymmetry (LoRA+ for Muon)

PEFT inits $B = 0$, so for early steps $\delta B \cdot A$ dominates
$\Delta W$ and $A$'s update is wasted unless $B$ grows fast. A
[LoRA+](glossary.md#lora-setup) multiplier $m$ on $B$'s lr should close
the gap.

Sweep `muon_loraplus_2k` (η ∈ {1e-3, 3e-3, 1e-2} × m ∈ {4, 16}) ran to
2k; m=1 baseline reused from `muon_lowlr_2k` and
`new_optimizers_high_eta_2k`. Final eval loss @ 2000:

| η     | m=4    | m=16   |
|-------|--------|--------|
| 1e-3  | 0.7674 | 0.7679 |
| 3e-3  | 0.7759 | 0.8460 |
| 1e-2  | 0.9611 | 1.4209 |

Best is η=1e-3 m=4 → 0.7674, essentially tied with the m=1 baseline
(0.7675 at η=3e-3). H1 is real but only reparameterizes — m shifts the
optimal η downward by ~1/m, but the achievable floor is unchanged. H1 is
confirmed but does not on its own beat AdamW (0.7579). LoRA+ is a free
1-line tweak so we keep `lr_b_multiplier=4` as a sensible default in
subsequent configs; the lever for beating AdamW has to come from
geometry (H2) or Adam direction (H4). A 500-step pilot
(`muon_loraplus_500`) ranked η identically to the 2k results, confirming
500-step pilots are reliable proxies for η-ranking selection only.

## H2 — Wrong proximal geometry (ProductMuonLoRA)

`ProductMuonLoRA` orthogonalizes the merged-$W$ direction projected
onto the LoRA subspace rather than the factors independently:
form a gauge-invariant rank-r proxy
$D = (1/\text{scale})\,m_B\,(AA^\top + \delta I)^{-1}\,A$, NS via thin
QR plus a small-r NS core, then recover $(\delta A, \delta B)$ by
[Sylvester gauge lift](glossary.md#channel-coordinates-and-lift-terms)
(mirrors `AdamLinLoRA`). At $B = 0$ init, the min-Frobenius Sylvester
solution correctly sets $\delta A = 0$, so the LoRA+ asymmetric
multiplier becomes redundant.

The 500-step pilot (`product_muon_500`, η ∈ {3e-4, 1e-3, 3e-3, 1e-2} ×
m ∈ {1, 4}) showed two structural facts. First, no divergence at
η=1e-2 (vs `muon-lora` m=16 at the same η, which ran to 1.32) — the
product-spectral cap correctly bounds $\|\Delta W\|_\text{op}$ at large
η. Second, m=1 wins at high η and m=4 only helps at low η, consistent
with the algorithm's automatic gauge handling. Slope-based extrapolation
to step 2000 lands at ~0.760, between the H1 floor (0.7674) and AdamW
(0.7579) — geometrically correct but insufficient to strictly beat AdamW
on its own. The natural follow-up was the H2 ⊗ H4 hybrid below.

## H3 — NS irrelevance sanity (falsified)

If `ns_steps=0` (raw momentum SGD) matched `muon-lora`, the
orthogonalization would not be pulling its weight. Sweep
`muon_nsoff_2k` (η ∈ {3e-4, 1e-3, 3e-3, 1e-2}, m=1, ns_steps=0)
final eval loss @ 2000:

| η     | loss   |
|-------|--------|
| 3e-4  | 1.1844 |
| 1e-3  | 1.1658 |
| 3e-3  | 1.0672 |
| 1e-2  | 0.9499 |

Best is η=1e-2 → 0.9499, vs `muon-lora` with NS at 0.7675. NS
contributes ≥ 0.18 nats — clearly essential. H3 falsified;
geometric orthogonalization is essential and no pivot away from the
Muon framework is needed on these grounds.

## H4 — Adam direction + Muon spectral cap (AdamMuonLoRA)

`AdamMuonLoRA` applies NS to Adam's $\hat m / (\sqrt{\hat v} +
\varepsilon)$ per-factor instead of raw momentum, decoupling diagonal
preconditioning from spectral capping. Cheaper than `ProductMuonLoRA` —
no Sylvester recovery; direction is per-factor.

500-step pilot (η ∈ {1e-4, 3e-4, 1e-3, 3e-3} × m ∈ {1, 4}) selected
η=3e-3 m=4 (0.7962 at step 500). Extension probes confirmed the
η-optimum: η=1e-2 m=1 → 0.7986, η=1e-2 m=4 → 0.8327, η=3e-2 m=4 →
1.2017 (diverging). The 2k confirmation at the selected configs:

| η   | m | step 2000 |
|-----|---|-----------|
| 1e-3 | 4 | 0.7670 |
| 3e-3 | 4 | **0.7557** |

Best is **0.7557**, strictly below AdamW (0.7579) and `adam-lin-lora`
(0.7581). H4 confirmed and decisive at r=16 m=4. Subsequent
disentanglement (`adam_muon_clean_2k`) showed the m=1 result at r=16 is
0.7624 — the headline win is partly LoRA+, but the r=64 m=1 result
(0.7515) carries the family without LoRA+.

## H4-reverse — MuonAdamLoRA (NS-first, Adam-second), with caveat

`AdamMuonLoRA` does Adam → NS. The reverse — NS the per-factor raw
gradient first, then run Adam EMA + per-element preconditioning on the
NS output — is "H4-reverse" in this doc. Different mechanism: NS already
kills magnitude variation across the spectrum, but per-element entries
still vary, so Adam's $\hat v$ on the NS output may add a useful
second-stage correction.

`muon_adam_2k` (η ∈ {1e-3, 3e-3} × m ∈ {1, 4}) at step 1600:

| η     | m=1     | m=4    |
|-------|---------|--------|
| 1e-3  | 0.7851  | 0.8846 (stuck) |
| 3e-3  | 1.0087 (rising) | 1.5766 (diverged) |

This is **not** a clean falsification of the polar-first composition
family — the implementation here is the *unstabilized* port that the
AdaMuon paper warns against. It is missing all three of AdaMuon's
stabilizers:

1. **`sign(M_t)` before NS.** This implementation feeds NS the raw
   gradient; AdaMuon feeds NS the sign of the momentum buffer.
2. **Only $V_t$ on the NS output, no $M_t$ on it.** This implementation
   runs full Adam $(m, v)$ on the NS output — double smoothing.
3. **No RMS-align.** Step magnitude is unbounded vs AdaMuon's
   $\gamma_t = 0.2 \sqrt{mn} / \|\tilde O_t\|_F$.

The originally-cited mechanism ("uncorrelated NS outputs cancel
$\hat m$, inflate $\sqrt{\hat v}$") describes exactly this port's
failure mode, which is exactly what AdaMuon's stabilizers were designed
to prevent. The proper test of the polar-first composition family is the
[AdaMuon-faithful](glossary.md#optimizer-composition) implementations:
`adamuon_lora_2k` (vanilla NS, AdaMuon-faithful) and
`adamuon_polar_product_2k` (AdaMuon-faithful with spectral-product
geometry instead of vanilla NS) — those are the load-bearing log groups
for the polar-first family. Until those are read, treat the polar-first
family as **open**, not falsified.

## H2 ⊗ H4 hybrid — AdamProductMuonLoRA

Combines `ProductMuonLoRA`'s gauge-invariant geometric pipeline (the
$D$-proxy, NS in factored form, Sylvester recovery) with
`AdamLinLoRA`'s Adam EMA on the recovered $(\text{precond}_A,
\text{precond}_B)$. Pilot group `adam_product_muon_500` (η ∈ {3e-4,
1e-3, 3e-3, 1e-2} × m ∈ {1, 4}). The motivation is to combine the two
mechanisms that separately got the closest to AdamW: H2 (geometry,
removes divergence at high η) and H4 (Adam preconditioning, the headline
H-family win).

## Code

- Optimizers: `lora_playground/optim.py` — `MuonLoRA` (extended),
  `ProductMuonLoRA`, `AdamMuonLoRA`
- CLI: `--optimizer {muon-lora,product-muon-lora,adam-muon-lora}`,
  `--lora_plus_multiplier <m>`, `--muon_ns_steps <k>` (0 disables NS)
- Tests: `tests/test_new_optimizers.py` — NS scale-invariance, LoRA+
  B-step scaling, gauge invariance for `ProductMuonLoRA`, ns=0 =
  momentum-SGD identity
- Sweep configs:
  `params/{muon_loraplus,muon_nsoff,product_muon,adam_muon}_2k.json`
- Sweep script: `scripts/sweep/sweep_muon_2k.sh`
- Notebook: `notebooks/sweep_analysis.ipynb` — section "Muon-LoRA
  variants — beat AdamW campaign"
