# Glossary — LoRA optimizer project

Quick definitions of recurring project-specific terms. When a doc uses one of these for the first time, it should link here rather than redefining inline.

## LoRA setup

- **LoRA pair.** $A \in \mathbb{R}^{r \times n}$, $B \in \mathbb{R}^{m \times r}$. Effective weight $W + (\alpha/r)\,BA$. Project default: $\alpha = r$ (so $\alpha/r = 1$, absorbed). PEFT init: $A$ Kaiming-normal, $B = 0$.
- **PEFT convention.** $A$ is the right factor (input side, $r \times n$), $B$ is the left factor (output side, $m \times r$). Adapter contribution = $BA$. Throughout this project's docs and code; do not mix with Hu-et-al ordering.
- **Rank $r$.** Project sweeps $r \in \{16, 64\}$ as the canonical comparison; $r=128, 256$ used for rank-extension diagnostics. Multi-seed deferred; all reported numbers single-seed at the canonical 2k-step horizon.
- **LoRA+ multiplier $m$.** Independent learning-rate scale on the $B$ factor (LoRA+: $\eta_B = m \cdot \eta_A$). Project default: $m = 1$ (no LoRA+). $m \ne 1$ entangles the optimizer effect with LoRA+ effect; report both axes if used.

## Common shorthand

Used throughout the algorithm pseudocode below. Per LoRA pair, per step:

- $G_A := \nabla_A F \in \mathbb{R}^{r \times n}$, $G_B := \nabla_B F \in \mathbb{R}^{m \times r}$ — raw factor gradients.
- $S_A := A A^\top + \delta I \in \mathbb{R}^{r \times r}$, $S_B := B^\top B + \delta I \in \mathbb{R}^{r \times r}$ — Picard's spectral preconditioners; $\delta = 10^{-6}$ throughout the project.
- $u_A, u_B$ — Adam-preconditioned **covectors** (bias-corrected first/second moment): $u = \hat m / (\sqrt{\hat v} + \varepsilon)$. Used in place of raw $G$ as the linear cost; this breaks gradient compatibility.
- $\eta$ — learning rate.
- $B = Q_L R_L$, $A^\top = Q_R R_R^\top$ — thin QR factorizations; $Q_L \in \mathbb{R}^{m \times r}$ and $Q_R \in \mathbb{R}^{n \times r}$ column-orthonormal, $R_L, R_R \in \mathbb{R}^{r \times r}$ upper-triangular.
- $\mathrm{polar}(M) := UV^\top$ for compact SVD $M = U\Sigma V^\top$ — the operator-norm-saturate direction. Implemented in this repo by **Newton-Schulz (NS)** iteration (see `_newton_schulz` in `lora_playground/optim.py:1449`); NS returns a Frobenius-norm-preserving polar approximation, not the magnitude-preserving one.
- $\mathrm{clip}_\tau(M) := U \cdot \mathrm{diag}(\min(\sigma_i, \tau)) \cdot V^\top$ — singular-value clipping; the exact prox of the spectral-norm-ball Frobenius projection.

---

## Optimizer algorithms

Every named optimizer in `OPTIMIZER_CHOICES` (`lora_playground/optim.py:177`) gets an explicit per-step rule below, organized by family. Cross-references point to `lora_playground/optim.py:<class-name>` for the implementing code and to `build_optimizer()` (`optim.py:4275`) for the CLI-name → class mapping. All optimizer math runs in float32; updates are cast back to parameter dtype before applying.

### Baselines

#### AdamW (`adamw`)

The project baseline. PyTorch's `torch.optim.AdamW` applied uniformly to all trainable LoRA factors (no factor-aware geometry). At $r=16$, $\eta=3\mathrm{e}{-4}$ this lands at eval loss 0.7579 over 2k steps — the bar all custom optimizers must clear.

```text
state per parameter θ: m, v, step
input: G = ∇θ F
m   ← β₁·m + (1−β₁)·G
v   ← β₂·v + (1−β₂)·G⊙G
m̂   = m / (1 − β₁^step)
v̂   = v / (1 − β₂^step)
Δθ  = -η · m̂ / (√v̂ + ε)   − η · wd · θ      (decoupled weight decay; project default wd=0)
θ   ← θ + Δθ
```

#### Plain Adam / SGD / SGD-M (`sgd`, `sgd-m`)

`sgd`: PyTorch SGD with `momentum=0` on all trainable parameters. `sgd-m`: same with `momentum=0.9`. Reference implementations; not used as competitive baselines.

```text
sgd:    θ ← θ − η·G
sgd-m:  v ← μ·v + G;  θ ← θ − η·v        (μ = 0.9)
```

#### LoRAPlusAdamW (`adamw` with `--lora_plus_multiplier ≠ 1`, class `LoRAPlusAdamW`, `optim.py:1318`)

AdamW with split learning rate: `lr_A = η`, `lr_B = m·η`. `m=1` collapses to plain AdamW (project default). $m\ne 1$ entangles optimizer effect with LoRA+ effect.

### Pre-Adam compositions (geometry → Adam)

Pattern: apply a LoRA-aware geometric solve to the **raw** gradient, then run Adam on the resulting direction. The geometric structure is partly destroyed by Adam's per-coordinate $\sqrt{\hat v}$ rescale (the H1 critique).

#### ScaledLoRA (`scaled-lora`, `optim.py:226`)

Plain SGD with Gram preconditioning, no Adam. Per pair:

```text
S_B = B^T·B + δI;  S_A = A·A^T + δI                  (r×r)
ΔA = −η · S_B^{-1} · G_A
ΔB = −η · G_B · S_A^{-1}
```

#### LinLoRA (`lin-lora`, `optim.py:287`)

SGD with Sylvester gauge solve (the linearized least-squares min-Frobenius lift on the LoRA tangent). Per pair:

```text
S_B = B^T·B + δI;  S_A = A·A^T + δI
RHS = −η · (G_A · A^T)
solve Sylvester:  S_B·K + K·S_A = RHS              (K ∈ ℝ^{r×r})
ΔA = − S_B^{-1} · (η·G_A + K·A)
ΔB = − (η·G_B + B·K) · S_A^{-1}
```

#### AdamLinLoRA (`adam-lin-lora`, `optim.py:359`)

Sylvester gauge solve on raw gradients, then Adam on the resulting factor direction. The class docstring flags this as gauge-incoherent but kept as an empirical baseline.

```text
state per pair: m_A, v_A, m_B, v_B, step
S_B = B^T·B + δI;  S_A = A·A^T + δI
RHS = −(G_A·A^T)
solve  S_B·K + K·S_A = RHS                          (Sylvester, lr-free)
v_A = S_B^{-1} · (G_A + K·A)                        # preconditioned direction
v_B = (G_B + B·K) · S_A^{-1}
# Adam on (v_A, v_B):
m_A ← β₁·m_A + (1−β₁)·v_A;  v²_A ← β₂·v²_A + (1−β₂)·v_A⊙v_A
m_B ← β₁·m_B + (1−β₁)·v_B;  v²_B ← β₂·v²_B + (1−β₂)·v_B⊙v_B
ΔA = −η · m̂_A/(√v̂_A + ε);   ΔB = −m·η · m̂_B/(√v̂_B + ε)
```

#### AdamScaledLoRA (`adam-scaled-lora`, `optim.py:723`)

Gram solve on raw gradients, then Adam on the result.

```text
state per pair: m_A, v_A, m_B, v_B, step
S_B = B^T·B + δI;  S_A = A·A^T + δI                (cached as Cholesky, refresh every K steps)
v_A = S_B^{-1}·G_A;   v_B = G_B·S_A^{-1}
# Adam EMA on (v_A, v_B):
m_A ← β₁·m_A + (1−β₁)·v_A;  v²_A ← β₂·v²_A + (1−β₂)·v_A²    (analog for B)
ΔA = −η · m̂_A/(√v̂_A + ε);   ΔB = −η · m̂_B/(√v̂_B + ε)
```

#### AdamLinCoreLoRA (`adam-lin-core-lora`, `optim.py:565`)

Variant of AdamLinLoRA that maintains Adam state in the **core/tangent** representation of the linearized step rather than per-factor. Classified internally as the "principled Adam-of-LinLoRA" but it has not won on the leaderboard. See `optim.py:565` docstring for the exact channel-coordinate state.

### Post-Adam compositions (Adam → geometry)

Pattern: run Adam on raw gradients to obtain the unitless covector $u$, then apply LoRA geometry to $u$. Adam's per-coordinate scaling is **upstream** of the geometry, so the geometry is not erased.

#### AdamLinLoRAPost (`adam-lin-lora-post`, `optim.py:1011`)

```text
state per pair: m_A, v_A, m_B, v_B, step
m_A ← β₁·m_A + (1−β₁)·G_A;  v_A ← β₂·v_A + (1−β₂)·G_A²    (analog for B)
u_A = m̂_A/(√v̂_A + ε);  u_B = m̂_B/(√v̂_B + ε)
S_B = B^T·B + δI;  S_A = A·A^T + δI
RHS = −γ·(u_A · A^T)                              (γ = (d_in/r)^{1/2} if scaled_metric else 1)
solve  S_B·K + γ²·K·S_A = RHS
geo_A = − S_B^{-1} · (u_A + γ·K·A)                 (lr-free direction)
geo_B = − (u_B + (1/γ)·B·K) · S_A^{-1}
# RMS-align magnitude (cribbed from AdaMuon):
ΔA = η · ‖u_A‖_F / ‖geo_A‖_F · geo_A
ΔB = m·η · ‖u_B‖_F / ‖geo_B‖_F · geo_B
```

#### AdamScaledLoRAPost (`adam-scaled-lora-post`, `optim.py:890`)

Adam covector → Gram solve → RMS-align.

```text
m_A ← β₁·m_A + (1−β₁)·G_A;  v_A ← β₂·v_A + (1−β₂)·G_A²   (analog for B)
u_A = m̂_A/(√v̂_A + ε);  u_B = m̂_B/(√v̂_B + ε)
S_B = B^T·B + δI;  S_A = A·A^T + δI
geo_A = −S_B^{-1}·u_A;   geo_B = −u_B·S_A^{-1}
ΔA = η · ‖u_A‖_F / ‖geo_A‖_F · geo_A
ΔB = m·η · ‖u_B‖_F / ‖geo_B‖_F · geo_B
```

### Matrix-Adam (per-pair scalar second moment)

Replace per-coordinate $\hat v$ with one scalar per LoRA pair. Tested as the H5 lever (whether per-coord $\sqrt{\hat v}$ is what shreds geometric structure).

#### AdamScaledLoRAMatrix (`adam-scaled-lora-matrix`, `optim.py:1159`)

```text
state per pair: m_A, m_B, v_pair (scalar), n_total = numel(A)+numel(B), step
S_B = B^T·B + δI;  S_A = A·A^T + δI
v_A_dir = S_B^{-1}·G_A;   v_B_dir = G_B·S_A^{-1}
m_A ← β₁·m_A + (1−β₁)·v_A_dir;   m_B ← β₁·m_B + (1−β₁)·v_B_dir
sqmean = (‖v_A_dir‖² + ‖v_B_dir‖²) / n_total
v_pair ← β₂·v_pair + (1−β₂)·sqmean
denom = √(v_pair / (1 − β₂^step)) + ε
ΔA = −η/denom · m̂_A;   ΔB = −η/denom · m̂_B
```

#### AdamLinLoRAMatrix (`adam-lin-lora-matrix`, `optim.py:1238`)

Same scalar-$v$ idea applied to AdamLinLoRA's Sylvester direction. See class for full pseudocode; structurally identical to AdamScaledLoRAMatrix with the LinLoRA Sylvester replacing the Gram solve.

### Muon family

Newton-Schulz orthogonalization (canonical Muon: pre-normalize, run NS, do **not** scale back to input magnitude).

#### MuonLoRA (`muon-lora`, `optim.py:1472`)

Raw momentum + per-factor NS, no Adam.

```text
state per pair: m_A, m_B
m_A ← β·m_A + (1−β)·G_A;   m_B ← β·m_B + (1−β)·G_B
D_A = NS(m_A, T)  if ns_steps > 0 else m_A
D_B = NS(m_B, T)  if ns_steps > 0 else m_B
ΔA = −η · D_A;   ΔB = −m·η · D_B
```

#### AdamMuonLoRA (`adam-muon-lora`, `optim.py:1656`)

Adam covector then per-factor NS — the cheap analog of AdamLinLoRA in Muon space (Tier 4 of the Muon campaign).

```text
m_A ← β₁·m_A + (1−β₁)·G_A;  v_A ← β₂·v_A + (1−β₂)·G_A²   (analog for B)
adam_A = m̂_A/(√v̂_A + ε);    adam_B = m̂_B/(√v̂_B + ε)
D_A = NS(adam_A, T);          D_B = NS(adam_B, T)
ΔA = −η · D_A;                ΔB = −m·η · D_B
```

#### MuonAdamLoRA (`muon-adam-lora`, `optim.py:2534`)

Reverse order: NS first, then Adam.

```text
ns_A = NS(G_A, T);  ns_B = NS(G_B, T)
m_A ← β₁·m_A + (1−β₁)·ns_A;   v_A ← β₂·v_A + (1−β₂)·ns_A²    (analog for B)
ΔA = −η · m̂_A/(√v̂_A + ε);    ΔB = −m·η · m̂_B/(√v̂_B + ε)
```

#### AdaMuonLoRA (`adamuon-lora`, class `AdaMuonLoRA`, `optim.py:2404`)

Faithful port of AdaMuon (arxiv 2507.11005, Algorithm 1) to per-factor LoRA — sign-stabilized polar with variance on the polar output and RMS-aligned magnitude. **No bias correction** on the EMA (matches canonical Muon recipe).

```text
state per factor X ∈ {A, B}: M, V, step
M_X ← β·M_X + G_X                                       # plain SGD momentum (no (1−β) factor)
O_X = NS(sign(M_X), T)                                  # sign-stabilized polar (paper Thm 1)
V_X ← β·V_X + (1−β)·O_X⊙O_X                             # variance on polar output
Õ_X = O_X / (√V_X + ε)                                  # elementwise normalize, no bias-corr
γ_X = 0.2·√(rows·cols) / ‖Õ_X‖_F                        # RMS-align to Adam magnitude
ΔX = −η·γ_X·Õ_X     (apply m·η for X = B)
```

#### ProductMuonLoRA (`product-muon-lora`, `optim.py:1535`)

Gauge-invariant rank-$r$ proxy of the merged-weight gradient → NS via thin QR on the small $r\times r$ core → Sylvester recovery. Uses only $G_B$ (the $G_A$ channel would re-introduce gauge dependence).

```text
state per pair: m_left  (∈ ℝ^{m×r})
S_A = A·A^T + δI                                        (r×r)
Z   = solve_spd(S_A, A)                                 # (A·A^T+δI)^{-1}·A   ∈ ℝ^{r×n}
m_left ← β·m_left + (1−β)·G_B
left  = (1/scale)·m_left                                # rank-r merged-direction proxy:  D = left @ Z
right = Z
# Thin QR + small NS on the r×r core
Q_L, R_L = qr(left);  Q_R, R_R = qr(right^T)
C  = R_L · R_R^T                                        (r×r)
C_ns = NS(C, T)  if ns_steps > 0 else C
# Sylvester recovery so that B·δA + δB·A ≈ −η·(Q_L·C_ns·Q_R^T):
grad_A_eq = (B^T·Q_L · C_ns)·Q_R^T                      (r, n)
grad_B_eq = Q_L · (C_ns · Q_R^T·A^T)                    (m, r)
S_B = B^T·B + δI
solve  S_B·K + K·S_A = −(grad_A_eq · A^T)
ΔA = −η · S_B^{-1}·(grad_A_eq + K·A)
ΔB = −m·η · (grad_B_eq + B·K)·S_A^{-1}
```

#### AdamProductMuonLoRA (`adam-product-muon-lora`, class `AdamProductMuonLoRA`, `optim.py:1720`)

H2⊗H4 hybrid: ProductMuonLoRA's gauge-invariant geometry **followed by** Adam EMA on the recovered factor steps.

```text
# Steps 1–4 identical to ProductMuonLoRA but no momentum on left (left = G_B/scale).
# Recover (precond_A, precond_B) via Sylvester as above.
m_A ← β₁·m_A + (1−β₁)·precond_A;   v_A ← β₂·v_A + (1−β₂)·precond_A²    (analog for B)
ΔA = −η · m̂_A/(√v̂_A + ε)
ΔB = −m·η · m̂_B/(√v̂_B + ε)
```

#### AdamuonPolarProductLoRA (`adamuon-polar-product-lora`, class `AdamuonPolarProductLoRA`, `optim.py:2229`)

**Polar-first** composition with the spectral-product geometry: plain momentum, optional sign-stabilization (AdaMuon Thm 1), polar-product geometry on the (signed) momentum, variance accumulated on the polar output, RMS-align.

```text
state per pair: m_A, m_B, v_A, v_B, step
m_A ← β₁·m_A + (1−β₁)·G_A;   m_B ← β₁·m_B + (1−β₁)·G_B
sA = sign(m_A)  if sign_stabilize else m_A;   sB = sign(m_B)  if sign_stabilize else m_B
S_A^{−1/2}, S_B^{−1/2} via eigh or Higham NS               (refresh every K steps)
P_B = NS(sB · S_A^{−1/2}, T);   D_B = P_B · S_A^{−1/2}      (∈ ℝ^{m×r})
P_A = NS(S_B^{−1/2} · sA, T);   D_A = S_B^{−1/2} · P_A      (∈ ℝ^{r×n})
v_A ← β₂·v_A + (1−β₂)·D_A⊙D_A;   v_B ← β₂·v_B + (1−β₂)·D_B⊙D_B
Õ_A = D_A / (√(v_A / (1 − β₂^step)) + ε);    (analog for B)
target_A = 0.2·√(r·n);  target_B = 0.2·√(m·r)              (AdaMuon §3.3)
γ_A = target_A / ‖Õ_A‖_F;   γ_B = target_B / ‖Õ_B‖_F
ΔA = −η·γ_A·Õ_A;             ΔB = −m·η·γ_B·Õ_B
```

### Polar-product / hybrid Picard family (project leaders)

The strongest optimizer family in this repo. **Adam-first** composition: Adam EMA on raw gradients → spectral-product polar block solve → optional cross-coupling Picard iterations → RMS-align.

#### AdamPolarProductLoRA (`adam-polar-product-lora`, `optim.py:1892`)

Uncoupled (`picard_iters = 1`) variant. Class is shared with the coupled variant; CLI distinguishes via `build_optimizer()` (`optim.py:4439`). Eval = 0.7546 at $r=16$, $\eta=3\mathrm{e}{-4}$ (current $r=16$ leader).

```text
state per pair: m_A, v_A, m_B, v_B, step
# Adam EMA on RAW gradient
m_A ← β₁·m_A + (1−β₁)·G_A;   v_A ← β₂·v_A + (1−β₂)·G_A²        (analog for B)
u_A = m̂_A/(√v̂_A + ε);   u_B = m̂_B/(√v̂_B + ε)
# Spectral-square-root preconditioners (refresh every K steps)
S_A^{−1/2} = (A·A^T + δI)^{−1/2};   S_B^{−1/2} = (B^T·B + δI)^{−1/2}
# Polar-product block solve (PolarProductLoRA structure, applied to u, not G):
P_B = NS(u_B · S_A^{−1/2}, T);   geo_B = P_B · S_A^{−1/2}
P_A = NS(S_B^{−1/2} · u_A, T);   geo_A = S_B^{−1/2} · P_A
# RMS-align step magnitude to ‖u‖_F (prevents σ_min(S)-driven drift)
ΔA = −η · (‖u_A‖_F / ‖geo_A‖_F) · geo_A
ΔB = −m·η · (‖u_B‖_F / ‖geo_B‖_F) · geo_B
```

#### AdamPolarProductLoRACoupled (`adam-polar-product-lora-coupled`, same class, `picard_iters = 2` by default)

Adds Picard fixed-point iteration on the joint normal equations. Eval = 0.7382 at $r=64$, $\eta=3\mathrm{e}{-4}$ (current $r=64$ leader); loses to AdamW at $r=16$ (0.7616 vs 0.7579).

The joint normal equations of the adjacent variational formulation (under spectral-product metric) are:

$$S_B \cdot \Delta A + B^\top \cdot \Delta B \cdot A = -\eta \cdot u_A,\qquad \Delta B \cdot S_A + B \cdot \Delta A \cdot A^\top = -\eta \cdot u_B.$$

Block-diagonal drops the cross-terms (this is `picard_iters=1`). Picard restores them by feeding the previous block iterate's contribution back as a correction:

```text
# Same Adam EMA + S^{−1/2} setup as uncoupled.
ΔA_prev ← 0;   ΔB_prev ← 0
for k in 1 .. picard_iters:
    if k == 1:
        u_A_eff = u_A;   u_B_eff = u_B
    else:
        u_A_eff = u_A + α·(B^T · ΔB_prev · A) / η                 # cross-coupling correction
        u_B_eff = u_B + α·(B · ΔA_prev · A^T) / η
    P_B = NS(u_B_eff · S_A^{−1/2}, T);   geo_B = P_B · S_A^{−1/2}
    P_A = NS(S_B^{−1/2} · u_A_eff, T);   geo_A = S_B^{−1/2} · P_A
    ΔA  = −η · (‖u_A_eff‖_F / ‖geo_A‖_F) · geo_A                  # standard mode
    ΔB  = −m·η · (‖u_B_eff‖_F / ‖geo_B‖_F) · geo_B
    ΔA_prev = ΔA;   ΔB_prev = ΔB
# Apply final ΔA, ΔB.
```

`picard_alpha` $\alpha$: cross-coupling damping. $\alpha = 1$ standard Picard (default); $\alpha = 0$ disables cross-term (collapses to `picard_iters=1`). Sweep at $r=16$ found interior $\alpha \in \{0.25, 0.5, 0.75\}$ all worse than both endpoints.

#### `adam-polar-product-lora-coupled-endrms` (same class, `end_rms_align=True`)

Variant where each Picard iterate is rescaled to the **original** $\|u_A\|_F$, $\|u_B\|_F$ (Adam-direction norms before the cross-term) rather than to $\|u_{A,\text{eff}}\|_F$. Prevents the cross-term from inflating step magnitude.

### Joint-core E-family (archive of the dead joint-operator-norm direction)

These all solve the joint operator-norm formulation (Case 3 of `polar_product/theory.md`) via the projected-quotient-polar core solver of `polar_product/theory.md`. Empirically falsified — see `polar_product/investigations.md` for full timeline. Listed for completeness; do **not** use for new work.

#### PolarCoupledCoreLoRA (`polar-coupled-core-lora`, `optim.py:3750`)

The §2.1 baseline solver — variant 1 of the Section 6 ladder. Pure projected-quotient-polar on raw factor gradients; no Adam, no momentum.

```text
# Build active core Ĥ_t from G_A, G_B and bases (Q_L, U) on col-space and (Q_R, V) on row-space.
# Π = projector that zeros the (2,2) block (forbidden corner) of an r×r core.
Compute Ĥ_t with the (C_L + C_R)/2 symmetrization (gauge-incoherent factor inputs are
projected back to the compatible subspace by averaging).
Z_+ = polar(Π(core_obj))                   # projected-quotient-polar
core_step = −η · ‖Ĥ_t‖_* · Z_+
Lift core_step → (ΔA, ΔB) via the §4 Sylvester gauge formula (min-Frobenius).
```

#### Joint-core E1–E8 variants (gauge / momentum / magnitude axes)

All wrap `PolarCoupledCoreLoRA` or related classes with one gauge / momentum / magnitude axis flipped. Refer to `polar_product/investigations.md` for evaluation; pseudocode follows the same scaffold as PolarCoupledCoreLoRA with one piece swapped:

| CLI name | class | what changes vs §2.1 baseline |
|---|---|---|
| `polar-coupled-core-imbalance-scalar-lora` | PolarCoupledCoreLoRA | `gauge="imbalance-preserve-scalar"` |
| `polar-coupled-core-imbalance-lora` | PolarCoupledCoreLoRA | `gauge="imbalance-preserve"` (iLoRA-style, fully matrix-valued) |
| `polar-coupled-core-imbalance-restore-lora` | PolarCoupledCoreLoRA | `gauge="imbalance-restore"` |
| `polar-coupled-core-balanced-scalar-lora` | PolarCoupledCoreLoRA | `gauge="balanced-scalar"` |
| `polar-coupled-core-state-rebalanced-lora` | PolarCoupledCoreLoRA | post-step `(A,B) ← (R^{-1}A, BR)` rebalance so $A A^\top \approx \rho B^\top B$ |
| `polar-coupled-core-sign-lora` | PolarCoupledCoreLoRA | `pre_polar_normalize="sign"` (per-coord sign-norm in core space, no EMA) |
| `polar-coupled-core-sign-rebalanced-lora` | PolarCoupledCoreLoRA | sign + state rebalance (compound) |
| `polar-coupled-core-factor-adam-lora` | PolarCoupledCoreFactorAdamLoRA (`optim.py:3836`) | factor-Adam on $G_A, G_B$ → core solver on $u_A, u_B$ |
| `polar-coupled-core-factor-adam-rebalanced-lora` | PolarCoupledCoreFactorAdamLoRA | factor-Adam + state rebalance |
| `muon-coupled-core-lora` | MuonCoupledCoreLoRA (`optim.py:3957`) | variant 1 + transported core EMA + Nesterov lookahead (canonical Muon) |
| `muon-coupled-core-imbalance-scalar-lora` / `-imbalance-lora` / `-balanced-scalar-lora` / `-state-rebalanced-lora` / `-sign-lora` / `-sign-rebalanced-lora` | MuonCoupledCoreLoRA | variant 2 + corresponding gauge / norm tweak |

There is also `MuonRMSCoupledCoreLoRA` (`optim.py:4119`) — variant 3, adds scalar RMS magnitude normalization on top of variant 2 — but it has no top-level CLI alias in `OPTIMIZER_CHOICES`.

### SVD oracle modes (full-finetune projected onto rank $r$)

Used with `--training_mode svd_step_oracle` or `svd_cumulative_oracle`; operate on dense target weights, not LoRA factors.

#### SVDStepAdamW (`svd-step-adamw`, `optim.py:1351`)

Per step, take a dense AdamW step on each target weight, then project the **per-step displacement** $\Delta W_t = \tilde W - W_t$ to rank $r$ via truncated SVD. Cumulative displacement from initialization is **not** rank-constrained.

```text
for each target weight W:
    W_before = clone(W)
    AdamW.step()                              # mutates W in place
    raw_delta = W − W_before
    Π_r(raw_delta) = truncated_svd(raw_delta, r)
    W ← W_before + Π_r(raw_delta)
```

#### SVDCumulativeAdamW (`svd-cumulative-adamw`, `optim.py:1397`)

AdamW proposals are accumulated in a full-rank float32 buffer $C_t$; the live weight is always $W_0 + \Pi_r(C_t)$, so cumulative displacement from initialization stays rank $r$.

```text
state per target: accumulator C  (full-rank float32, initialized to 0)
for each target weight W with base_weight W₀:
    W_before = clone(W)
    AdamW.step()
    raw_delta = W − W_before
    C ← C + raw_delta
    Π_r(C) = truncated_svd(C, r)
    W ← W₀ + Π_r(C)
```

### GaLore (subspace-projected Adam in dense space)

#### GaLoreAdamW (`galore-adamw`, `optim.py:2983`)

Project each dense gradient onto a rank-$r$ subspace (top singular vectors of $G$), run Adam in that subspace, project back. Faithful port of [galore_torch](https://github.com/jiaweizzhao/GaLore). For $W \in \mathbb{R}^{d_\text{out}\times d_\text{in}}$, `proj_type="std"`, $d_\text{out} \ge d_\text{in}$:

```text
state per W: P ∈ ℝ^{r × d_in}, m ∈ ℝ^{d_out × r}, v ∈ ℝ^{d_out × r}, step
every update_proj_gap steps:  P ← top-r right singular vectors of G
R = G · P^T                                         # project onto subspace
m ← β₁·m + (1−β₁)·R;  v ← β₂·v + (1−β₂)·R²
ΔW = −scale · m̂/(√v̂ + ε) · P                       # project back to full space
```

(For tall-vs-wide $W$ the projector lands on the smaller side; see class for both branches.)

### Diagonal K-FAC family

Layer-level diagonal statistics from forward inputs and backward output-gradients, updated via PyTorch hooks.

#### DiagScaledLoRA (`diag-scaled-lora`, `optim.py:2694`)

Pure diagonal K-FAC scaling on independent $G_A, G_B$. **NOT** the PSI-LoRA paper — deliberate ablation isolating just the diagonal stats.

```text
hook captures:  X (forward input, shape (B, d_in)),  S (output grad, shape (B, d_out))
state per pair: D_V ∈ ℝ^{d_in},  D_U ∈ ℝ^{d_out}     (init to 1)
D_V ← β₂·D_V + (1−β₂)·diag(X^T·X / B)
D_U ← β₂·D_U + (1−β₂)·diag(S^T·S / B)
sv = (D_V + δ)^{−γ}                       (d_in,)
su = (D_U + δ)^{−γ}                       (d_out,)
ΔA = G_A ⊙ sv          (broadcast on d_in axis)
ΔB = (su.unsqueeze(1)) ⊙ G_B
A ← A − η·ΔA;  B ← B − η·ΔB
```

#### KronGradLoRA (`kron-grad-lora`, `optim.py:2760`)

DiagScaledLoRA + an extra $r \times r$ Kronecker factor from gradient outer products. Custom variant, not from any paper.

```text
H_A ← β₂·H_A + (1−β₂)·G_A·G_A^T              (r×r)
H_B ← β₂·H_B + (1−β₂)·G_B^T·G_B              (r×r)
ΔA = (H_A + δI)^{−γ} · G_A · diag((D_V + δ)^{−γ})
ΔB = diag((D_U + δ)^{−γ}) · G_B · (H_B + δI)^{−γ}
A ← A − η·ΔA;  B ← B − η·ΔB
```

#### PSILoRA (`psi-lora`, `optim.py:2832`)

Faithful port of Almansoori et al. 2026 PSI-LoRA (arxiv 2602.16456) Algorithm 3: F-LoRSUM proximal subspace iteration with diagonal K-FAC metrics and full-weight low-rank momentum. See class docstring for the F-LoRSUM ALS inner iteration; one-line summary:

```text
hook:  X (B×d_in), S (B×d_out)
update D_V, D_U as in DiagScaledLoRA.
factors = [(A, B), (X, S^T), (M_A, M_B)]
coeffs  = [1, −η·(1−α₁), −η·α₁]
(A_new, B_new) = F-LoRSUM(factors, coeffs, D_U, D_V, K iters, ρ; gamma, delta)
if α₁ > 0:  (M_A, M_B) ← LoRSUM([(M_A, M_B), (X, S^T)], (α₁, 1−α₁), K, ρ)
A ← A_new;  B ← B_new
```

### Clipping-prox (proposed, not yet implemented)

Algorithm sketch from `polar_product/proposal.md` §2.2–2.5. Direct competitor to AdamPolarProductLoRACoupled at $k = 2$: replace polar (saturate every active mode) with **singular-value clipping** (truncate only modes that exceed $\tau$) — the exact prox of the spectral-norm-ball Frobenius projection. The lift discipline is committed to QR basis + min-Frobenius gauge + descent sign.

```text
state per pair: m_A, v_A, m_B, v_B, step
# 1. Adam covector (same as Picard; gradient compatibility intentionally broken).
m_A ← β₁·m_A + (1−β₁)·G_A;   v_A ← β₂·v_A + (1−β₂)·G_A²    (analog for B)
u_A = m̂_A/(√v̂_A + ε);   u_B = m̂_B/(√v̂_B + ε)
# 2. Channel coordinates from thin QR.
B = Q_L·R_L;   A^T = Q_R·R_R^T
# 3. Block-coordinate Picard (Jacobi; mirror hybrid Picard's k semantics).
ΔA_prev ← 0;   ΔB_prev ← 0
for k in 1 .. K:
    # A-block subproblem in channel coords X = Q_L^T·B·ΔA ∈ ℝ^{r×n}.
    if k == 1:  T_A = 0
    else:       T_A = −Q_L^T · ΔB_prev · A
    L0_A = R_L^{-T} · u_A
    X_unc = T_A − λ·L0_A                      (λ = η)
    # SVD then clip:  X⋆ = U·diag(min(σ_i, τ))·V^T,  τ = c·σ_max(X_unc)
    X_star_A = clip_τ(X_unc)
    # B-block subproblem (symmetric): channel coords Y = ΔB·A·Q_R ∈ ℝ^{m×r}.
    if k == 1:  T_B = 0
    else:       T_B = −B·ΔA_prev·Q_R
    L0_B = u_B · R_R^{-1}
    Y_unc = T_B − λ·L0_B
    Y_star_B = clip_τ(Y_unc)
    # 4. Lift channel-coord results back to (ΔA, ΔB) via Sylvester min-Frobenius gauge.
    #    Solve  S_L·K_lift + K_lift·S_R = R_L^T · X_star_A · R_R^T   for K_lift ∈ ℝ^{r×r}.
    #    Recover (ΔA, ΔB) so that B·ΔA + ΔB·A matches the channel target.
    ΔA, ΔB = lift(X_star_A, Y_star_B; Q_L, R_L, Q_R, R_R, S_L = R_L·R_L^T, S_R = R_R^T·R_R)
    ΔA_prev = ΔA;   ΔB_prev = ΔB
# 5. Apply (no Picard RMS-rescale — natural-prox magnitude with λ = η).
A ← A + ΔA;   B ← B + ΔB
```

`c` is the only new shape parameter (clip threshold $\tau = c \cdot \sigma_{\max}(X_\text{unc})$): $c \to \infty$ recovers the Frobenius-coupled Sylvester closed form (no clip); $c \to 0$ recovers the polar saturate-all direction up to magnitude. **Single value of $(c, k)$ to be shipped across ranks**; no per-rank tuning.

---

## Optimizer concepts

Cross-cutting machinery used by multiple algorithms above. Pseudocode for the optimizers themselves lives in **Optimizer algorithms**; this section defines the building blocks they share.

- **Hybrid Picard.** The polar-product family (uncoupled $k=1$ and coupled $k\ge 2$) — see ## Optimizer algorithms / AdamPolarProductLoRA, AdamPolarProductLoRACoupled.
- **Picard fixed-point iteration.** Block-coordinate iteration on the joint normal equations of the adjacent variational formulation — see AdamPolarProductLoRACoupled pseudocode.
- **`picard_iters` $k$.** Number of inner cross-coupling passes. $k=1$ disables cross-coupling (uncoupled spectral-product). $k=2$ is the default coupled variant. Within-family rank-dependent: $k=1$ wins at $r=16$, $k=2$ wins at $r=64$.
- **`picard_alpha` $\alpha$.** Damping coefficient on the cross-coupling correction. $\alpha = 1$ is full coupling (default at $k \ge 2$); $\alpha = 0$ disables it (equivalent to $k=1$). Sweep at $r=16$ found interior $\alpha \in \{0.25, 0.5, 0.75\}$ all worse than both endpoints.
- **Polar block solve.** $\mathrm{polar}(M) = UV^\top$ for compact SVD $M = U\Sigma V^\top$. Saturates every active singular direction to magnitude 1; the operator-norm-solve direction. Implemented by Newton-Schulz, not exact SVD.
- **Singular-value clipping.** $\mathrm{clip}_\tau(M) = U \cdot \mathrm{diag}(\min(\sigma_i, \tau)) \cdot V^\top$. Truncates only modes that exceed $\tau$; sub-threshold modes retained at unconstrained magnitudes. The exact prox of the spectral-norm-ball Frobenius projection. Used by the proposed clipping-prox optimizer.
- **Spectral preconditioner.** $S_B := B^\top B + \delta I$, $S_A := AA^\top + \delta I$ with $\delta = 10^{-6}$. Picard whitens $u_A$ by $S_B^{-1/2}$ before the polar.
- **RMS-align (Picard's magnitude rule).** Rescale a candidate factor step $D$ to Frobenius norm $\eta \|u\|_F$: $\Delta = -\eta \cdot \|u\|_F / \|D\|_F \cdot D$. Inherits step magnitude from the Adam covector.
- **Adam covector $u$.** $u_A = \hat m_A / (\sqrt{\hat v_A} + \varepsilon)$ where $\hat m, \hat v$ are bias-corrected first/second moments of $G_A$. Symmetric for $B$. Used in place of the raw gradient $G_A$ as the linear cost in the variational program; this breaks gradient compatibility (see below).
- **Gradient compatibility.** $G_A A^\top = B^\top G_B$, holding for raw autograd gradients. Broken under per-factor Adam preconditioning; both Picard and the clipping-prox optimizer accept this break as the price of using Adam covectors.
- **Min-Frobenius gauge.** The unique $(\Delta A, \Delta B)$ representative of a fixed first-order tangent $J = B \Delta A + \Delta B\, A$ minimizing $\|\Delta A\|_F^2 + \|\Delta B\|_F^2$. Characterized by $B^\top \Delta B = \Delta A\, A^\top$. Computed by solving a small Sylvester equation in $K \in \mathbb{R}^{r \times r}$.
- **Sylvester gauge lift.** Given a tangent target in core coordinates, solve $S_L K + K S_R = R_L^\top X R_R^\top$ for $K$, then assemble $\Delta A, \Delta B$ from $X, K$ and the QR factors. Implemented as `solve_sylvester` in `lora_playground/utils.py`.
- **Newton-Schulz (NS).** Iterative polynomial approximation to the polar factor; canonical Muon recipe (`_newton_schulz` at `optim.py:1449`). Pre-normalize input by $\|X\|_F$ to bring singular values into the basin of attraction near 1, run $X \leftarrow 1.5\,X - 0.5\,X X^\top X$ for `ns_steps` iterations, do **not** rescale back to input magnitude.

## Variational formulations

- **Joint operator-norm formulation (Case 3).** $\min \langle G, J \rangle + (1/2\lambda)\|J\|_F^2$ s.t. $\|J\|_2 \le \lambda$, where $J = B \Delta A + \Delta B A$. Empirically falsified — see E1–E7 in `polar_product/investigations.md`.
- **Adjacent formulation.** $\min \langle G, J \rangle + (1/2\lambda)\|J\|_F^2$ s.t. $\|B \Delta A\|_2 \le \lambda$ and $\|\Delta B\, A\|_2 \le \lambda$ (per-channel constraints, Frobenius coupling). Live target; the formulation Picard's iteration implicitly tries to solve. Exact block solve is singular-value clipping.
- **Frobenius case (Case 1).** Same objective with no spectral constraint. Closed-form Sylvester solution; the $c \to \infty$ limit of the clipping variant.

## Channel coordinates and lift terms

- **Thin QR.** $B = Q_L R_L$ with $Q_L \in \mathbb{R}^{m \times r}$ column-orthonormal and $R_L \in \mathbb{R}^{r \times r}$ upper-triangular. Symmetric $A = R_R Q_R^\top$.
- **Channel coordinates.** Working in the QR basis. The $A$-block subproblem reduces to one over $X = Q_L^\top B \Delta A \in \mathbb{R}^{r \times n}$ (the projection of the channel update onto $\mathrm{col}(B)$).
- **Core covector $L_0$.** $L_0 := R_L^{-\top} u_A$ in channel coordinates. The Adam covector expressed in the QR basis.
- **Cross-coupling target $T$.** $T := -Q_L^\top \Delta B_\text{prev} A$ for the $A$-block. Encodes the previous block-coordinate iterate's effect on the current block's normal equation.

## Joint-core solver terminology (archive of the dead family)

These terms appear in `polar_product/theory.md` and `polar_product/theory.md`; they describe the joint operator-norm core solver that lost.

- **Active core $\widehat H$.** The $(r+t) \times (r+s)$ matrix combining the gradient projections onto $\mathrm{col}(B), \mathrm{col}(B)^\perp$ and $\mathrm{row}(A), \mathrm{row}(A)^\perp$.
- **Forbidden corner / $(2,2)$ block.** The $(\mathrm{col}(B)^\perp, \mathrm{row}(A)^\perp)$ block of the active core. Feasible tangent updates have zero $(2,2)$ block (the rank-$r$ tangent cannot reach this corner); the joint-core solver projects the polar to enforce this.
- **$\Pi$ projection.** The operation that zeros the $(2,2)$ block of the polar and renormalizes.
- **Symmetrization $C = \tfrac{1}{2}(C_L + C_R)$.** A step in the joint-core solver that combines two projections of the gradient. Project-specific; used only in the joint-operator-norm direction.
- **`compat`.** $\|C_L - C_R\|_F / (\|C_L\|_F + \|C_R\|_F + \varepsilon)$ — diagnostic measuring how much the joint-core solver is throwing away by symmetrizing.
- **`align_mom`, `align_inst`.** Cosine similarities used as diagnostics in core-space momentum experiments (E5, E8). `align_mom` < `align_inst` indicated that EMA in the rotating $(Q_L, Q_R)$ basis was incoherent — structurally broken.

## Optimizer composition

- **Adam-first vs polar-first.** Composition order. Adam-first (`adam-polar-product-lora`): Adam EMA on factor gradients, then polar of preconditioned covectors. Polar-first (`adamuon-polar-product-lora`): NS polar on raw gradients, then Adam-style accumulation. Adam-first wins at both ranks in this project.
- **`*-Post` (RMS-align-Post).** Variant where RMS-align is applied at the *end* of the step pipeline rather than per-block.
- **AdaMuon-faithful.** A specific Muon-style implementation that mirrors the canonical `~/modded-nanogpt/train_gpt.py` recipe (no bias correction on EMA, Nesterov lookahead before NS orthogonalization). Distinguished from the pre-Adam variant. Implemented as `AdaMuonLoRA` (CLI `adamuon-lora`).

## Hypothesis labels

- **H1.** Per-factor polar with min-Frobenius gauge lift. Gauge axis. Folded into the clipping-prox proposal as an ablation.
- **H2.** $(2,2)$-block un-zeroed (joint-core variant). Specific to the dead joint-operator-norm family.
- **H3.** Step-magnitude trajectory. Diagnostic.
- **H4.** Gauge axis (paired with H1).
- **H5.** Picard's $k$-iteration is doing the work. Empirically closed at $r=16$ — `picard_iters` $k \in \{1, 2, 3, 4\}$ at $r=16$ shows $k=1$ best, $k \ge 2$ worse.
- **H6.** Implementation bug. Sanity-check; passes basic synthetic equivalence test.
