# Algorithm 2′ — chord-tight-clean ($k=2$, gram-NS): implementation walkthrough

The canonical optimizer in this campaign is **`spectral_chord_tight_clean`** with **`picard_iters = 2`** and **`ns_form = "gram"`**. This doc walks one step of that configuration through `AdamPolarProductLoRA._chord_tight_clean_polar_pipeline` (`lora_playground/optim.py:3408`), names every tensor that crosses an iteration boundary, and counts FLOPs in conceptual form. Algorithm-derivation companion: `algorithm_tight_chord.md` §10.

## 1. Shapes and notation

One LoRA pair: $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$. Inside `_step_batched_group`, all $N$ pairs of the same shape are stacked:

$$
A : (N, r, d_{\text{in}}), \qquad B : (N, d_{\text{out}}, r).
$$

Let $d := \max(d_{\text{in}}, d_{\text{out}})$ and $D := d_{\text{in}} + d_{\text{out}}$. We track only matmul terms; elementwise $O(N r D)$ is subdominant once $r \ge 16$.

**Persistent state, per group.**

- Adam moments $m_A, v_A$ shape $(N, r, d_{\text{in}})$; $m_B, v_B$ shape $(N, d_{\text{out}}, r)$.
- Whitening matrices $S_A^{-1/2}, S_B^{-1/2}$ shape $(N, r, r)$, refreshed every `precond_refresh_every` steps.
- Warm-start top singular vectors (smaller side is $r$ for every matrix below, so all warm-starts have shape $(N, r)$):

  | key | matrix whose $\sigma_{\max}$ is computed | consumer |
  |---|---|---|
  | `v_sigma_A` | $A$ | $\rho = \eta / s$ |
  | `v_sigma_B` | $B$ | $\rho = \eta / s$ |
  | `v_sigma_XA` | $X_A = S_B^{-1/2} u_A$ | pre-rescale |
  | `v_sigma_XB` | $X_B = u_B\, S_A^{-1/2}$ | pre-rescale |
  | `v_op_geoA_slots[n]` | $\text{geo}_A^{(n)} = S_B^{-1/2} P_A^{(n)}$ | magnitude rescale, per Picard iter $n$ |
  | `v_op_geoB_slots[n]` | $\text{geo}_B^{(n)} = P_B^{(n)} S_A^{-1/2}$ | magnitude rescale, per Picard iter $n$ |

  The `_slots` lists are length-`picard_iters` so each Picard iter $n$ warms its own σ-power-iter from its own prior step (not from iter $n-1$ of the current step). Implemented as preallocated lists rather than f-string dict keys so `torch.compile` can specialize the kernel; see `optim.py:3558`.

## 2. One step at $k = 2$ gram

The pipeline below is the full body of `_chord_tight_clean_polar_pipeline`. Steps marked **refresh** run on a schedule (`precond_refresh_every`); everything else runs every step.

### 2.1 Adam direction

In-place EMA on $(m_A, v_A, m_B, v_B)$ followed by bias correction:

$$
u_A = \frac{m_A/(1-\beta_1^t)}{\sqrt{v_A/(1-\beta_2^t)} + \varepsilon}, \qquad
u_B = \frac{m_B/(1-\beta_1^t)}{\sqrt{v_B/(1-\beta_2^t)} + \varepsilon}.
$$

### 2.2 Tight-tangent radius

$$
\sigma_A := \sigma_{\max}(A), \quad
\sigma_B := \sigma_{\max}(B), \quad
s := \sigma_A + \sigma_B, \quad
\rho := \eta / s.
$$

Power iter with `n_iters = 8`, warm-started from `v_sigma_A` / `v_sigma_B`.

### 2.3 Whitening refresh (scheduled)

$$
S_A := A A^\top, \quad S_B := B^\top B, \qquad
S_A^{-1/2}, \, S_B^{-1/2} \;=\; \mathrm{Higham}(S_A), \, \mathrm{Higham}(S_B).
$$

Higham (`spd_inv_sqrt_higham_batched`, `utils.py:147`) is 10 coupled Newton–Schulz quintic iterations on the $(N, r, r)$ Gram, in fp32 with TF32 disabled, plus one internal $\lambda_{\max}$ power iter. Results are written back into `gs['SA_half_inv']`, `gs['SB_half_inv']`.

### 2.4 Whiten the Adam direction (once)

$$
X_A := S_B^{-1/2}\, u_A \in \mathbb{R}^{N \times r \times d_{\text{in}}}, \qquad
X_B := u_B\, S_A^{-1/2} \in \mathbb{R}^{N \times d_{\text{out}} \times r}.
$$

Computed once per step. The $n=0$ polar input reuses this matrix; the $n=1$ Picard correction perturbs it additively (§2.6).

### 2.5 Pre-rescale (this is what makes Alg 2′ ≠ Alg 2)

$$
X_A \;\leftarrow\; X_A \,/\, \sigma_{\max}(X_A), \qquad
X_B \;\leftarrow\; X_B \,/\, \sigma_{\max}(X_B),
$$

and the same scalar divisors are applied to $u_A, u_B$ so the Picard cross-coupling at $n \ge 1$ stays consistent. Two power iters at 8 iterations each, warm-started.

### 2.6 Picard loop, $n = 0, 1$

The loop body produces the polar map of the (possibly cross-coupling-corrected) whitened direction and computes its magnitude-rescaled unwhitening.

**$n = 0$** (no cross-coupling; $\mathrm dA = \mathrm dB = 0$):
$$
X_A^{(0)} = X_A, \qquad X_B^{(0)} = X_B.
$$

**$n = 1$** (Lemma 1 cross-coupling, coefficient $1/\eta$):
$$
\begin{aligned}
u_A^{(1)} &= u_A + \tfrac{1}{\eta}\, B^\top\, \mathrm dB^{(0)}\, A, \\
u_B^{(1)} &= u_B + \tfrac{1}{\eta}\, B\, \mathrm dA^{(0)}\, A^\top, \\
X_A^{(1)} &= S_B^{-1/2}\, u_A^{(1)}, \qquad X_B^{(1)} = u_B^{(1)}\, S_A^{-1/2}.
\end{aligned}
$$

**Both iters**, after $X^{(n)}$ is in hand:

$$
\begin{aligned}
P_A^{(n)} &= \mathrm{polar}_{\text{NS-gram}}\bigl(X_A^{(n)}\bigr), \qquad
P_B^{(n)} = \mathrm{polar}_{\text{NS-gram}}\bigl(X_B^{(n)}\bigr), \\
\text{geo}_A^{(n)} &= S_B^{-1/2}\, P_A^{(n)}, \qquad
\text{geo}_B^{(n)} = P_B^{(n)}\, S_A^{-1/2}, \\
\mathrm dA^{(n)} &= -\rho \cdot \text{geo}_A^{(n)} / \sigma_{\max}\bigl(\text{geo}_A^{(n)}\bigr), \\
\mathrm dB^{(n)} &= -\rho \cdot \alpha_{\text{LoRA+}} \cdot \text{geo}_B^{(n)} / \sigma_{\max}\bigl(\text{geo}_B^{(n)}\bigr).
\end{aligned}
$$

Newton–Schulz uses `_newton_schulz_gram_batched` (`optim.py:1918`), with restart at $\tau = 2$; see §5. `ns_steps = 5` quintic iterations.

### 2.7 Apply

$A \mathrel{+}= \mathrm dA^{(1)}$, $B \mathrel{+}= \mathrm dB^{(1)}$.

## 3. FLOP budget

Per group of $N$ pairs, per step. Refresh-only rows fire at the configured cadence; everything else fires every step. The polar-map row assumes $K = 5$ NS iters in gram form (§5).

| block | shape of inner op | FLOPs |
|---|---|---|
| $\sigma_{\max}(A), \sigma_{\max}(B)$ | matvec on $(N, r, d)$ | $32\, N r D$ |
| Whitening Grams ($A A^\top, B^\top B$), *refresh* | $(N, r, d) \cdot (N, d, r)$ | $2\, N r^2 D$ |
| Higham, *refresh* | $(N, r, r)$ | ${\approx}\, 60\, N r^3$ per side, $\approx 120\, N r^3$ total |
| Whiten input $X_A, X_B$ | $(N, r, r) \cdot (N, r, d)$ | $2\, N r^2 D$ |
| $\sigma_{\max}(X_A), \sigma_{\max}(X_B)$ | matvec | $32\, N r D$ |
| Polar map, gram NS, $K = 5$, **per Picard iter** | $(N, r, d) \cdot (N, d, r)$ initial + $(N, r, r)$ recur | $4\, N r^2 D + 30\, N r^3$ per polar call |
| Cross-coupling, **only at $n = 1$** | four bmms | $8\, N r^2 D$ |
| Unwhiten $\text{geo}_A, \text{geo}_B$, **per Picard iter** | $(N, r, r) \cdot (N, r, d)$ | $2\, N r^2 D$ per side |
| $\sigma_{\max}(\text{geo})$, **per Picard iter** | matvec | $32\, N r D$ |

**At $k = 2$, gram NS, $K = 5$**, the dominant per-pair cost (drop the group-size factor $N$) collects to

$$
C_{\text{opt}} \;\approx\; \underbrace{24\, r^2 D}_{\text{matmuls (whitening + 2 polar calls + 2 unwhitens + cross-coupling)}} \;+\; \underbrace{60\, r^3}_{\text{2 polar calls (gram recurrence)}} \;+\; \underbrace{120\, r^3 / n_{\text{refresh}}}_{\text{Higham, amortized over refresh cadence}}.
$$

The matmul coefficient breaks down as: 2 (whiten input) + 8 (cross-coupling) + $2 \cdot 4$ (two gram-polar calls at $4 r^2 D$ each) + $2 \cdot 2$ (unwhiten on both Picard iters) = 22, plus residual elementwise $\approx 2$, giving the $24$ above.

## 4. Overhead vs AdamW — conceptual

Symbols used in this section:

- $C_{\text{opt}}$ — per-pair optimizer FLOPs per step (from §3).
- $C_{\text{AdamW}} \approx 4\, r D$ — per-pair AdamW FLOPs per step (EMAs + bias correct + update, all elementwise).
- $M$ — number of LoRA pairs in the model.
- $P_{\text{LoRA}} = M\, r\, D$ — total LoRA parameter count.
- $P_{\text{model}}$ — base-model parameter count.
- $T$ — tokens per step.
- $n_{\text{refresh}}$ — whitening-refresh cadence, in steps.
- $c_{\text{fb}} \approx 4$ — fwd+bwd FLOPs per token per base-model parameter.

### 4.1 Per-pair ratio

$$
\frac{C_{\text{opt}}}{C_{\text{AdamW}}} \;\approx\; 6 r \;+\; 15\, \frac{r^2}{D} \;+\; \frac{30\, r^2}{n_{\text{refresh}}\, D}.
$$

When $r \ll D$ the first term dominates: **optimizer FLOPs per pair scale linearly in $r$ relative to AdamW**, with subleading $r^2/D$ from the gram recurrence and the amortized Higham (these become relevant only when $r$ approaches model width).

### 4.2 Fraction of step FLOPs

Forward + backward dominate when the base model is the bulk of the parameter count: per-step cost $\approx c_{\text{fb}}\, P_{\text{model}}\, T + M \cdot C_{\text{opt}}$. Substituting $C_{\text{opt}} \approx 24\, r^2 D$ (the leading matmul term from §3) gives

$$
\frac{\text{opt FLOPs}}{\text{step FLOPs}}
\;\approx\; \frac{M \cdot 24\, r^2 D}{c_{\text{fb}}\, P_{\text{model}}\, T}
\;=\; \frac{24}{c_{\text{fb}}} \cdot \frac{M\, r^2\, D}{P_{\text{model}}\, T}.
$$

This still hides one dependency: $P_{\text{model}}$ itself scales with $D$. For a depth-$L$ transformer with width $D$, per-layer parameters are $O(D^2)$, so $P_{\text{model}} \sim \kappa\, L\, D^2$ with $\kappa$ a placement-dependent constant (≈ 12 for standard attention + 2-layer MLP with 4× expansion). LoRA placements scale with depth too: $M = \mu\, L$ where $\mu$ is the number of target modules per layer (typically 7 for the q/k/v/o + gate/up/down convention). Substituting:

$$
\frac{\text{opt FLOPs}}{\text{step FLOPs}}
\;\approx\; \frac{24}{c_{\text{fb}}} \cdot \frac{\mu}{\kappa} \cdot \frac{r^2}{D\, T}.
$$

The primitive scalings are then:

- **$\propto r^2 / D$** in the matrix shape. Doubling LoRA rank quadruples overhead; doubling model width halves it (because $P_{\text{model}}$ grows as $D^2$ but per-pair LoRA cost only as $D$).
- **$\propto 1/T$** in tokens per step; bigger batches amortize the optimizer.
- **$\propto \mu / \kappa$** in placement: the ratio of target modules per layer to total parameter density per layer. Independent of depth $L$.

Larger base models therefore dilute optimizer overhead at fixed $r$ — the $r^2/(D T)$ form is what to keep in mind, not the bare $r^2$ that the earlier per-pair view suggested.

### 4.3 Reconciliation with measured wall time

Measured wall overhead vs AdamW is larger than the FLOP overhead implies. The asymmetry is hardware utilization, not arithmetic:

- fwd+bwd: large bf16 matmuls at near tensor-core peak.
- Optimizer: fp32 Higham (no tensor cores), small $r \times r$ matmuls (launch-bound), σ-power-iter matvecs (launch-bound).

The wall-per-FLOP ratio between optimizer and fwd+bwd is roughly an order of magnitude on Blackwell at packed_v1 shapes. The optimization levers that move wall time attack utilization, not FLOP count: gram NS shifts polar work from rectangular bf16 onto small $r^3$ matmuls, fp16+restart keeps it on tensor cores, and CUDA-graph / compile attacks launch overhead in the small-matmul regime. Concrete numbers live in `docs/notes/polar_product/walltime_profile.md` § "Gram-NS + k=2 wall-time".

## 5. Gram-form Newton–Schulz (Dao 2026)

The polar map is the dominant block in §3. Standard ("rect") NS iterates the polynomial $p(X) = a X + b X X^\top X + c (X X^\top)^2 X$ on the rectangular $X \in \mathbb{R}^{r \times d}$. The gram form (`_newton_schulz_gram_batched`, `optim.py:1918`; Dao 2026 §3, `docs/papers/gram_newton_schulz_dao_2026.md`) factors $p_t(x) = x \cdot h_t(x^2)$ and iterates on the smaller-side Gram $R = X X^\top \in \mathbb{R}^{r \times r}$ while accumulating a single transform $Q \in \mathbb{R}^{r \times r}$:

$$
R_0 = X_0 X_0^\top, \quad Q_0 = I, \qquad
\begin{aligned}
Z_t &= h_t(R_{t-1}) \;=\; a_t I + b_t R_{t-1} + c_t R_{t-1}^2, \\
Q_t &= Q_{t-1} Z_t, \\
R_t &= Z_t R_{t-1} Z_t,
\end{aligned}
\qquad X_T = Q_T X_0.
$$

All interior work is on $(N, r, r)$ matrices; only two $r \times d$ matmuls survive ($R_0$ and the final reconstruction).

### 5.1 Restart at $\tau = 2$ (Dao Algorithm 3)

Gram NS has two failure modes that rect NS does not, because $R_t$ is *cumulative state*:

1. **Spurious negative eigenvalues.** $R_0$ in finite precision has eigenvalues $\approx -\varepsilon$ where the true value is 0. The scalar recurrence has $r_t \approx h_t(0)^2 r_{t-1}$ near the origin, which amplifies any small negative magnitude until $|r| \sim 1$, at which point the polynomial leaves its basin and $Z_t$ blows up.
2. **Eigenvector drift.** In exact arithmetic $Q_T$ shares $X_0$'s left singular vectors. Finite-precision drift breaks this, so $Q_T X_0$ no longer captures $X_0$'s dominant directions.

The restart at $\tau = 2$ is

$$
X_0 \leftarrow Q_\tau X_0, \quad R_\tau \leftarrow X_0 X_0^\top, \quad Q_\tau \leftarrow I,
$$

which resets the noise floor on the *current* iterate and zeroes the $R \leftrightarrow Q$ eigenvector drift. Cost: two extra matmuls per restart.

### 5.2 Precision

Gram NS runs in **fp16, not bf16**. After Frobenius pre-norm (safety factor 1.05) the iterates sit near $[0, 1]$, so fp16's narrow dynamic range does not bind, and its 10-bit mantissa shrinks the spurious-negative noise floor by $\sim 10\times$ vs bf16's 7 bits. Rect NS does not compound state, so bf16 is fine there; gram reverses that priority.

A safety-mode fp32 path (TF32 disabled, mirroring `spd_power_batched`) is exposed for diagnostics; restart is unnecessary there because the fp32 noise floor never reaches the basin.

### 5.3 Verification

- Equivalence: `tests/test_ns_gram.py` — gram-fp32 matches rect-fp32 within $10^{-5}$ rel-err on random, real (Tier-1 chord-tight $r{=}64$ snapshots), and synthetic $\mathrm{cond} = 10^4$ inputs.
- fp16+restart: matches rect-fp32 within $5 \times 10^{-2}$ rel-err on the Tier-1 corpus (top-20 worst-conditioned $X^{\text{eff}}$, max measured $\mathrm{cond}(G) = 1.4 \times 10^5$).
- End-to-end smoke: 5-step OLMo-2-1B at $r{=}64, k{=}2$ — gram eval_loss $= 1.1759$, rect $= 1.1792$, $|\Delta| < 5 \times 10^{-3}$.

## 6. Power-iter call sites

Four σ-power-iter sites fire per step at $k = 2$:

| site | matrix | warm-start key | role |
|---|---|---|---|
| **A1** | $A$ | `v_sigma_A` | $\rho = \eta / s$ |
| **A2** | $B$ | `v_sigma_B` | $\rho = \eta / s$ |
| **B** | $X_A, X_B$ | `v_sigma_XA`, `v_sigma_XB` | pre-rescale (§2.5) |
| **C** ($\times 2$ Picard iters) | $\text{geo}_A^{(n)}, \text{geo}_B^{(n)}$ | `v_op_geoA_slots[n]`, `v_op_geoB_slots[n]` | magnitude rescale (§2.6) |

All sites currently use `n_iters = 8` unconditionally (see §7).

## 7. Outstanding cleanup

- **Hoist $\lambda_{\max}(S_A)$ out of Higham.** $\sigma_{\max}(A)$ is already computed at site A1, and $\lambda_{\max}(S_A) = \sigma_{\max}(A)^2$. Higham currently re-derives $\lambda_{\max}$ internally. The FLOP win is small ($(N, r, r)$ matmuls); the architectural win is one canonical source for $\sigma_{\max}(A)$ per step. Requires adding a `lam_max` kwarg to the Higham helper (non-breaking for other callers).
- **Power-iter warm-start at `n_iters = 3`.** Currently `n_iters = 8` at all four sites. Helper docstring claims `n_iters = 3` suffices with a good warm start. Adoption gated on a per-pair residual test against the `n_iters = 8` reference on a representative snapshot; not run.
- **Unit-test coverage for `_chord_tight_clean_polar_pipeline`.** No `tests/test_*.py` currently exercises this method. The end-to-end smoke (§5.3) catches gross regressions but not silent drift in the cross-coupling formula, the Picard-iter-keyed warm-start, or the per-step σ-rescale. Minimum useful coverage: a tiny ($r=4$, $d=8$, $N=2$) fixture asserting bit-identical output between a Python-reference implementation of §2.1–2.7 and the batched method at $k = 1$, plus a separate $k = 2$ trajectory test under fixed seeds.
