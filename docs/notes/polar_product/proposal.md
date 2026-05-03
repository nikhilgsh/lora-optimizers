# Polar-product LoRA — Sylvester min-Frob lift architecture (clip deferred)

## 0. Background and vocabulary

This proposal works in one specific corner of the LoRA-optimizer design space. To see which corner, the reader needs (i) the setup, (ii) a survey of natural variational formulations and what's been tried in each, (iii) the specific program this proposal targets, (iv) the operator the program uses, and (v) the open question about how to set the operator's threshold.

### 0.1 Setup

A LoRA pair is

$$A \in \mathbb{R}^{r \times n},\qquad B \in \mathbb{R}^{m \times r},$$

contributing $BA$ to the effective weight $W + BA$ ($\alpha/r$ folded in at use-time). One optimizer step on a pair produces an update $(\Delta A, \Delta B)$.

Each factor carries its own Adam state. The Adam updates are

$$u_A := \hat m_A / (\sqrt{\hat v_A} + \varepsilon),\qquad u_B := \hat m_B / (\sqrt{\hat v_B} + \varepsilon).$$

The polar-product family uses $u_A, u_B$ in place of the raw gradients $G_A, G_B$ throughout the per-block solves below — this is the "Adam-covector compromise" (§2.6).

The **spectral preconditioners** are

$$S_A\ :=\ AA^\top + \delta I \in \mathbb{R}^{r \times r},\qquad S_B\ :=\ B^\top B + \delta I \in \mathbb{R}^{r \times r},\qquad \delta = 10^{-6}.$$

Both are used in derivations below.

The **joint tangent** is

$$J\ :=\ B \Delta A + \Delta B\, A \in \mathbb{R}^{m \times n},$$

i.e. the contribution of one optimizer step to the merged-weight change.

### 0.2 Survey of variational formulations for LoRA factor updates

**Why a per-step variational framing at all.** LoRA factor updates have geometric structure that per-coordinate Adam ignores: the rank-$r$ tangent at $(A, B)$ is the span of $J$. Per-coordinate Adam treats $\Delta A, \Delta B$ independently and can produce updates misaligned with the actual $J$. Several optimizer families exploit this structure; each corresponds to a different per-step variational program.

**The umbrella variational form** ($\star$ in `theory.md`):

$$\min_{\Delta A,\, \Delta B}\ \langle G_A,\, \Delta A\rangle + \langle G_B,\, \Delta B\rangle + \frac{1}{2\lambda}\, \rho\!\bigl(B \Delta A + \Delta B\, A\bigr)$$

(or the constrained analogue $\|B \Delta A + \Delta B\, A\| \le \lambda$). The penalty/constraint $\rho$ is the design freedom — different $\rho$ gives different optimizer families.

**Two binary choices for $\rho$ generate four natural programs:**

- **Constraint type:** Frobenius-norm penalty (smooth; no spectral cap) vs operator-norm constraint (caps the largest singular direction).
- **Constraint scope:** apply to the *joint tangent* $J = B \Delta A + \Delta B\, A$ vs apply *per block* to $B \Delta A$ and $\Delta B A$ separately.

These map to `theory.md`'s named cases as: (i) Joint Frobenius = Case 1; (ii) Joint operator-norm = Case 3; (iv) Per-block operator-norm = "Adjacent formulation" (the live target). Plus a fifth, partial restriction (v) = Case 2 (out of the 2x2). Cell (iii) — Per-block Frobenius — is not named in `theory.md` but is the natural fourth corner of the 2x2; the existing `adam-scaled-lora` family is its closest implementation.

The Sylvester min-Frob gauge lift shared by Cases 1, 3, and the per-block formulation is in `theory.md`. The cells below add implementation status and per-variant numbers on top of these references.

**Implementation sub-axes (orthogonal to which program).** Every program below is written with raw gradients $G_A, G_B$. Implemented variants differ on three further axes (each choice independent of the program):

- **Linear cost:** raw $G_A, G_B$ vs Adam-preconditioned covectors $u_A, u_B$ (the §2.6 compromise; raw-gradient closed forms become heuristic when Adam moments are substituted).
- **Adam ordering** (when Adam is used): **pre** (Adam on the geometric step's input gradient) or **post** (Adam on raw $G$ first, then geometric step on the Adam direction). The `-post` suffix in optimizer names denotes the post variant.
- **Magnitude rule:** none / per-coord Adam $\hat m / (\sqrt{\hat v} + \varepsilon)$ / per-pair scalar Adam (`-matrix` suffix) / Frobenius rescale to $\eta\|u\|_F$ ("RMS-align", used by `adam-polar-product-lora`, AdaMuon, NorMuon).

**Survey of the four programs.** Each gets a subsection with: program statement (raw $G$), closed-form update, implementation variants in this repo with their sub-axis choices, best eval @ 2k. AdamW baseline for reference: 0.7579 ($r=16$), 0.7550 ($r=64$).

#### (i) Joint Frobenius

Program:

$$\min_{\Delta A, \Delta B}\ \langle G_A, \Delta A\rangle + \langle G_B, \Delta B\rangle + \tfrac{1}{2\eta}\|J\|_F^2.$$

Closed-form update — solve a small $r \times r$ Sylvester equation, then per-block apply the correction (`LinLoRA` in `optim.py`):

$$
\begin{aligned}
S_B\, K + K\, S_A &\ =\ -\eta\, G_A\, A^\top \qquad \text{(solve for }K \in \mathbb{R}^{r \times r}\text{)} \\
\Delta A &\ =\ -\,S_B^{-1}\bigl(\eta\, G_A + K\, A\bigr) \\
\Delta B &\ =\ -\bigl(\eta\, G_B + B\, K\bigr)\, S_A^{-1}.
\end{aligned}
$$

- **Implemented variants:**

  `lin-lora` — literal closed form on raw $G$ (the equations above).

  `adam-lin-lora` (pre-Adam): preconditioned-gradient $\to$ Adam.

  $$
  \begin{aligned}
  K &= \mathrm{solve}\bigl(S_B K + K S_A = -G_A A^\top\bigr) \\
  v_A &= S_B^{-1}(G_A + K A),\qquad v_B = (G_B + B K)\, S_A^{-1} \\
  (\hat m, \hat v) &\leftarrow \text{Adam EMA on }(v_A, v_B) \\
  \Delta A &= -\eta\, \hat m_A / (\sqrt{\hat v_A} + \varepsilon),\qquad \Delta B = -\eta\, \hat m_B / (\sqrt{\hat v_B} + \varepsilon).
  \end{aligned}
  $$

  `adam-lin-lora-post` (post-Adam): Adam-direction $\to$ Sylvester.

  $$
  \begin{aligned}
  u_A, u_B &\leftarrow \text{Adam EMA on raw }(G_A, G_B) \\
  K &= \mathrm{solve}\bigl(S_B K + K S_A = -\eta\, u_A A^\top\bigr) \\
  \Delta A &= -S_B^{-1}(\eta\, u_A + K A),\qquad \Delta B = -(\eta\, u_B + B K)\, S_A^{-1}.
  \end{aligned}
  $$

  Both orderings are heuristic — the variational form is not invariant under either.

  `adam-lin-lora-matrix` — same as `adam-lin-lora` but with a single per-pair scalar second moment $\hat v_\text{pair} \in \mathbb{R}$ instead of per-coord $\hat v$.
- **Per-variant best @ 2k:**

  | variant | $r=16$ | $r=64$ |
  |---|---|---|
  | `lin-lora` | 0.7776 ($\eta = 3$) | — |
  | `adam-lin-lora` | **0.7581** ($\eta = 10^{-3}$) | **0.7527** ($\eta = 3\mathrm{e}{-4}$) |
  | `adam-lin-lora-post` | 0.7641 ($\eta = 3\mathrm{e}{-4}$) | — |
  | `adam-lin-lora-matrix` | 0.7744 ($\eta = 10^{-3}$) | 0.7690 ($\eta = 10^{-3}$) |

#### (ii) Joint operator-norm

Program (Case 3 of `theory.md`); two equivalent forms — **same direction, magnitudes differ by a constant**:

$$\min_{\Delta A, \Delta B}\ \langle G_A, \Delta A\rangle + \langle G_B, \Delta B\rangle \quad \text{s.t.}\ \|J\|_2 \le \tau \qquad \text{(constrained form)}$$

$$\min_{\Delta A, \Delta B}\ \langle G_A, \Delta A\rangle + \langle G_B, \Delta B\rangle + \tfrac{1}{2\eta}\|J\|_2^2 \qquad \text{(squared-penalty form)}$$

Closed-form update — direction is $\mathrm{polar}$ of a single $W$-space matrix, magnitude is $-\tau$ (constrained) or $-\eta\,\|G\|_*$ (squared-penalty, $\|G\|_*$ = nuclear norm of $G$ on the rank-$r$ tangent); recover the factor pair via the Sylvester min-Frob lift:

$$
\begin{aligned}
P &\ =\ \mathrm{polar}(D) \qquad \text{(}D\text{ a joint }W\text{-space gradient/momentum estimate)} \\
B\, \Delta A + \Delta B\, A &\ =\ -\eta\, P,\qquad \text{s.t.}\ B^\top \Delta B = \Delta A\, A^\top \\
&\hspace{2em}\downarrow\ \text{Sylvester lift (small }r \times r\text{ solve, see §2.5)} \\
&\ \ (\Delta A, \Delta B).
\end{aligned}
$$

- **Implemented variants:**

  `product-muon-lora` — EMA-momentum on $\partial L/\partial B$, then polar in $W$-space, then Sylvester recover.

  $$
  \begin{aligned}
  m_B &\leftarrow \beta\, m_B + (1 - \beta)\, \partial L/\partial B \\
  D &= \tfrac{r}{\alpha}\, m_B\, A^\top (A A^\top + \delta I)^{-1} \quad \in \mathbb{R}^{d_\text{out} \times d_\text{in}} \\
  P &= \mathrm{polar}(D) \quad (\text{Newton–Schulz}) \\
  (\Delta A, \Delta B) &\ \text{recover s.t.}\ B\, \Delta A + \Delta B\, A = -\eta\, P\ \text{with}\ B^\top \Delta B = \Delta A\, A^\top.
  \end{aligned}
  $$

  `adam-product-muon-lora` — Adam moments (per-coord $\hat m_B / (\sqrt{\hat v_B} + \varepsilon)$) substituted for $m_B$ in $D$. Same polar + Sylvester downstream.

  **E1–E7** in `investigations.md` — joint-operator-norm core solvers in tangent space, with various per-coord/per-pair scalings (E4 adds elementwise sign-norm in core space, E3 uses a wider lr scan, etc.).

  No 2k runs of `product-muon-lora` / `adam-product-muon-lora` at the canonical horizon at the time of writing.
- **Per-variant best @ 2k (from `investigations.md` Table in §3):**

  | variant | $r=16$ | $r=64$ |
  |---|---|---|
  | E1 (baseline §2.1 solver) | 0.8188 ($\eta = 3\mathrm{e}{-3}$) | 0.7821 ($\eta = 3\mathrm{e}{-3}$) |
  | E2 (post-step state rebalance) | 0.8104 | 0.7686 |
  | E3 (E1 + wider $\eta$ scan) | 0.8049 | **0.7490** ($\eta = 3\mathrm{e}{-2}$) |
  | E4 (core sign-norm) | **0.7680** ($\eta = 10^{-4}$) | diverges |
  | E6 (E4 + E5 + E2 compounds) | 0.7684 | 0.9440 |
  | E7 (factor-Adam preconditioning) | 0.7846 ($\eta = 10^{-4}$) | 1.097 (cancelled — falsified) |
  | `product-muon-lora` | — (no 2k runs) | — |
  | `adam-product-muon-lora` | — (no 2k runs) | — |

#### (iii) Per-block Frobenius

Program:

$$\min_{\Delta A, \Delta B}\ \langle G_A, \Delta A\rangle + \langle G_B, \Delta B\rangle + \tfrac{1}{2\eta}\bigl(\|B\Delta A\|_F^2 + \|\Delta B\, A\|_F^2\bigr).$$

Closed-form update — no coupling between blocks; each block's first-order condition gives a damped Gram-inverse precondition (`ScaledLoRA`):

$$\Delta A\ =\ -\eta\, S_B^{-1}\, G_A,\qquad \Delta B\ =\ -\eta\, G_B\, S_A^{-1}.$$

- **Implemented variants:**

  `scaled-lora` — literal closed form on raw $G$ (the equations above).

  `adam-scaled-lora` (pre-Adam):

  $$
  \begin{aligned}
  v_A = S_B^{-1} G_A,\qquad v_B &= G_B\, S_A^{-1} \\
  (\hat m, \hat v) &\leftarrow \text{Adam EMA on }(v_A, v_B) \\
  \Delta A = -\eta\, \hat m_A / (\sqrt{\hat v_A} + \varepsilon),\qquad \Delta B &= -\eta\, \hat m_B / (\sqrt{\hat v_B} + \varepsilon).
  \end{aligned}
  $$

  `adam-scaled-lora-post` (post-Adam, the H4 design):

  $$
  \begin{aligned}
  u_A, u_B &\leftarrow \text{Adam EMA on raw }(G_A, G_B) \\
  \Delta A = -\eta\, S_B^{-1} u_A,\qquad \Delta B &= -\eta\, u_B\, S_A^{-1}.
  \end{aligned}
  $$

  `adam-scaled-lora-matrix` — same as `adam-scaled-lora` but with per-pair scalar $\hat v_\text{pair}$ instead of per-coord $\hat v$.
- **Caveat:** exact correspondence between these implementations and the per-block-Frobenius variational program is not formally derived in this repo; treated as the closest match.
- **Per-variant best @ 2k:**

  | variant | $r=16$ | $r=64$ |
  |---|---|---|
  | `scaled-lora` | 0.7706 ($\eta = 3$) | — |
  | `adam-scaled-lora` | 0.7590 ($\eta = 10^{-3}$) | **0.7506** ($\eta = 3\mathrm{e}{-4}$) |
  | `adam-scaled-lora-post` | **0.7570** ($\eta = 3\mathrm{e}{-4}$) | — |
  | `adam-scaled-lora-matrix` | 0.7760 ($\eta = 10^{-3}$) | 0.7723 ($\eta = 10^{-3}$) |

#### (iv) Per-block operator-norm — this proposal works here

Program:

$$\min_{\Delta A, \Delta B}\ \langle G_A, \Delta A\rangle + \langle G_B, \Delta B\rangle + \tfrac{1}{2\eta}\|J\|_F^2 \quad \text{s.t.}\ \|B\Delta A\|_2 \le \tau,\ \|\Delta B\, A\|_2 \le \tau.$$

Closed-form update — per-block prox under spectral-norm constraint, in whitened coordinates (§2.1 derives this; full skeleton in §0.3):

$$
\begin{aligned}
&X_{A,\text{unc}} = T_A - \eta\, R_B^{-\top}\, G_A,\qquad X_A^\star = \mathrm{clip}(X_{A,\text{unc}};\, \tau)\\
&Y_{B,\text{unc}} = T_B - \eta\, G_B\, R_A^{-\top},\qquad Y_B^\star = \mathrm{clip}(Y_{B,\text{unc}};\, \tau) \\
&(\Delta A, \Delta B) = \mathrm{Lift}(X_A^\star, Y_B^\star) \quad \text{(Sylvester min-Frob gauge, §2.5)}
\end{aligned}
$$

(Cross-coupling targets $T_A, T_B$ are zero at $k=1$ and feed the previous inner-iterate at $k \ge 2$; see §0.3.)

- **Implemented variants** (all use polar, not clip — see §0.4):

  `adam-polar-product-lora` ($k = 1$, uncoupled). Different whitening from the QR above ($S^{-1/2}$ symmetric; Frobenius-isometric per §2.3), and **no** Sylvester lift:

  $$
  \begin{aligned}
  u_A, u_B &\leftarrow \text{Adam EMA on raw }(G_A, G_B) \\
  \widetilde u_A = S_B^{-1/2} u_A,\qquad \widetilde u_B &= u_B\, S_A^{-1/2} \\
  g_A = S_B^{-1/2}\, \mathrm{polar}(\widetilde u_A),\qquad g_B &= \mathrm{polar}(\widetilde u_B)\, S_A^{-1/2} \\
  \Delta A = -\eta\, \tfrac{\|u_A\|_F}{\|g_A\|_F}\, g_A,\qquad \Delta B &= -\eta\, \tfrac{\|u_B\|_F}{\|g_B\|_F}\, g_B.
  \end{aligned}
  $$

  `adam-polar-product-lora-coupled` ($k \ge 2$). For $j \ge 2$ on each Picard pass, augment the input before whitening:

  $$
  u_A^\text{eff} = u_A + \tfrac{1}{\eta} B^\top \Delta B_\text{prev} A,\qquad u_B^\text{eff} = u_B + \tfrac{1}{\eta} B\, \Delta A_\text{prev} A^\top
  $$

  then run the uncoupled pipeline on $(u_A^\text{eff}, u_B^\text{eff})$. Rescale uses augmented norms.

  `adam-polar-product-lora-coupled-endrms` — same as `-coupled` but rescale uses the original $\|u_A\|_F, \|u_B\|_F$ (not augmented).

  **This proposal** is a new implementation in (iv) using a Sylvester min-Frob lift to recombine the per-block prox outputs (instead of `adam-polar-product-lora`'s per-factor RMS-align with no recombination); polar (clip deferred — §0.5); RMS-align on the lifted output. The change vs `adam-polar-product-lora` is detailed in §3.
- **Per-variant best @ 2k:**

  | variant | $r=16$ | $r=64$ |
  |---|---|---|
  | `polar-product-lora` (no Adam, raw $G$) | 0.7975 ($\eta = 10^{-3}$) | 0.7786 ($\eta = 10^{-3}$) |
  | `adam-polar-product-lora` ($k=1$) | **0.7546** ($\eta = 3\mathrm{e}{-4}$) | 0.7454 ($\eta = 3\mathrm{e}{-4}$) |
  | `adam-polar-product-lora-coupled` ($k=2$) | 0.7557 ($\eta = 3\mathrm{e}{-4}$) | 0.7382 ($\eta = 3\mathrm{e}{-4}$) |
  | `adam-polar-product-lora-coupled-endrms` ($k=2$) | 0.7615 ($\eta = 3\mathrm{e}{-4}$) | **0.7379** ($\eta = 3\mathrm{e}{-4}$) |

#### (v) One-factor operator-norm (partial restriction; outside the 2x2)

A fifth natural program from `theory.md` §"Case 2": hold one factor (say $A$) fixed and constrain the operator norm of the other factor's contribution alone:

$$\min_{\Delta B}\ \langle G_B, \Delta B\rangle \quad \text{s.t.}\ \|\Delta B\, A\|_2 \le \tau.$$

Closed-form update — via the SVD $A = U_R \Sigma_R V_R^\top$, reparametrize $\Delta B = X\, \Sigma_R^{-1} U_R^\top$, then $X^\star = -\eta\, \mathrm{polar}\bigl(G_B\, U_R\, \Sigma_R^{-1}\bigr)$:

$$\Delta B = -\eta\, \mathrm{polar}\bigl(G_B\, U_R\, \Sigma_R^{-1}\bigr)\, \Sigma_R^{-1} U_R^\top.$$

This is a *partial* tangent restriction (only one factor is in the optimization at a time), so it doesn't fit the 2x2 Joint vs Per-block × Frob vs Op-norm classification — included here for completeness because the historical empirical table in `theory.md` §"Empirical context" has Case 2 numbers.

- **Implementation status:** No current implementation in `optim.py` matches Case 2 exactly. (`polar-product-lora` applies polar per-block to *both* factors symmetrically, which is cell (iv), not Case 2.)
- **Result reported in `theory.md`** (table at end of doc, possibly stale relative to the current 2k canonical numbers): 0.755 ($r = 16$), 0.745 ($r = 64$). Same doc's table also reports older polar-product baseline numbers (0.762, 0.738) that don't match current `load_runs` output (0.7546, 0.7379) — treat the Case 2 numbers as stale and likely needing a re-run if revisited.

#### Read-out

| program | $r=16$ best | $r=64$ best |
|---|---|---|
| (i) Joint Frob | 0.7581 | 0.7527 |
| (ii) Joint op-norm | 0.7680 | 0.7490 |
| (iii) Per-block Frob | 0.7570 | 0.7506 |
| **(iv) Per-block op-norm** | **0.7546** | **0.7379** |
| AdamW baseline | 0.7579 | 0.7550 |

- Per-rank winners both live in cell (iv).
- (iii) and (i) are within ~0.005 of (iv) at $r=64$ but lose by 0.003–0.020 at $r=16$.
- (ii) loses to all others at both ranks (in the E-series form measured here).

**Why work in (iv) here.** Empirical: that's where the best numbers are. Not a theoretical claim — (ii) has a clean Muon-on-$W$-space story that this proposal does not refute. Re-litigating which of (i)–(iv) is right is out of scope.

The rest of this doc specializes to (iv).

### 0.3 What the optimizer is solving (program, block-coordinate iteration, skeleton)

**The coupled program.** For each LoRA pair, the polar-product family targets the per-step program

$$\min_{\Delta A,\, \Delta B}\ \underbrace{\langle u_A,\, \Delta A\rangle + \langle u_B,\, \Delta B\rangle}_{\text{linear cost (Adam direction)}}\ +\ \underbrace{\frac{1}{2\eta}\,\|B \Delta A + \Delta B\, A\|_F^2}_{\text{Frobenius coupling on the joint tangent}} \quad \text{s.t.}\ \underbrace{\|B \Delta A\|_2 \le \tau,\ \ \|\Delta B\, A\|_2 \le \tau}_{\text{per-block spectral constraint}}.$$

Three pieces:

- **Linear cost.** Inner product of the Adam covector with the update. (Variational form has raw $G_A, G_B$; the substitution to $u_A, u_B$ is §2.6's compromise.)
- **Frobenius coupling.** Penalty on $J$. The only term that couples the two blocks: a step on $\Delta A$ alone is meaningful only via $B \Delta A$.
- **Per-block spectral constraint.** Caps the operator norm of each block's contribution to $J$ — the LoRA analogue of Muon's spectral cap on the $W$-space update.

`theory.md` derives this program as the formulation `adam-polar-product-lora` implicitly targets. The radius $\tau$ is the only quantity the program leaves free; how to set it is a research blocker for the operator we'd otherwise use (§0.5).

**Why iterate over blocks.** Solving jointly in $(\Delta A, \Delta B)$ requires a large eigendecomposition per step — too expensive. The family does **block-coordinate (Picard) iteration**:

- Hold $\Delta B$ fixed at its previous inner-iterate; solve the $A$-subproblem.
- Hold $\Delta A$ fixed; solve the $B$-subproblem.
- Repeat for $k$ inner iterations.

**The cross-coupling target.** The off-block enters the on-block only through a single $r \times n$ matrix:

$$T_A\ :=\ -Q_B^\top\, \Delta B_\text{prev}\, A,\qquad T_B\ :=\ -B\, \Delta A_\text{prev}\, Q_A$$

($Q_B, Q_A$ from the QR of $B$ and $A$, defined just below.)

**Iteration order.** Jacobi:

- Both blocks read the *previous pass's* $\Delta A, \Delta B$.
- Both update together at the end of the pass.

(Gauss–Seidel — feed the fresh $\Delta A$ into $B$ within the same pass — is not used.)

Naming:

- $k = 1$ — **uncoupled**. Cross-coupling targets are zero on the first pass.
- $k \ge 2$ — **coupled**. Each pass after the first sees a nonzero target.

**Each block subproblem has three pieces.** The $A$-subproblem (with $\Delta B$ held at $\Delta B_\text{prev}$) is a quadratic in $\Delta A$ with a spectral-norm constraint. It decomposes as:

1. **Whitening.** The Frobenius coupling $\|B \Delta A + \Delta B_\text{prev} A\|_F^2$ has $B$ in front. Take the thin QR

   $$B = Q_B R_B,\qquad Q_B \in \mathbb{R}^{m \times r}\ \text{column-orthonormal},\quad R_B \in \mathbb{R}^{r \times r}\ \text{upper-triangular},$$

   and change variables to $X := R_B \Delta A \in \mathbb{R}^{r \times n}$. Then $B \Delta A = Q_B R_B \Delta A = Q_B X$ and the coupling collapses to $\|X - T_A\|_F^2$. The linear cost becomes $\eta \langle R_B^{-\top} u_A, X \rangle$ (since $\Delta A = R_B^{-1} X$ and $\langle u_A, R_B^{-1} X \rangle = \langle R_B^{-\top} u_A, X \rangle$).

2. **Unconstrained block prox.** Setting the gradient of (linear cost) + $(1/2\eta)$(coupling) to zero:

   $$X_{A,\text{unc}}\ =\ T_A\ -\ \eta\, R_B^{-\top}\, u_A.$$

3. **Spectral operator $\mathcal{P}$.** The constraint $\|X\|_2 \le \tau$ is enforced by an operator $\mathcal{P}(\cdot;\, \tau)$. §2.1 derives the variationally correct $\mathcal{P}$ as **singular-value clip**, but this campaign uses **polar** (clip is deferred — §0.5). The architectural changes (§3) are in the whitening and the lift, not in $\mathcal{P}$.

The $B$-subproblem is symmetric: row QR $A = R_A Q_A^\top$ ($Q_A \in \mathbb{R}^{n \times r}$ col-orthonormal, $R_A \in \mathbb{R}^{r \times r}$); decision variable $Y := \Delta B\, R_A$; target $T_B := -B \Delta A_\text{prev} Q_A$; unconstrained $Y_{B,\text{unc}} = T_B - \eta\, u_B R_A^{-\top}$.

**The lift.** After $\mathcal{P}$ returns $(X_A^\star, Y_B^\star)$, recover a factor pair $(\Delta A, \Delta B)$ whose joint tangent matches the prox solution:

- The map $(\Delta A, \Delta B) \mapsto B \Delta A + \Delta B\, A$ is many-to-one (gauge freedom).
- The family picks the **min-Frobenius representative** — the unique pair minimizing $\|\Delta A\|_F^2 + \|\Delta B\|_F^2$.
- Computed via a small Sylvester solve (formula in §2.5).

For the skeleton, treat $\mathrm{Lift}(X_A^\star, Y_B^\star) \to (\Delta A, \Delta B)$ as a black box.

**Skeleton.** Per step per pair:

$$
\begin{aligned}
&\textbf{inputs: }\ A,\ B,\ u_A,\ u_B,\ \eta,\ k,\ \mathcal{P} \\[2pt]
&\text{QR: }\ B = Q_B R_B,\quad A = R_A Q_A^\top \\[2pt]
&\Delta_A \leftarrow 0,\quad \Delta_B \leftarrow 0 \\[2pt]
&\textbf{for }\ j = 1, \dots, k\ \textbf{ do} \\
&\quad T_A \leftarrow -Q_B^\top\, \Delta_B\, A &&\text{(cross-coupling target;\ }= 0\text{ on }j=1\text{)} \\
&\quad T_B \leftarrow -B\, \Delta_A\, Q_A &&\text{(cross-coupling target;\ }= 0\text{ on }j=1\text{)} \\
&\quad X_{A,\text{unc}} \leftarrow T_A\ -\ \eta\, R_B^{-\top}\, u_A &&\text{(unconstrained }A\text{-block prox)} \\
&\quad Y_{B,\text{unc}} \leftarrow T_B\ -\ \eta\, u_B\, R_A^{-\top} &&\text{(unconstrained }B\text{-block prox)} \\
&\quad X_A^\star \leftarrow \mathcal{P}(X_{A,\text{unc}};\, \tau_A) &&\text{(}\mathcal{P} = \text{polar in this campaign; clip deferred — §0.5)} \\
&\quad Y_B^\star \leftarrow \mathcal{P}(Y_{B,\text{unc}};\, \tau_B) \\
&\quad (\Delta_A,\, \Delta_B) \leftarrow \mathrm{Lift}(X_A^\star,\, Y_B^\star) &&\text{(min-Frobenius gauge, §2.5)} \\
&\textbf{end for} \\[2pt]
&\text{(magnitude rule applied to }\Delta_A, \Delta_B; \text{ §5)} \\
&A \leftarrow A + \Delta_A,\quad B \leftarrow B + \Delta_B
\end{aligned}
$$

Every line was motivated above. Under polar, $\tau_A, \tau_B$ are not meaningful (polar's $UV^\top$ has unit singular values; magnitude is set by the post-Lift rescale, §5). Under clip, $\tau_A, \tau_B$ are the threshold the variational form requires but doesn't pick (§0.5).

### 0.4 The operator $\mathcal{P}$ — polar vs clip

$\mathcal{P}$ acts on a matrix $X$ via its compact SVD $X = U \Sigma V^\top$:

$$\mathcal{P}_\text{polar}(X)\ =\ U V^\top \qquad \text{(every }\sigma_i \to 1;\ \text{magnitude is set externally)}$$

$$\mathcal{P}_\text{clip}(X;\, \tau)\ =\ U\,\mathrm{diag}\bigl(\min(\sigma_i, \tau)\bigr)\,V^\top \qquad \text{(only }\sigma_i > \tau\text{ are capped)}$$

`adam-polar-product-lora`'s current implementation uses $\mathcal{P}_\text{polar}$ (computed via Newton–Schulz). $\mathcal{P}_\text{clip}$ is the variationally correct operator (§2.1) but is deferred for this campaign (§0.5).

Where they differ on $X$'s singular structure:

- On any direction with $\sigma_i < \tau$ (under clip's threshold): polar sets the singular value to 1 (then external magnitude scales it); clip leaves it at $\sigma_i$.
- On directions with $\sigma_i \ge \tau$: clip sets to $\tau$; polar sets to 1.

### 0.5 The threshold $\tau$ — research blocker for clip

The variational form (§2.1) requires a numerical value for $\tau$. We do not have a rule that picks it defensibly across the workload distribution — neither from theory nor from calibration. Sweeping $\tau$ would re-introduce a per-problem hyperparameter, which is what we're trying to avoid. **This is the research blocker for shipping clip** (Q2 in §4).

**This campaign therefore uses $\mathcal{P} = \mathcal{P}_\text{polar}$ in the new architecture (Sylvester min-Frob lift, §3) — no $\tau$ needed.** Polar is the standing baseline; the architecture question (Q1) stands or falls without $\tau$.

The $\tau$-rule (Q2) is not addressed here. The polar runs log spectrum diagnostics (§7) that may inform a future calibration sub-campaign — no claim those alone resolve Q2.

## 1. TL;DR

**Goal.** Test one architectural change to `adam-polar-product-lora` in cell (iv) of the §0.2 survey: replace per-factor RMS-align (which leaves $\Delta A, \Delta B$ off the gauge surface) with a Sylvester min-Frob lift (which puts them on $B^\top \Delta B = \Delta A A^\top$). Operator stays $\mathcal{P}_\text{polar}$.

**Two open questions.**

- **Q1 (testable here).** Does the Sylvester-lift variant beat `adam-polar-product-lora` at both $r=16$ (< 0.7546, currently uncoupled $k=1$) and $r=64$ (≤ 0.7379, currently `-coupled-endrms` $k=2$)? Tested by the §5 sweep.
- **Q2 (research blocker, NOT addressed here).** How is $\tau$ set so clip ships with one fixed defensible value across the workload distribution? Gates clip; not resolved by this campaign.

**Mechanism hypothesis (Q3, speculative).** The off-gauge mismatch in `adam-polar-product-lora`'s pair compounds across Picard iterations at $k \ge 2$. Putting the pair on-gauge via the lift may help especially when $k=2$ hurts `adam-polar-product-lora` at small $r$. Diagnostics in §7 are consistent-with-but-not-proof-of.

**Falsification.** Phase 1 of §5: $k \in \{1, 2\}$ sweep at $r=16$. If no $k$ beats 0.7546 across the $\eta$ grid, the lift doesn't carry. End the campaign.

## 2. Details deferred from §0

§0 stated the program (§0.3) and the algorithmic skeleton (§0.3). This section fills in the pieces §0 deferred:

- §2.1 — proof that $\mathcal{P}_\text{clip}$ is the variationally correct operator. *Deferred-clip theory*; preserved here for the follow-up campaign on Q2.
- §2.3 — choice of the $1/(2\eta)$ penalty weight.
- §2.5 — explicit Sylvester formula for $\mathrm{Lift}$.
- §2.6 — the Adam-covector compromise.
- §2.7 — the $B = 0$ init boundary.

### 2.1 Why $\mathcal{P}_\text{clip}$ is the variationally correct operator

**Theory only — clip is deferred for this campaign (§0.5); this section establishes the variational claim for the follow-up.**

The program of §0.3 was shown by `theory.md` to be the formulation `adam-polar-product-lora` implicitly targets via block-coordinate descent.

Within one block-coordinate step on the $A$-subproblem, fix $\Delta B = \Delta B_\text{prev}$. Apply the change of variables from §0.3: $X := R_B \Delta A$, $T_A := -Q_B^\top \Delta B_\text{prev} A$, $L_0 := R_B^{-\top} u_A$. The subproblem is

$$\min_{X \in \mathbb{R}^{r \times n}}\ \eta\,\langle L_0,\, X\rangle + \tfrac{1}{2\eta}\,\|X - T_A\|_F^2 \quad \text{s.t.}\ \|X\|_2 \le \tau.$$

Steps:

- Drop the constraint. Setting the gradient to zero gives

  $$X_{A,\text{unc}}\ =\ T_A\ -\ \eta\, L_0\ =\ T_A\ -\ \eta\, R_B^{-\top} u_A,$$

  matching the skeleton's $X_{A,\text{unc}}$ line.

- Re-add the constraint $\|X\|_2 \le \tau$. The objective becomes (constant) $+ (1/2\eta) \|X - X_{A,\text{unc}}\|_F^2$ (complete the square). So the constrained problem is the Frobenius projection onto the spectral-norm ball of radius $\tau$ centered at $X_{A,\text{unc}}$.
- That projection is **singular-value clipping**: SVD $X_{A,\text{unc}} = U\Sigma V^\top$, return

  $$X^\star\ =\ U\,\mathrm{diag}\bigl(\min(\sigma_i, \tau)\bigr)\,V^\top.$$

  (Standard fact: the closest matrix in Frobenius norm to $X$ subject to $\|\cdot\|_2 \le \tau$ caps each singular value at $\tau$ and leaves the singular vectors fixed.)

So $\mathcal{P}_\text{clip}$ is the closed form. Polar is a different operator (every $\sigma_i \to 1$, with external magnitude); it has a different fixed point.

The $B$-subproblem is symmetric: row QR $A = R_A Q_A^\top$, decision variable $Y := \Delta B\, R_A$, target $T_B$, linear cost $\eta\, u_B R_A^{-\top}$.

### 2.3 The $1/(2\eta)$ penalty weight

The Frobenius coupling in §0.3 has weight $1/(2\lambda)$ for $\lambda = \eta$. Why $\lambda = \eta$:

- In the limit $T_A = 0$, $\tau \to \infty$ (no cross-coupling, no clipping), the program reduces to the Frobenius-coupled Sylvester closed form.
- Its step Frobenius norm scales as $\eta \cdot \|R_B^{-\top} u_A\|_F$.
- `adam-polar-product-lora`'s update has step Frobenius norm $\eta \|u_A\|_F$, in a different basis ($S_B^{-1/2}$ rather than $R_B^{-\top}$).
- These two bases are inverse-square-roots of $S_B$ that differ by an orthogonal rotation — Frobenius-isometric — so the unclipped-limit step magnitudes match.

$\lambda = \eta$ is the choice that makes the unclipped variational limit step-magnitude-comparable to `adam-polar-product-lora`.

### 2.5 Lift formula

§0.3 introduced $\mathrm{Lift}$ as a black box returning the min-Frobenius representative of the joint tangent. Closed form: solve the small Sylvester equation

$$S_B K + K S_A\ =\ R_B^\top X^\star R_A^\top, \qquad K \in \mathbb{R}^{r \times r}$$

(with $S_B = R_B^\top R_B$, $S_A = R_A R_A^\top$), then

$$\Delta A\ =\ S_B^{-1}\,\bigl(R_B^\top X^\star - K R_A\bigr)\,Q_A^\top \qquad \text{(symmetric for }\Delta B\text{)}.$$

This is the §4 lift formula in `theory.md` with the off-block-diagonal extensions zeroed out — a rank-$r$ tangent has no component orthogonal to $\mathrm{col}(B)$ on the left or $\mathrm{row}(A)$ on the right.

Sylvester solve and spectral preconditioners use existing utilities (`solve_sylvester`, `spdify` in `lora_playground/utils.py`); no new math infrastructure.

### 2.6 The Adam-covector compromise

The §0.3 program was written with raw factor gradients $G_A, G_B$ as the linear cost. The polar-product family substitutes $u_A, u_B$ instead. This breaks an algebraic identity and turns the clean program into a heuristic. Spelled out:

**The identity (raw gradients).** Let $W = BA$ and $G_W := \partial L / \partial W$. The chain rule gives

$$G_A\ =\ B^\top G_W,\qquad G_B\ =\ G_W A^\top,\qquad \therefore\ G_A A^\top\ =\ B^\top G_W A^\top\ =\ B^\top G_B.$$

The identity $G_A A^\top = B^\top G_B$ says the two blocks see consistent linear cost projected through the bilinear parametrization.

**Why it breaks.** Adam preconditioning is applied independently per factor — $u_A$ uses $A$'s own moments, $u_B$ uses $B$'s. There is no reason $u_A A^\top = B^\top u_B$.

**Why we do it anyway.** `adam-polar-product-lora` already lives with this compromise; AdaMuon (arXiv:2507.11005) and NorMuon (arXiv:2510.05491) do too. Using the same compromise across `adam-polar-product-lora` and the new architecture keeps the experiment a clean A/B test of the architectural changes.

### 2.7 Init boundary: $B = 0$ at PEFT step 1

**Where the problem is.** The skeleton in §0.3 contains the line

$$X_{A,\text{unc}}\ =\ T_A\ -\ \eta \cdot R_B^{-\top}\, u_A.$$

PEFT initializes $B = 0$, so at step 1 the QR of $B$ has $R_B = 0$ and $R_B^{-\top}$ does not exist. The skeleton is undefined as written.

**The fallback.** `adam-polar-product-lora`'s implementation replaces the whitening matrix at step 1 by $\delta^{-1/2} I$ — the limit of $S_B^{-1/2}$ as $B \to 0$. Plugged into the skeleton:

$$X_{A,\text{unc}}\ =\ -\eta \cdot \delta^{-1/2}\, u_A,\qquad \Delta B = 0,\qquad \Delta A \propto u_A,$$

i.e. a plain Adam step on $A$ alone, with $B$ held at zero. From step 2 onward, $B \ne 0$ and the standard skeleton runs unchanged.

**Both polar and clip variants inherit this fallback** — step 1 has no information for the operator to act on; step $\ge 2$ runs the full skeleton with whichever $\mathcal{P}$.

**Smoke check.** For early steps where $\sigma_\text{min}(R_B)$ is positive but small, $R_B^{-\top}$ can amplify $u_A$. Log $\|R_B^{-\top} u_A\|_F$ for the first ~50 steps; if it stays within an order of magnitude of $\|u_A\|_F$ the whitening is well-conditioned in practice.

## 3. Current state

All numbers single-seed, 2k-step horizon, sourced from logs.

| rank | current best | $\eta$ | eval @ 2k | source group |
|---|---|---|---|---|
| 16 | uncoupled `adam-polar-product-lora` ($k=1$) | 3e-4 | **0.7546** | `diag_h1234_2x2` |
| 64 | `adam-polar-product-lora-coupled-endrms` ($k=2$) | 3e-4 | **0.7379** | `adam_polar_product_coupled_endrms_2k` |
| 16 | AdamW (baseline) | 3e-4 | 0.7579 | `lr_sweep_2k` |
| 64 | AdamW (baseline) | 3e-4 | 0.7550 | `h3_rsweep_2k` |

Gap to close vs AdamW: +0.0033 at $r=16$, +0.0171 at $r=64$.

**Within-family rank-dependence.** Same family wins both ranks but with different optimal $k$:

| $r$ | $k=1$ | $k=2$ | $k=3$ | $k=4$ |
|---|---|---|---|---|
| 16 | **0.7546** | 0.7616 | 0.7557 | 0.7594 |
| 64 | 0.7453 | **0.7382** ($k=2$ regular) / **0.7379** (-coupled-endrms) | — | — |

Sources for the $k$-sweep at $r=16$: `diag_h1234_2x2`, `adam_polar_product_coupled_rsweep_2k`, `picard_iters_sweep_2x2`. Sources for $r=64$: `polar_product_r64_diag_2k`, `adam_polar_product_coupled_r64_2k`, `adam_polar_product_coupled_endrms_2k`.

**At $r=16$, $k=2$ coupled loses to AdamW** (0.7616 vs 0.7579). No existing config wins at both ranks — the bidirectional goal is unmet.

### Closed dead-ends within cell (iv)

| approach | result | source |
|---|---|---|
| `adam-polar-product-lora` $k$-sweep at $r=16$ ($k \in \{1, 2, 3, 4\}$) | $k=1$ best (0.7546); $k \ge 2$ worse, $k=2$ loses to AdamW | logs above |
| `picard_alpha` damping at $k=2$, $r=16$, $\alpha \in \{0.25, 0.5, 0.75\}$ | 0.7562 / 0.7582 / 0.7600 — interior worse than $\alpha \in \{0, 1\}$ | `logs/alpha_sweep_2x2/` |
| Polar-first composition (`adamuon-polar-product-lora`) | $r=16$ 0.7653, $r=64$ 0.7486 — worse than Adam-first at both | `optimizer_synthesis.md` leaderboard |

(See §0.2 for closed dead-ends in cells (i), (ii), (iii).)

### How this proposal differs from existing variants

Existing variants are written out as math in §0.2. This section says what this proposal changes vs `adam-polar-product-lora`.

#### This proposal — Sylvester min-Frob lift variant in cell (iv)

**One algorithmic change vs `adam-polar-product-lora`** (operator $\mathcal{P} = \mathrm{polar}$ stays the same): replace per-factor RMS-align with a Sylvester min-Frob lift.

$$
\begin{aligned}
\text{Baseline (independent factors):}\quad
&\Delta A = -\eta\, \tfrac{\|u_A\|_F}{\|g_A\|_F}\, g_A,\quad
\Delta B = -\eta\, \tfrac{\|u_B\|_F}{\|g_B\|_F}\, g_B \\
&\text{where }g_A = S_B^{-1/2}\, \mathrm{polar}(S_B^{-1/2} u_A),\ g_B = \mathrm{polar}(u_B\, S_A^{-1/2})\, S_A^{-1/2} \\[4pt]
\text{This proposal (Sylvester lift):}\quad
&(\Delta A, \Delta B) = \mathrm{Lift}\bigl(\mathrm{polar}(\widetilde u_A),\ \mathrm{polar}(\widetilde u_B)\bigr) \\
&\text{s.t.}\ B^\top \Delta B = \Delta A\, A^\top \quad \text{(min-Frob gauge)}
\end{aligned}
$$

`adam-polar-product-lora`'s pair sits off the min-Frob gauge surface ($B^\top \Delta B \ne \Delta A A^\top$ in general); the Sylvester lift projects it onto the surface.

**Note on whitening basis (presentational only).** The proposal's derivation uses QR-whitening $\widetilde u_A = R_B^{-\top} u_A$. `adam-polar-product-lora` uses symmetric-whitening $\widetilde u_A = S_B^{-1/2} u_A$. These two differ by an orthogonal rotation $U$ (since both are inverse-square-roots of $S_B$); polar and clip are gauge-equivariant under such rotation, and the Sylvester lift's RHS satisfies $R_B^\top X^\star R_A^\top = S_B^{1/2} X'^\star S_A^{1/2}$ (where $X^\star, X'^\star$ are the operator outputs in the two bases). So the lift's $K$ — and hence the final $(\Delta A, \Delta B)$ — is the same in either basis. **The whitening basis is a derivation choice with no algorithmic content**; the QR form is used because §2.1 reads cleaner that way (the change of variables $X = R_B \Delta A$ makes the coupling penalty collapse to $\|X - T_A\|_F^2$ with no leftover orthogonal factors).

**Why the Sylvester lift specifically:** it is forced by the §2.1 variational derivation. Clip is the closed form of the per-block prox only when the factor pair satisfies the min-Frob gauge (otherwise the clip is on the wrong representative of the joint tangent). Adopting the lift now is what positions us to swap clip in once Q2 is resolved (§0.5).

**Algorithm.** QR factorize $B = Q_B R_B$ and $A = R_A Q_A^\top$ once per step, then Picard-iterate:

$$
\begin{aligned}
&\textbf{for } j = 1, \dots, k: \\
&\quad T_A = -Q_B^\top\, \Delta B\, A, \qquad T_B = -B\, \Delta A\, Q_A \\
&\quad X_{A,\text{unc}} = T_A - \eta\, R_B^{-\top} u_A, \qquad Y_{B,\text{unc}} = T_B - \eta\, u_B\, R_A^{-\top} \\
&\quad X_A^\star = \mathrm{polar}(X_{A,\text{unc}}), \qquad Y_B^\star = \mathrm{polar}(Y_{B,\text{unc}}) \\
&\quad (\Delta A,\, \Delta B) = \mathrm{Lift}(X_A^\star,\, Y_B^\star) \quad \text{(Sylvester, min-Frob gauge — §2.5)} \\
&\textbf{end for} \\
&\text{RMS-align: } \Delta A \leftarrow \tfrac{\eta\,\|u_A\|_F}{\|\Delta A\|_F}\,\Delta A;\ \Delta B \leftarrow \tfrac{\eta\,\|u_B\|_F}{\|\Delta B\|_F}\,\Delta B.
\end{aligned}
$$

The post-Lift RMS-align is **forced** (not a free choice): $\mathrm{polar}$'s $UV^\top$ has unit singular values, so the lifted update has a magnitude that does not match learning-rate-schedule expectations. AdaMuon / NorMuon / `adam-polar-product-lora` all apply post-spectral rescaling for the same reason. Natural-prox is not an option under polar.

#### Structural contrast

Each row asks one design question; columns show what each variant picked.

| design choice (question) | `adam-polar-product-lora` ($k = 1, 2$) | ProductMuon (cell (ii)) | this proposal (cell (iv) new arch) |
|---|---|---|---|
| **What rotation is applied to the linear cost $u$ before the operator runs?** (presentational; not algorithmic — see note above) | $S^{-1/2}$ symmetric square-root, per block | none — operator runs on a joint $W$-space matrix | $R_B^{-\top}, R_A^{-\top}$ from QR, per block (equivalent to $S^{-1/2}$ under polar + lift) |
| **Does the operator act on each block separately, or on a single matrix combining both?** | per block | joint ($W$-space matrix $D$) | per block |
| **What is the operator?** | $\mathrm{polar}$ (Newton–Schulz) | $\mathrm{polar}$ (Newton–Schulz) | $\mathrm{polar}$ (this campaign); $\mathrm{clip}$ is the variationally correct choice but deferred (§0.5) |
| **How is the overall step magnitude set?** | post-hoc Frobenius rescale of each factor to $\eta\|u\|_F$ | scalar $-\eta$ multiplying the joint $\mathrm{polar}$ output | post-Lift Frobenius rescale (forced under polar; same form as `adam-polar-product-lora`) |
| **Are $\Delta A, \Delta B$ produced independently, or recombined into a self-consistent pair?** | independently — no recombination | recombined via Sylvester (min-Frob gauge) | recombined via Sylvester (min-Frob gauge) |
| **Through what does the off-block influence the on-block?** | only via shared $S_A, S_B$ ($k=1$); plus a cross-term added to $u^\text{eff}$ ($k=2$) | implicit (joint $W$ matrix is unfactored) | explicit cross-coupling target $T$ + Sylvester lift |

#### Implication for interpretation

A win in this campaign tests one thing: whether putting the $(\Delta A, \Delta B)$ pair on the min-Frob gauge surface changes empirical performance. At $k=1$, that is the only difference from `adam-polar-product-lora`. At $k \ge 2$, the lift also changes how cross-coupling is computed (the previous-pass-lifted pair feeds the cross-term, and the augmentation sits in $X$-space rather than $u$-space) — but those are downstream of the same lift change, not independent design knobs.

## 4. Open questions

- **Q1 (architecture, testable here).** Does the Sylvester min-Frob lift variant (polar, otherwise matching `adam-polar-product-lora`) beat `adam-polar-product-lora` at $r=16$ (< 0.7546) and at $r=64$ ($\le 0.7379$)? Phase 1 of §5 is decisive at $r=16$; Phase 2 at $r=64$.
- **Q2 (research blocker for clip, NOT addressed here).** How is $\tau$ set so that clip ships with one fixed defensible value across the workload distribution? Possible families of rules to investigate later (none committed to):
  - $\tau \propto \eta\|u_A\|_F \cdot f(r)$ with $f$ chosen so $\tau$ matches AdaMuon's RMS-align scale (analogy to validated work).
  - $\tau \propto$ median $\sigma_i(X_{\text{unc}})$ (data-derived).
  - $\tau$ chosen so a fixed fraction of singular values are clipped.

  Resolving Q2 requires either (a) theoretical derivation or (b) a separate calibration sub-campaign that defines what "right $\tau$" means on a controlled problem and measures it. The polar runs in this campaign log spectrum diagnostics (§7) that *may* be useful input for a future calibration. **No claim that they alone resolve Q2.** Q2 explicitly remains open at the end of this campaign even if Q1 succeeds.
- **Q3 (mechanism, speculative).** Why does coupled $k=2$ hurt at $r=16$ in `adam-polar-product-lora`? Does the Sylvester lift fix it? Diagnostics: cross-coupling magnitude $\|T_A\|_F / \|u_A\|_F$ vs $r$; gauge-deviation $\|B^\top \Delta B - \Delta A A^\top\|_F / \|\Delta A A^\top\|_F$ logged for `adam-polar-product-lora` runs (which lacks the lift) to quantify how far off-gauge the per-factor pair is. Treat as evidence consistent with the mechanism, not proof.

**Out-of-scope follow-up.** Clip is gated on Q2. If Q1 also fails (architecture doesn't carry), the polar-product family may be hyperparameter-saturated within the per-block-spectral program, and the next campaign should look at a different program (cells (i), (ii), (iii) of §0.2) or a different parametrization altogether (DoRA, periodic adapter merging, rank-adaptive).

## 5. Sweep design

**Primary axes:**

| axis | values | meaning |
|---|---|---|
| $r$ | $\{16, 64\}$ | LoRA rank |
| $k$ | $\{1, 2\}$ | Picard inner iterations (§0.3) |
| $\eta$ | $\{1\mathrm{e}{-4},\ 3\mathrm{e}{-4},\ 1\mathrm{e}{-3}\}$ (extend on boundary) | learning rate |

**Magnitude rule (forced).** Polar requires a magnitude rule (its $UV^\top$ output has unit singular values). This campaign uses post-Lift RMS-align to $\eta\|u_A\|_F, \eta\|u_B\|_F$ — the same rule `adam-polar-product-lora` uses, and the same pattern AdaMuon (arXiv:2507.11005) and NorMuon (arXiv:2510.05491) use. No alternative is included.

**Sequencing.**

1. **Phase 1 — $r=16$, $k$-sweep.** $k \in \{1, 2\}$, $\eta$ over the 3 values = 6 cells. Decisive on Q1 at $r=16$. Extend $\eta$ by one value on boundary.
2. **Phase 2 (conditional).** If Phase 1's best beats 0.7546, run $r=64$, $k \in \{1, 2\}$, $\eta$ at the best from Phase 1 (extend on boundary). Decisive on Q1 at $r=64$.

If Phase 1 fails, end the campaign without resolving Q2 (clip is still gated regardless).

Q3 diagnostics ride along on every cell; Q2 spectrum diagnostics ride along on every cell as raw data for a possible future τ-calibration sub-campaign.

## 6. Decision rule

Shipped optimizer must satisfy **both** thresholds with **the same $k$**:

| rank | threshold |
|---|---|
| $r=16$ | $< 0.7546$ |
| $r=64$ | $\le 0.7379$ |

(Strict at $r=16$ because that's where the family currently has no winning config; non-strict at $r=64$ because the existing winner is in this family.)

Outcomes (all assume Q2 still unresolved):

- **Both met, same $k$.** Rank-stability holds on this workload — necessary condition for shipping the polar variant satisfied. Cross-workload stability is a separate follow-up; clip remains gated on Q2.
- **Both met, different $k$ per rank.** Problem-tunable on the rank axis; do not ship.
- **One met, other regresses.** Rank-locked; do not ship.
- **Neither met.** The lift doesn't carry on this workload; do not ship.

## 7. Diagnostics

Logged per pair every 200 steps. Strict inequalities ($\sigma > \tau$, not $\ge$) where applicable.

- **Cross-coupling magnitude (Q3):** $\|T_A\|_F / \|u_A\|_F$, measured per Picard iteration. Compare $r=16$ vs $r=64$ to test whether cross-coupling magnitude correlates with the $k=2$-hurts-at-small-$r$ pattern.
- **Gauge-deviation (Q3):** $\|B^\top \Delta B - \Delta A\, A^\top\|_F / \|\Delta A\, A^\top\|_F$. Measured under `adam-polar-product-lora` (no lift) — quantifies how far off-gauge the per-factor pair is. Should be ≈ 0 in the new architecture (sanity check on the lift).
- **Finite-step ratio:** $\|\Delta B \Delta A\|_F / \|B \Delta A + \Delta B A\|_F$. Bilinear second-order term, recorded for cross-run comparison.
- **Spectrum statistics of $X_{A,\text{unc}}, Y_{B,\text{unc}}$ (Q2 raw data, NOT a Q2 answer):** $\sigma_\text{max}$, $\sigma_\text{min}$, P10/P50/P90, participation ratio $(\sum \sigma_i)^2 / \sum \sigma_i^2$, ratio $\sigma_\text{max} / (\eta\|u_A\|_F)$. Logged for completeness; framed as raw inputs that *might* inform a future $\tau$-calibration sub-campaign, not as a measurement that resolves Q2.

## 8. Reproducibility

- Submission via `slurm_scripts/submit.sh` with `SWEEP_SCOPE=ext_compare,polar_family` and explicit purpose string.
- New optimizer registered as `adam-polar-product-lora-sylvester` (or similar) in `OPTIMIZER_CHOICES`; entries added to `OPTIM_COLORS` and at least one `OPTIM_FAMILIES` set in `lora_playground/plot_utils.py`.
- Analysis via `lora_playground.loader.load_runs(where=…)`; never hand-typed group lists.
- Unit tests for the new architecture (in `tests/test_polar_product.py`):
  1. **Determinism** on a tiny tensor with fixed seed.
  2. **Min-Frobenius gauge:** $\|B^\top \Delta B - \Delta A\, A^\top\|_F \le 10^{-5} \cdot \|\Delta A\, A^\top\|_F$ after the lift.
  3. **Sign:** with $T_A = 0$ (i.e. $k = 1$) and the linear cost dominating, $\langle u_A, \Delta A \rangle < 0$.
  4. **K isolates the lift change.** At $k=1$ with $T_A = T_B = 0$, artificially set the Sylvester correction $K = 0$ in the lift. The resulting $(\Delta A, \Delta B)$ — before RMS-align — must match `adam-polar-product-lora`'s per-factor geometric step $g_A, g_B$ up to the (Frobenius-isometric) basis rotation between $S^{-1/2}$ and $R_B^{-\top}$. Validates that $K \ne 0$ is the only algorithmic difference.

(Tests for the deferred clip variant — Sylvester limit, polar limit — belong in the follow-up campaign once Q2 is resolved.)
