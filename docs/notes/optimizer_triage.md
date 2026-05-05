# Optimizer triage — DRAFT for manual review

The leaderboard in `notebooks/sweep_analysis.ipynb` gets crowded because (a) hyperparameter sweeps create many entries that share a parent algorithm and (b) all plots use the same global color scheme. The fix is structural, not a registry purge:

- **Leaderboard cell** collapses HP-children to their parent's row; an extra "tuned" row appears only if best-HP differs from default.
- **Non-leaderboard panels** drop the global color scheme — default matplotlib colors disambiguated by linestyle/marker, with AdamW pinned black.

Given that, the only triage axis we need is **"does this optimizer get its own leaderboard row?"** Retirement to a historical doc is a separate decision and not required for the leaderboard to be clean.

Two buckets:

- **A — Leaderboard-eligible.** Scientifically-distinct optimizer (current go-to OR required baseline OR active investigation).
- **B — HP-child.** Differs from a Bucket-A parent only by hyperparameters; folds into parent's row.

A third "**Other**" section lists everything else. These stay registered, may appear in dedicated within-family panels (e.g. a gauge sub-figure or a sign-norm sub-figure), and are simply not on the leaderboard. Retire-to-historical-doc is a future call you can make per-entry.

Anchors: `docs/notes/optimizer_synthesis.md`, `docs/notes/optimizer_design_review.md`, `docs/notes/polar_product/anderson_2026_05_03.md`, `docs/notes/polar_product/preconditioning_saturation_2026_05_03.md`, `lora_playground/optim.py:264-323`.

User: please re-bucket as needed. **"User-flag"** lines mark cells where I'm least confident.

---

## Bucket A — Leaderboard-eligible

### Baselines

| optimizer | reason |
|---|---|
| `adamw` | Reference baseline. Required for every Δ. Multi-seed σ source (`logs/adamw_multiseed/`). |
| `adafactor` | Memory-efficient, non-Adam-V baseline. |

### Current single-seed leaders

| optimizer | reason |
|---|---|
| `adam-polar-product-lora` | r=16 single-seed leader at k=1 (0.7546). Headline "Adam-then-polar" composition. |
| `adam-polar-product-lora-coupled` | r=64 single-seed leader at k=3 (0.7364, ~13σ improvement, landed 2026-05-03 in commit dadea5d per `polar_product/anderson_2026_05_03.md`). Project headline result. **Note:** `optimizer_synthesis.md` still cites the older k=2 / 0.7382 number; that table needs an update separately. |

### Scientific baselines (load-bearing for mechanism stories)

| optimizer | reason |
|---|---|
| `adam-lin-lora` | Pre-Adam Sylvester preconditioning. Anchor for the H1 cos-diagnostic story (Adam's $\sqrt{\hat v}$ erases upstream geometric rotation). r=64 → 0.7527. |
| `adam-scaled-lora` | Pre-Adam Gram-solve. Pair to `adam-lin-lora`. r=64 → 0.7506. |
| `adamuon-lora` | AdaMuon-faithful port. Canonical "polar-first with stabilizers" baseline. r=64 → 0.7515. Required to disambiguate ordering effects vs `adam-polar-product-lora`. |
| `adam-muon-lora` | Muon-NS applied to the Adam direction. Simplest Muon×Adam composition; r=64 → 0.7515. |
| `muon-lora` | Pure Muon (no Adam V). The "no-Adam Muon" anchor — without it the NS-contribution-Δ claim has no reference (synthesis: ns=0 baseline at 0.95+, NS contributes ≥0.18 nat). |

---

## Bucket B — HP-child (folds into parent's row)

Strictly HP-only: differs from a Bucket-A parent by a single scalar/integer hyperparameter value, *not* by a structural change to the update operator. Leaderboard groupby key becomes `(parent_label, lora_r)`; an extra `(parent_label, lora_r, "tuned")` row appears only when best-HP differs from default.

Clip-vs-polar, gauge-fix, RMS-align placement, etc. are **not** HP folds — they are separate algorithms (different update operator) and live in "Other" below.

| variant | folds into | varied HP |
|---|---|---|
| `adam-polar-product-lora-coupled` (k=1, 2, 4 sweep cells) | `adam-polar-product-lora-coupled` (best-k = 3 default) | `picard_iters` |
| `adam-muon-lora` LoRA+ m=4 sweep cells | `adam-muon-lora` (m=1 default) | `lora_plus_m`. Per CLAUDE.md "don't stack LoRA+", default is m=1. |
| AdamW LoRA+ m sweep cells | `adamw` (m=1 default) | `lora_plus_m` |
| `muon-lora` ns_iters sweep cells | `muon-lora` (ns_iters=5 default) | `ns_iters` |

---

## Other — not on leaderboard

Stays in the registry. May appear in dedicated within-family panels (e.g. a gauge-axis sub-figure for the polar family, a sign-norm sub-figure for the coupled-core family) or get retired to a historical doc later. No leaderboard row either way. Brief blurbs so the manual review can decide whether each should appear in *some* plot or just sit in the registry.

### No-Adam variants

| optimizer | note |
|---|---|
| `lin-lora` | No-Adam Sylvester. Superseded by `adam-lin-lora`. |
| `scaled-lora` | No-Adam Gram. Superseded by `adam-scaled-lora`. |
| `polar-product-lora` | No-Adam polar. Superseded by `adam-polar-product-lora`. |

### Closed branches per `optimizer_design_review.md`

| optimizer | note |
|---|---|
| `adam-lin-lora-post` | Diagnosis 1: post-Adam Sylvester correction falsified. |
| `adam-scaled-lora-post` | Diagnosis 1: RMS-aligned `*-scaled-post` ties AdamW within 0.0009 — synthesis calls it in-noise tie, not a win. |
| `adam-lin-lora-matrix` | Diagnosis 3: matrix-Adam decisively worse than per-coord. |
| `adam-scaled-lora-matrix` | Diagnosis 3: same as `adam-lin-lora-matrix`. |
| `muon-adam-lora` | Diagnosis 2: predecessor of `adamuon-lora`, missing AdaMuon stabilizers. |
| `adamuon-polar-product-lora` | Diagnosis 4: polar-first ordering, loses to Adam-first at every measured (r, η). |

### Bucket3-weak per `OPTIM_FAMILIES`

| optimizer | note |
|---|---|
| `product-muon-lora` | Gauge-invariant Muon attempt. Theoretically promising, empirically weak. |
| `adam-product-muon-lora` | Same family. |
| `diag-scaled-lora` | Diagonal K-FAC. Synthesis: 0.8153, ~0.06 nat worse than AdamW. |
| `kron-grad-lora` | Diagonal K-FAC. Synthesis: 0.8263. |
| `psi-lora` | `bucket3_weak`. Not on leaderboard. |
| `galore-adamw` | ~3× slower per step, doesn't beat plain LoRA at matched compute. |

### Coupled-core (joint-operator-norm) cluster

Investigation E1–E7 in `polar_product/investigations.md`. The cluster shares one code path: the projected-quotient-polar core solver in `_polar_coupled_core_step` / `_polar_coupled_core_lift` (`optim.py`), which solves the joint operator-norm problem in core space rather than running Picard's per-factor polar iteration. Variants differ along three axes:

- **gauge** (KKT lift constraint): `min-frobenius` (default), `imbalance-preserve-scalar`, `imbalance-preserve`, `imbalance-restore`, `balanced-scalar` — each picks a different point in the freedom group.
- **pre_polar_normalize**: `None` or `sign` (per-coord sign-normalize the core covector before polar — Adam-like adaptivity in core space without EMA).
- **state_rebalance**: post-step factor rotation to enforce iLoRA invariant $AA^\top = (r/m) B^\top B$, preserving $BA$ exactly.

Plus three base-momentum classes:
- `PolarCoupledCoreLoRA` — no momentum (raw factor grads).
- `MuonCoupledCoreLoRA` — Muon-style transported core EMA + Nesterov on the rotating $(Q_L, Q_R)$ basis.
- `PolarCoupledCoreFactorAdamLoRA` — Adam EMA on factor grads, then the core solver (closest direct analog of Picard).

**Per-variant best 2k eval (loaded via `load_runs`; AdamW reference: r=16 = 0.7579, r=64 = 0.7550; Picard r=64 best = 0.7364 at k=3).**

| optimizer | E-id | what it does | r=16 | r=64 | takeaway |
|---|---|---|---|---|---|
| `polar-coupled-core-lora` | E1 / E3 | §2.1 baseline: min-Frobenius gauge, no momentum, no rebalance. E3 = same code at wider lr (3e-2). | 0.8049 (lr 3e-2) | 0.7490 (lr 3e-2) | Best of cluster at r=64 but +0.013 vs Picard. r=16 catastrophic. Ceiling of "raw-grad core solver." |
| `polar-coupled-core-state-rebalanced-lora` | E2 | E1 + post-step iLoRA rebalance every step. | 0.8104 | 0.7686 | Worse than E1 at both ranks. Imbalance is not the lever. |
| `polar-coupled-core-imbalance-scalar-lora` | gauge variant | E1 with `imbalance-preserve-scalar` gauge: keep $\|A\|/\|B\|$ ratio invariant in lift. Ablates the gauge axis vs E1's `min-frobenius`. | (no canonical-horizon logs) | (no logs) | Registered but never sweep-run at 2k. |
| `polar-coupled-core-imbalance-lora` | gauge variant | E1 with `imbalance-preserve` gauge: full per-pair imbalance preserved. Ablates gauge. | (no logs) | (no logs) | Registered but never run. |
| `polar-coupled-core-imbalance-restore-lora` | gauge variant | E1 with `imbalance-restore` gauge: actively restore initial imbalance. Ablates gauge. | (no logs) | (no logs) | Registered but never run. |
| `polar-coupled-core-balanced-scalar-lora` | gauge variant | E1 with `balanced-scalar` gauge: enforce $\|A\| = \|B\|$ scalar balance. Ablates gauge. | (no logs) | (no logs) | Registered but never run. |
| `polar-coupled-core-sign-lora` | E4 | E1 + sign-normalize core covector before polar (per-coord adaptivity in core space, no EMA). | 0.7680 (lr 1e-4) | 0.9395 (lr 1e-4) | Best of cluster at r=16 (+0.010 vs AdamW); blows up at r=64. Per-coord adaptivity helps small r, kills high r. |
| `polar-coupled-core-sign-rebalanced-lora` | E6 | E4 ⊕ E2: sign-norm + rebalance compound. | (no logs) | (no logs) | Registered; not run separately at 2k (E6 numbers in investigations.md come from `muon-coupled-core-sign-rebalanced-lora`). |
| `polar-coupled-core-factor-adam-lora` | E7 | Adam-EMA on factor grads, then projected-quotient-polar core solver. Tests whether factor-Adam is the ingredient Picard wins on (substituting our core solver for Picard's iteration with the rest matched). | 0.7846 (lr 1e-4) | 1.097 (diverged) | Falsifies "factor-Adam is the Picard secret" — the core-solver consumes the adaptivity differently than Picard's iteration. CLAUDE.md flagged this composition as theoretically suspect ("factor-Adam-then-compatibility-project"). |
| `polar-coupled-core-factor-adam-rebalanced-lora` | E7+E2 | E7 + post-step rebalance. | 0.8147 | (no r=64 log) | No improvement over E7 at r=16. |
| `muon-coupled-core-lora` | E5 | Muon-style transported core EMA + Nesterov on $(Q_L, Q_R)$ basis. | 0.8546 (lr 3e-3) | 0.8077 (lr 3e-3) | Far worse than E1. Transport residual eats the momentum signal; canonical Muon's "constant basis" assumption breaks under rotating $(Q_L, Q_R)$. |
| `muon-coupled-core-imbalance-scalar-lora` | gauge × E5 | E5 with `imbalance-preserve-scalar` gauge instead of `min-frobenius`. Ablates the gauge axis under transported core EMA. | (no logs) | (no logs) | Registered but never run. |
| `muon-coupled-core-imbalance-lora` | gauge × E5 | E5 with `imbalance-preserve` gauge. Ablates the gauge axis under transported core EMA. | (no logs) | (no logs) | Registered but never run. |
| `muon-coupled-core-balanced-scalar-lora` | gauge × E5 | E5 with `balanced-scalar` gauge. Ablates the gauge axis under transported core EMA. | (no logs) | (no logs) | Registered but never run. |
| `muon-coupled-core-state-rebalanced-lora` | E5+E2 | E5 + post-step rebalance. | 0.8641 | 0.8248 | Worse than E5 alone. |
| `muon-coupled-core-sign-lora` | E5+E4 | E5 + sign-norm. | 0.7858 | 1.2402 (diverged) | Sign-norm helps at r=16 over plain E5; high-r divergence is a sign-norm consequence, not a Muon one. |
| `muon-coupled-core-sign-rebalanced-lora` | E5+E4+E2 | E5 + sign-norm + rebalance (full stack). | 0.7684 | 0.9440 | Matches E6 numbers in investigations.md. r=16 best of muon family; r=64 diverged. |

**Cluster takeaway.** The whole joint-operator-norm direction loses to Picard. Best r=64 in the cluster (E1 at lr=3e-2) is 0.7490 vs Picard's 0.7364 at k=3. r=16 best (E4 sign-norm) is 0.7680 vs AdamW 0.7579 — still a loss. Sign-norm is the only ingredient that helps at r=16 but it diverges at r=64; everything else (rebalance, imbalance gauges, Muon transport, factor-Adam) is neutral-to-harmful. Six variants were never run at the 2k horizon — registered but inert in the registry.

### Polar-family algorithm variants (separate operator, not HP)

These share the polar-product backbone but change the update operator (clip replaces polar, gauge fix adds a structural step, RMS-align placement is a structural choice). Not HP folds. Currently not on leaderboard; revisit individually if any becomes a current go-to.

| optimizer | note |
|---|---|
| `adam-polar-product-lora-coupled-endrms` | RMS-align placement variant. Structural choice, not a parameter. |
| `adam-polar-product-lora-gauge` | Adds gauge-fix step over `adam-polar-product-lora`. |
| `adam-polar-product-lora-gauge-coupled` | Gauge-fix on coupled variant. |
| `adam-polar-product-lora-clip-gauge` | Gauge-fix + clip combo. |
| `adam-polar-product-lora-clip-gauge-coupled` | Gauge-fix + clip on coupled variant. |
| `adam-clip-product-lora` | Replaces polar block solve with clip operator. Separate algorithm. |
| `adam-clip-product-lora-coupled` | Clip operator on coupled variant. |
| `adam-clip-product-lora-coupled-endrms` | Clip + RMS-align placement. |

### Other variants not on leaderboard

| optimizer | note |
|---|---|
| `adam-ucv-core-lora` | Orthogonal-core UCV parameterization (factorize the LoRA update as $UCV^\top$ with $U, V$ orthogonal, $C$ free). Recently added (commit 72b7dfb); project conclusion so far is weak — does not beat polar/Picard. |
| `adam-lin-core-lora` | `adam-lin-lora` variant that runs the Sylvester preconditioner in core space rather than on factors. Ablates the operator-application location (factor-space vs core-space) for the lin family. Not on leaderboard. |
| `adam-soap-polar-product-lora` | SOAP-style (Shampoo-on-momentum) preconditioner before the polar block. Ablates the upstream preconditioner choice for the polar family (Adam vs SOAP). Not on leaderboard. |
| `adafactor-polar-product-lora` | Adafactor (rank-1 second-moment) preconditioner before the polar block. Ablates the upstream preconditioner choice (Adam vs Adafactor). Not on leaderboard. |
| `sign-momentum-polar-product-lora` | Sign-of-momentum (no second moment) before the polar block. Ablates the upstream preconditioner choice (Adam vs sign-momentum). Not on leaderboard. |

### Placeholders / non-LoRA-optimizer modes

| optimizer | note |
|---|---|
| `sgd` | Placeholder. |
| `sgd-m` | Placeholder. |
| `svd-step-adamw` | Not a LoRA optimizer — uses `--training_mode svd_step_oracle`. Belongs in its own oracle-mode section. |
| `svd-cumulative-adamw` | Same as above with `svd_cumulative_oracle`. |

---

## Confidence

- **High** (synthesis/anderson/closeout/investigations-doc-backed): Bucket A entries; all "Other" closed branches including the 17 coupled-core (E1–E7) variants.
- **Medium**: Bucket B HP-variant assignments — mechanically correct, but you may want to promote a "headline tuned" row to A for visibility.
