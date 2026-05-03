# lin-lora / scaled-lora investigation

## TL;DR

- At r=16, both `adam-lin-lora` (0.7581) and `adam-scaled-lora` (0.7572) sit essentially tied with AdamW (0.7579) at the canonical 2k-step horizon; `adam-muon-lora` (0.7557) is the only family member that clearly beats AdamW at r=16.
- Diagnostics confirm Adam's per-coordinate $\hat v^{-1/2}$ erases the geometric ($S_B^{-1}$) correction throughout training (cos_B $\geq 0.94$ from step 20), so pre-precondition compositions reduce to $\varepsilon$-perturbed AdamW by construction.
- At r=64, both `adam-lin-lora` (0.7527) and `adam-scaled-lora` (0.7506) beat AdamW (0.7550); `adam-scaled-lora` at r=64 is the strongest entry seen in this investigation. Cosine diagnostics at r=64 are not yet collected.

> **Label scope note.** This doc's hypothesis labels (H1–H5) are local to the lin/scaled investigation and do not correspond to the H1–H6 labels in `docs/notes/glossary.md` (which are anchored to the coupled-polar / Picard family). Cross-doc citations should be scoped, e.g. "lin-scaled H3".

## Leaderboard (best η per optimizer, r=16, 2k steps)

For the full cross-investigation leaderboard see `docs/notes/optimizer_synthesis.md`.

| rank | optimizer            | best η  | eval loss | source                      | beats AdamW?    |
|------|----------------------|---------|-----------|-----------------------------|-----------------|
| 1    | adam-muon-lora       | 3e-3    | 0.7557    | `adam_muon_2k`              | yes, Δ = −0.0022 |
| 2    | adam-lin-lora        | 1e-3    | 0.7581    | `h1_pre_probe_2k`           | tied             |
| 3    | adam-scaled-lora     | 1e-3    | 0.7572    | `optim_compare_high_eta_2k` | tied             |
| 4    | adamw                | 3e-4    | 0.7579    | `lr_sweep_2k`               | baseline         |

Target for any productive new entry: below AdamW's 0.7579 single-seed at the 2k-step r=16 horizon.

---

## Working hypothesis

Composing Adam *after* the geometric (Sylvester / Gram) solve lets Adam's per-coordinate $\hat v^{-1/2}$ normalize away the cross-coordinate scale structure the geometric step just installed. "Gram solve" here means rescaling by $S_B^{-1}$ (or $S_A^{-1}$) where $S_B := B^\top B + \delta I$ is the Gram matrix of the $B$ factor — the spectral preconditioner of the [glossary](glossary.md#optimizer-concepts). We use the arrow notation $X \to Y$ for composition order, where $X$ is applied first and $Y$ second on the result; e.g. `Adam → NS` means Adam EMA on factor gradients followed by Newton–Schulz polar of the resulting Adam covector, and `Adam → S⁻¹` means Adam covector followed by Gram-inverse rescaling. Per `feedback_beat_dont_match.md`, a productive change must come in below AdamW's 0.7579 single-seed at the canonical 2k-step horizon. Multi-seed verification is deferred.

---

## H1 — Adam's $\hat v^{-1/2}$ wipes out the geometric correction (CONFIRMED)

The diagnostic run (`logs/h1_diag_2k`) instrumented `adam-lin-lora` and `adam-scaled-lora` at η=1e-3, r=16, with `--log_optim_diagnostics` writing per-pair cos(Δ_lin, Δ_adamw), Frobenius norms, $\|A\|_F / \|B\|_F$, and $\sigma_{\min}/\sigma_{\max}$ of $S_A, S_B$ every 20 steps. The falsifier was median cos > 0.95 (geometric step ≈ AdamW step → confirmed) vs cos < 0.7 throughout with losses still matching (falsified, look elsewhere).

Final values at step 2000:

| optimizer        | cos_A | cos_B | $\|dA_\text{lin}\|/\|dA_\text{raw}\|$ | $\sigma_{\min}(S_B)$ | $\|B\|_F$ | final eval |
|------------------|-------|-------|----------------------------------------|-----------------------|-----------|------------|
| adam-lin-lora    | 0.84  | 0.94  | 0.25                                   | 1.08                  | 7.68      | 0.7581     |
| adam-scaled-lora | 0.88  | 0.97  | 0.27                                   | 1.14                  | 6.52      | 0.7592     |

Trajectory (adam-lin-lora): cos_A rises 0.46 → 0.84 in the first ~500 steps then plateaus; cos_B stays in $[0.94, 0.99]$ throughout. $\sigma_{\min}(S_B)$ climbs $0.011 \to 1.08$ driven by $\|B\|^2$ growth — Gram conditioning *improves* over training, exactly when most of the loss reduction happens.

Verdict: H1 is confirmed. Adam's per-coordinate $\sqrt{\hat v}$ erases the geometric correction throughout training (cos_B $\geq 0.94$ from step 20). The only meaningful direction divergence is on $A$ in the first ~500 steps, before $B$ leaves zero, and even there the geometric step is consistently a quarter the magnitude of plain AdamW. Pre-precondition compositions are $\varepsilon$-perturbed AdamW by construction. The productive change must reorder Adam and the geometric solve, which motivates H4.

## H3 — Benefit at small r vs large r

The r-sweep (`logs/h3_rsweep_2k`) covered $r \in \{2, 4, 64\}$ × {adamw, adam-lin-lora, adam-scaled-lora} × η ∈ {3e-4, 1e-3} via disBatch (r=16 was already in `optim_compare_2k_1ep` / `lr_sweep_2k`). The falsifier for H3's premise was: if the gap stays $< 0.005$ at r=2, conditioning is not the bottleneck for this base+dataset.

Final eval at η=3e-4, step 2000:

| r  | adamw  | adam-lin-lora | adam-scaled-lora |
|----|--------|---------------|-------------------|
| 2  | 0.7920 | 0.8150        | 0.8134            |
| 4  | 0.7807 | 0.8024        | 0.8001            |
| 64 | 0.7550 | 0.7527        | 0.7506            |

At r=2 and r=4, lin/scaled lose to AdamW by ~0.02 — the small-r premise is wrong, consistent with H1's mechanism (per-coord $\hat v$ dominates). At r=64, both lin/scaled beat AdamW, with adam-scaled-lora at 0.7506 the strongest entry in this investigation. A working hypothesis for why H1 doesn't fully apply at r=64: at higher r the LoRA factor matrices are larger, so per-coord $\hat v$ cannot fully wash out the cross-coordinate scale structure that $S_B^{-1}$ installs. Cosine diagnostics were not enabled at r=64 — a clean follow-up.

## H4 — Productive: Adam on raw grads, geometric solve on Adam step (FALSIFYING)

H4 swaps the composition order to `Adam → S⁻¹`: Adam state on raw $(\nabla A, \nabla B)$; compute the unitless Adam direction $u = \hat m / (\sqrt{\hat v} + \varepsilon)$; feed $u$ as a synthetic gradient through the LinLoRA / ScaledLoRA geometric step; apply lr afterwards. The intent is for $\hat v$ to adapt to natural gradient distribution while the geometry installs the $(A, B)$-coupled rotation post-hoc. New optimizers `AdamLinLoRAPost`, `AdamScaledLoRAPost` were added to `lora_playground/optim.py` with 7 unit tests in `tests/test_optim_post.py`. The η-sweep (`logs/h4_post_2k`) covered η ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} × 2 optimizers, r=16, 2k steps.

In-flight result at step 1400, η=1e-3: `adam-lin-lora-post` 0.7923, `adam-scaled-lora-post` 0.8421.

Verdict: H4 is falsifying. Applying $S_B^{-1}$ to a sign-like Adam step does not produce a useful direction. Compare to `adam-muon-lora`, which uses the same `Adam → geometry` composition order but with Newton–Schulz polar instead of $S^{-1}$ and *does* beat AdamW (0.7557). Provisional rule: post-Adam corrections work iff they are structurally meaningful on a sign-magnitude input — Newton–Schulz polar (spectral cap) qualifies, $S^{-1}$ (Gram-inverse rescaling) does not.

## H5 — Productive: per-pair scalar second moment (matrix-Adam) — IN FLIGHT

H5 (in flight as of 2026-04-30) keeps `AdamLinLoRA` / `AdamScaledLoRA`'s flow (geometry-then-Adam composition) but replaces per-coord $\hat v$ with a single scalar EMA per $(A, B)$ pair tracking $\|\text{precond}_A\|_F^2 + \|\text{precond}_B\|_F^2$. Direction comes from per-element $\hat m$; only magnitude is adaptively rescaled per pair. We refer to this preconditioner shape as **matrix-Adam** (one second-moment scalar per matrix-pair, in contrast to per-coordinate Adam). New optimizers `AdamLinLoRAMatrix`, `AdamScaledLoRAMatrix` were added to `lora_playground/optim.py` with 6 additional unit tests in `tests/test_optim_post.py`. The sweep (`logs/h5_matrix_2k`) covers η ∈ {3e-5, 1e-4, 3e-4, 1e-3, 3e-3} × 2 optimizers, r=16, 2k steps.

A first attempt was broken: at η ∈ {3e-5, 1e-4} step 1000+, eval stayed at 1.187 (random init). The bug was per-pair $v_\text{pair}$ tracking $\sum g^2$ instead of mean, giving $\sqrt{\hat v} \approx \sqrt{N} \cdot \text{RMS}(g)$ and effective lr $= \text{lr}/\sqrt{N} \approx \text{lr}/700$ for typical LoRA shapes — no learning at the standard η range. The fix (commit ac81bba) divides by $N_\text{total} = \text{numel}(A) + \text{numel}(B)$ so $\hat v$ tracks mean square; verified at η=1e-3 step 50 → eval 0.886 (was 1.187 at step 1000 broken). The corrected sweep is currently in flight; final values pending.
