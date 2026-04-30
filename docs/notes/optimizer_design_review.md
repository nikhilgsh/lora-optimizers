# Critical design review: failing optimizer variants

Triggered by user's question on whether the failures look like bugs or
fundamental flaws. This doc enumerates the suspicious variants, diagnoses
the most likely cause for each, and proposes a falsifiable fix.

## Status of each "failure"

| optimizer                    | status     | best loss | gap vs AdamW | suspicion level |
|------------------------------|------------|-----------|--------------|------------------|
| adam-lin-lora-post           | losing     | 0.79+ (in flight, η=1e-3) | +0.03 | **High — likely magnitude bug** |
| adam-scaled-lora-post        | losing     | 0.84+ (in flight, η=1e-3) | +0.06 | **High — likely magnitude bug** |
| adam-lin-lora-matrix         | resubmitted with v_pair=mean fix; pending | tbd | tbd | medium — initial submission was a different bug, fix needs validation |
| adam-scaled-lora-matrix      | resubmitted; pending      | tbd | tbd | medium |
| muon-adam-lora               | losing     | 0.80 at η=1e-3 step 600 | +0.02 trending | **High — three known divergences from AdaMuon paper** |
| product-muon-lora            | losing     | ~0.81 at step 500 (extrapolated) | +0.05 | medium — geometry might be right but magnitude is off |
| original adam-{lin,scaled}-lora | tied with AdamW | 0.756 / 0.757 | ≈ 0 | not a bug — H1 explains why |

The first two are the highest-suspicion items because they were *meant* to
fix the H1 mechanism but lost worse than the unfixed originals — that's
inconsistent with the simple narrative "Adam → geometry preserves geometric
information that Adam → no-geometry doesn't."

## Diagnosis 1 — H4 *-Post: step magnitude varies by 100× over training

### Symptom

`adam-{lin,scaled}-lora-post` at η=1e-3 (the best η for the original
adam-lin-lora) at step 1400: lin-post 0.79, scaled-post 0.84. Both worse
than plain AdamW (0.78 at the same step). At smaller η no different — at
η=3e-4 step 1600 lin-post is 0.83, scaled-post 0.86.

### Root cause

In *-Post, the geometric step is `S_B⁻¹ · u_A` where `u_A = m̂_A /
(√v̂_A + ε)` is Adam's sign-like direction with ‖u_A‖_F ≈ √(r·d_in) ≈ 256
for r=16, d_in=4096.

- `‖S_B⁻¹ u_A‖_F = ‖u_A‖_F / σ_min(S_B)` (worst case).
- From H1 trajectory: σ_min(S_B) climbs **0.011 → 1.08 over 2000 steps**,
  driven by ‖B‖² growth.
- So `‖step‖_F = lr · ‖S_B⁻¹ u_A‖_F` varies by ~**100×** over training at
  fixed lr.

**Effective lr drifts:** the same lr=1e-3 produces a step of ‖_F ≈ 23000
(*lr·256/0.011*) at step 20 and ‖_F ≈ 240 (*lr·256/1.08*) at step 1500.
Either we're badly overshooting early (causing bad early dynamics that the
optimizer never recovers from), or the η we picked is sized for late
training but is gigantic early, or the η is sized for early training but
is tiny late. Any way it's cut, lr means different things at different
steps.

This is a classic problem solved in the Muon literature by **RMS-aligned
rescaling** — see AdaMuon (arxiv 2507.11005) Algorithm 1 line: γ_t =
0.2·√(mn) / ‖Õ‖_F. Idea: rescale the geometric direction so its Frobenius
norm equals a target (the bare Adam step's ‖_F). Decouples geometry from
magnitude.

### Fix (implemented as commit pending — see below)

For both `AdamScaledLoRAPost` and `AdamLinLoRAPost`, replace the bare
geometric step with an RMS-aligned variant:

```python
# Compute lr-free geometric direction
geo_A = solve_spd(SB, u_A)           # for Scaled
# or, for Lin: geo_A = -S_B^{-1}(u_A + γ K' A) with K' lr-factored

# Rescale: ‖ΔA‖_F = lr · ‖u_A‖_F
dA = lr * (‖u_A‖_F / ‖geo_A‖_F) * geo_A
```

With this, `‖ΔA‖_F = lr · ‖u_A‖_F` regardless of S_B's conditioning.
Geometry only contributes *direction*; magnitude is set by lr.

### Falsifiable predictions

If the magnitude drift was the dominant issue:
1. **Best η for *-Post should now be similar to AdamW's best** (≈ 3e-4)
   instead of 1e-3.
2. **Eval at the new best η should be ≤ 0.7579 ± 0.005** (tying or
   beating plain AdamW).
3. **Trajectory should match plain AdamW closely early in training**
   (because S_B⁻¹ on a sign vector is approximately a rotation when σ_min
   is tiny — the geometric correction shouldn't kick in meaningfully
   until S_B has structure).
4. If after rescale the loss is still ≥ 0.005 above AdamW: the geometric
   direction itself is uninformative on a sign-input, *not* a magnitude
   issue.

These give clean pass/fail criteria.

### How to test

Smoke at η=3e-4 for 100 steps on the canonical fixture: should drop train
loss from ~3.5 to ~1.0 in line with what plain AdamW does. If the rescaled
optimizer crashes / NaNs / fails to learn, fix is wrong. Then a 4-run mini
sweep (η ∈ {1e-4, 3e-4, 1e-3} × {scaled-post, lin-post}, 500 steps) on the
cluster decides whether the fix transfers to scale.

## Diagnosis 2 — muon-adam-lora: missing AdaMuon's three innovations

### Symptom

`muon-adam-lora` (NS → Adam) at η=1e-3 m=1 step 600: 0.80 (worse than
AdamW 0.78). At η=3e-3 m=4: 1.5045 (diverging).

### Root cause

Our implementation is "naïve NS-then-Adam":
- NS the raw gradient
- Run full Adam (m, v, bias correction, m̂/√v̂) on the NS output

AdaMuon (arxiv 2507.11005) does this composition successfully and
identifies three details that matter:

1. **Sign-stabilize before NS:** O_t = NS(sign(M_t), T), not NS(M_t).
   sign() bounds the input magnitude per-coord so NS sees a stationary
   distribution and produces a more consistent O_t.
2. **Don't run Adam's first moment on NS output:** AdaMuon only tracks v
   on O_t, not m. The "first momentum" is just gradient momentum (M_t) on
   raw G; that's what feeds NS. Running an additional m̂ EMA on the NS
   output is *double smoothing* — slows response without adding signal.
3. **RMS-align the step:** γ_t = 0.2·√(mn)/‖Õ_t‖_F, again for magnitude
   stability.

We do none of these. Each one alone could plausibly explain the gap.

### Proposed fix

Implement a faithful AdaMuon port: `AdaMuonLoRA`. Per-factor independently
(matching the rest of our LoRA-mode optimizers):
- M_t = β·M_{t-1} + G_t  (gradient momentum)
- O_t = NewtonSchulz(sign(M_t), T)
- V_t = β·V_{t-1} + (1−β)·O_t ⊙ O_t
- Õ_t = O_t ⊘ (√V_t + ε)
- γ_t = 0.2·√(mn) / ‖Õ_t‖_F
- ΔW = -η·γ_t·Õ_t

Reuse `_newton_schulz` from the existing MuonLoRA impl.

### Falsifiable predictions

If the failures of muon-adam-lora are due to the three missing pieces:
1. AdaMuon-style port should beat muon-adam-lora at every η.
2. Best η for AdaMuon should match AdamW's best (3e-4) due to RMS-align.
3. Final eval should be ≤ adam-muon-lora's 0.7557 (because AdaMuon is
   essentially adam-muon-lora's reverse-order with better stabilization,
   so we expect comparable performance).

If AdaMuon-style still loses to AdamW: something specific to the NS-on-
sign-of-momentum composition doesn't transfer to the LoRA fine-tune
regime.

## Diagnosis 3 — pre-style adam-{lin,scaled}-lora-matrix

### Symptom

First submission (sum-of-squares v_pair): didn't learn at all
(eval=1.187 throughout = random init).

### Root cause

v_pair tracked Σ g² (sum) instead of mean. √v̂ ≈ √N · RMS(g), effective lr
= lr/√N ≈ lr/700 for r=16 LoRA shapes. Step magnitude was ~1/700 of what
the η suggested → near-zero updates.

### Fix (already shipped: commit ac81bba)

Track mean square: v_pair = β·v_pair + (1-β)·(Σ g² / N_total).

### Validation

Verified: η=1e-3 step 50 → eval 0.886 (was 1.187 broken). Resubmitted as
job 6312759 — pending in queue.

### Outstanding question

Even with the magnitude fix, matrix-Adam's *direction* is the
geometrically preconditioned EMA m̂_A scaled by per-pair magnitude. Per
H1, m̂_A on `precond_A = S_B⁻¹ ∇A` should preserve the geometric
direction (unlike per-coord v̂ which would normalize it away). So
matrix-Adam should produce a *different* step than plain Adam. Whether
that step is *better* is what the sweep will show.

## What to actually run next

In priority order:

1. **Smoke RMS-aligned *-Post on local A6000.** Pass: training loss drops
   from random init to ≤ 1.5 in 50 steps at η=3e-4 (expected for any
   reasonable optimizer at this scale). Cost ~ 2 minutes wall.

2. **If smoke passes, queue a 500-step pilot** at η ∈ {1e-4, 3e-4, 1e-3}
   × {lin-post, scaled-post}, 6 runs. Decision criterion: **best final
   eval < 0.78**. If yes → resubmit full 2k sweep at the winning η. If no
   → conclude the geometric direction is uninformative on a sign-input and
   close H4.

3. **Implement AdaMuonLoRA** (Diagnosis 2). 1-class addition. Smoke + 500-
   step pilot in parallel with (1)-(2). Comparable cost.

4. **Wait for matrix-Adam (H5 resubmitted)** — already running fix; not
   actionable until results land. ~2h.

5. **Park the original H4 sweep (job 6312277).** It's still running with
   the unfixed *-Post code. Either let it finish (deterministic
   falsification at the unfixed parameter values, useful for the doc) or
   cancel to free GPUs. **Recommendation: let it finish** — it's mostly
   done, the data point "unfixed *-Post loses at every η" is publishable
   evidence that the magnitude drift is the dominant failure mode.

## Provisional acceptance criteria for "fix succeeded"

- *-Post (after RMS-align): final eval < 0.7479 at any η. Below 0.7579
  counts as a partial success, ≥ 0.7579 counts as falsification of the
  *-Post family on this benchmark.
- AdaMuonLoRA (after implementation): final eval < 0.7557 (current
  headline). Below AdamW (0.7579) is a partial success.
- matrix-Adam (after fix): same as *-Post.

Each is a single number with no ambiguity.
