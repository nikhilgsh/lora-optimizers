# Critical design review — resolved snapshot

This doc was a mid-investigation triage of failing optimizer variants. All
four diagnoses below have since resolved (validated, ported, or closed).
Current standings, leaderboard numbers, and live mechanism questions live
in `docs/notes/optimizer_synthesis.md`. Term definitions (Hybrid Picard,
RMS-align, Adam covector, polar block solve, etc.) are in
`docs/notes/glossary.md`.

## Diagnoses — outcomes

- **Diagnosis 1 — `*-Post` step-magnitude drift (`adam-{lin,scaled}-lora-post`).**
  Validated. Without RMS-align, σ_min(S_B) climbed 0.011→1.08 over training,
  varying the step magnitude ~100× at fixed lr. The RMS-align fix shipped;
  `adam-scaled-lora-post` at η=3e-4 reaches 0.7570 (within 0.0009 of AdamW
  at r=16). See `optimizer_synthesis.md` Bucket 1 entry "H4 RMS-aligned
  *-Post".

- **Diagnosis 2 — `muon-adam-lora` missing AdaMuon stabilizers
  (sign(M) before NS, V on NS-output only, RMS-align).** Validated. The
  AdaMuon-faithful port `adamuon-lora` is registered in `optim.py`. At
  r=64 it reaches 0.7515 (Δ=−0.0035 vs AdamW); at r=16 it ties AdamW at
  0.7603. The polar-first family is no longer falsified, but in this
  project Adam-then-polar wins among orderings. See
  `optimizer_synthesis.md`:H1 narrative and the Bucket 2 entries.

- **Diagnosis 3 — matrix-Adam (`adam-{lin,scaled}-lora-matrix`).** Closed
  as not productive. After the v_pair=mean fix, r=16 best is 0.7744 and
  r=64 best is 0.7723 — both decisively worse than the corresponding
  per-coord Adam variants. Trading per-coord Adam for direction
  preservation costs more than it buys.

- **Diagnosis 4 — polar-product ordering (`adam-polar-product-lora` vs
  `adamuon-polar-product-lora`).** Tested. Adam-then-polar beats
  polar-then-V at every measured (r, η): at r=64 single-seed, 0.7453 vs
  0.7486; at r=16, 0.7546 vs 0.7653. AdaMuon's pretraining-scale design
  argument does not transfer to LoRA fine-tune scale on this benchmark.
  Spectral-product geometry is the load-bearing piece across both
  orderings; Adam-first is the headline.

## Pointers

- Current leaderboard, buckets, and live mechanism questions:
  `docs/notes/optimizer_synthesis.md`.
- Term definitions: `docs/notes/glossary.md`.
- Coupled polar (the family that contains the current single-seed
  leaders): `docs/notes/polar_product/investigations.md`.
