# Data-loading pipeline — known issues + next-session followups

Captured 2026-05-07 at end of A0.7 DDP refactor session. Several issues with
the current data path interact badly with the cheap-wins stack (compile, FA-2/4)
and the campaign protocol (seq=2048, single-pass, paper-grade horizon). Fix
these before Phase B/C campaign launch.

> **Status (2026-05-08): implemented as `packed_v1`.**
> Issues 1–4 below are addressed by the new pipeline:
> sequence packing on the train side (issues 1, 2, 4), pad-to-max
> on eval (issue 1), prompt-masked loss (issue 3), SDPA-only
> doc-aware attention (issue 4). The CLI flag
> `--data_pipeline_version` (default `packed_v1`, fallback
> `unpacked_v0` for back-compat) selects the path; the value is
> recorded in cfg events and manifests. Runs prior to 2026-05-08
> are tagged `unpacked_v0`. Re-anchoring of AdamW noise floor under
> `packed_v1` is the gating follow-up before the campaign.
>
> See `lora_playground/data.py`, `tests/test_data_pipeline.py`
> (forward + backward equivalence proven on a tiny Llama), and
> `scripts/data/backfill_pipeline_version.py` (one-shot manifest
> tagger, already run on 174 historical sweeps).

## The current path

`prepare_data.py` tokenizes Magicoder examples one-per-row, with truncation
and **right-padding to longest-in-batch** via HuggingFace's
`DataCollatorForLanguageModeling`. Each example carries an `input_ids` sequence
that's `min(actual_doc_length, max_seq_length)` tokens, padded only in the
collator at batch time.

Two consequences flow from this:

1. **Per-batch shapes are dynamic.** Different batches have different padded
   lengths (longest doc in that batch). Compile-graph cache key includes shape,
   so this triggers `torch.compile` to recompile per shape.
2. **Padding fraction is high at large seq.** Magicoder docs average ~663
   tokens; at seq=2048 the average padded fraction is ~70%. Compute spent on
   padding is wasted (loss is masked, but fwd+bwd kernels still run).

## Issue 1: dynamo recompile pathology with `--compile` + dynamic shapes

**Observed:** at 8B + `--compile` + DDP + the standard collator, dynamo
recompiles every step or every-few-steps because input shapes vary. Same
pathology was hit with FA-4 earlier in the session. Recompile takes 60-90s
each time; over a 6k-step run that's hours of wasted time and the wall
budget table in `walltime_profile.md` (which assumed compile delivered its
~1.3× speedup) becomes optimistic.

**Why we missed it in A0.1:** the bench script (`bench_optimizer_step.py`)
uses a fixed-shape random-token batch, so dynamo compiles once and never
recompiles. The bench was honest about per-step compute given a static
shape, but the production path through `train.py` has dynamic shapes that
recompile. The 1B baseline + AFTER profile rows in `walltime_profile.md`
were measured via the bench, NOT via train.py — so they're slightly
optimistic for the true production path.

**Action item:** before Phase B/C, either:
- (a) Switch to a **pad-to-`max_seq_length` collator** so every batch has
  identical shape `(batch, max_seq_length)`. Cost: ~70% padding compute waste
  at seq=2048 (same as today, just made explicit). Compile recompiles only
  once. Easiest fix; preserves existing tokenization.
- (b) Switch to **sequence packing** (preferred). Concat all docs, optionally
  with EOS separators, slice into N×`max_seq_length` chunks. Each chunk has
  full signal density (no padding). Compile recompiles once because all
  batches have the same shape. Saves the wasted compute AND fixes the recompile.
- (c) Disable `--compile` for the campaign. Simplest but gives up the 1.15-1.3×
  E2E speedup measured in A0.1 and undoes part of the cheap-wins gain.

(b) is what we want long-term. (a) is the conservative interim if (b) is
risky to ship before campaign start.

## Issue 2: Padding waste at seq=2048

At seq=2048 with Magicoder's ~663-token average docs:
- Useful tokens per sample: ~663
- Padded tokens per sample: ~1385 (68%)
- Effective signal-to-compute ratio: ~32%

For 8B at 6k steps × batch_eff=16 × seq=2048 = 196M forward-pass tokens, only
~63M are signal. We're paying for 196M tokens of compute but training on 63M.

**Sequence packing fixes this completely.** With packing:
- Each 2048-token chunk is fully populated (modulo a small last-chunk leftover)
- 196M forward-pass tokens ≈ 196M signal tokens
- Effectively 3× more signal per GPU-hour at the same step count

This is also the path that brings our planned 270M-token horizon into the
"feasible on single 24h SLURM wall at 8B with cheap wins" regime.

## Issue 3: Loss masking under packing

When packing concatenates `[doc1_tokens, doc2_tokens, doc3_tokens, ...]` into
a single `seq_length` chunk, all tokens contribute to the loss by default.
This is the standard SFT-with-packing recipe (Schulman, Biderman both use it).
For Magicoder specifically:
- Each example is `Instruction:\n{instr}\n\nResponse:\n{resp}`
- Best practice is to mask the prompt and only compute loss on the response
  tokens (using `labels = -100` for prompt positions)
- Current `format_example` concatenates instruction+response into one text
  blob with no prompt/response boundary, so all tokens contribute. This is
  fine for SFT-as-LM but loses the "instruction-tune" framing.

**Decision needed at campaign time:** match Biderman/Schulman by adding
prompt-masking, or accept the LM-style loss and document the choice. If we
add prompt-masking, the packing implementation needs to track per-document
boundaries to produce the right `labels` tensor.

## Issue 4: Cross-document attention under packing

Standard packing concatenates documents into one seq=2048 chunk, but the
default causal attention mask lets each token attend to ALL preceding tokens
in the chunk — including tokens from earlier documents. This is incorrect
attention; tokens should only attend within their own document.

**Fix:** use a **document-aware attention mask** (also called "block
attention" or "intra-document" attention) that resets attention boundaries
between documents. Flash Attention 2 supports this via the
`attn_implementation="flash_attention_2"` path with `position_ids` reset at
document boundaries. For sdpa, need to construct a custom 4D mask.

This is the small implementation cliff that makes "packing" a real engineering
task rather than a 1-hour feature. Frameworks like axolotl, llm-foundry, and
Schulman's vLLM repro all have working implementations to reference.

## Recommended next-session order

1. **Decide between pad-to-max (a) and packing (b).** Packing is better but
   harder. If campaign launch is imminent, ship (a) first as a quick fix,
   then upgrade to (b) for the actual production runs.
2. **Implement** the chosen path in `prepare_data.py` and `train.py`'s
   collator. Test with a unit test that verifies shapes are constant across
   batches and (for packing) loss matches single-pass training within float
   noise on a tiny fixture.
3. **Re-run A0.1-style profile** under the new data path to get the TRUE
   compile + sdpa speedup numbers (the bench-script measurements were too
   optimistic; production path was hitting recompiles silently).
4. **Re-run the DDP scaling smokes** (1B/3B/8B with compile) once shapes
   are static — those measurements were waiting on this fix.
5. Update `walltime_profile.md` with the post-fix numbers; verify Phase B/C
   wall-budget verdicts still hold.

## Reference implementations to mine

- **HuggingFace `transformers`**: `DataCollatorForSeq2Seq` and the recent
  `pack_dataset` utility in datasets ≥ 2.x.
- **axolotl**: production-grade SFT packing with prompt-masking,
  document-aware attention, and intra-document EOS handling.
- **Schulman's vLLM repro** (`~/lora-without-regret-repros/lora-without-regret`):
  reference implementation matching the recipe we're targeting.
- **llm-foundry**: their `streaming` data loader + `IcyDataset`-style packing.

Mining their implementations will save us reinventing the document-aware
attention mask, which has subtle correctness traps.
