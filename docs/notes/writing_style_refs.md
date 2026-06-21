# Writing-style references for the experiments section

Exemplars to emulate when drafting `paper/manuscript/main.tex` §4 (Experiments).
Style only — borrow structure and phrasing discipline, not content. PDFs in `docs/papers/`.

## Read closely

**SOAP (`soap_2409.11321`) — §5 + §6.1, pp. 7–8.** Best template for this paper.
- §5 "Experimental Methodology": three bolded-lead paragraphs (Hyperparameter tuning /
  Throughput Measurement / Efficiency Benefits), each one protocol decision + the *why*.
  Model for converting our §4.1 bullet dump into labeled paragraphs.
- "Efficiency Benefits" paragraph + Fig. 2: defines "steps/wall to reach the baseline's
  final loss" precisely *before* showing it, reads it off with dashed crossing lines.
  This is our speedup metric, done well. (We already borrow its §6.1 title
  "Measuring Efficiency Benefits".)

**LoRA-RITE (`lora_rite_2410.20625`) — §5, pp. 8–9.** Closest genre (published LoRA optimizer).
- Optimizer roster: "We compare the following optimizers:" → one bullet per baseline,
  one line of "what it is" each. Our body is missing this (baselines currently live only
  in figure captions).
- LR-sweep stated in one sentence + "rank r=16 based on the ablation"; exhaustive config
  pushed to the appendix.

## Skim when drafting (not yet read here)

- **iMuon (`imuon_2605.09238`)** — direct competitor, same spectral-LoRA framing; see how a
  same-subfield, same-month paper frames the comparison.
- **Mousse (`mousse_2603.09697`)** — same whiten–polar–unwhiten family; dense full-rank member.
