# arXiv submission metadata — PoLoRA

**arXiv:2607.17620** (https://arxiv.org/abs/2607.17620), submitted 2026-07-20 as
`submit/7844106` (TeX Live 2023 selected). Paper password `43aam` — coauthors
claim authorship with the id + password via arXiv's "Claim Ownership with a
password" form.

Paste-ready fields, checked against arXiv's Title / Authors / Abstract conventions
(https://info.arxiv.org/help/prep.html). Kept in `paper/` because
`arxiv_submission/` is deleted and rebuilt by `paper/make_arxiv.sh`.

## Title

```
PoLoRA: A Preconditioned Orthogonalized LoRA Optimizer
```

Mixed case, no unicode, no TeX macros.

## Authors

```
Nikhil Ghosh, Tetiana Parshakova, Robert M. Gower
```

Firstname Lastname order, comma-separated, no honorifics, no truncation, no
unicode. Affiliations are optional and omitted here since all three authors
share one institute and the title page already states it; if added, arXiv
requires them in parentheses after each name.

## Abstract

Plain text, 1244 characters (limit 1920). No leading whitespace on any line, no
unicode, no TeX macros, does not begin with the word "Abstract".

```
Low-rank adaptation (LoRA) makes finetuning large language models cheaper by adding to each weight matrix a trainable low-rank update parameterized as the product of two matrices. These matrices are usually trained with Adam, which treats them as a single flat vector of parameters and ignores both the matrix and product structure of LoRA. Applying a matrix-aware optimizer such as Muon to each factor does not consistently improve over Adam, and neither do the product-aware Muon variants proposed in concurrent works. To realize consistent gains, we introduce PoLoRA, a Preconditioned Orthogonalized LoRA optimizer built from three ingredients: a product-aware spectral update direction, curvature preconditioning derived from controlling the per-sample loss change, and a magnitude rule that controls the sizes of both the factor and merged updates. We evaluate PoLoRA on instruction-tuning datasets for code and math across models from 1B to 8B parameters, and find that it reaches the final held-out loss achieved by tuned Adam in 1.2-1.7 times fewer steps, while adding at most 3% per-step overhead. Compared to Adam, PoLoRA is also less sensitive to the learning rate, and its optimal learning rate is stable across ranks.
```

Note: the en dash in "1.2-1.7" from the manuscript is replaced by a hyphen here
(no unicode in metadata).

## Categories

- Primary: `cs.LG`
- Cross-list: `math.OC` — the proofs appendix is optimization content (constrained
  steepest descent, LMO solutions, norm duality), and it reaches the
  Muon/Scion/Shampoo theory audience.
- Cross-list: `cs.CL` — every experiment is LLM instruction tuning; LoRA-RITE
  (arXiv:2410.20625), the closest prior LoRA optimizer, used cs.LG + cs.AI + cs.CL.
- Skipped: `cs.AI` (too generic to route readers), `stat.ML` (content is
  optimization, not statistics), `math.NA` (the Newton-Schulz and power-iteration
  appendix is implementation detail, not a numerical-analysis contribution).

## Comments field

```
27 pages. Code: https://github.com/nikhilgsh/polora
```

## Processing options at upload

- **TeX Live version: 2023.** The shipped `main.bbl` is biblatex bbl format 3.2;
  TeX Live 2023 accepts 3.2 and 3.3, TeX Live 2025 accepts only 3.3. Selecting
  2025 fails with `File 'main.bbl' is wrong format version - expected 3.3`,
  which empties the bibliography and leaves every citation undefined. The
  cluster's `texlive/20240312` module emits 3.2, so producing a 3.3 bbl requires
  recompiling under TeX Live 2025 (e.g. via Overleaf's TeX Live setting).
- arXiv writes `00README.json` itself to record these choices; do not author it
  by hand.

## Before uploading a revision

1. `./paper/sync.sh pull` to pick up coauthor edits; commit, then
   `./paper/sync.sh publish`.
2. Rerun `paper/make_arxiv.sh` and confirm the tarball postdates the last
   manuscript commit.

## After the arXiv id arrives

1. Fill `eprint` and `url` in the citation block of `~/polora/README.md`; commit
   and push.
2. Make the polora GitHub repository public.
