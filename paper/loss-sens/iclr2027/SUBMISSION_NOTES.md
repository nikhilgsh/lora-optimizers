# ICLR 2027 submission scaffold — status and gaps

Build directory for an ICLR 2027 submission of `paper/manuscript/main.tex`.
Nothing in `paper/manuscript/` was modified; this tree is a separate build.

Everything below was read off the ICLR 2027 pages and the official style bundle
on 2026-09-21. Sources are listed per fact in the last section.

## 1. Deadlines

| Item | Date | Source |
|---|---|---|
| Abstract submission | **Sep 18, 2026, 11:59 PM AOE** | CFP |
| Full paper + supplementary material | **Sep 25, 2026, 11:59 PM AOE** | CFP / Author Guidelines |
| Submission site | <https://openreview.net/group?id=ICLR.cc/2027/Conference> | Author Guidelines |

AOE = Anywhere on Earth, UTC−12.

> **The abstract deadline has already passed.** Today is 2026-09-21; abstracts
> were due 2026-09-18. ICLR requires the abstract to be registered ahead of the
> paper deadline, so if no abstract was registered on OpenReview for this paper
> by Sep 18, a Sep 25 paper submission is not possible. Confirm on OpenReview
> before spending any more effort on the items below. Everything else here
> assumes an abstract was registered.

## 2. Page limit verdict

| | |
|---|---|
| ICLR 2027 limit, initial submission | **9 pages**, strict, main text only |
| This build, main text | **10 pages** (References begin 72% down page 10) |
| **Overage** | **1 page** |

The main text has to lose about three quarters of a page of content to fit.

What does *not* count toward the limit, so none of it is part of the 10:
references (unlimited), appendices (unlimited, placed after the bibliography —
this draft already does that), and the AI use / ethics / reproducibility
statements. The limit rises to 10 pages for rebuttal and camera-ready, so the
current draft would fit *after* acceptance but not at submission.

Page map of the 25-page build: main text 1–10, references 10–12, appendix
contents 13, appendices A–H 14–25.

## 3. Compile status

Compiles clean with the pinned module:

```
source /etc/profile.d/modules.sh && module load texlive/20240312
latexmk -pdf -interaction=nonstopmode main.tex
```

- `latexmk` exit 0, `main.pdf`, 25 pages.
- 0 unresolved references, 0 unresolved citations.
- **0 `Overfull \hbox`**, 0 `Overfull \vbox`. None in display math.
  (`iclr2027_conference.sty` sets `\sloppy`, which resolves tight lines as loose
  ones instead. 3 `Underfull \hbox` at body lines 298, 659, 752, and 2 underfull
  `\vbox` from `\flushbottom` — all cosmetic.)
- Title, abstract, display math, lemmas, algorithms, figures, tables, `cleveref`
  links and ICLR line numbering all render correctly; pages 1 and 4 were
  inspected as images.

## 4. Unmet ICLR requirements

Ordered by severity. None of these were fixed here — items 1 and 2 are prose
changes, which were out of scope for this scaffold.

### Desk-reject risks

- [ ] **De-anonymization: the GitHub URL in the introduction.** The rendered PDF
  contains `https://github.com/nikhilgsh/polora` on page 2 (`main.tex` line
  155). The path carries the first author's GitHub username. ICLR: *"Any paper
  where author identity is revealed in either the main text or the supplementary
  material will be desk rejected."* Replace with an anonymized repository
  (e.g. anonymous.4open.science) or drop the URL and point at supplementary
  material instead.

- [ ] **Main text is 1 page over the 9-page limit.** See §2. ICLR describes it
  as a *"strict upper limit"*.

- [ ] **AI use statement is missing.** ICLR 2027 requires it, and it does not
  count toward the page limit. It must state which disclosure-required tasks
  generative AI was used for, which it was not used for, and which are not
  applicable; plus recommended-disclosure tasks and how AI-assisted work was
  reviewed. A fill-in template sits in `iclr2027_conference.tex` under the
  heading "AI use statement". Disclosure is also required separately on the
  OpenReview submission form.

### Recommended, not mandatory

- [ ] **Reproducibility statement is missing.** Strongly encouraged: one
  paragraph at the end of the main text, before the references, that
  *references* where the reproducibility material lives (Appendix A already has
  the full configuration, Appendix B the proofs) rather than restating it. Does
  not count toward the page limit.

- [ ] **Ethics statement is missing.** Optional; needed only if the work raises
  Code of Ethics questions. Probably not applicable here, but decide explicitly.

### Mechanical / to verify before upload

- [ ] **Author block still holds real names in the source.** `main.tex` lines
  103–108 still carry the three author names, the corresponding-author email and
  the Flatiron Institute affiliation. Nothing leaks today: with `\iclrfinalcopy`
  commented out, `iclr2027_conference.sty` ignores `\author` entirely and prints
  "Anonymous authors / Paper under double-blind review" instead — verified, the
  strings `Flatiron`, `nghosh` and `Gower` appear zero times in the rendered
  PDF. But the block is one uncommented macro away from being published, so
  replace it with the anonymous placeholder before upload and restore it only
  for camera-ready.

- [ ] **Confirm reciprocal reviewing is satisfied.** Every submission needs at
  least one author registered to review at least 3 papers; authors on 3+
  submissions must review at least 6. No author may be a co-author on more than
  20 submissions.

- [ ] **Decide on the citation style.** This build keeps the manuscript's
  `biblatex` numeric-comp citations rather than switching to ICLR's
  `iclr2027_conference.bst` author-year default. ICLR does not mandate a
  citation style and numeric is common at the venue, but the style file ships
  the `.bst` for the house style. Switching means replacing `\printbibliography`
  with `\bibliographystyle{iclr2027_conference}\bibliography{references}`,
  dropping the natbib shim, and re-checking the page count — author-year
  citations are longer and will cost main-text space.

### Non-issues, checked

- **arXiv preprint is fine.** The paper is on arXiv as arXiv:2607.17620
  (submitted 2026-07-20) under the real author names. ICLR explicitly permits
  this: posting to arXiv during review is allowed, and it does not violate
  anonymity as long as the submission does not point at it. Do not cite it as
  the authors' own work.
- **Dual submission is fine**, provided the paper is not under review at another
  conference or journal. arXiv and non-archival workshops are exempt.
- **Self-citations are in third person.** `Ghosh` (3×) and `Parshakova` (1×)
  appear in the rendered PDF only as bibliography entries and third-person
  citations, which ICLR allows.
- **Appendices already sit after the bibliography**, as ICLR requires.

## 5. What this build changes relative to the manuscript

The body is byte-identical to `paper/manuscript/main.tex`; verified with `diff`.
Only the preamble differs, and only where the class change forces it:

- `\documentclass[11pt]{article}` → `\documentclass{article}` (the style file
  sets 10pt).
- Dropped `\usepackage[letterpaper,margin=1in]{geometry}`; the style file sets
  the page box (5.5in × 9.0in text block).
- Dropped `libertinus`, added `times`, per the ICLR template.
- Added `\usepackage{iclr2027_conference,times}`.
- Added a natbib shim. `iclr2027_conference.sty` does `\RequirePackage{natbib}`,
  and `biblatex` aborts with *"Incompatible package 'natbib'"* if natbib is
  loaded. The style file's only use of natbib is one `\setcitestyle` call on its
  line 23, so the shim marks natbib as pre-loaded (making the `\RequirePackage`
  a no-op), stubs `\setcitestyle`, then clears the marker again before
  `biblatex` loads. `natbib.sty` is never read. The manuscript's bibliography
  setup then works unchanged.

The author block, `\printbibliography`, and all prose are untouched.

Directory contents: the official style bundle (`iclr2027_conference.sty`,
`.bst`, `.bib`, the template `iclr2027_conference.tex`, `math_commands.tex`,
`fancyhdr.sty`, `natbib.sty`) unpacked from `iclr-2027-style-files.zip`;
`main.tex`; `references.bib` (copy); `figs/` (symlink to
`../manuscript/figs`).

## 6. Sources

| Fact | URL |
|---|---|
| Abstract and paper deadlines, AOE, double-blind, arXiv allowed | <https://iclr.cc/Conferences/2027/CallForPapers> |
| 9-page limit, references/appendix excluded, dual submission, arXiv, reproducibility and ethics statements, co-authorship and reciprocal reviewing, OpenReview URL, supplementary deadline | <https://iclr.cc/Conferences/2027/AuthorGuidelines> |
| AI use policy: required disclosure tasks, mandatory in-paper section, also on the submission form | <https://iclr.cc/Conferences/2027/AIPolicyForAuthors> |
| Style bundle (official 2027 release — not a placeholder) | <https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip> |
| "strict upper limit of 9 pages … expanded to 10 pages for rebuttal/camera ready"; AI use statement "required"; ethics and reproducibility "recommended" | `iclr2027_conference.tex` lines 131, 399, 416, 430 (in this directory) |
