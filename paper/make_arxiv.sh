#!/bin/bash
# Rebuild the arXiv submission package from paper/manuscript/.
#
# Produces polora_arxiv.tar.gz at the repo root, containing main.tex (with
# internal provenance comments and the todonotes author macros stripped),
# main.bbl, and only the figures main.tex actually includes. No .bib is
# shipped: arXiv would otherwise re-run BibTeX and regenerate the .bbl under a
# different biblatex version.
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/paper/manuscript"
DST="$REPO/arxiv_submission"
TARBALL="$REPO/polora_arxiv.tar.gz"

rm -rf "$DST"
mkdir -p "$DST/figs"

python3 - "$SRC" "$DST" <<'PY'
import pathlib, re, shutil, sys
src, dst = (pathlib.Path(p) for p in sys.argv[1:3])
s = (src / "main.tex").read_text()

# author-note macros and the package they need: unused in the submitted version
drop = """\\usepackage[colorinlistoftodos,bordercolor=orange,backgroundcolor=orange!20,linecolor=orange]{todonotes}

\\newcommand{\\nikhil}[1]{\\todo[inline]{\\textbf{Nikhil: }#1}}
\\newcommand{\\rob}[1]{\\todo[inline]{\\textbf{Robert: }#1}}
\\newcommand{\\tanya}[1]{\\todo[inline]{\\textbf{Tanya: }#1}}
"""
if drop in s:
    s = s.replace(drop, "")
elif re.search(r"\\(nikhil|rob|tanya)\{|\\todo\[", s):
    sys.exit("author-note macros are in use; refusing to strip them")

# internal provenance comments (arXiv distributes the source)
internal = re.compile(
    r"^%.*(lora_playground|polora bench|loader-verified|config\.json|C_opt|C_fb"
    r"|counterfactual|Underlying counts|per-model step speedup|wall = step"
    r"|Rows in code-descending|order so each math|independent at our fixed"
    r"|\d+r\^2 D \+)"
)
kept = [ln for ln in s.split("\n") if not internal.match(ln.strip())]
(dst / "main.tex").write_text("\n".join(kept))

shutil.copy(src / "main.bbl", dst / "main.bbl")
figs = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", s)
for f in sorted(set(figs)):
    shutil.copy(src / "figs" / f, dst / "figs" / f)
print(f"main.tex ({len(s.splitlines()) - len(kept)} comment lines stripped), "
      f"main.bbl, {len(set(figs))} figures")
PY

# the package must build from its own contents alone
( cd "$DST" && module load texlive/20240312 >/dev/null 2>&1 || true
  for i in 1 2 3; do pdflatex -interaction=nonstopmode main.tex >"pass$i.log" 2>&1; done
  grep -q "^!" pass3.log && { echo "BUILD FAILED"; grep -m5 "^!" pass3.log; exit 1; }
  grep -qi "undefined" pass3.log && { echo "UNDEFINED REFERENCES"; exit 1; }
  echo "isolated build clean: $(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $2}') pages" )

( cd "$DST" && rm -f pass*.log main.aux main.log main.out main.ptc main.pdf
  tar -czf "$TARBALL" main.tex main.bbl figs )
echo "wrote $TARBALL"
