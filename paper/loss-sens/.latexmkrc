# Overleaf/latexmk can need extra passes for biblatex, cleveref,
# hyperref bookmarks, and appendix standalone cross-references.
$max_repeat = 10;

# Needed for PDF reverse search (Cmd-click from the PDF back to source).
$pdflatex = 'pdflatex -synctex=1 %O %S';
