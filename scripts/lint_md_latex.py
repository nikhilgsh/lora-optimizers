#!/usr/bin/env python3
"""Lint LaTeX-in-Markdown for the failure modes that keep producing broken renders.

Run: python scripts/lint_md_latex.py docs/notes/**/*.md
Exit non-zero if any issue is found, so it can gate a handoff.

Failure modes caught (each has bitten related_work_2026_05.md at least once):
  E1  Bare-operator span: $+$ $=$ $\\approx$ $\\sim$ $\\times$ $\\gtrsim$ $\\le$ $\\ge$ standing
      alone. Renderers re-pair the stray $ with the next $, corrupting a whole line.
  E2  sim/approx glued to a digit across the math boundary: "${\\sim}$10", "$\\approx$120".
      The number sits outside math; renders as literal "10" jammed onto the glyph.
  E3  digit/letter adjacent to a math glyph split across $: "1$'$", "d$=$64", "$\\sim$2200".
  E4  Odd number of unescaped $ on a non-display line (unbalanced inline math).
  H1  History/self-correction narrative: "earlier draft", "I earlier mis-stated", "the correction
      to", "we used to", "previously", "as I noted above". Docs describe CURRENT state, not their own
      edit history — a reader does not care that a prior draft was wrong. State the correct thing.
      (CLAUDE.md "Documentation": docs describe current state, not history; default move is DELETE.)
"""
import re
import sys
import glob

# Operators that, alone in a span, are the classic split-span bug.
BARE_OPS = [r"\+", r"=", r"\\approx", r"\\sim", r"\\times", r"\\gtrsim",
            r"\\lesssim", r"\\le", r"\\ge", r"\\to", r"\\pm", r"\\cdot"]
# allow optional LaTeX thin-space \, around the operator (still renders as a lone glyph)
E1 = re.compile(r"(?<!\$)\$\s*(?:\\,\s*)?(?:" + "|".join(BARE_OPS) + r")(?:\s*\\,)?\s*\$(?!\$)")
# {\sim} or \approx etc. immediately followed by an alnum char outside math.
E2 = re.compile(r"\$\s*(?:\{?\\(?:sim|approx|gtrsim|lesssim|le|ge|times)\}?)\s*\$[0-9A-Za-z]")
# alnum immediately before $...$, or $...$ immediately before alnum, where the span is a glyph/op.
E3 = re.compile(r"(?:[0-9A-Za-z]\$[^$]{1,6}\$)|(?:\$[^$]{1,6}\$[0-9A-Za-z])")
# H1: self-referential edit-history / correction narrative. Docs state current truth, not their past.
H1 = re.compile(
    r"\b("
    r"earlier draft|prior draft|previous (?:draft|version)|earlier version|"
    r"(?:i|we|my)\s+(?:earlier|previously|wrongly|incorrectly|mis)\w*\b"
    r"(?:[- ]?(?:stated|said|claimed|wrote|noted|framed))?|"
    r"mis-?stated|the correction to|this corrects|as (?:i|we) (?:said|noted|stated) (?:above|earlier)|"
    r"(?:we|i)\s+used to|formerly|in (?:the|an) earlier|the (?:old|former) (?:draft|framing|version)"
    r")\b",
    re.IGNORECASE,
)


def check(path):
    issues = []
    in_block = False
    for n, line in enumerate(open(path), 1):
        s = line.rstrip("\n")
        # Track $$...$$ display blocks; skip math-mode lines for inline checks.
        if s.strip().startswith("$$"):
            in_block = not in_block if s.strip() == "$$" else in_block
            # one-line $$...$$ stays display; multi-line toggles
            if s.strip() == "$$":
                continue
        if in_block or s.strip().startswith("$$"):
            continue
        if s.strip().startswith("```"):
            continue
        for tag, rx in (("E1", E1), ("E2", E2)):
            for m in rx.finditer(s):
                issues.append((n, tag, m.group(0)))
        # E3: only flag if the inner span looks like a bare glyph/op (heuristic: no letters-as-word)
        for m in E3.finditer(s):
            inner = m.group(0)
            # Hyphen-after-math ("$A$-step", "$r$-dim") is fine; only flag bare glyph/op spans
            # ($=$, $'$, $+$) directly glued to an alnum char.
            if re.search(r"\\(approx|sim|times|to|pm|cdot|gtrsim)|[0-9A-Za-z]\$[=+'][^$]?\$|\$[=+'][^$]?\$[0-9A-Za-z]", inner):
                issues.append((n, "E3", inner))
        # E4: balance of unescaped, non-display $
        bare = re.sub(r"\$\$", "", re.sub(r"\\\$", "", s))
        if bare.count("$") % 2 == 1:
            issues.append((n, "E4", f"odd $ count: {bare.count('$')}"))
        # H1: history / self-correction narrative (prose lines only; skip code fences handled above)
        for m in H1.finditer(s):
            issues.append((n, "H1", m.group(0)))
    return issues


def main(argv):
    paths = []
    for a in argv:
        paths.extend(glob.glob(a, recursive=True))
    if not paths:
        print("no files matched", file=sys.stderr)
        return 2
    total = 0
    for p in sorted(set(paths)):
        for n, tag, frag in check(p):
            print(f"{p}:{n}: {tag}: {frag}")
            total += 1
    if total:
        print(f"\n{total} issue(s)", file=sys.stderr)
        return 1
    print("clean", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
