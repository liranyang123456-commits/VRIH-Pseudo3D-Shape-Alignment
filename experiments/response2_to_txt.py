#!/usr/bin/env python3
r"""Convert revision2_materials/response2.tex into a clean plain-text version

tailored for the online submission response box (e.g., Editorial Manager /
ScholarOne: "Please respond to specific reviewer and editor comments in the box below").

Features:
- Preserves full reviewer comments and responses point-by-point.
- Strips LaTeX markup while producing human-readable math and plain formatting.
- Explicitly notes where visual figures, 3D surface plots, and complex benchmark
  tables reside in the official PDF response (response2.pdf) and the revised paper
  (VRIH_Paper_R2.pdf).
- Fixes symbol anomalies (e.g., ^circ -> deg, \nabla -> grad, \to -> ->, \{ -> {).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R2DIR = ROOT / "revision2_materials"
SRC = R2DIR / "response2.tex"
OUT = R2DIR / "response2_plain.txt"

GREEK_AND_MATH = {
    r"\rightarrow": "->",
    r"\leftarrow": "<-",
    r"\mapsto": "->",
    r"\to": "->",
    r"\pm": "+-",
    r"\times": "x",
    r"\leq": "<=",
    r"\geq": ">=",
    r"\le": "<=",
    r"\ge": ">=",
    r"\neq": "!=",
    r"\approx": "~",
    r"^\circ": " deg",
    r"\circ": " deg",
    r"\nabla": "grad ",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\lambda": "lambda",
    r"\tau": "tau",
    r"\eta": "eta",
    r"\epsilon": "epsilon",
    r"\rho": "rho",
    r"\in": " in ",
    r"\lor": " or ",
    r"\land": " and ",
    r"\ldots": "...",
    r"\dots": "...",
    r"\cdot": " * ",
    r"\mid": " | ",
    r"\min": "min",
    r"\max": "max",
    r"\exp": "exp",
    r"\log": "log",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tanh": "tanh",
    r"\infty": "inf",
    r"\top": "T",
    r"\!": "",
}


def grab_braced(text: str, start: int) -> tuple[str, int]:
    """Return content of balanced {...} starting at text[start]=='{' and end index."""
    assert text[start] == "{"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    raise ValueError("unbalanced braces")


def clean_text(text: str) -> str:
    t = text

    # Protect literal / escaped braces and bars
    t = t.replace(r"\{", "__LBRACE__").replace(r"\}", "__RBRACE__")
    t = t.replace(r"\|", "|")

    # List environments -> dash bullets
    t = re.sub(r"\\begin\{(itemize|enumerate)\}(?:\[[^\]]*\])?", "", t)
    t = re.sub(r"\\end\{(itemize|enumerate)\}", "", t)
    t = re.sub(r"\\item(?:\[[^\]]*\])?\s*", "\n- ", t)

    # Protect degree symbol before command stripping
    t = re.sub(r"\^\s*\\circ\b", " deg", t)
    t = t.replace(r"\circ", " deg")

    # Accents: only replace \c{c} with c, do NOT match \circ or other macro prefixes
    t = re.sub(r"\\c\{([A-Za-z])\}", r"\1", t)
    t = re.sub(r'\\["\'`^~=.]{?([A-Za-z])}?', r"\1", t)

    # Text-mode ellipsis (math-mode \ldots is handled in demath)
    t = re.sub(r"\\(ldots|dots)\b", "...", t)

    # Spacing and line-breaks
    t = t.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ").replace(r"\ ", " ")
    t = re.sub(r"\\\.\s*", ". ", t)
    t = re.sub(r"\\\.", ".", t)
    t = re.sub(r"\\\\\s*", "\n", t)

    # Unwrap formatted content
    for cmd in (
        "revised", "textbf", "emph", "textit", "texttt", "boldsymbol",
        "mathbf", "mathrm", "bm", "textcolor", "uline"
    ):
        while True:
            m = re.search(r"\\" + cmd + r"(?:\[[^\]]*\])?\{", t)
            if not m:
                break
            inner, end = grab_braced(t, m.end() - 1)
            t = t[: m.start()] + inner + t[end:]

    # Math mode converter
    def demath(m: re.Match) -> str:
        s = m.group(1)
        # Handle font wrappers inside math
        s = re.sub(r"\\(hat|bar|tilde)\{([A-Za-z0-9])\}", r"\2", s)
        s = re.sub(r"\\(hat|bar|tilde)([A-Za-z0-9])", r"\2", s)
        s = re.sub(r"\\(mathcal|mathbb|mathfrak)\{([A-Za-z0-9]+)\}", r"\2", s)
        s = re.sub(r"\\(mathrm|operatorname|text)\{([^}]*)\}", r"\2", s)
        s = re.sub(r"\\(subsec|ref|label)\{[^}]*\}", "", s)

        # Substitute known symbols
        for k, v in GREEK_AND_MATH.items():
            s = s.replace(k, v)

        # Drop leftover backslash commands
        s = re.sub(r"\\[a-zA-Z]+", "", s)
        s = s.replace("{", "").replace("}", "")

        # Clean subscripts and superscripts
        s = re.sub(r"\s*_\s*([A-Za-z0-9]+)", r"_\1", s)
        s = re.sub(r"\s*_\s*\(([^)]+)\)", r"_(\1)", s)
        s = re.sub(r"\^\s*(-?\d+)", r"^\1", s)
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" _", "_").replace("( ", "(").replace(" )", ")")
        return s.strip()

    t = re.sub(r"\$\$([^$]+)\$\$", lambda m: " " + demath(m) + " ", t)
    t = re.sub(r"\$([^$]+)\$", lambda m: " " + demath(m) + " ", t)

    # URLs and citations
    t = re.sub(r"\\url\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"\2 (\1)", t)
    t = re.sub(r"\\cite\{([^}]*)\}", lambda m: "[" + m.group(1) + "]", t)

    # Quotes and dashes
    t = t.replace("``", '"').replace("''", '"')
    t = t.replace("---", " - ").replace("--", "-")
    t = t.replace("~", " ")
    t = t.replace(r"\&", "&").replace(r"\%", "%").replace(r"\#", "#").replace(r"\_", "_")

    # Remove residual dangling commands
    t = re.sub(r"\\[a-zA-Z]+\*?", "", t)
    t = t.replace("{", "").replace("}", "")

    # Restore escaped braces
    t = t.replace("__LBRACE__", "{").replace("__RBRACE__", "}")

    # Clean residual trailing backslashes like "rot.\ "
    t = re.sub(r"\\\s+", " ", t)
    t = re.sub(r"\\$", "", t, flags=re.M)

    # Typography polishing
    t = re.sub(r"in\{", "in {", t)
    # Add space after comma inside set braces only, e.g. {50,60,68} -> {50, 60, 68}
    def add_set_spaces(m):
        inner = m.group(1)
        items = [x.strip() for x in inner.split(",")]
        return "{" + ", ".join(items) + "}"
    t = re.sub(r"\{([0-9,\s]+)\}", add_set_spaces, t)
    t = re.sub(r"(\d+)\s+-(frame|pixel|step|way|fold)", r"\1-\2", t)
    t = re.sub(r"(\d+)\s+%", r"\1%", t)

    # Normalize whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


VISUAL_NOTE = """
[----------------------------------------------------------------------------------
NOTE ON SUPPORTING FIGURES AND TABLES:
While graphical visual images and 3D surface plots cannot be directly rendered in
this online text box, the two full quantitative benchmark tables (Table 1 and
Table 2) are presented below in plain-text format for direct review. Detailed
high-resolution graphical illustrations and the fully formatted response are
provided in the PDF version of this response letter ("response2_compiled.pdf" /
"response2.pdf", pages 6-8) and the revised manuscript ("VRIH_Paper_R2_clean.pdf" /
"VRIH_Paper_R2.pdf", Figure 4, page 8). Specifically:

  - Figure 1 (response2_compiled.pdf, p. 6): Gradient-percentile thresholding panels
    comparing six operational cut levels (median/p50, p60, p68/t_main, mean,
    p90/t90, p95) across representative self-acquired endoscopic, public SCARED,
    and calibration checkerboard frames.
  - Figure 2 (response2_compiled.pdf, p. 7): 3D surface plot of endoscopic gradient
    magnitude with horizontal threshold cut planes at the six levels, visualizing
    the smooth low-frequency structural bulk separated from salient boundary ridges.
  - Figure 4 in the revised manuscript (VRIH_Paper_R2_clean.pdf, p. 8): Empirical gradient
    component distributions (zero-mean heavy-tailed histograms), per-sequence t90
    threshold stability (CV 0.014-0.171), and downstream relative rotation
    insensitivity (bandwidth <= 0.015 deg).

Please refer directly to response2_compiled.pdf and VRIH_Paper_R2_clean.pdf to view
the graphical figures. The supporting quantitative tables from the response letter
are provided below:

===================================================================================
Table 1 (from response2.pdf): Operating-point sweep of the detection percentile on
the 3,482-frame pool (3,002 self-acquired + 480 public SCARED frames).
All values are medians over frames except the retained-pixel fraction, which is
fixed by construction. Energy share: fraction of total gradient magnitude carried
by the retained pixels. Dice: overlap of consecutive-frame masks. HCR: fraction of
retained pixels inside the specular-highlight mask (grayscale >= 95th percentile).
===================================================================================
+-------------------+-----------------+--------------+----------------+-------+
| Threshold         | Retained Pixels | Energy Share | Dice (consec.) | HCR   |
+-------------------+-----------------+--------------+----------------+-------+
| p50 (median)      |       50%       |    0.881     |     0.793      | 0.073 |
| p60               |       40%       |    0.828     |     0.759      | 0.080 |
| p68 (t_main) [*]  |       32%       |    0.774     |     0.732      | 0.087 |
| p75               |       25%       |    0.718     |     0.710      | 0.096 |
| p80               |       20%       |    0.669     |     0.695      | 0.105 |
| p85               |       15%       |    0.609     |     0.678      | 0.119 |
| p90 (t90)         |       10%       |    0.531     |     0.640      | 0.145 |
| p95               |        5%       |    0.413     |     0.572      | 0.209 |
+-------------------+-----------------+--------------+----------------+-------+
[*] Selected operating point (t_main): favorable trade-off between energy
    retention (77.4%) and specular-highlight suppression (HCR < 0.09).

===================================================================================
Table 2 (from response2.pdf): Multi-method tracking comparison from the companion
study (11 standardized checkerboard sequences, 3,848 frames), reproduced here as
auxiliary evidence. IoU is the coverage-aware BBox IoU (a prediction fully covering
the ground-truth box scores 1.0; otherwise the standard Jaccard index), reported as
mean +- SD over the 11 sequences. **: paired two-sided Wilcoxon signed-rank test
versus the contour-based tracker on per-sequence IoU, p < 0.01. Transformer trackers
(OSTrack, MixFormer, LMTrack, Cutie) use randomly initialized backbones (offline
setting, architecture reference). The reference masks of this protocol are
constructed with the same operator family as t_main (bilateral filtering, Sobel
gradient magnitude, 68th-percentile threshold).
===================================================================================
+-----------------------------------------------+------------------+-----------+---------+------------------+
| Method                                        |     IoU (up)     | Precision | Success | Center Err. (px) |
+-----------------------------------------------+------------------+-----------+---------+------------------+
| Contour-driven tracker (Scheme C + NanoTrack) |  0.809 +- 0.21   |   0.648   |  0.888  |       70.8       |
| MixFormer                                     |  0.540** +- 0.11 |   0.019   |  0.440  |      217.6       |
| LMTrack                                       |  0.536** +- 0.18 |   0.017   |  0.444  |      219.1       |
| OSTrack                                       |  0.523** +- 0.13 |   0.014   |  0.425  |      225.9       |
| CSRT                                          |  0.356** +- 0.33 |   0.129   |  0.291  |      231.5       |
| SAM                                           |  0.333** +- 0.27 |   0.226   |  0.323  |      341.4       |
| YOLO-seg                                      |  0.305** +- 0.17 |   0.091   |  0.301  |      544.0       |
| Cutie                                         |  0.288** +- 0.22 |   0.049   |  0.219  |      237.0       |
+-----------------------------------------------+------------------+-----------+---------+------------------+
----------------------------------------------------------------------------------]
""".strip()


ONLINE_BOX_HEADER = """
======================================================================
RESPONSE TO REVIEWERS (ONLINE SUBMISSION PLAIN-TEXT VERSION)
Manuscript No.: VRIH-D-26-00061R1 (Second-Round Revision)
Title: Nonrigid-Assisted Pseudo-3D Shape Alignment for Endoscopic Image Sequences
Journal: Virtual Reality & Intelligent Hardware (VRIH)
======================================================================

[SPECIAL NOTE FOR EDITORIAL OFFICE AND REVIEWERS:
This plain-text document contains our complete, point-by-point responses to all
comments from Reviewer #2, along with our acknowledgement to Reviewer #3 and
a comprehensive summary of round-2 manuscript revisions.

Because this online submission text box does not support embedded graphical
figures, 3D surface visualizations, formatted multi-column tables, or color-coded
markup, we have uploaded the fully formatted official PDF documents:
  - "response2_compiled.pdf" (or "response2.pdf"): Contains all full-resolution
    supporting figures (Figures 1-2), ablation tables (Tables 1-2), mathematical
    formatting, and visual panels.
  - "VRIH_Paper_R2_clean.pdf" (or "VRIH_Paper_R2.pdf"): Clean final manuscript.
  - "VRIH_Paper_R2_markedup.pdf": Main paper showing all round-2 additions and
    modifications highlighted in dark red.

Where graphical panels or benchmark tables are discussed below, explicit pointers
are provided directing to the corresponding pages of response2_compiled.pdf and the
revised manuscript. For your immediate reading convenience, the supporting benchmark
tables from the response letter are directly formatted and presented as clean text
tables within this document (see Comment 2.3). We kindly encourage reviewers to consult
response2_compiled.pdf (or response2.pdf) for the optimal visual reading experience.]
""".strip()


def build_plain_text() -> str:
    raw_tex = SRC.read_text(encoding="utf-8")

    # Protect visual and tabular block with a placeholder so its whitespace and ASCII borders stay intact
    PLACEHOLDER = "___VISUAL_TABLES_BLOCK_PLACEHOLDER___"
    tex = re.sub(
        r"% RESP-VISUAL-BEGIN.*?% RESP-VISUAL-END",
        PLACEHOLDER,
        raw_tex,
        flags=re.S,
    )

    # Standardize comment macro
    tex = tex.replace(r"\rcomment{", r"\comment{")
    tex = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", tex)

    # Extract document body
    body = tex.split(r"\begin{document}", 1)[1]
    body = body.split(r"\end{document}")[0]

    lines_out: list[str] = []
    lines_out.append(ONLINE_BOX_HEADER)
    lines_out.append("")

    # Split into sections
    parts = re.split(r"\\section\*?\{", body)
    preamble = parts[0]

    # Clean greeting
    greeting = preamble.split(r"\hrule")[0]
    greeting = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", greeting, flags=re.S)
    greeting = re.sub(r"\\(medskip|bigskip|smallskip|vspace\{[^}]*\}|hrule|noindent)", "", greeting)
    greeting_lines = [l for l in greeting.splitlines() if not l.lstrip().startswith("%")]
    greeting_clean = clean_text("\n".join(greeting_lines))
    if greeting_clean:
        lines_out.append(greeting_clean)
        lines_out.append("")

    for part in parts[1:]:
        title, _, rest = part.partition("}")
        title = title.replace(r"\#", "#").strip()
        rest = "\n".join(l for l in rest.splitlines() if not l.lstrip().startswith("%"))
        rest = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", rest)

        lines_out.append("=" * 70)
        lines_out.append(title.upper())
        lines_out.append("=" * 70)
        lines_out.append("")

        pos = 0
        cmt_no = 0
        while pos < len(rest):
            m = re.search(r"\\(comment|response)\{", rest[pos:])
            if not m:
                tail = rest[pos:]
                tail = re.sub(r"\\(begin|end)\{(itemize|enumerate|center|commentbox)\}(?:\[[^\]]*\])?", "", tail)
                tail = re.sub(r"\\item(?:\[[^\]]*\])?", "\n- ", tail)
                tail = clean_text(tail)
                if tail:
                    lines_out.append(tail)
                break

            kind = m.group(1)
            start = pos + m.end() - 1
            inner, end = grab_braced(rest, start)
            pre = rest[pos : pos + m.start()]
            pre = re.sub(r"\\(begin|end)\{(itemize|enumerate|center|commentbox)\}(?:\[[^\]]*\])?", "", pre)
            pre = re.sub(r"\\item(?:\[[^\]]*\])?", "\n- ", pre)
            pre = clean_text(pre)
            if pre:
                lines_out.append(pre)
                lines_out.append("")

            inner_clean = clean_text(inner)
            if kind == "comment":
                cmt_no += 1
                rev_no = title.split()[-1].lstrip("#")
                lines_out.append("-" * 70)
                lines_out.append(f"COMMENT {rev_no}.{cmt_no}: {inner_clean}")
                lines_out.append("-" * 70)
            else:
                lines_out.append(f"RESPONSE: {inner_clean}")
                lines_out.append("")
            pos = end

    text = "\n".join(lines_out)
    text = text.replace(PLACEHOLDER, VISUAL_NOTE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main() -> None:
    text = build_plain_text()
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"Successfully generated: {OUT} ({len(text)} characters, {len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
