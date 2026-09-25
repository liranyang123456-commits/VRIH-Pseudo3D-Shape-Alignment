#!/usr/bin/env python3
"""Build the third-round (R3, minor revision) manuscript variants.

Reads the fixed master revision3_materials/VRIH_Paper_R3.tex and writes

  * VRIH_Paper_R3_clean.tex     -- byte-identical copy of the master
  * VRIH_Paper_R3_markedup.tex  -- master with the single R3 change wrapped in
                                   dark-red markup (relative to the R2 submission)

The only textual change requested by Reviewer #2 in round 3 is the rewording of
one sentence in Section 2 (Related Work); the keyword issue lives in the
Editorial Manager metadata, not in the manuscript.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "revision3_materials"

# Highlight colour for the R3 change. The journal does not prescribe a colour;
# blue is used here (R2 used dark red). Keep NAME/RGB in sync with the
# \definecolor{changecolor} line in response3.tex.
REV_COLOR_NAME = "blue"
REV_COLOR_RGB = "0,70,180"

NEW_SENTENCE = (
    "estimates a relative shape-alignment transform through geometric registration "
    "of the resulting pseudo-3D meshes---dense optical-flow correspondences, robust "
    "RANSAC/Kabsch fitting, and ICP refinement (Section~\\ref{subsec:pose})---with "
    "highlight robustness quantified directly by the HCR metric rather than achieved "
    "by explicit highlight removal."
)
OLD_SENTENCE_FRAGMENT = "through differentiable mesh registration"


def main() -> None:
    master = (ROOT / "VRIH_Paper_R3.tex").read_text(encoding="utf-8")

    if OLD_SENTENCE_FRAGMENT in master:
        raise SystemExit("master still contains the R2 wording: " + OLD_SENTENCE_FRAGMENT)
    if NEW_SENTENCE not in master:
        raise SystemExit("master does not contain the expected R3 sentence")

    # clean = master
    (ROOT / "VRIH_Paper_R3_clean.tex").write_bytes((ROOT / "VRIH_Paper_R3.tex").read_bytes())

    # marked-up
    preamble_add = (
        "\n%---- R3 revision markup (third-round Reviewer-#2 consistency items) ----\n"
        "\\definecolor{revchange}{RGB}{" + REV_COLOR_RGB + "}\n"
        "\\newcommand{\\rev}[1]{{\\color{revchange}#1}}\n"
    )
    marked = master.replace(
        r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}",
        r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}" + preamble_add,
        1,
    )
    note = (
        "\n\\noindent{\\footnotesize\\rev{Note: " + REV_COLOR_NAME.capitalize() + " text highlights the "
        "third-round (R3, minor revision) change made in response to Reviewer \\#2's consistency "
        "comment: the description of the proposed method in Section~2 (Related Work) now states that "
        "the relative shape-alignment transform is obtained by geometric registration (dense optical "
        "flow, robust RANSAC/Kabsch fitting, ICP refinement), matching Section~3. All other text, "
        "tables, figures, and results are unchanged relative to the second revised manuscript. The "
        "second-round keyword change (``Robust registration'') is now also reflected in the submission "
        "metadata.}}\n\\vspace{0.6em}\n\n"
    )
    marked = marked.replace(r"\maketitle", r"\maketitle" + note, 1)
    marked = marked.replace(NEW_SENTENCE, r"\rev{" + NEW_SENTENCE + "}", 1)

    out = ROOT / "VRIH_Paper_R3_markedup.tex"
    out.write_text(marked, encoding="utf-8")
    print("written", ROOT / "VRIH_Paper_R3_clean.tex")
    print("written", out, "| rev{ count:", marked.count("\\rev{"))


if __name__ == "__main__":
    main()
