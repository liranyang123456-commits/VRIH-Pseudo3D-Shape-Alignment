#!/usr/bin/env python3
r"""Convert revision3_materials/response3.tex into a plain-text version for the
Editorial Manager response box ("Please respond to specific reviewer and editor
comments in the box below").

Re-uses the LaTeX-to-text cleaner of response2_to_txt.py. The R3 response
contains no figures or tables, so no visual placeholder is needed; the header
still points to the uploaded PDF files by their exact names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from response2_to_txt import clean_text, grab_braced  # noqa: E402

ROOT = HERE.parent
R3DIR = ROOT / "revision3_materials"
SRC = R3DIR / "response3.tex"
OUT = R3DIR / "response3_plain.txt"

HEADER = """
======================================================================
RESPONSE TO REVIEWERS (ONLINE SUBMISSION PLAIN-TEXT VERSION)
Manuscript No.: VRIH-D-26-00061R2 (Third-Round / Minor Revision)
Title: Nonrigid-Assisted Pseudo-3D Shape Alignment for Endoscopic Image Sequences
Journal: Virtual Reality & Intelligent Hardware (VRIH)
======================================================================

[NOTE FOR THE EDITORIAL OFFICE AND REVIEWER:
This plain-text box contains our complete point-by-point response to the two
consistency comments of Reviewer #2 and a summary of all third-round changes.
The same response, with typographic formatting, is uploaded as
"response3.pdf". The revised manuscript is uploaded as
"VRIH_Paper_R3_clean.pdf" (clean) and "VRIH_Paper_R3_markedup.pdf" (only the
single third-round change highlighted, in blue, relative to the second revised
manuscript). Text quoted from the manuscript is enclosed in double quotes
below; superseded wording is labelled [OLD] and revised wording [NEW].]
""".strip()


def build() -> str:
    tex = SRC.read_text(encoding="utf-8")
    tex = tex.replace(r"\rcomment{", r"\comment{")
    tex = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", tex)
    # The greeting explains the PDF colour legend; restate it for plain text.
    tex = tex.replace(
        "the reviewer's comments are shown in shaded boxes",
        "the reviewer's comments are set between dashed rules",
    )
    tex = tex.replace(
        r"\revised{text added to or revised in the manuscript is highlighted in blue}",
        "text added to or revised in the manuscript is marked [NEW]",
    )
    tex = tex.replace(r"Superseded wording is shown in \oldtext{grey}.", "Superseded wording is marked [OLD].")
    # old/new markers so the box reader can tell superseded from revised wording
    tex = tex.replace(r"\oldtext{", r"\textbf{[OLD] ")
    tex = tex.replace(r"\revised{", r"\textbf{[NEW] ")

    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}")[0]

    out: list[str] = [HEADER, ""]

    parts = re.split(r"\\section\*?\{", body)
    greeting = parts[0]
    greeting = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", greeting, flags=re.S)
    greeting = re.sub(r"\\(medskip|bigskip|smallskip|vspace\{[^}]*\}|hrule|noindent)", "", greeting)
    greeting = "\n".join(l for l in greeting.splitlines() if not l.lstrip().startswith("%"))
    greeting = clean_text(greeting)
    if greeting:
        out += [greeting, ""]

    for part in parts[1:]:
        title, _, rest = part.partition("}")
        title = title.replace(r"\#", "#").strip()
        rest = "\n".join(l for l in rest.splitlines() if not l.lstrip().startswith("%"))
        out += ["=" * 70, title.upper(), "=" * 70, ""]

        pos, n = 0, 0
        while pos < len(rest):
            m = re.search(r"\\(comment|response)\{", rest[pos:])
            if not m:
                tail = clean_text(re.sub(r"\\item(?:\[[^\]]*\])?", "\n- ", rest[pos:]))
                if tail:
                    out.append(tail)
                break
            kind = m.group(1)
            start = pos + m.end() - 1
            inner, end = grab_braced(rest, start)
            pre = clean_text(re.sub(r"\\item(?:\[[^\]]*\])?", "\n- ", rest[pos : pos + m.start()]))
            if pre:
                out += [pre, ""]
            inner_clean = clean_text(inner)
            if kind == "comment":
                n += 1
                out += ["-" * 70, f"REVIEWER #2, COMMENT {n}: {inner_clean}", "-" * 70]
            else:
                out += [f"RESPONSE: {inner_clean}", ""]
            pos = end

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main() -> None:
    text = build()
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"written {OUT} ({len(text)} chars, {len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
