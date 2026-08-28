#!/usr/bin/env python3
"""Convert response.tex into a clean plain-text version for the submission system.

Strips LaTeX markup while preserving structure: reviewer sections, numbered
comments, responses, bullet lists, and readable math (unicode symbols).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised")
SRC = ROOT / "response.tex"
OUT = ROOT / "response_plain.txt"

SYMBOLS = {
    r"\rightarrow": "->",
    r"\leftarrow": "<-",
    r"\mapsto": "->",
    r"\pm": "+-",
    r"\times": "x",
    r"\leq": "<=",
    r"\geq": ">=",
    r"\neq": "!=",
    r"\approx": "~",
    r"\circ": "deg",
    r"\mu": "mu",
    r"\sigma": "sigma",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\lambda": "lambda",
    r"\tau": "tau",
    r"\eta": "eta",
    r"\epsilon": "epsilon",
    r"\rho": "rho",
    r"\in": "in",
    r"\lor": "or",
    r"\ldots": "...",
    r"\emph": "",
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


def clean(text: str) -> str:
    t = text
    # list environments -> dash bullets (may be nested inside \response{...})
    t = re.sub(r"\\begin\{(itemize|enumerate)\}(?:\[[^\]]*\])?", "", t)
    t = re.sub(r"\\end\{(itemize|enumerate)\}", "", t)
    t = re.sub(r"\\item(?:\[[^\]]*\])?\s*", "\n- ", t)
    # accent escapes -> plain ascii
    t = re.sub(r'\\["\'`^~=.]{?([A-Za-z])}?', r"\1", t)
    t = re.sub(r"\\c\{?([A-Za-z])}?", r"\1", t)
    # spacing / line-break escapes
    t = t.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ").replace(r"\ ", " ")
    t = t.replace("\\.", ".")
    t = re.sub(r"\\\\\s*", " ", t)
    # structural commands with braced content -> keep inner text
    for cmd in ("revised", "textbf", "emph", "textit", "boldsymbol", "mathbf", "mathrm", "bm", "textcolor"):
        while True:
            m = re.search(r"\\" + cmd + r"(?:\[[^\]]*\])?\{", t)
            if not m:
                break
            inner, end = grab_braced(t, m.end() - 1)
            t = t[: m.start()] + inner + t[end:]
    # math mode -> readable unicode-ish text
    def demath(m: re.Match) -> str:
        s = m.group(1)
        s = re.sub(r"\\(hat|bar|tilde|bar)\{([A-Za-z])\}", r"\2", s)
        s = re.sub(r"\\(hat|bar|tilde)([A-Za-z])", r"\2", s)
        s = re.sub(r"\\(mathcal|mathbb|mathfrak)\{([A-Za-z])\}", r"\2", s)
        s = re.sub(r"\\(mathrm|operatorname)\{([^}]*)\}", r"\2", s)
        s = re.sub(r"\\(subsec|ref|label)\{[^}]*\}", "", s)
        for k, v in SYMBOLS.items():
            s = s.replace(k + " ", f" {v} ").replace(k, f" {v} ")
        s = re.sub(r"\\[a-zA-Z]+", "", s)  # drop remaining commands
        s = s.replace("{", "").replace("}", "")
        s = re.sub(r"\s*_\s*([A-Za-z0-9]+)", r"_\1", s)
        s = re.sub(r"\s*_\s*\(([^)]+)\)", r"_(\1)", s)
        s = re.sub(r"\^\s*(-?\d+)", r"^\1", s)
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" _", "_").replace("( ", "(").replace(" )", ")")
        return s.strip()

    t = re.sub(r"\$\$([^$]+)\$\$", lambda m: " " + demath(m) + " ", t)
    t = re.sub(r"\$([^$]+)\$", lambda m: " " + demath(m) + " ", t)
    # url / href
    t = re.sub(r"\\url\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"\2 (\1)", t)
    # quotes and dashes
    t = t.replace("``", '"').replace("''", '"')
    t = t.replace("---", " - ").replace("--", "-")
    t = t.replace("~", " ")
    t = t.replace(r"\&", "&").replace(r"\%", "%").replace(r"\#", "#").replace(r"\_", "_")
    t = re.sub(r"\\(cite)\{([^}]*)\}", lambda m: "[" + m.group(2) + "]", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", "", t)
    t = t.replace("{", "").replace("}", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def main() -> None:
    src = SRC.read_text(encoding="utf-8")
    body = src.split(r"\begin{document}", 1)[1]
    body = body.split(r"\end{document}")[0]

    lines_out: list[str] = []
    lines_out.append("RESPONSE TO REVIEWERS")
    lines_out.append("Manuscript No.: VRIH-D-26-00061")
    lines_out.append("Nonrigid-Assisted Pseudo-3D Shape Alignment for Endoscopic Image Sequences")
    lines_out.append("")

    # split into sections
    parts = re.split(r"\\section\*?\{", body)
    preamble = parts[0]
    # greeting before first section
    greeting = preamble.split(r"\hrule")[0]
    greeting = re.sub(r"\\begin\{center\}.*?\\end\{center\}", "", greeting, flags=re.S)
    greeting = re.sub(r"\\(medskip|bigskip|smallskip|vspace\{[^}]*\}|hrule|noindent)", "", greeting)
    greeting = clean(greeting)
    if greeting:
        lines_out.append(greeting)
        lines_out.append("")

    for part in parts[1:]:
        title, _, rest = part.partition("}")
        title = title.replace(r"\#", "#").strip()
        # strip LaTeX comment lines (% ...) from the section body
        rest = "\n".join(l for l in rest.splitlines() if not l.lstrip().startswith("%"))
        lines_out.append("=" * 70)
        lines_out.append(title.upper())
        lines_out.append("=" * 70)
        lines_out.append("")
        rest = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", rest)

        # tokenize \comment{...} and \response{...} and itemize blocks in order
        pos = 0
        cmt_no = 0
        while pos < len(rest):
            m = re.search(r"\\(comment|response)\{", rest[pos:])
            if not m:
                tail = rest[pos:]
                tail = re.sub(r"\\(begin|end)\{(itemize|enumerate|center|commentbox)\}(?:\[[^\]]*\])?", "", tail)
                tail = re.sub(r"\\item(?:\[[^\]]*\])?", "\n- ", tail)
                tail = clean(tail)
                if tail:
                    lines_out.append(tail)
                break
            kind = m.group(1)
            start = pos + m.end() - 1
            inner, end = grab_braced(rest, start)
            pre = rest[pos : pos + m.start()]
            pre = re.sub(r"\\(begin|end)\{(itemize|enumerate|center|commentbox)\}(?:\[[^\]]*\])?", "", pre)
            pre = re.sub(r"\\item(?:\[[^\]]*\])?", "\n- ", pre)
            pre = clean(pre)
            if pre:
                lines_out.append(pre)
                lines_out.append("")
            inner_clean = clean(inner)
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
    text = re.sub(r"\n{3,}", "\n\n", text)
    OUT.write_text(text + "\n", encoding="utf-8")
    print("written", OUT, len(text), "chars")


if __name__ == "__main__":
    main()
