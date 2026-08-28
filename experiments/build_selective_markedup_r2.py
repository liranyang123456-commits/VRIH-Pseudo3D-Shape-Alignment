#!/usr/bin/env python3
"""Build the Round-2 selective marked-up manuscript (R1 -> R2 changes only).

Highlights in dark red only the substantive Round-2 revisions responding to
Reviewer #2's ten comments. Routine wording is not marked.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised")
R2DIR = ROOT / "revision2_materials"


def wrap_exact(text: str, old: str, new: str | None = None) -> str:
    if old not in text:
        raise SystemExit("MISSING BLOCK:\n" + old[:200].replace("\n", " ") + "...")
    if new is None:
        new = r"\rev{" + old + "}"
    return text.replace(old, new, 1)


def main() -> None:
    src = (R2DIR / "VRIH_Paper_R2_clean.tex").read_text(encoding="utf-8")

    preamble_add = r"""
%---- Selective revision markup (Round-2 Editor/Reviewer items only) ----
\definecolor{revchange}{RGB}{150,0,0}
\newcommand{\rev}[1]{{\color{revchange}#1}}
\newcommand{\revstart}{\color{revchange}}
\newcommand{\revend}{\normalcolor}
"""
    if "Selective revision markup" not in src:
        src = src.replace(
            r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}",
            r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}" + preamble_add,
            1,
        )

    note = r"""
\noindent{\footnotesize\rev{Note: Dark-red text highlights the substantive Round-2 revisions responding to Reviewer \#2 (pose-relationship analysis, rendering-role clarification, distribution-free threshold justification with empirical evidence, metric definitions, ablation/sensitivity section, splitting protocol, and implementation details). Routine wording edits are not marked.}}
\vspace{0.6em}

"""
    if "Dark-red text highlights the substantive Round-2" not in src:
        src = src.replace(r"\maketitle", r"\maketitle" + note, 1)

    # 1. Physical-pose relationship paragraph (R2.1)
    m = re.search(
        r"\\textbf\{Relationship to the physical camera pose\.\}.*?not a learned or fitted mapping from \$h\$ to depth\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: pose-relationship block missing")

    # 2. Rendering role clarification (R2.2)
    m = re.search(
        r"To be explicit about the role of rendering:.*?no rendering-based pose optimization is performed at inference\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: rendering-role sentence missing")

    # 3. Section 3.1 distribution-free passage + statistics figure (R2.3)
    m = re.search(
        r"The percentile thresholds are defined on the empirical cumulative histogram.*?\\label\{fig:gradient_statistics\}\n\\end\{center\}",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: distribution-free block missing")

    # 4. Physical-justification sentence update (R2.3)
    m = re.search(
        r"The use of gradient magnitude as a deformation-height proxy is justified by the distribution-free percentile construction above:.*?measured directly on \$3\{,\}482\$ real endoscopic frames rather than assumed\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: physical-justification sentence missing")

    # 5. PE collision note (R2.8)
    m = re.search(
        r"We note that the scalar \$\(x\{\+\}y\)\$ encoding assigns identical values along anti-diagonals.*?reported in Section~\\ref\{subsec:ablation\}\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: PE note missing")

    # 6. Training protocol paragraph (R2.6/R2.7)
    m = re.search(
        r"\\textbf\{Training protocol\.\}.*?independent of the training corpus\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: training protocol missing")

    # 7. Metric definitions paragraph (R2.4)
    m = re.search(
        r"\\textbf\{Metric definitions\.\}.*?before ATE computation\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: metric definitions missing")

    # 8. Implementation details extension (R2.7)
    m = re.search(
        r"The robust fitting uses RANSAC \(300 iterations.*?sampled uniformly during training\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: implementation details missing")

    # 9. Loss-weight definition (R2.7)
    m = re.search(
        r"The term \$\\mathcal\{L\}_\\mathrm\{reg\}\$ collects standard regularization.*?released with the training script in the public repository\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: loss-weight sentence missing")

    # 10. Success-rate column headers in Tables 2/3 (R2.4)
    src = src.replace(
        r"Translation-direction error ($^\circ$) & Success rate ($<1^\circ$)",
        r"Translation-direction error ($^\circ$) & \rev{Success rate ($<1^\circ$)}",
    )

    # 11. Section 4.5 ablation and sensitivity (R2.5/R2.8/R2.9)
    i1 = src.find(r"\subsection{Ablation and sensitivity analysis}")
    i2 = src.find(r"\section{Conclusion}")
    if i1 < 0 or i2 < 0:
        raise SystemExit("ablation section markers missing")
    src = src[:i1] + r"\revstart" + "\n" + src[i1:i2] + r"\revend" + "\n" + src[i2:]

    out = R2DIR / "VRIH_Paper_R2_markedup.tex"
    out.write_text(src, encoding="utf-8")
    print("written", out)
    print("revstart", src.count(r"\revstart"), "rev{", src.count(r"\rev{"))


if __name__ == "__main__":
    main()
