#!/usr/bin/env python3
"""Build the R2 marked-up manuscript: highlight second-round (R1->R2) revisions.

All R2 changes are Reviewer-#2-driven, so the changed/new blocks are wrapped in
the dark-red \\rev markup. Base = current master (R2 state).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised")
OUT = ROOT / "revision2"


def wrap_exact(text: str, old: str, new: str | None = None) -> str:
    if old not in text:
        raise SystemExit("MISSING BLOCK:\n" + old[:200].replace("\n", " ") + "...")
    if new is None:
        new = r"\rev{" + old + "}"
    return text.replace(old, new, 1)


def main() -> None:
    src = (ROOT / "VRIH_Paper.tex").read_text(encoding="utf-8")

    preamble_add = r"""
%---- R2 revision markup (second-round Reviewer-#2 items) ----
\definecolor{revchange}{RGB}{150,0,0}
\newcommand{\rev}[1]{{\color{revchange}#1}}
\newcommand{\revstart}{\color{revchange}}
\newcommand{\revend}{\normalcolor}
"""
    if "R2 revision markup" not in src:
        src = src.replace(
            r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}",
            r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}" + preamble_add,
            1,
        )

    note = r"""
\noindent{\footnotesize\rev{Note: Dark-red text highlights the second-round (R2) revisions made in response to Reviewer \#2's comments (pose-relationship analysis, differentiable-rendering removal, empirical threshold justification, enriched metrics, ablation/sensitivity, implementation details, positional-encoding correction, condition-stratified analysis).}}
\vspace{0.6em}

"""
    if "second-round (R2) revisions" not in src:
        src = src.replace(r"\maketitle", r"\maketitle" + note, 1)

    # 1. Keywords (R2.2)
    src = wrap_exact(
        src,
        r"\keywords{Relative motion estimation; Pseudo-3D shape alignment; Robust registration; Non-rigid deformation; Endoscopic surgical navigation}",
        r"\keywords{Relative motion estimation; Pseudo-3D shape alignment; \rev{Robust registration}; Non-rigid deformation; Endoscopic surgical navigation}",
    )

    # 2. Intro framing (R2.2)
    src = wrap_exact(
        src,
        "this study proposes a pseudo-3D shape-alignment method based on gradient-domain structural and implicit-representation components.",
        r"this study proposes a pseudo-3D shape-alignment method based on \rev{gradient-domain structural} and implicit-representation components.",
    )

    # 3. Supportive technologies (R2.2)
    src = wrap_exact(
        src,
        "frame-to-frame pseudo-3D mesh alignment; implicit neural representation",
        r"frame-to-frame pseudo-3D mesh alignment\rev{}\rev{",
    ) if False else src  # placeholder no-op; block handled below
    m = re.search(
        r"\\textbf\{Supportive technologies\.\}.*?drift control\.", src, flags=re.S
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: supportive technologies block missing")

    # 4. Methods overview DR removal (R2.2)
    m = re.search(
        r"The alignment problem is reformulated as a frame-to-frame pseudo-3D mesh alignment task:.*?presence of non-rigid surface motion\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: methods overview block missing")

    # 5. Empirical gradient statistics paragraph + figure (R2.3)
    m = re.search(
        r"The effectiveness of \$t_\{90\}\$ in delineating soft-tissue deformation regions is supported empirically.*?\\label\{fig:gradient_statistics\}\n\\end\{figure\}",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: gradient statistics block missing")

    # 6. Physical justification sentence (R2.3)
    m = re.search(
        r"The use of gradient magnitude as a deformation-height proxy is justified by the empirical evidence above:.*?isolated by \$t_\{90\}\$\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: physical justification sentence missing")

    # 7. PE paragraph (R2.8)
    m = re.search(
        r"We propose integrating positional encodings into the attention computation.*?during feature extraction\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: PE block missing")

    # 8. Loss weights + L_reg + diffusion time (R2.7)
    m = re.search(
        r"Specifically, the RGB reconstruction loss.*?not a physical time\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: loss block missing")

    # 9. Training protocol + splitting (R2.6/R2.7)
    m = re.search(
        r"\\textbf\{Training protocol\.\}.*?do not involve the trained network\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: training protocol block missing")

    # 10. Relationship paragraph (R2.1)
    m = re.search(
        r"\\textbf\{Relationship between the pseudo-3D transform and the physical camera pose\.\}.*?never claim metric pose recovery\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: relationship block missing")

    # 11. Tables 2 and 3 (R2.4) — wrap whole table envs
    for label in ("tab:chessboard_pose", "tab:scared_pose"):
        i = src.find("\\label{" + label + "}")
        if i < 0:
            print(f"WARN: {label} missing")
            continue
        start = src.rfind(r"\begin{table}", 0, i)
        # include \needspace line if present
        ns = src.rfind(r"\needspace", 0, start)
        if ns >= 0 and src[ns:start].strip().startswith(r"\needspace"):
            start = ns
        end = src.find(r"\end{table}", i) + len(r"\end{table}")
        block = src[start:end]
        src = src[:start] + r"\revstart" + "\n" + block + "\n" + r"\revend" + src[end:]

    # 12. Condition-stratified paragraph (R2.9)
    m = re.search(
        r"\\textbf\{Performance under challenging conditions\.\}.*?Chess-Line2\} case above\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: condition paragraph missing")

    # 13. Ablation subsection (R2.5)
    m = re.search(
        r"\\subsection\{Component ablation and hyperparameter sensitivity\}.*?no per-sequence tuning is needed\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\revstart" + "\n" + m.group(0) + "\n" + r"\revend", 1)
    else:
        print("WARN: ablation subsection missing")

    # 14. Implementation table + runtime sentence (R2.7)
    m = re.search(
        r"Table~\\ref\{tab:implementation\} lists the full configuration.*?intra-operative deployment\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: runtime sentence missing")
    i = src.find(r"\begin{table}[H]" + "\n" + r"\centering" + "\n" + r"\caption{Implementation configuration.")
    if i < 0:
        i = src.find("Implementation configuration.")
        if i >= 0:
            start = src.rfind(r"\begin{table}", 0, i)
            end = src.find(r"\end{table}", i) + len(r"\end{table}")
            block = src[start:end]
            src = src[:start] + r"\revstart" + "\n" + block + "\n" + r"\revend" + src[end:]
        else:
            print("WARN: implementation table missing")
    else:
        end = src.find(r"\end{table}", i) + len(r"\end{table}")
        block = src[i:end]
        src = src[:i] + r"\revstart" + "\n" + block + "\n" + r"\revend" + src[end:]

    out = OUT / "VRIH_Paper_R2_markedup.tex"
    out.write_text(src, encoding="utf-8")
    print("written", out)
    print("revstart", src.count(r"\revstart"), "rev{", src.count(r"\rev{"))


if __name__ == "__main__":
    main()
