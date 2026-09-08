#!/usr/bin/env python3
"""Build the R2 marked-up manuscript inside revision2_materials.

Reads the fixed master revision2_materials/VRIH_Paper_R2.tex and wraps every
second-round (Reviewer-#2-driven) changed/new block in dark-red markup.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised") / "revision2_materials"


def wrap_exact(text: str, old: str, new: str | None = None) -> str:
    if old not in text:
        raise SystemExit("MISSING BLOCK:\n" + old[:200].replace("\n", " ") + "...")
    if new is None:
        new = r"\rev{" + old + "}"
    return text.replace(old, new, 1)


def wrap_span(src: str, pattern: str, name: str) -> str:
    m = re.search(pattern, src, flags=re.S)
    if not m:
        print(f"WARN: block missing: {name}")
        return src
    return src.replace(m.group(0), "\\revstart\n" + m.group(0) + "\n\\revend", 1)


def main() -> None:
    src = (ROOT / "VRIH_Paper_R2.tex").read_text(encoding="utf-8")

    preamble_add = r"""
%---- R2 revision markup (second-round Reviewer-#2 items) ----
\definecolor{revchange}{RGB}{150,0,0}
\newcommand{\rev}[1]{{\color{revchange}#1}}
\newcommand{\revstart}{\color{revchange}}
\newcommand{\revend}{\normalcolor}
"""
    src = src.replace(
        r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}",
        r"\pdfstringdefDisableCommands{\renewcommand*{\bm}[1]{#1}}" + preamble_add,
        1,
    )

    note = r"""
\noindent{\footnotesize\rev{Note: Dark-red text highlights the second-round (R2) revisions made in response to Reviewer \#2's comments (pose-proxy terminology and pose-relationship analysis, removal of the differentiable-rendering formulation, empirical distribution-free threshold justification, median/RPE metric enrichment, geometric back-end decomposition and sensitivity analysis, full loss/implementation configuration, positional-encoding full-network ablation, extended SCARED pose-ground-truth evaluation, condition-stratified analysis, and clinical-case windows).}}
\vspace{0.6em}

"""
    src = src.replace(r"\maketitle", r"\maketitle" + note, 1)

    # Keywords (R2.2)
    src = wrap_exact(
        src,
        r"\keywords{Relative motion estimation; Pseudo-3D shape alignment; Robust registration; Non-rigid deformation; Endoscopic surgical navigation}",
        r"\keywords{Relative motion estimation; Pseudo-3D shape alignment; \rev{Robust registration}; Non-rigid deformation; Endoscopic surgical navigation}",
    )

    # Intro framing (R2.2)
    src = wrap_exact(
        src,
        "this study proposes a pseudo-3D shape-alignment method based on gradient-domain structural and implicit-representation components.",
        r"this study proposes a pseudo-3D shape-alignment method based on \rev{gradient-domain structural} and implicit-representation components.",
    )

    # Scope and terminology (R2.1)
    src = wrap_exact(
        src,
        "Consequently, throughout this paper we describe our own output as a relative shape-alignment transform or pose \\emph{proxy}, and reserve the term ``camera pose'' for external ground truth and for baselines that genuinely estimate it;",
        r"\rev{Consequently, throughout this paper we describe our own output as a relative shape-alignment transform or pose \emph{proxy}, and reserve the term ``camera pose'' for external ground truth and for baselines that genuinely estimate it;}",
    )

    # Supportive technologies (R2.2)
    src = wrap_span(
        src,
        r"\\textbf\{Supportive technologies\.\}.*?drift control\.",
        "supportive technologies",
    )

    # Methods overview DR removal (R2.2)
    src = wrap_span(
        src,
        r"The alignment problem is reformulated as a frame-to-frame pseudo-3D mesh alignment task:.*?geometric estimation \(optical flow with RANSAC/Kabsch fitting and ICP refinement\)\.",
        "methods overview",
    )

    # Sobel realization of the derivative field (R2.3 consistency)
    src = wrap_exact(
        src,
        r"In the released implementation this derivative-magnitude field is realized by a $3\times3$ Sobel stencil after the bilateral filter---the same discrete operator used for the $3{,}482$-frame empirical statistics and the operating-point sweep below.",
    )

    # Threshold definition (R2.3)
    src = wrap_exact(
        src,
        "In practice, a main threshold $t_{\\text{main}}$ at the 68th percentile ($p=0.68$) is used for standard deformation detection, while a higher threshold $t_{90}$ at the 90th percentile ($p=0.90$) is employed to generate high-sensitivity gradient maps highlighting pronounced deformation regions. Both values are empirical percentile choices on the cumulative histogram---no distributional assumption is attached to either---selected on validation data; the downstream alignment is insensitive to the exact percentile (Section~\\ref{subsec:ablation}).",
    )

    # Empirical gradient statistics paragraph + figure (R2.3)
    src = wrap_span(
        src,
        r"The percentile thresholds are defined on the empirical cumulative histogram.*?\\label\{fig:gradient_statistics\}\n\\end\{center\}",
        "gradient statistics",
    )

    # Design intuition paragraph (R2.3)
    src = wrap_span(
        src,
        r"\\textbf\{Design intuition: information bulk versus salient tail\.\}.*?coverage--repeatability--contamination trade-off\.",
        "design intuition",
    )

    # Physical justification paragraph (R2.3)
    src = wrap_span(
        src,
        r"\\textbf\{Physical justification and robustness of the pseudo-3D representation\.\}.*?contribute little to the final shape-alignment estimate\.",
        "physical justification",
    )

    # PE design paragraph (R2.8)
    src = wrap_span(
        src,
        r"We propose integrating positional encodings into the attention computation.*?reported in Section~\\ref\{subsec:ablation\}\.",
        "PE design",
    )

    # Loss specification (R2.7)
    src = wrap_span(
        src,
        r"All terms and weights are specified exactly as released\..*?released with the training script in the public repository\.",
        "loss specification",
    )

    # Training protocol + split (R2.6/R2.7)
    src = wrap_span(
        src,
        r"\\textbf\{Training protocol\.\}.*?does not change the descriptor's re-ranking behavior\.",
        "training protocol",
    )

    # AE role + ablation + gate (R2.5)
    src = wrap_span(
        src,
        r"To quantify what the autoencoder adds.*?front-end screening role\.",
        "AE ablation",
    )

    # Relationship paragraph (R2.1)
    src = wrap_span(
        src,
        r"\\textbf\{Relationship to the physical camera pose\.\}.*?calibrated stereo baseline \(Section~4\.3\)\.",
        "relationship",
    )

    # Pairwise-transform wording (R2.1 residual cleanup)
    src = wrap_exact(
        src,
        "Therefore, the method yields pairwise shape-alignment transforms $\{T_{t-1\\leftarrow t}\}$ and composes them into a cumulative trajectory $\{T_{0\\leftarrow t}\}$. Accordingly, the resulting trajectory should be interpreted as a rigid pose \\emph{proxy} in the pseudo-3D $(u,v,h)$ space. Along with these transforms, we report per-frame diagnostics (e.g., inlier ratio and robust residual statistics for flow-guided fitting, or fitness/RMSE for ICP-based registration) and the corresponding gate/recovery decisions to support reproducibility and failure analysis.",
    )

    # Implementation paragraph (R2.7/R2.4 runtime)
    src = wrap_span(
        src,
        r"\\textbf\{Implementation and compute\.\}.*?it is not a physical time\.",
        "implementation",
    )

    # Data inventory table (R2.6)
    src = wrap_span(
        src,
        r"Table~\\ref\{tab:data_inventory\} lists every corpus.*?does not introduce newly collected videos or camera-pose ground truth\.\}",
        "data inventory",
    )

    # Metric definitions (R2.4)
    src = wrap_span(
        src,
        r"\\textbf\{Metric definitions\.\}.*?Sim\(3\) protocol before ATE computation\.",
        "metric definitions",
    )

    # Tables 2 and 3 (R2.4)
    for label in ("tab:chessboard_pose", "tab:scared_pose"):
        i = src.find("\\label{" + label + "}")
        if i < 0:
            print(f"WARN: {label} missing")
            continue
        start = src.rfind(r"\begin{table}", 0, i)
        end = src.find(r"\end{table}", i) + len(r"\end{table}")
        block = src[start:end]
        src = src[:start] + "\\revstart\n" + block + "\n\\revend" + src[end:]

    # Extended SCARED paragraph + table (R2.9, Plan 2+3)
    src = wrap_span(
        src,
        r"\\textbf\{Extended SCARED sequences with pose ground truth\.\}.*?\\label\{tab:scared_extended\}[\s\S]*?\\end\{table\}",
        "extended SCARED",
    )

    # Mean/median gap sentence (R2.4)
    src = wrap_exact(
        src,
        "The gap between the feature baselines' pooled medians ($0.37$--$0.40^\\circ$) and their means ($6.5$--$10.3^\\circ$) shows that their typical per-pair behavior is acceptable but punctuated by catastrophic matching failures, whereas the proposed variants have mean and median within $0.04^\\circ$ of each other, i.e., no heavy failure tail.",
    )

    # Rotation attribution aligned with Table 5 (R2.5)
    src = wrap_exact(
        src,
        "We therefore attribute the framework's rotation stability to dense optical-flow correspondences with Kabsch fitting rather than to the pseudo-height field itself; RANSAC outlier rejection, quality gating, and ICP refinement are retained as protective mechanisms (Section~\\ref{subsec:ablation}) and do not drive the accuracy on these sequences.",
    )

    # Ablation subsection (R2.5/R2.9)
    src = wrap_span(
        src,
        r"\\subsection\{Ablation and sensitivity analysis\}.*?\\label\{tab:clinical_cases\}[\s\S]*?\\end\{table\}",
        "ablation subsection",
    )

    # Fig18 caption sentence (R2.9/R2.10)
    src = wrap_exact(
        src,
        "This figure is a qualitative generalization illustration only; it does not enter any quantitative table.",
    )

    # Conclusion opening (R2.10)
    src = wrap_exact(
        src,
        "This paper presented a deformation-aware framework for pseudo-3D shape alignment in dynamic endoscopic sequences, integrating gradient-domain mesh generation, a hybrid semantic autoencoder, and spatiotemporally constrained non-rigid registration. The method adaptively localizes deformation-relevant regions and produces interpretable contour descriptors with lower highlight contamination than the evaluated classical operators. Five-sequence checkerboard and six-sequence public SCARED experiments reveal a consistent texture-dependent pattern: on weak-texture endoscopic sequences the framework provides the most stable relative-rotation/alignment cue among the evaluated baselines, whereas on feature-rich calibration scenes detector-based and structure-from-motion pipelines (DetectorFreeSfM, Reloc3r) achieve better trajectory-shape agreement. Accordingly, the contribution of the framework is an interpretable deformation-aware alignment front end for correspondence screening and reconstruction initialization, not a replacement for calibrated metric pose estimation.",
    )

    # Limitations added sentence (R2.6)
    src = wrap_exact(
        src,
        "Fourth, the autoencoder training corpus uses a frame-level split, so temporally adjacent frames of the same video can fall into different splits; although no quantitative pose, contour, or stereo table depends on the trained network, a sequence-level retrain under an identical budget yields essentially the same reconstruction error and re-ranking behavior (Section~4.5), so the split does not drive the descriptor's utility.",
    )
    src = wrap_exact(
        src,
        "Fifth, the extended evaluation of Table~\\ref{tab:scared_extended} uses additional \\emph{released} SCARED keyframes (datasets~5--7) rather than newly collected patient data, and the hard-quartile rows of Table~\\ref{tab:conditions} and the clinical-case windows of Table~\\ref{tab:clinical_cases} reuse the existing SCARED and self-acquired corpora; the self-acquired windows do not constitute a new ground-truth pose set.",
    )

    out = ROOT / "VRIH_Paper_R2_markedup.tex"
    out.write_text(src, encoding="utf-8")
    print("written", out)
    print("revstart", src.count("\\revstart"), "rev{", src.count("\\rev{"))


if __name__ == "__main__":
    main()
