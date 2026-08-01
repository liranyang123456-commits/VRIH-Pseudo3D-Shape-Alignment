#!/usr/bin/env python3
"""Build a selective marked-up manuscript highlighting only Editor/Reviewer items."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised")


def wrap_exact(text: str, old: str, new: str | None = None) -> str:
    if old not in text:
        raise SystemExit("MISSING BLOCK:\n" + old[:200].replace("\n", " ") + "...")
    if new is None:
        new = r"\rev{" + old + "}"
    return text.replace(old, new, 1)


def main() -> None:
    src = (ROOT / "VRIH_Paper_clean.tex").read_text(encoding="utf-8")

    preamble_add = r"""
%---- Selective revision markup (Editor/Reviewer items only; not full latexdiff) ----
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
\noindent{\footnotesize\rev{Note: Dark-red text highlights substantive revisions that specifically address the Editor and Reviewers (scope clarification, new metrics/experiments, baselines, calibration literature, limitations/SDG, and figure originality). Routine wording edits are not marked.}}
\vspace{0.6em}

"""
    if "Dark-red text highlights substantive revisions" not in src:
        src = src.replace(r"\maketitle", r"\maketitle" + note, 1)

    # Title
    src = wrap_exact(
        src,
        r"\title{Nonrigid-Assisted Pseudo-3D Shape Alignment for Endoscopic Image Sequences}{Nonrigid-Assisted Pseudo-3D Shape Alignment for Endoscopic Image Sequences}",
        r"\title{\rev{Nonrigid-Assisted Pseudo-3D Shape Alignment for Endoscopic Image Sequences}}{\rev{Nonrigid-Assisted Pseudo-3D Shape Alignment for Endoscopic Image Sequences}}",
    )

    # Abstract
    m = re.search(r"\\abstract\{(.*?)\}", src, flags=re.S)
    if not m:
        raise SystemExit("no abstract")
    src = src.replace(m.group(0), r"\abstract{\rev{" + m.group(1) + "}}", 1)

    # Keywords
    src = wrap_exact(
        src,
        r"\keywords{Relative motion estimation; Pseudo-3D shape alignment; Differentiable rendering; Non-rigid deformation; Endoscopic surgical navigation}",
        r"\keywords{\rev{Relative motion estimation; Pseudo-3D shape alignment; Differentiable rendering; Non-rigid deformation; Endoscopic surgical navigation}}",
    )

    # Scope / supportive / organization
    block = r"""\textbf{Scope and terminology.} To avoid ambiguity, we clarify at the outset that the proposed method recovers the \emph{relative} inter-frame motion as a shape-alignment transform defined in a pseudo-3D $(u,v,h)$ space, in which $h$ is a non-metric pseudo-height derived from gradient information rather than calibrated metric depth. Consequently, throughout this paper the term ``camera pose'' denotes this relative pose proxy; metric interpretation is obtained only through calibration-based checkerboard ground truth (Section~\ref{sec:experiments}). This distinction is made explicit to avoid overstating the results as true metric $SE(3)$ extrinsic recovery.

\textbf{Supportive technologies.} The proposed framework is built upon several enabling techniques: bilateral filtering and Gaussian-derivative gradient analysis for deformation-region detection; differentiable rendering as the frame-to-frame mesh alignment engine; implicit neural representation via a hybrid convolution--attention--positional-encoding autoencoder; dense optical flow with RANSAC and SVD-based Kabsch robust fitting for correspondence estimation; and iterative closest point (ICP) geometric refinement together with quality gating and keyframe relocalization for drift control.

The remainder of this paper is organized as follows. Section~\ref{sec:related} reviews related work on feature description/matching and camera pose estimation. Section~\ref{sec:methods} details the proposed methodology, including adaptive pseudo-3D mesh generation, the hybrid autoencoder, and non-rigid-assisted pose estimation. Section~\ref{sec:experiments} presents experiments and validation on diverse datasets, and Section~\ref{sec:conclusion} concludes with limitations, industrial and societal implications, and future work."""
    src = wrap_exact(src, block, r"\revstart" + "\n" + block + "\n" + r"\revend")

    # Calibration / 2026 literature (R1/R3)
    stereo_frag = (
        "For binocular endoscopes and depth-aware navigation, stereo camera calibration "
        "is equally fundamental: it establishes the inter-camera epipolar geometry and "
        "metric scale through rectification and joint intrinsic/extrinsic estimation, "
        "and remains a prerequisite for accurate metric 3D reconstruction~\\cite{43,26}."
    )
    if stereo_frag in src:
        src = src.replace(stereo_frag, r"\rev{" + stereo_frag + "}", 1)
    else:
        print("WARN: stereo frag missing")

    bouguet = "Bouguet's Camera Calibration Toolbox~\\cite{42}"
    if bouguet in src:
        src = src.replace(bouguet, r"\rev{" + bouguet + "}", 1)

    m = re.search(
        r"Most recently, deformation-aware pose estimation and reconstruction for endoscopy have advanced rapidly\..*?our method addresses it from a complementary, interpretable pseudo-3D perspective that requires neither dense depth input nor per-scene neural training\.",
        src,
        flags=re.S,
    )
    if m:
        src = src.replace(m.group(0), r"\rev{" + m.group(0) + "}", 1)
    else:
        print("WARN: 2026 paragraph missing")

    cite49 = (
        ", and, more recently, semantic-guided area-to-point matching with geometric "
        "consistency for accurate correspondence under ambiguous textures~\\cite{49}"
    )
    if cite49 in src:
        src = src.replace(cite49, r"\rev{" + cite49 + "}", 1)

    # Physical justification (R2.2)
    phys = r"""\textbf{Physical justification and robustness of the pseudo-3D representation.}
The pseudo-3D mesh is not intended as a metric or photometric reconstruction of the tissue surface; rather, it serves as a \emph{structure-encoding representation} that lifts repeatable 2D structural saliency into an auxiliary dimension so that dense correspondence and rigid fitting become better conditioned than in the raw image plane. The use of gradient magnitude as a deformation-height proxy is justified by the statistical argument above: after bilateral filtering and Gaussian-derivative smoothing, the first-order gradient response is approximately Gaussian (Lindeberg--Feller central limit theorem), so stable low-frequency anatomy concentrates within $[\mu-\sigma,\mu+\sigma]$ while deformation-salient contours occupy the distribution tail isolated by $t_{90}$. The height field therefore encodes \emph{where structural change is repeatable across consecutive frames}, and motion is recovered by cross-frame shape consistency rather than by absolute geometric fidelity. We explicitly acknowledge that soft-tissue appearance is affected by lighting, specular reflection, fluid, smoke, and non-Lambertian effects, and that a high image gradient does not always correspond to a real 3D surface deformation. These confounders are mitigated by three mechanisms: (i) bilateral filtering and foreground masking that suppress isolated high-frequency noise and background clutter; (ii) the HCR evaluation in Table~\ref{tab:hcr}, which quantifies the proportion of contour pixels contaminated by highlights; and (iii) robust RANSAC/IRLS fitting on the in-plane residual together with the quality-gating and recovery logic in Section~\ref{subsec:pose}, which down-weight or reject correspondences dominated by highlights or transient artifacts. Consequently, pixels that are high-gradient purely due to specular highlights contribute little to the final shape-alignment estimate."""
    src = wrap_exact(src, phys, r"\revstart" + "\n" + phys + "\n" + r"\revend")

    # Autoencoder I/O (R2.3)
    io = r"""\textbf{Inputs, outputs, supervision, and role in pose estimation.}
To make the function of the network explicit, we summarize its interface and its concrete contribution to the pipeline. The autoencoder takes as input the RGB endoscopic frame together with its deformation-region crop, and produces four outputs: a compact implicit descriptor $z$, a predicted contour map $\hat{C}$, a predicted heatmap $\hat{H}$, and the reconstructed image $\hat{I}$. Crucially, the ground-truth contour maps $C_b$ and heatmaps $H_b$ are \emph{not} manually annotated; they are generated automatically by the proposed gradient-domain method (the $t_{90}$ contour and its distance-transform heatmap), so that supervision is self-consistent with the deformation-detection stage and requires no costly manual labeling, which is advantageous for scalability. Within the overall framework, the implicit descriptor $z$ and the predicted contour drive the spatiotemporally constrained target search of Section~\ref{subsec:pose}---the fast coarse screening and contour semantic-aware fine ranking---thereby providing robust inter-frame correspondences and a stable initialization under weak texture and noise. The subsequent geometric back-end (dense optical flow with RANSAC/Kabsch fitting and ICP refinement) estimates the relative shape-alignment transform. In short, the autoencoder acts as the weak-texture-tolerant matching front-end, while the registration module performs the actual pose-proxy estimation; this data flow is indicated by the pipeline in Fig.~\ref{fig:Fig1}."""
    src = wrap_exact(src, io, r"\revstart" + "\n" + io + "\n" + r"\revend")

    # HCR evaluation + table (R3.2)
    hcr_start = r"\textbf{Highlight-contamination evaluation.}"
    idx = src.find(hcr_start)
    if idx < 0:
        raise SystemExit("HCR start missing")
    idx2 = src.find(r"\end{table}", idx)
    if idx2 < 0:
        raise SystemExit("HCR table end missing")
    idx2 += len(r"\end{table}")
    block = src[idx:idx2]
    src = src[:idx] + r"\revstart" + "\n" + block + "\n" + r"\revend" + src[idx2:]

    # Pose baselines / metrics (R2.4, R3.2/3.3)
    q_start = r"\subsection{Quantitative evaluation of relative shape alignment}"
    q_end = r"\subsection{Calibrated stereo validation of pseudo-height-assisted matching}"
    i1, i2 = src.find(q_start), src.find(q_end)
    if i1 < 0 or i2 < 0:
        raise SystemExit(f"quant section markers missing {i1} {i2}")
    src = src[:i1] + r"\revstart" + "\n" + src[i1:i2] + r"\revend" + "\n" + src[i2:]

    # Stereo validation (R2.2 follow-up)
    s_start = r"\subsection{Calibrated stereo validation of pseudo-height-assisted matching}"
    s_end = r"\section{Conclusion}"
    i1, i2 = src.find(s_start), src.find(s_end)
    if i1 < 0 or i2 < 0:
        raise SystemExit("stereo section markers missing")
    src = src[:i1] + r"\revstart" + "\n" + src[i1:i2] + r"\revend" + "\n" + src[i2:]

    # Conclusion / limitations / SDG / future (R1, R3.5)
    c_start = r"\section{Conclusion}"
    c_end = r"\section*{Declaration of Competing Interest}"
    i1, i2 = src.find(c_start), src.find(c_end)
    if i1 < 0 or i2 < 0:
        raise SystemExit("conclusion markers missing")
    src = src[:i1] + r"\revstart" + "\n" + src[i1:i2] + r"\revend" + "\n" + src[i2:]

    # Figure originality (R1.12)
    figo = r"""\section*{Figure Originality}
All schematic diagrams, plots, tables, and comparison layouts in this manuscript were created by the authors; no third-party copyrighted figure is reproduced. For comparative results, the displayed baseline outputs were generated by the authors using publicly available implementations or models under their stated licenses. The corresponding methods are cited at their first appearance in the captions and text."""
    src = wrap_exact(src, figo, r"\revstart" + "\n" + figo + "\n" + r"\revend")

    out = ROOT / "VRIH_Paper_markedup.tex"
    out.write_text(src, encoding="utf-8")
    (ROOT / "revision_submission_materials" / "VRIH_Paper_markedup.tex").write_text(src, encoding="utf-8")
    print("written", out)
    print("bytes", out.stat().st_size)
    print("revstart", src.count(r"\revstart"))
    print("rev{", src.count(r"\rev{"))


if __name__ == "__main__":
    main()
