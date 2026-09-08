#!/usr/bin/env python3
"""Final point-by-point audit of the R2 package against Reviewer #2's comments."""
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised") / "revision2_materials"
paper = (ROOT / "VRIH_Paper_R2.tex").read_text(encoding="utf-8")
clean = (ROOT / "VRIH_Paper_R2_clean.tex").read_text(encoding="utf-8")
marked = (ROOT / "VRIH_Paper_R2_markedup.tex").read_text(encoding="utf-8")
resp = (ROOT / "response2.tex").read_text(encoding="utf-8")
plain = (ROOT / "response2_plain.txt").read_text(encoding="utf-8")
cover = (ROOT / "coverletter_R2.tex").read_text(encoding="utf-8")

results = []


def check(name, ok):
    results.append(("PASS" if ok else "FAIL", name))


def ws(text):
    """Whitespace-normalized text for cross-line phrase matching."""
    return " ".join(text.split())


paper_ws, resp_ws = ws(paper), ws(resp)


# ---- Comment 1 ----
check("1 paper: 'reserve the term camera pose'", "reserve the term ``camera pose''" in paper)
check("1 paper: old alias removed", "denotes this relative pose proxy" not in paper)
check("1 paper: no rigorous mapping stated", "there is no rigorous mapping from this transform to a physical camera extrinsic" in paper)
check("1 paper: pinhole/roll claim removed", "coincides with the physical camera roll" not in paper)
check("1 paper: R^2=0.015 in relationship para", "affine fit $R^2=0.015$" in paper)
check("1 resp: quotes match paper", "never convert the estimated transform into a physical rotation--translation pair" in paper_ws and "never convert the estimated transform into a physical rotation--translation pair" in resp_ws)
check("1 paper: no leftover 'pose estimate(s)'", "pose estimates" not in paper_ws and "a pose estimate" not in paper_ws and "yields a pose estimate" not in paper_ws)
check("1 paper: pairwise shape-alignment transforms", "yields pairwise shape-alignment transforms" in paper_ws)
check("1 paper: no leftover R/t-parameter wording", "relative rotation and translation parameters" not in paper)
check("1 paper: methods uses pairwise transform wording", "transformed by a pairwise shape-alignment transform" in paper)
check("1 resp: residual cleanup noted", "pairwise shape-alignment transforms" in resp_ws)
check("1 highlights: relative-rotation cue", "relative-rotation cue" in (ROOT / "Highlights_revised.txt").read_text(encoding="utf-8"))

# ---- Comment 2 ----
check("2 paper: keywords Robust registration", "Pseudo-3D shape alignment; Robust registration;" in paper)
check("2 paper: 'following the differentiable-rendering formulation' removed", "following the differentiable-rendering formulation" not in paper)
check("2 paper: explicit no-renderer statement", "no differentiable renderer and no rendering-based loss participate" in paper)
check("2 paper: 'rendering losses' removed", "rendering losses" not in paper)
check("2 resp: quote matches paper", "no differentiable renderer and no rendering-based loss participate in estimating" in resp_ws)

# ---- Comment 3 ----
check("3 paper: zero-sum addressed head-on", "as dictated by the zero-sum derivative kernel" in paper)
check("3 paper: distribution-free statement", "no distributional assumption is attached to either" in paper)
check("3 paper: 3,482 frames", "3{,}482" in paper)
check("3 paper: CV 0.014--0.171", "0.014$--$0.171" in paper)
check("3 paper: kurtosis 112", "kurtosis $112$" in paper)
check("3 paper: 0.015 deg tightened", "at most $0.015^\\circ$" in paper)
check("3 paper: no 68% Gaussian remnants", "68.27" not in paper and "inflection" not in paper and paper.count("Gaussian probability") == paper.count("no Gaussian probability"))
check("3 paper: no CLT/Lindeberg remnants", "Lindeberg" not in paper)
check("3 resp: acknowledges E[S_N]=0", "\\mathbb{E}[S_N]=\\mu\\sum_{u,v}w_{u,v}=0" in resp)
check("3 resp: magnitude/Rayleigh point", "Rayleigh-type" in resp)
check("3 resp: iScience companion evidence", "coverage-aware IoU of $0.809$" in resp)
check("3 resp: 0.015 synced", "most $0.015^\\circ$ across" in resp)
# ---- Comment 3: this-round additions (admission-first logic, intuition, sweep, visuals) ----
check("3 resp: admission-first opening", "we admit them\nplainly: the derivation in the previous version was mathematically erroneous" in resp.replace("\r\n", "\n"))
check("3 resp: motivation confession", "explain honestly how the flawed derivation came\nabout" in resp.replace("\r\n", "\n"))
check("3 resp: hypothesis does not survive data", "does not survive contact with the data" in resp_ws)
check("3 resp: operating-point sweep bullet", "Operating-point sweep for $t_{\\text{main}}$ (new experiment)" in resp)
check("3 resp: sweep numbers", "carry $77\\%$ of the total gradient magnitude" in resp and "Dice of $0.73$" in resp)
check("3 resp: marginal cost doubling", "$0.006$ (over $p{=}50\\!\\to\\!68$) to $0.011$" in resp and "$0.024$" in resp)
check("3 resp: mean at 78th percentile", "mean lands at the 78th percentile in the median" in resp)
check("3 resp: figures present", "\\label{fig:resp_thresholds}" in resp and "\\label{fig:resp_3d}" in resp)
check("3 resp: tables present", "\\label{tab:resp_sweep}" in resp and "\\label{tab:resp_iscience}" in resp)
check("3 resp: iScience operator-family precision", "same operator family" in resp and "not the tracker itself" in resp_ws)
check("3 resp: tracker not sole cue", "not} as its sole cue" in resp or "not as its sole cue" in resp_ws)
check("3 resp: Scheme C + NanoTrack named", "Scheme~C + NanoTrack" in resp)
check("3 paper: Sobel realization disclosed", "realized by a $3\\times3$ Sobel stencil" in paper)
check("3 resp: image files exist", (ROOT / "images" / "Fig_threshold_panels.png").exists() and (ROOT / "images" / "Fig_gradient_3d_cutplanes.png").exists())
check("3 paper: design intuition paragraph", "Design intuition: information bulk versus salient tail" in paper)
check("3 paper: motivation-only disclaimer", "recorded here as motivation only---no step of the method relies on it" in paper)
check("3 paper: bulk-versus-tail core", "distribution-free core---the bulk-versus-tail separation" in paper)
check("3 paper: sweep sentence in 3.1", "An operating-point sweep over detection percentiles" in paper and "carry $77\\%$ of the total gradient magnitude" in paper)
check("3 paper: sweep marginal costs", "$0.006$ per percentile over $50\\!\\to\\!68$" in paper)
check("3 paper/resp: intuition quote synced", "recorded here as motivation\nonly---no step of the method relies on it" in resp.replace("\r\n", "\n"))
check("3 markedup: design intuition marked", "Design intuition: information bulk versus salient tail" in marked and marked.count("\\revstart") == 18)
check("3 plain: visual block replaced cleanly", "PDF version of this response letter" in plain and "includegraphics" not in plain and "toprule" not in plain)
# ---- Comment 3: precision upgrades (single-component fact, mixture explanation) ----
check("3 resp: single-component 68.27% fact", "68.27\\%" in resp and "39.3\\%" in resp and "1.51\\sigma" in resp)
check("3 resp: breaks at component-to-magnitude transfer", "transfer from component to magnitude" in resp_ws)
check("3 resp: could not succeed in principle", "could not have succeeded even in principle" in resp_ws)
check("3 resp: no CLT applies", "no central-limit argument applies" in resp_ws)
check("3 resp: order-statistic residue", "by construction as an order statistic" in resp_ws)
check("3 paper: mixture explanation", "necessarily yields a sharply peaked, heavy-tailed aggregate distribution" in paper_ws)
check("3 paper: split-by-construction clause", "a split that holds by construction for any underlying distribution" in paper_ws)
check("3 paper/resp: mixture quote synced", "an endoscopic frame is a mixture of large smooth regions, which produce near-zero derivative responses" in paper_ws and "an endoscopic frame is a mixture of large smooth regions, which produce near-zero derivative responses" in resp_ws)
check("3 paper: 68.27 kept out of paper", "68.27" not in paper and "39.3" not in paper and "1.51" not in paper)
# ---- Comment 3: invariance property + noise-floor anchoring ----
check("3 paper: invariance property", "percentiles are equivariant under any strictly increasing rescaling" in paper_ws)
check("3 paper: invariance conclusion", "invariant to global illumination or contrast gain---a robustness no fixed absolute threshold offers" in paper_ws)
check("3 paper: noise-floor analysis", "the bulk boundary sits at the 81st percentile" in paper_ws and "interquartile range 79--83" in paper_ws)
check("3 paper: complementary roles", "recall-oriented---the salient tail is never truncated" in paper_ws and "precision-oriented" in paper_ws)
check("3 resp: invariance quote synced", "percentiles are equivariant under any strictly increasing rescaling" in resp_ws)
check("3 resp: noise-floor bullet", "Noise-floor anchoring of the two thresholds (new analysis)" in resp_ws and "81st percentile" in resp_ws)
check("3 resp: recall/precision roles", "recall-oriented" in resp_ws and "precision-oriented" in resp_ws)
check("3 data: bulk_fraction.json exists", (Path(__file__).resolve().parent / "results" / "bulk_fraction.json").exists())

# ---- Comment 4 ----
check("4 paper: RPE correspondence", "standard relative pose error (RPE) at a frame interval of one" in paper)
check("4 paper: median column chessboard", "0.896" in paper and "Median rot" in paper)
check("4 paper: median column scared", "0.272" in paper and "0.420" in paper)
check("4 paper: mean-vs-median analysis", "punctuated by catastrophic matching failures" in paper)
check("4 paper: Sim(3) protocol stated", "aligned with the same Sim(3) protocol before ATE computation" in paper)
check("4 resp: medians match paper", "$0.896^\\circ$" in resp and "$0.272^\\circ$" in resp and "$0.420^\\circ$" in resp)

# ---- Comment 5 ----
check("5 paper: pipeline ablation table", "\\label{tab:pipeline_ablation}" in paper)
check("5 paper: gate 0/474", "$0$ of $474$" in paper)
check("5 paper: honest attribution to flow", "originates from the dense-flow correspondence field itself" in paper)
check("5 paper: diffusion-block statement", "diffusion-inspired denoising block is an internal component" in paper)
check("5 paper: sensitivity table", "\\label{tab:sensitivity}" in paper)
check("5 paper: sensitivity d1_k1 d3_k2", "d1\\_k1" in paper and "d3\\_k2" in paper and "0.276" in paper)
check("5 paper: sensitivity band", "0.273$--$0.288" in paper and "0.012^\\circ$ from the default" in paper)
check("5 paper: Kabsch vs RANSAC disclosed", "plain SVD-based Kabsch" in paper and "0.302$--$0.303" in paper)
check("5 paper: rotation attribution Kabsch", "dense optical-flow correspondences with Kabsch fitting" in paper)
check("5 resp: numbers match (0.302/0.303)", "$0.302\\pm0.061^\\circ$" in resp and "$0.303\\pm0.062^\\circ$" in resp)
check("5 resp: sensitivity band", "0.273$--$0.288" in resp and "default $0.276^\\circ$" in resp)
check("5 resp: dense flow separate control", "sparse-feature baselines" in resp)
check("5 resp: recovery bullet", "Recovery strategy" in resp)
check("5 paper: ICP-not-swept structural reason", "invariant to the ICP parameters by construction" in paper_ws and "never fires ($0$ of $474$ pairs" in paper_ws)
check("5 resp: ICP-not-swept bullet", "Why ICP parameters are not in the main sweep" in resp_ws and "ungated ICP is unstable" in resp_ws)
check("5 resp: ICP config disclosed", "point-to-plane minimization, maximum correspondence distance $10$ in pixel units" in resp_ws)
check("5 paper/resp: ICP quote synced", "invariant to the ICP parameters by construction" in resp_ws)

# ---- Comment 6 ----
check("6 paper: frame-level split disclosed", "temporally adjacent frames may therefore appear in different splits" in paper)
check("6 paper: sequence-level retrain done", "sequence-level (video-level) split (four sequences for training" in paper and "does not change the descriptor's re-ranking behavior" in paper)
check("6 paper: limitations fourth item", "Fourth, the autoencoder training corpus uses a frame-level split" in paper)
check("6 paper: Fig6 qualitative-only caption", "does not enter any quantitative table" in paper)
check("6 paper: data inventory table", "\\label{tab:data_inventory}" in paper)
check("6 paper: six sequences 3002", "six self-acquired endoscopic sequences" in paper and "3{,}002" in paper)
check("6 resp: data inventory bullet", "Data inventory (new Table~1)" in resp)

# ---- Comment 7 ----
check("7 paper: RANSAC 8px", "$8$-pixel inlier threshold" in paper)
check("7 paper: 3.0px removed", "3.0$-pixel" not in paper and "3.0-pixel" not in paper)
check("7 paper: ICP units", "in $(u,v,h)$ units, i.e., pixel scale" in paper)
check("7 paper: quality gate thresholds", "\\tau_\\eta=0.15" in paper and "\\tau_\\epsilon=12" in paper)
check("7 paper: recon composite weights", "$1.0/0.5/0.1/0.3$" in paper)
check("7 paper: contour composite weights", "$1.0/1.0/0.3$" in paper)
check("7 paper: aux weight schedule", "decays linearly from $0.2$ to a floor of $0.1$" in paper)
check("7 paper: L_reg = weight decay", "realized as AdamW weight decay" in paper)
check("7 paper: lambda_s=0 in descriptor training", "$\\lambda_s=0$" in paper)
check("7 paper: diffusion time meaning", "normalized diffusion-time index" in paper)
check("7 paper: params/memory/runtime", "225.6" in paper and "3.8" in paper and "141" in paper)
check("7 resp: RANSAC correction disclosed", "inconsistent $3.0$-pixel figure in the previous draft was corrected" in resp)

# ---- Comment 8 ----
check("8 paper: anti-diagonal collision admitted", "identical values along anti-diagonals" in paper)
check("8 paper: released encoder is separate 2D", "released encoder uses a conventional separable two-dimensional" in paper)
check("8 paper: draft formula not used", "that draft formula is \\emph{not} used in the released encoder" in paper)
check("8 paper: three-variant ablation numbers", "72.3" in paper and "91.6" in paper and "74.7" in paper)
check("8 paper: full-network retrain", "full} $225.6$\\,M released network---not a compact proxy" in paper)
check("8 paper: bounded claim", "neither better nor worse than the alternatives" in paper)
check("8 resp: synced quotes", "neither better nor worse than the alternatives" in resp_ws and "none of the quantitative pose tables" in paper_ws and "none of the quantitative pose tables" in resp_ws)

# ---- Comment 9 ----
check("9 paper: conditions table relabeled", "More-affected half" in paper and "severity vs" in paper)
check("9 paper: six conditions present", all(s in paper for s in ["Weak texture (low mean", "Specular reflection (highlight fraction)", "Large deformation (mask-IoU change)", "Rapid camera motion (mean flow)", "Motion blur (low Laplacian variance)", "Partial occlusion (support change)"]))
check("9 paper: significance markers", "$+0.24^{**}$" in paper and "$+0.13^{**}$" in paper)
check("9 resp: numbers match table", "$0.368^\\circ$" in resp and "$0.236^\\circ$" in resp and "$0.324^\\circ$" in resp)
check("9 paper: reproduced condition numbers", "0.368" in paper and "0.236" in paper and "$-0.20^{**}$" in paper)
check("9 paper: hard-quartile numbers", "0.425" in paper and "0.442" in paper and "0.410" in paper and "0.483" in paper)
check("9 paper: not new GT sequences", "released} SCARED keyframes" in paper and "do not constitute a new ground-truth pose set" in paper)
check("9 paper: extended SCARED table", "\\label{tab:scared_extended}" in paper and "0.270" in paper and "6.55" in paper)
check("9 data: extended/abrupt json", (ROOT.parent / "experiments" / "results" / "scared_extended.csv").exists() and (ROOT.parent / "experiments" / "results" / "scared_abrupt.json").exists())
check("9 paper: clinical-case table", "\\label{tab:clinical_cases}" in paper and "no camera-pose ground truth" in paper)
check("9 paper: clinical-case residuals", "0.568" in paper and "1.619" in paper and "0.971" in paper)
check("9 data: condition summary exists", (ROOT.parent / "experiments" / "results" / "scared_condition_summary.json").exists())
check("9 data: hard-quartile json", '"deformation_q4"' in (ROOT.parent / "experiments" / "results" / "scared_condition_summary.json").read_text(encoding="utf-8"))
check("9 data: clinical-case json", (ROOT.parent / "experiments" / "results" / "self_acquired_case_summary.json").exists())
check("9 resp: hard quartile and extended GT", "0.425" in resp and "held-out} SCARED keyframes from datasets~5--7" in resp_ws and "0.270" in resp)
check("10 resp: ten tables listed", "ten tables" in resp and "Table~1 data inventory" in resp and "Table~10" in resp and "Table~5 extended" in resp)

# ---- Comment 10 ----
import re
labels = re.findall(r"\\label\{tab:([a-z_0-9]+)\}", paper)
check("10 paper: 10 tables in expected order", labels == ["data_inventory", "hcr", "chessboard_pose", "scared_pose", "scared_extended", "stereo_pseudo3d", "pipeline_ablation", "sensitivity", "conditions", "clinical_cases"])
figs = re.findall(r"\\label\{fig:([A-Za-z_0-9]+)\}", paper)
check("10 paper: 14 figures", len(figs) == 14)
check("10 paper: multicolumn count fixed", "\\multicolumn{5}{l}" not in paper)
check("10 resp: YOLOv9 ECCV citation", "10.1007/978-3-031-72751-1" in resp and "ECCV 2024" in paper)

# ---- Package consistency ----
check("pkg: clean identical to master", clean == paper)
check("pkg: markedup derived from master (contains all master tables)", all(("\\label{tab:%s}" % l) in marked for l in labels))
check("pkg: markedup has rev markup", "\\revstart" in marked and "revchange" in marked)
check("pkg: plain txt has 10 comments", all(("COMMENT 2.%d:" % i) in plain for i in range(1, 11)))
check("pkg: plain txt R1-vs-R2 numbers synced", "3,482" in plain and "0.014" in plain and "8 -pixel" in plain.replace("$", "") or "8-pixel" in plain or "8 pixel" in plain)
check("pkg: cover letter is R2 (mentions R1 number)", "VRIH-D-26-00061R1" in cover)
check("pkg: cover letter reviewer #3", "Reviewer \\#3" in cover)
check("pkg: response addresses reviewer #3 note", "Note on Reviewer \\#3" in resp)

fails = [r for r in results if r[0] == "FAIL"]
for status, name in results:
    print(status, "|", name)
print()
print(f"TOTAL {len(results)} checks, {len(fails)} FAIL")
