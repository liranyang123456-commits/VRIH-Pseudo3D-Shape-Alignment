#!/usr/bin/env python3
"""Consistency audit for the third-round (minor revision) package.

Checks that the two Reviewer-#2 consistency items are closed in every
document of revision3_materials and that nothing else drifted from R2.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R2 = ROOT / "revision2_materials"
R3 = ROOT / "revision3_materials"

paper = (R3 / "VRIH_Paper_R3.tex").read_text(encoding="utf-8")
clean = (R3 / "VRIH_Paper_R3_clean.tex").read_text(encoding="utf-8")
marked = (R3 / "VRIH_Paper_R3_markedup.tex").read_text(encoding="utf-8")
resp = (R3 / "response3.tex").read_text(encoding="utf-8")
plain = (R3 / "response3_plain.txt").read_text(encoding="utf-8")
cover = (R3 / "coverletter_R3.tex").read_text(encoding="utf-8")
r2paper = (R2 / "VRIH_Paper_R2.tex").read_text(encoding="utf-8")

fails = 0
total = 0


def check(name: str, ok: bool) -> None:
    global fails, total
    total += 1
    print(("PASS" if ok else "FAIL"), "|", name)
    if not ok:
        fails += 1


KEYWORDS = r"\keywords{Relative motion estimation; Pseudo-3D shape alignment; Robust registration; Non-rigid deformation; Endoscopic surgical navigation}"
NEW = "through geometric registration of the resulting pseudo-3D meshes---dense optical-flow correspondences, robust RANSAC/Kabsch fitting, and ICP refinement (Section~\\ref{subsec:pose})"

# ---- Comment 1: Related Work wording ----
check("1 paper: 'differentiable mesh registration' removed", "differentiable mesh registration" not in paper)
check("1 paper: new geometric-registration sentence present", NEW in paper)
check("1 paper: sentence sits in Related Works", paper.find(NEW) > paper.find(r"\section{Related Works}") and paper.find(NEW) < paper.find(r"\section{Methods}"))
check("1 paper: label subsec:pose exists", r"\label{subsec:pose}" in paper)
# every remaining 'differentiable' must be about other work; none may co-occur with our own method words in the same sentence
own = re.compile(r"(?i)(our|the proposed|the present work|this study|this paper|we )")
bad = []
for sent in re.split(r"(?<=[.!?])\s+", paper):
    if re.search(r"(?i)differentiable", sent) and own.search(sent):
        # allowed: explicit negations ("no differentiable renderer ...") and 'non-differentiable formulations' of classical descriptors
        if "no differentiable renderer" in sent or "non-differentiable formulations" in sent:
            continue
        bad.append(sent[:120])
check("1 paper: no self-description as differentiable (residual scan)", not bad)
for b in bad:
    print("      residual:", b)
check("1 paper: Methods overview still has explicit no-renderer statement", "no differentiable renderer and no rendering-based loss participate" in paper)
check("1 resp: quotes old wording", "differentiable mesh registration" in resp)
check("1 resp: quotes new wording", "robust RANSAC/Kabsch fitting, and ICP refinement" in resp)
check("1 resp: full-text re-audit bullet", "Full-text re-audit" in resp)
check("1 markedup: single rev block on the new sentence", marked.count("\\rev{") == 2 and r"\rev{estimates a relative shape-alignment transform through geometric registration" in marked)
check("1 markedup: R3 note present", "third-round (R3, minor revision) change" in marked)

# ---- Comment 2: keywords / metadata ----
check("2 paper: keyword line is the five agreed keywords", KEYWORDS in paper)
kw_line = next(l for l in paper.splitlines() if l.startswith(r"\keywords{"))
check("2 paper: 'Differentiable rendering' absent from keyword line", "ifferentiable" not in kw_line)
check("2 paper: remaining 'Differentiable rendering' mentions are background only", all(
    ("introduced new possibilities" in s or "advanced 2D and 3D reconstruction" in s or "has likewise progressed" in s)
    for s in re.split(r"(?<=[.!?])\s+", paper) if re.search(r"(?i)differentiable[- ]rendering", s)))
check("2 resp: metadata keyword replacement stated", "``Differentiable rendering''" in resp and "``Robust registration''" in resp)
check("2 resp: five keywords listed verbatim", "Relative motion estimation; Pseudo-3D shape alignment; Robust\nregistration; Non-rigid deformation; Endoscopic surgical navigation" in resp)
check("2 cover: metadata keyword change stated", "Editorial Manager" in cover and "Robust registration" in cover)

# ---- Highlight colour consistency (blue in R3; no stale 'dark red' legend) ----
check("colour: markedup defines revchange as blue", r"\definecolor{revchange}{RGB}{0,70,180}" in marked)
check("colour: response defines changecolor as the same blue", r"\definecolor{changecolor}{RGB}{0,70,180}" in resp)
check("colour: markedup note says blue", "Blue text highlights the third-round" in marked)
check("colour: no 'dark red' legend left in R3 docs", all("dark red" not in d for d in (resp, cover, plain, marked)))
check("colour: legend says blue in response/cover/plain", "highlighted in blue" in resp and "highlighted in blue" in cover and "in blue" in plain)

# ---- Package integrity ----
check("pkg: clean identical to master", clean == paper)
diff_lines = [l for l in paper.splitlines() if l not in r2paper.splitlines()]
check("pkg: master differs from R2 in exactly one line", len(diff_lines) == 1 and NEW in diff_lines[0])
check("pkg: markedup derived from master (same tables)", marked.count(r"\begin{table}") == paper.count(r"\begin{table}"))
check("pkg: response has 2 comments", resp.count(r"\rcomment{") == 2)
check("pkg: response says nothing left unaddressed", "no rebuttal is required" in resp)
check("pkg: response lists round-3 summary", r"\section{Summary of Changes in Round 3}" in resp)
check("pkg: plain txt has 2 comments", "COMMENT 1:" in plain and "COMMENT 2:" in plain and "COMMENT 3:" not in plain)
check("pkg: plain txt names uploaded files", "response3.pdf" in plain and "VRIH_Paper_R3_clean.pdf" in plain and "VRIH_Paper_R3_markedup.pdf" in plain)
check("pkg: plain txt has no LaTeX residue", "\\" not in plain and "\\item" not in plain and "textbf" not in plain)
check("pkg: cover letter is R2 manuscript number", "VRIH-D-26-00061R2" in cover and "VRIH-D-26-00061R2" in resp)
check("pkg: highlights carried over", (R3 / "Highlights_revised.txt").exists() and (R3 / "Highlights_revised.docx").exists())
for pdf in ("VRIH_Paper_R3_clean.pdf", "VRIH_Paper_R3_markedup.pdf", "response3.pdf", "coverletter_R3.pdf"):
    check(f"pkg: {pdf} exists and non-trivial", (R3 / pdf).exists() and (R3 / pdf).stat().st_size > 50_000)

print(f"\nTOTAL {total} checks, {fails} FAIL")
raise SystemExit(1 if fails else 0)
