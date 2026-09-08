#!/usr/bin/env python3
"""One-off sub-item verification against the FULL text of Reviewer #2's comments."""
import re
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised") / "revision2_materials"
paper = (ROOT / "VRIH_Paper_R2.tex").read_text(encoding="utf-8-sig")
resp = (ROOT / "response2.tex").read_text(encoding="utf-8")
pw = " ".join(paper.split())
rw = " ".join(resp.split())


def grab(cmd: str) -> str:
    i = paper.find("\\" + cmd + "{")
    if i < 0:
        return ""
    start = i + len(cmd) + 2
    depth = 1
    for k in range(start, len(paper)):
        if paper[k] == "{":
            depth += 1
        elif paper[k] == "}":
            depth -= 1
            if depth == 0:
                return " ".join(paper[start:k].split())
    return ""


print("== C1: title/abstract/contributions/conclusion terminology ==")
title = grab("title")
print("title:", title)
ab = grab("abstract")
print("abstract len:", len(ab))
print("abstract 'camera pose' occurrences:", ab.count("camera pose"))
print("abstract proxy/shape-alignment:", "proxy" in ab, "shape-alignment" in ab.lower() or "shape alignment" in ab.lower())
print("abstract snippet:", ab[:500])
print()
i = pw.find("Conclusion")
concl = pw[i:i + 2600] if i >= 0 else ""
print("conclusion mentions shape-alignment/proxy:", "shape-alignment" in concl or "proxy" in concl)
print("conclusion mentions camera pose claim:", "accurate camera rotation" in concl)
print("conclusion snippet:", concl[:600])
print()
print("paper-wide 'accurate camera rotation and translation':", "accurate camera rotation and translation" in pw)
print("paper-wide 'accurate rotation and translation':", "accurate rotation and translation" in pw)

print()
print("== C4 sub-items ==")
print("Reloc3r in paper:", pw.count("Reloc3r"), "| DetectorFreeSfM:", pw.count("DetectorFreeSfM"))
print("'>10K' or '10K' remnant:", "10K" in pw, "| '10{,}000':", "10{,}000" in pw)
print("'RMSE' occurrences:", pw.count("RMSE"))
for m in re.finditer(r"RMSE", pw):
    print("   ...", pw[max(0, m.start() - 90):m.end() + 90])
print("success rate:", "success rate" in pw.lower())
print("translation-direction error:", "translation-direction error" in pw)
print("std reported (pm in tables):", "$\\pm$" in paper or "\\pm" in paper)
print("Figure 7 caption wording 'quantitative comparison'?:")
for m in re.finditer(r"[Tt]rajector", pw):
    seg = pw[max(0, m.start() - 60):m.end() + 60]
    if "Figure" in seg or "caption" in seg:
        print("   ...", seg)
        break
print("Table 3 exists (scared_pose label):", "\\label{tab:scared_pose}" in paper)

print()
print("== C5 sensitivity coverage ==")
i = pw.find("one at a time around their default values")
print("sensitivity intro:", pw[i - 100:i + 700] if i >= 0 else "NOT FOUND")
print("pseudo-depth/z-scale in sensitivity:", "z-scale" in pw or "pseudo-depth scal" in pw or "height scale" in pw)
print("RANSAC threshold in sensitivity:", "RANSAC" in pw[i:i + 1200] if i >= 0 else False)
print("ICP params in sensitivity:", "ICP" in pw[i:i + 1200] if i >= 0 else False)
print("recovery strategy ablated/discussed:", "recovery" in pw.lower())

print()
print("== C6 sub-item: purely endoscopic test set separately ==")
print("resp mentions purely endoscopic separate:", "purely endoscopic" in rw)
print("paper SCARED-only tables separate from chessboard:", "\\label{tab:scared_pose}" in paper and "\\label{tab:chessboard_pose}" in paper)

print()
print("== C7 sub-items ==")
for kw in ["Farneback", "RAFT", "DIS ", "optical-flow", "AdamW", "learning rate", "batch size", "epoch", "resolution", "FPS", "frames per second"]:
    print(f"  '{kw}':", kw in pw or kw in paper)

print()
print("== C10 sub-items ==")
print("contour table now Table 1 (hcr first):", paper.find("\\label{tab:hcr}") < paper.find("\\label{tab:chessboard_pose}"))
print("Section 4.2 language: resp mentions language polish:", "language" in rw.lower())
print()
print("== rcomment full-text match (keyword spot checks vs pasted review) ==")
checks = [
    ("C1", "the effects of camera intrinsics, pixel-coordinate scale, and the manually defined pseudo-height scale"),
    ("C2", "This appears more similar to a robust geometric registration pipeline"),
    ("C3", "the expected value under a zero-sum derivative kernel"),
    ("C4", "Figure 7 is described as a quantitative comparison"),
    ("C5", "temporal gating, and recovery strategy"),
    ("C6", "patient, video, or sequence level"),
    ("C7", "the meaning of the continuous time variable"),
    ("C8", "if they share the same coordinate sum"),
    ("C9", "weak texture, specular reflection, large non-rigid deformation, rapid camera motion, motion blur, and partial occlusion"),
    ("C10", "the reference associated with YOLOv9"),
]
for tag, phrase in checks:
    print(f"  {tag} phrase in resp:", phrase in rw)
