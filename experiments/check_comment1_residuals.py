#!/usr/bin/env python3
"""Comment-1 leftover-language scan after the residual cleanup."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised") / "revision2_materials"
paper = (ROOT / "VRIH_Paper_R2.tex").read_text(encoding="utf-8-sig")
bib = paper.find(r"begin{thebibliography}")
body = paper[:bib] if bib >= 0 else paper
pw = " ".join(body.split())

FORBIDDEN = [
    "accurate camera rotation",
    "accurate camera translation",
    "accurate rotation and translation",
    "camera rotation and translation",
    "true camera pose",
    "yields a pose estimate",
    "Along with the pose estimates",
    "relative rotation and translation parameters",
    "most stable rotation on weak-texture",
]
REQUIRED = [
    "reserve the term ``camera pose''",
    "there is no rigorous mapping from this transform to a physical camera extrinsic",
    "we never convert the estimated transform into a physical rotation--translation pair",
    "yields pairwise shape-alignment transforms",
    "transformed by a pairwise shape-alignment transform",
    "not a replacement for calibrated metric pose estimation",
    "must not be interpreted as metric camera-pose recovery",
]

print("== forbidden leftovers in manuscript body ==")
fails = 0
for s in FORBIDDEN:
    hit = s in body or s in pw
    print(("FAIL" if hit else "PASS"), s)
    fails += int(hit)

print("\n== required C1 sentences ==")
for s in REQUIRED:
    hit = s in body or s in pw
    print(("PASS" if hit else "FAIL"), s[:80])
    fails += int(not hit)

print("\n== remaining 'pose estimate' tokens (should be none except 'pose estimation') ==")
for m in re.finditer(r"pose estimates?\b", body, flags=re.I):
    ctx = " ".join(body[max(0, m.start() - 70) : m.end() + 70].split())
    if re.match(r"pose estimation\b", body[m.start() :], flags=re.I):
        continue
    print("LEFTOVER:", ctx)
    fails += 1

print("\n== 'camera pose' contexts (body) ==")
for m in re.finditer(r"camera pose", body, flags=re.I):
    ctx = " ".join(body[max(0, m.start() - 80) : m.end() + 90].split())
    print("-", ctx)
    print()

print(f"\nRESIDUAL FAILS: {fails}")
