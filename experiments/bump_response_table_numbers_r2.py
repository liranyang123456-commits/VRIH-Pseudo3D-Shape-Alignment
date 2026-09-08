#!/usr/bin/env python3
"""Shift hardcoded Table N citations after inserting the extended-SCARED table.

New table order in the manuscript:
  1 data_inventory, 2 hcr, 3 chessboard_pose, 4 scared_pose, 5 scared_extended,
  6 stereo_pseudo3d, 7 pipeline_ablation, 8 sensitivity, 9 conditions,
  10 clinical_cases.

The response previously used: 5 stereo, 6 pipeline, 7 sensitivity, 8
conditions, 9 clinical. Those must become 6,7,8,9,10. Tables 1-4 are unchanged.

The 2.9 rewrite already references the new Table~5 (extended) and Table~10
(clinical); those are protected by sentinel replacement before the bump.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "revision2_materials" / "response2.tex"


def find_macro_blocks(src: str, macro: str) -> list[tuple[int, int]]:
    key = "\\" + macro + "{"
    blocks = []
    i = 0
    while True:
        start = src.find(key, i)
        if start < 0:
            break
        depth = 0
        k = start + len(key) - 1
        while k < len(src):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append((start, k + 1))
                    i = k + 1
                    break
            k += 1
        else:
            break
    return blocks


def apply_outside(src: str, blocks: list[tuple[int, int]], fn) -> str:
    out = []
    last = 0
    for a, b in blocks:
        out.append(fn(src[last:a]))
        out.append(src[a:b])
        last = b
    out.append(fn(src[last:]))
    return "".join(out)


def bump(s: str) -> str:
    # old 5..9 -> new 6..10 (descending to avoid collisions)
    for n in range(9, 4, -1):
        s = s.replace(f"Table~{n}", f"Table~TMP{n + 1}")
        s = s.replace(f"Tables~{n}", f"Tables~TMP{n + 1}")
    return s.replace("Table~TMP", "Table~").replace("Tables~TMP", "Tables~")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    # Protect the already-correct new references in the 2.9 rewrite.
    text = text.replace("reported in the new Table~5 (extended SCARED)",
                        "reported in the new Table~EXT5 (extended SCARED)")
    text = text.replace("Table~10 reports internal", "Table~CLIN10 reports internal")
    text = text.replace("reproduces Table~10.", "reproduces Table~CLIN10.")

    # Do not touch the reviewer-comment boxes.
    comments = find_macro_blocks(text, "rcomment")
    text = apply_outside(text, comments, bump)

    # Restore protected references.
    text = text.replace("Table~EXT5", "Table~5")
    text = text.replace("Table~CLIN10", "Table~10")

    # Update the table-count sentence and the enumerated list.
    text = text.replace("contains nine tables", "contains ten tables")
    text = text.replace(
        "Table~4 SCARED alignment; Table~5 stereo\nvalidation; Table~6 geometric back-end decomposition; Table~7 hyperparameter\nsensitivity; Table~8 condition-stratified analysis; Table~9 no-GT\nclinical-case windows)",
        "Table~4 SCARED alignment; Table~5 extended\nSCARED; Table~6 stereo validation; Table~7 geometric back-end\ndecomposition; Table~8 hyperparameter sensitivity; Table~9\ncondition-stratified analysis; Table~10 no-GT clinical-case windows)",
    )
    text = text.replace(
        "Table~4 SCARED alignment; Table~5 stereo validation; Table~6 geometric back-end\ndecomposition; Table~7 hyperparameter sensitivity; Table~8\ncondition-stratified analysis with a hard-quartile block; Table~9 no-GT clinical-case\nwindows)",
        "Table~4 SCARED alignment; Table~5 extended SCARED; Table~6 stereo\nvalidation; Table~7 geometric back-end decomposition; Table~8\nhyperparameter sensitivity; Table~9 condition-stratified analysis with a\nhard-quartile block; Table~10 no-GT clinical-case windows)",
    )

    PATH.write_text(text, encoding="utf-8")
    print("bumped table numbers in", PATH)


if __name__ == "__main__":
    main()
