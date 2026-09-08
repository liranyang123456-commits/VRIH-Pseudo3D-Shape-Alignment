#!/usr/bin/env python3
"""Shift hardcoded Table N citations after inserting Table 1 (data inventory)."""
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
    for n in range(7, 0, -1):
        s = s.replace(f"Table~{n}", f"Table~TMP{n + 1}")
        s = s.replace(f"Tables~{n}", f"Tables~TMP{n + 1}")
    return s.replace("Table~TMP", "Table~").replace("Tables~TMP", "Tables~")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    text = text.replace("new Table~1", "NEWTABLEONE")
    text = apply_outside(text, find_macro_blocks(text, "rcomment"), bump)
    text = text.replace("NEWTABLEONE", "new Table~1")
    text = text.replace(
        "contains seven tables (Table~2 HCR contour evaluation;",
        "contains eight tables (Table~1 data inventory; Table~2 HCR contour evaluation;",
    )
    text = text.replace(
        "contains seven tables, cited in order (Table~2 HCR contour evaluation;",
        "contains eight tables, cited in order (Table~1 data inventory; Table~2 HCR contour evaluation;",
    )
    text = text.replace(
        "Disclosed the frame-level 70/15/15 split, its non-sequence-independence,",
        "Disclosed the frame-level 70/15/15 split (six sequences / $3{,}002$ frames; new Table~1), its non-sequence-independence,",
    )
    PATH.write_text(text, encoding="utf-8")
    print("updated", PATH)


if __name__ == "__main__":
    main()
