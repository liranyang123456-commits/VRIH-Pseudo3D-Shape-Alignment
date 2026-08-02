#!/usr/bin/env python3
"""Renumber numeric references into first-appearance (citation) order.

Works on VRIH_Paper.tex / VRIH_Paper_clean.tex / VRIH_Paper_markedup.tex.
Rewrites every \\cite{a,b,c} group (sorted by new number, preserving any
non-numeric keys) and reorders/renumbers the \\bibitem blocks.
Prints the old->new mapping for cross-file updates (e.g. response.tex).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"E:/elsarticle-template-TMI_Revised")
FILES = ["VRIH_Paper.tex", "VRIH_Paper_clean.tex", "VRIH_Paper_markedup.tex"]


def first_appearance_order(body: str) -> list[int]:
    order: list[int] = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", body):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit() and int(part) not in order:
                order.append(int(part))
    return order


def split_bibliography(src: str) -> tuple[str, dict[int, str]]:
    m = re.search(r"\\begin\{thebibliography\}\{[^}]*\}", src)
    if not m:
        raise SystemExit("thebibliography begin not found")
    e = src.find(r"\end{thebibliography}")
    if e < 0:
        raise SystemExit("thebibliography end not found")
    bib_src = src[m.end():e]
    items: dict[int, str] = {}
    matches = list(re.finditer(r"\\bibitem\{(\d+)\}", bib_src))
    for k, mm in enumerate(matches):
        start = mm.start()
        end = matches[k + 1].start() if k + 1 < len(matches) else len(bib_src)
        items[int(mm.group(1))] = bib_src[start:end].strip()
    return src, items


def process(path: Path) -> dict[int, int]:
    src = path.read_text(encoding="utf-8")
    bmark = src.find(r"\begin{thebibliography}")
    if bmark < 0:
        raise SystemExit(f"{path.name}: no bibliography")
    body, tail = src[:bmark], src[bmark:]

    order = first_appearance_order(body)
    _, items = split_bibliography(src)
    old_ids = sorted(items)
    missing = [i for i in old_ids if i not in order]
    if missing:
        raise SystemExit(f"{path.name}: bibitems never cited: {missing}")
    if len(order) != len(old_ids):
        raise SystemExit(f"{path.name}: count mismatch cites={len(order)} bib={len(old_ids)}")

    mapping = {old: new for new, old in enumerate(order, start=1)}

    def repl_cite(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        nums = sorted((mapping[int(p)] for p in parts if p.isdigit()))
        others = [p for p in parts if not p.isdigit()]
        return "\\cite{" + ",".join(str(n) for n in nums + others) + "}"

    body_new = re.sub(r"\\cite\{([^}]*)\}", repl_cite, body)

    bib_new_parts = []
    for new_id in range(1, len(order) + 1):
        old_id = order[new_id - 1]
        block = items[old_id]
        block = re.sub(r"^\\bibitem\{\d+\}", r"\\bibitem{%d}" % new_id, block, count=1)
        bib_new_parts.append(block)
    em = re.search(r"\\begin\{thebibliography\}\{[^}]*\}", tail)
    head = tail[: em.end()]
    eidx = tail.find(r"\end{thebibliography}")
    bib_new = head + "\n\n" + "\n\n".join(bib_new_parts) + "\n\n" + tail[eidx:]

    path.write_text(body_new + bib_new, encoding="utf-8")
    return mapping


def verify(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    bmark = src.find(r"\begin{thebibliography}")
    body = src[:bmark]
    errs = []
    order = first_appearance_order(body)
    if order != sorted(order):
        errs.append("citation order not ascending")
    if order != list(range(1, len(order) + 1)):
        errs.append(f"citation ids not contiguous: {order[:10]}...")
    _, items = split_bibliography(src)
    if sorted(items) != list(range(1, len(order) + 1)):
        errs.append("bibitem ids not contiguous after renumber")
    return errs


def main() -> None:
    mapping = process(ROOT / FILES[0])
    print("old->new mapping:")
    for old in sorted(mapping):
        print(f"  {old:2d} -> {mapping[old]:2d}")
    for f in FILES[1:]:
        m2 = process(ROOT / f)
        if m2 != mapping:
            print(f"WARN: {f} mapping differs from main file!")
    print("\nverification:")
    ok = True
    for f in FILES:
        errs = verify(ROOT / f)
        print(f"  {f}: {'OK' if not errs else errs}")
        ok = ok and not errs
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
