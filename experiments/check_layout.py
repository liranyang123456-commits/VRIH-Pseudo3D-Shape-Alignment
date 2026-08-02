#!/usr/bin/env python3
"""Render the revised PDF to images for visual layout inspection."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

PDF = Path(r"E:\elsarticle-template-TMI_Revised\revision_submission_materials\VRIH_Paper.pdf")
OUT = Path(r"E:\elsarticle-template-TMI_Revised\experiments\results\layout_check")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if fitz is None:
        print("PyMuPDF not available; falling back to text extraction only")
        return
    doc = fitz.open(PDF)
    print("pages:", doc.page_count)
    pages = range(doc.page_count) if len(sys.argv) < 2 else [int(a) - 1 for a in sys.argv[1:]]
    for i in pages:
        doc[i].get_pixmap(dpi=60).save(str(OUT / f"page_{i + 1:02d}.png"))
    print(f"rendered pages {[i + 1 for i in pages]}")


if __name__ == "__main__":
    main()
