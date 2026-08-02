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
    for i in range(doc.page_count):
        page = doc[i]
        pix = page.get_pixmap(dpi=60)
        pix.save(str(OUT / f"page_{i + 1:02d}.png"))
        # report text overflow markers: content beyond margins is hard to detect
        # automatically, so we save pages for manual inspection
    print(f"rendered {doc.page_count} pages to {OUT}")


if __name__ == "__main__":
    main()
