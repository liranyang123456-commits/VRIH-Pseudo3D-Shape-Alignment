"""Detect near-blank pages in the compiled PDFs (little text and no drawings/images)."""
import sys

import fitz

for pdf in sys.argv[1:] or ["VRIH_Paper.pdf"]:
    doc = fitz.open(pdf)
    print(f"=== {pdf} ({len(doc)} pages) ===")
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        images = page.get_images()
        drawings = page.get_drawings()
        words = len(text.split())
        if words < 40 and not images and len(drawings) < 3:
            print(f"  page {i + 1}: NEAR-BLANK (words={words}, images={len(images)}, drawings={len(drawings)})")
        elif words < 15:
            print(f"  page {i + 1}: sparse (words={words}, images={len(images)}, drawings={len(drawings)})")
    # also report where each figure/table starts, for layout context
    print("  page word counts:", [len(p.get_text().split()) for p in doc])
