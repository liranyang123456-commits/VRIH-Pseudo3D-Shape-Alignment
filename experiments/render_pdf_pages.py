"""Render selected PDF pages to PNG for visual layout inspection."""
import sys

import fitz  # PyMuPDF

PDF = sys.argv[1] if len(sys.argv) > 1 else "VRIH_Paper.pdf"
PAGES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [23, 24, 25]
OUT = sys.argv[3] if len(sys.argv) > 3 else "experiments/results/layout_check"

import pathlib

pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
doc = fitz.open(PDF)
print("pages:", len(doc))
for p in PAGES:
    idx = p - 1
    if 0 <= idx < len(doc):
        pix = doc[idx].get_pixmap(dpi=100)
        out = f"{OUT}/{pathlib.Path(PDF).stem}_page_{p}.png"
        pix.save(out)
        print("saved", out)
