#!/usr/bin/env python3
"""Run response_to_txt.py logic against revision2_materials/response2.tex."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R2DIR = ROOT / "revision2_materials"

# Preprocess: adapt R2 macros to the converter's expectations.
tex = (R2DIR / "response2.tex").read_text(encoding="utf-8")
tex = tex.replace("\\rcomment{", "\\comment{")
tex = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", tex)
# Figures/tables cannot be rendered in plain text: replace the visual block
# with a pointer to the PDF version of the response letter.
tex = re.sub(
    r"% RESP-VISUAL-BEGIN.*?% RESP-VISUAL-END",
    "[Supporting figures and tables for this comment (threshold panels at "
    "median/p60/p68/mean/p90/p95, a 3D gradient surface with threshold cut "
    "planes, the full operating-point sweep table, and the companion-study "
    "multi-method tracking comparison) are provided in the PDF version of "
    "this response letter.]",
    tex,
    flags=re.S,
)
tmp = R2DIR / "_response2_pre.tex"
tmp.write_text(tex, encoding="utf-8")

src = (HERE / "response_to_txt.py").read_text(encoding="utf-8")
src = src.replace('SRC = ROOT / "response.tex"',
                  'SRC = ROOT / "revision2_materials/_response2_pre.tex"')
src = src.replace('OUT = ROOT / "response_plain.txt"',
                  'OUT = ROOT / "revision2_materials/response2_plain.txt"')
src = src.replace('lines_out.append("Manuscript No.: VRIH-D-26-00061")',
                  'lines_out.append("Manuscript No.: VRIH-D-26-00061R1 (Second-Round Revision)")')
# Strip LaTeX comment lines from the greeting too.
src = src.replace(
    'greeting = clean(greeting)',
    'greeting = "\\n".join(l for l in greeting.splitlines() if not l.lstrip().startswith("%"))\n'
    '    greeting = clean(greeting)',
)
exec(compile(src, "response_to_txt.py", "exec"))
tmp.unlink()
