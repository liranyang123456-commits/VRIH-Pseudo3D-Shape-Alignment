from pathlib import Path
import re

paper = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex').read_text(encoding='utf-8')

print("==================================================")
print("1. ALL 10 TABLES IN PAPER: CAPTIONS & DATA")
print("==================================================")
for m in re.finditer(r'\\begin\{table\}(.*?)\\end\{table\}', paper, re.S):
    t_text = m.group(1)
    label = re.search(r'\\label\{([^}]+)\}', t_text)
    caption = re.search(r'\\caption\{([^}]+)\}', t_text)
    lbl_str = label.group(1) if label else 'NO_LABEL'
    cap_str = caption.group(1) if caption else 'NO_CAPTION'
    print(f"\n--- Table [{lbl_str}] ---")
    print(f"Caption: {cap_str[:120]}...")
    # print headers and first 2 rows
    lines = [l.strip() for l in t_text.splitlines() if '&' in l and not l.strip().startswith('%')]
    for l in lines[:3]:
        print("   Row:", l[:100])

print("\n==================================================")
print("2. ALL 14 FIGURES IN PAPER: CAPTIONS & REFERENCES")
print("==================================================")
for m in re.finditer(r'(\\begin\{figure\}|\\captionof\{figure\}|\\includegraphics)(.*?)(?=\\end\{figure\}|\\captionof\{figure\}|\Z)', paper, re.S):
    pass

# Cleaner figure extraction:
fig_matches = re.finditer(r'(\\caption(?:of\{figure\})?\{([^}]+)\}\s*\\label\{([^}]+)\})|(\\label\{([^}]+)\}\s*\\caption(?:of\{figure\})?\{([^}]+)\})', paper)
for idx, m in enumerate(fig_matches, 1):
    txt = m.group(0)
    lbl = re.search(r'\\label\{([^}]+)\}', txt).group(1)
    cap = re.search(r'\\caption(?:of\{figure\})?\{([^}]+)\}', txt).group(1)
    print(f"\nFigure {idx:2d} [fig:{lbl}]:")
    print(f"   Caption: {cap[:110]}...")
    # find where this figure is cited in paper
    cites = [l.strip() for l in paper.splitlines() if f'\\ref{{{lbl}}}' in l]
    print(f"   Citations count: {len(cites)}")
    for c in cites[:2]:
        print(f"     Cite: {c[:90]}...")
