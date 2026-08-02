"""Cross-consistency checks for VRIH_Paper.tex: citation coverage, label/ref coverage."""
import re

SRC = r"E:\elsarticle-template-TMI_Revised\VRIH_Paper.tex"
src = open(SRC, encoding="utf-8").read()
body = src.split(r"\begin{thebibliography}")[0]

# --- 1. citation coverage -------------------------------------------------
cited = set()
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for part in m.group(1).split(","):
        part = part.strip()
        if part.isdigit():
            cited.add(int(part))
allrefs = set(range(1, 59))
print("uncited refs:", sorted(allrefs - cited))
print("cited but nonexistent:", sorted(cited - allrefs))

# --- 2. figure/table label-ref coverage -----------------------------------
labels = re.findall(r"\\label\{(fig:[^}]+|tab:[^}]+)\}", body)
print(f"\nlabels found: {len(labels)}")
for lb in labels:
    refs = len(re.findall(r"\\ref\{" + re.escape(lb) + r"\}", body))
    flag = "OK" if refs > 0 else "UNREFERENCED!"
    print(f"  {lb}: {refs} {flag}")

# --- 3. duplicate bibitems -------------------------------------------------
bibs = re.findall(r"\\bibitem\{(\d+)\}", src)
dup = [b for b in set(bibs) if bibs.count(b) > 1]
print("\nduplicate bibitems:", dup)

# --- 4. citation order (first appearance should be ascending) --------------
first_order = []
for m in re.finditer(r"\\cite\{([^}]*)\}", body):
    for part in m.group(1).split(","):
        part = part.strip()
        if part.isdigit() and int(part) not in first_order:
            first_order.append(int(part))
viol = [(a, b) for a, b in zip(first_order, first_order[1:]) if b < a]
print("first-appearance order violations (b<a):", viol)

# --- 5. suspicious wording --------------------------------------------------
patterns = {
    "reviewer/referee mention": r"(?i)\b(reviewer|referee)\b",
    "double space": r"(?<=[a-z,;])  +(?=[a-zA-Z])",
    "chessboard (spelled)": r"(?i)chessboard(?!_pose|_compare)",
    "TODO/XX leftover": r"(?i)\b(TODO|FIXME|xxx+)\b",
    "Section hardcode": r"Sections?~[0-9]",
    "unescaped underscore": r"(?<!\\)_",
}
print()
for name, pat in patterns.items():
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        if name == "unescaped underscore":
            line2 = re.sub(r"\\_", "", line)
            line2 = re.sub(r"\$[^$]*\$", "", line2)  # skip math
            if re.search(pat, line2):
                hits.append((i, line.strip()[:90]))
        elif re.search(pat, line):
            hits.append((i, line.strip()[:110]))
    print(f"{name}: {len(hits)}")
    for h in hits[:8]:
        print("   ", h)
