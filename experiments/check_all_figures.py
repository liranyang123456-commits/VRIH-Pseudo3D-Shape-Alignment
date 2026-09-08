from pathlib import Path
import re

paper = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex').read_text(encoding='utf-8')
root_dir = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials')

# Find all figures in paper
fig_blocks = []
# Match \begin{figure}...\end{figure} OR \begin{center}...\includegraphics...\captionof{figure}{...}\label{...}\end{center}
lines = paper.splitlines()

fig_labels = re.findall(r'\\label\{fig:([a-zA-Z0-9_]+)\}', paper)
print(f"Total figure labels found: {len(fig_labels)}")

for idx, lbl in enumerate(fig_labels, 1):
    # Find label context
    pos = paper.find(f"\\label{{fig:{lbl}}}")
    context = paper[max(0, pos-400):min(len(paper), pos+400)]
    # Extract image file
    img_match = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', context)
    img_file = img_match.group(1) if img_match else 'NO_IMAGE'
    # Extract caption
    cap_match = re.search(r'\\caption(?:of\{figure\})?\{([^}]+)\}', context)
    caption = cap_match.group(1) if cap_match else 'NO_CAPTION'
    
    # Check if image file exists
    img_path = root_dir / img_file
    exists = img_path.exists()
    
    # Check citations in paper
    cites = len(re.findall(r'\\ref\{fig:' + lbl + r'\}', paper))
    
    print(f"\nFigure {idx:2d}: [fig:{lbl}]")
    print(f"  File: {img_file} (exists: {exists})")
    print(f"  Citations in text: {cites}")
    print(f"  Caption: {caption[:100]}...")
