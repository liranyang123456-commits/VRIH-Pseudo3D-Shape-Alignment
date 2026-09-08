from pathlib import Path
import re

paper = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex').read_text(encoding='utf-8')
root_dir = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials')

fig_labels = re.findall(r'\\label\{fig:([a-zA-Z0-9_]+)\}', paper)

for idx, lbl in enumerate(fig_labels, 1):
    pos = paper.find(f"\\label{{fig:{lbl}}}")
    # find the enclosing environment: either begin{figure}...end{figure} or begin{center}...end{center}
    env_start = max(paper.rfind(r'\begin{figure}', 0, pos), paper.rfind(r'\begin{center}', 0, pos))
    env_end_fig = paper.find(r'\end{figure}', pos)
    env_end_ctr = paper.find(r'\end{center}', pos)
    
    ends = [e for e in [env_end_fig, env_end_ctr] if e > 0]
    env_end = min(ends) if ends else pos + 500
    
    env_text = paper[env_start:env_end+15]
    
    img = re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', env_text)
    cap = re.search(r'\\caption(?:of\{figure\})?\{([^}]+)\}', env_text)
    
    img_file = img.group(1) if img else 'NONE'
    caption = cap.group(1) if cap else 'NONE'
    exists = (root_dir / img_file).exists() if img_file != 'NONE' else False
    
    cites = len(re.findall(r'\\ref\{fig:' + lbl + r'\}', paper))
    
    print(f"Fig {idx:2d} | Label: fig:{lbl:28s} | File: {img_file:35s} | Exists: {str(exists):5s} | Cites: {cites}")
    print(f"       Cap: {caption[:90]}...")
