from pathlib import Path
import re

paper = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex').read_text(encoding='utf-8')
sec3_start = paper.find(r'\section{Methods}')
sec3_end = paper.find(r'\section{Experiments and validation}')
sec3 = paper[sec3_start:sec3_end]

symbols_to_check = [
    (r'|\nabla I|', 'gradient magnitude'),
    (r'\tilde{G}', 'normalized gradient magnitude'),
    (r't_{\text{main}}', 'main threshold 68th percentile'),
    (r'\tilde{X}_l', 'fused feature at level l'),
    (r'\text{GAP}', 'global average pooling'),
    (r'\text{MLPE}', 'MLP expansion'),
    (r'\text{ConvTE}', 'transposed convolution expansion'),
    (r'\mathcal{L}_\mathrm{total}', 'total loss'),
    (r'\mathcal{L}_\mathrm{RGB}', 'RGB reconstruction loss'),
    (r'\mathcal{L}_\mathrm{c}', 'contour loss'),
    (r'\mathcal{L}_\mathrm{h}', 'heatmap loss'),
    (r'\mathcal{L}_\mathrm{s}', 'Laplacian smoothness loss'),
    (r'\mathcal{L}_\mathrm{reg}', 'regularization loss'),
]

for sym, desc in symbols_to_check:
    print(f"{sym:30s} in Sec 3: {sym in sec3}")
