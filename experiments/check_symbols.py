from pathlib import Path
import re

paper = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex').read_text(encoding='utf-8')

# Focus on Section 3: Methods
sec3_start = paper.find(r'\section{Methods}')
sec3_end = paper.find(r'\section{Experiments and validation}')
sec3 = paper[sec3_start:sec3_end]

print(f"Section 3 length: {len(sec3)} chars")

# Extract all math expressions: $...$ and \[...\] and \begin{equation}...\end{equation}
inline_math = re.findall(r'\$([^\$]+)\$', sec3)
block_math = re.findall(r'\\begin\{equation\}(.*?)\\end\{equation\}', sec3, re.S)

print(f"Inline math segments: {len(inline_math)}")
print(f"Block equations: {len(block_math)}")

# Key symbols in Section 3 to verify definition
symbols_to_check = [
    # Sec 3.1
    ('I(i,j)', 'input intensity / pixel'),
    ('I_f(i,j)', 'bilateral filtered intensity'),
    ('w_s', 'spatial weight'),
    ('w_r', 'range weight'),
    ('\sigma_s', 'spatial Gaussian parameter'),
    ('\sigma_r', 'range Gaussian parameter'),
    ('G(x)', 'Gaussian kernel'),
    ("G'(x)", 'Gaussian derivative kernel'),
    ('G_x', 'horizontal gradient component'),
    ('G_y', 'vertical gradient component'),
    ('|\nabla I|', 'gradient magnitude'),
    ('\tilde{G}', 'normalized gradient magnitude'),
    ('G_{\min}', 'minimum gradient'),
    ('G_{\max}', 'maximum gradient'),
    ('t_p', 'percentile threshold'),
    ('t_{\text{main}}', 'main threshold 68th percentile'),
    ('t_{90}', 'high-sensitivity threshold 90th percentile'),
    ('F(x)', 'cumulative histogram'),
    ('s', 'mesh grid spacing / stride'),
    ('\mathcal{M}(t)', 'triangular mesh at time t'),
    ('\mathcal{V}(t)', 'vertex set'),
    ('\mathcal{F}', 'triangle connectivity'),
    ('\mathbf{v}_i(t)', 'vertex coordinates'),
    ('N_v', 'number of vertices'),
    ('I_{\mathrm{norm}}', 'normalized intensity'),
    ('z_{\mathrm{scale}}', 'pseudo-height scale'),
    (r'\beta', 'scale parameter (0.15)'),
    ('Z(x,y)', 'pseudo-height value'),
    ('M(x,y)', 'mask'),
    ('M\'(x,y)', 'resampled mask'),
    ('\chi(x,y)', 'indicator function'),
    ('Z_{\mathrm{masked}}', 'masked height field'),
    ('T_1', 'triangle 1'),
    ('T_2', 'triangle 2'),
    ('d_x', 'horizontal spacing'),
    ('d_y', 'vertical spacing'),
    ('\mathbf{c}', 'vertex color'),

    # Sec 3.2
    ('PE_l', 'positional encoding at level l'),
    ('\tilde{X}_l', 'fused feature at level l'),
    ('E_{l-1}', 'feature map from previous stage'),
    ('W_Q', 'query projection matrix'),
    ('W_K', 'key projection matrix'),
    ('W_V', 'value projection matrix'),
    ('Q', 'query embedding'),
    ('K', 'key embedding'),
    ('V', 'value embedding'),
    ('d_k', 'attention head dimensionality'),
    ('B', 'relative positional bias'),
    ('z', 'latent code vector'),
    ('W_z', 'latent projection weight'),
    ('b_z', 'latent projection bias'),
    ('\text{GAP}', 'global average pooling'),
    ("F'_L", 'deepest encoder feature map'),
    ('\hat{F}_L', 'reconstructed feature map'),
    ('\text{MLPE}', 'MLP expansion'),
    ('\text{ConvTE}', 'transposed convolution expansion'),
    ('t', 'normalized diffusion time'),
    ('t_{\mathrm{emb}}', 'time embedding'),
    ('h_1', 'first intermediate feature in denoising block'),
    ('h_2', 'modulated feature in denoising block'),
    ('\mathrm{Scale}', 'learnable affine scale parameter'),
    ('\mathrm{Shift}', 'learnable affine shift parameter'),
    ('\hat{I}', 'reconstructed image output'),
    ('\mathcal{L}_{\mathrm{total}}', 'total loss'),
    ('\mathcal{L}_{\mathrm{RGB}}', 'RGB reconstruction loss'),
    ('\mathcal{L}_c', 'contour loss'),
    ('\mathcal{L}_h', 'heatmap loss'),
    ('\mathcal{L}_s', 'Laplacian smoothness loss'),
    ('\mathcal{L}_{\mathrm{reg}}', 'regularization loss'),
    (r'\lambda_c', 'contour loss weight'),
    (r'\lambda_h', 'heatmap loss weight'),
    (r'\lambda_s', 'smoothness loss weight'),
    (r'\lambda_r', 'regularization weight'),
    (r'\lambda_{\mathrm{aux}}', 'auxiliary branch common weight'),
    ('C_b', 'ground-truth contour map'),
    ('\hat{C}_b', 'predicted contour map'),
    ('H_b', 'ground-truth heatmap'),
    ('\hat{H}_b', 'predicted heatmap'),
    ('v_i', 'mesh vertex i'),
    ('\mathcal{N}(i)', '1-ring neighborhood of vertex i'),

    # Sec 3.3
    ('\mathcal{T}_{\mathrm{fg}}', 'foreground extraction operator'),
    ('\mathcal{T}_{\mathrm{mesh}}', 'mesh construction operator'),
    ('\mathcal{T}_{\mathrm{rt}}', 'rigid transform estimation operator'),
    ('\mathcal{T}_{\mathrm{gate}}', 'gating/recovery operator'),
    ('\hat g_t^{\mathrm{8b}}', 'truncated 8-bit gradient response'),
    ('M_t', 'binary foreground mask at frame t'),
    ('\mathrm{bbox}_t', 'bounding box / ROI at frame t'),
    ('T_{t-1\leftarrow t}', 'pairwise shape-alignment transform'),
    ('T_{0\leftarrow t}', 'cumulative shape-alignment transform'),
    ('R', 'rotation in SO(3)'),
    ('\mathbf{t}', 'translation vector in R^3'),
    ('\Pi', 'projection operator to (u,v)'),
    ('e_i', 'in-plane residual'),
    ('\mathcal{P}_t', 'point cloud at frame t'),
    ('\mathcal{Q}_{t-1}', 'point cloud at frame t-1'),
    ('r_i(T)', 'residual for ICP'),
    (r'\rho', 'robust penalty function (e.g. Huber)'),
    ('\mathbf{f}', 'dense optical flow'),
    ('(u_i,v_i)', 'sampled correspondence coordinate'),
    ("(u_i',v_i')", 'warped correspondence coordinate'),
    ('\mathbf{x}_i', 'current pseudo-3D coordinate'),
    ('\mathbf{y}_i', 'previous pseudo-3D coordinate'),
    ('\eta', 'inlier ratio'),
    ('|\mathcal{I}|', 'number of inliers'),
    ('N', 'total correspondence count'),
    ('\epsilon_{\mathrm{med}}', 'median inlier residual'),
    ('d_t', 'center displacement'),
    ('\mathbf{c}_t', 'foreground centroid'),
    ('m_t', 'scale variation ratio'),
    ('r_t', 'foreground support ratio'),
    ('\mathcal{V}_t', 'mesh vertices at time t'),
    ('\mathcal{V}_t^+', 'foreground vertex set'),
    ('\epsilon_z', 'pseudo-height threshold'),
    ('\mathrm{jump}_t', 'temporal jump indicator'),
    (r'\tau_d', 'center displacement threshold'),
    (r'\tau_m', 'scale variation threshold'),
    ('\mathrm{Bad}_{\mathrm{icp}}', 'ICP bad indicator'),
    ('\mathrm{Bad}_{\mathrm{flow}}', 'flow bad indicator'),
    (r'\tau_f', 'ICP fitness threshold'),
    (r'\tau_r', 'ICP RMSE threshold'),
    (r'\tau_\eta', 'inlier ratio threshold'),
    (r'\tau_\epsilon', 'median residual threshold'),
    ('\mathcal{S}_t', 'candidate transform set'),
    ('J(T)', 'lexicographic objective function'),
]

print("=== CHECKING SYMBOL DEFINITIONS IN SECTION 3 ===")
undefined_count = 0
for sym, desc in symbols_to_check:
    # check if sym is in sec3
    found_sym = sym in sec3
    if not found_sym:
        # try without backslash
        clean_sym = sym.replace('\\', '').replace('{', '').replace('}', '')
        found_sym = clean_sym in sec3
    if not found_sym:
        print(f"MISSING SYMBOL: {sym:25s} | Expected: {desc}")
        undefined_count += 1

print(f"\nTotal symbols checked: {len(symbols_to_check)}, Missing: {undefined_count}")
