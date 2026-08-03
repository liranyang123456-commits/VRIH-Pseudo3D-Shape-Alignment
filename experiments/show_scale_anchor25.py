"""Print the 25-keyframe scale-anchor summary."""
import json
from pathlib import Path

d = json.load(open(Path(__file__).parent / "results" / "scared_scale_anchor_summary.json"))
print("n:", d["n_keyframes"])
for k in ["spearman_h_vs_invdepth", "spearman_h_vs_depthgrad", "top10_hit_rate", "affine_r2", "affine_median_absrel"]:
    v = d[k]
    print(f"{k}: mean={v['mean']:.4f} median={v['median']:.4f}")
