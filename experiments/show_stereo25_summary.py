"""Print the 25-keyframe stereo summary in a compact table."""
import csv
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).parent / "results" / "scared_stereo_density_quality_full25_summary.csv")))
print(f"{'method':22s} {'n':>3s} {'matches':>8s} {'prec':>7s} {'rot':>6s} {'tdir':>6s} {'abs_med':>8s} {'rmse_med':>9s} {'abs_mean':>9s}")
for r in rows:
    print(
        f"{r['method']:22s} {r['n_keyframes']:>3s} {float(r['mean_matches']):8.1f} "
        f"{float(r['mean_gt_match_precision']):7.4f} {float(r['mean_rotation_error_deg']):6.3f} "
        f"{float(r['mean_translation_direction_error_deg']):6.2f} {float(r['median_depth_absrel']):8.4f} "
        f"{float(r['median_depth_rmse']):9.3f} {float(r['mean_depth_absrel']):9.4f}"
    )
