#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

ROOT = Path(r"E:/elsarticle-template-TMI_Revised/experiments/results")
SEQS = [
    "scared_d1_k1",
    "scared_d1_k2",
    "scared_d2_k1",
    "scared_d2_k2",
    "scared_d3_k1",
    "scared_d3_k2",
]
DIRS = {
    "GradientHeight": "{seq}_gradient",
    "IntensityHeight": "{seq}_intensity",
    "ConstantHeight": "{seq}_constant",
    "Reloc3r": "reloc3r_{seq}",
    "SIFT": "sift_{seq}",
    "AKAZE": "akaze_{seq}",
    "ORB": "orb_{seq}",
    "DetectorFreeSfM": "detectorfreesfm_{seq}",
}
PAPER = {
    "GradientHeight": (1.781, 0.302, 118.732, 0.979, 0.272),
    "IntensityHeight": (1.689, 0.333, 123.371, 0.979, 0.308),
    "ConstantHeight": (1.771, 0.296, 119.791, 0.979, 0.258),
    "Reloc3r": (1.726, 0.525, 108.237, 0.772, 0.420),
    "SIFT": (2.344, 9.146, 94.698, 0.924, 0.371),
    "AKAZE": (1.637, 6.511, 92.144, 0.937, 0.376),
    "ORB": (1.805, 10.309, 95.535, 0.909, 0.397),
    "DetectorFreeSfM": (0.742, 0.508, 67.396, 0.848, 0.297),
}

p = ROOT / "scared_d1_k1_gradient" / "pose_metrics.json"
d = json.loads(p.read_text(encoding="utf-8"))
print("sample keys", list(d.keys()), "has series", "series" in d)

print("\n=== SCARED ===")
for method, tmpl in DIRS.items():
    all_e = []
    ates, rots, tdirs, succs, nposes = [], [], [], [], []
    for seq in SEQS:
        mp = ROOT / tmpl.format(seq=seq) / "pose_metrics.json"
        if not mp.exists():
            print(" MISSING", mp)
            continue
        payload = json.loads(mp.read_text(encoding="utf-8"))
        nposes.append(payload.get("n_poses"))
        if method == "DetectorFreeSfM" and payload.get("n_poses", 0) < 80:
            continue
        ates.append(payload["ate_after_sim3_gt_units"]["rmse"])
        rots.append(payload["relative_rotation_error_deg"]["mean"])
        tdirs.append(payload["relative_translation_direction_error_deg"]["mean"])
        if "series" in payload:
            err = np.asarray(payload["series"]["rotation_errors_deg"], dtype=float)
            all_e.extend(err.tolist())
            succs.append(float(np.mean(err < 1.0)))
        else:
            succs.append(float("nan"))
    arr = np.asarray(all_e, dtype=float)
    med = float(np.median(arr)) if arr.size else float("nan")
    ate_std = float(np.std(ates, ddof=1)) if len(ates) > 1 else 0.0
    rot_std = float(np.std(rots, ddof=1)) if len(rots) > 1 else 0.0
    print(
        f"{method:18s} npose={nposes} npairs={arr.size} "
        f"med={med:.4f} ATE {np.mean(ates):.3f}+/-{ate_std:.3f} "
        f"rot {np.mean(rots):.3f}+/-{rot_std:.3f} "
        f"tdir {np.mean(tdirs):.3f} succ {np.nanmean(succs):.3f}"
    )
    if method in PAPER:
        print("   paper", PAPER[method])
        print(
            "   match ATE/rot/tdir/succ/med",
            abs(np.mean(ates) - PAPER[method][0]) < 0.002,
            abs(np.mean(rots) - PAPER[method][1]) < 0.002,
            abs(np.mean(tdirs) - PAPER[method][2]) < 0.02,
            abs(np.nanmean(succs) - PAPER[method][3]) < 0.002,
            abs(med - PAPER[method][4]) < 0.002 if arr.size else False,
        )
