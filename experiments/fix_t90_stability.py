"""Recompute per-sequence t90 CV for the six self-acquired sequences separately
(the pooled corpus bar mixed six scenes) and regenerate the statistics figure."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASE = ROOT.parent / "vrih_experiment_hub" / "links" / "self_acquired_endoscopy"

per_seq = {}
for i in range(1, 7):
    d = BASE / f"extracted_frames{i}" / "extracted_frames"
    vals = []
    for p in sorted(d.glob("*.jpg")):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        g = gray.astype(np.float32) / 255.0
        smooth = cv2.bilateralFilter(g, 7, 0.1, 3.0)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        vals.append(float(np.percentile(np.hypot(gx, gy), 90.0)))
    v = np.array(vals)
    per_seq[f"self_seq{i}"] = {"mean": float(v.mean()), "cv": float(v.std() / v.mean()), "n": len(v)}
    print(f"self_seq{i}: CV={v.std()/v.mean():.3f} n={len(v)}", flush=True)

# merge into the statistics json
path = RESULTS / "gradient_statistics.json"
d = json.loads(path.read_text())
d["summary"]["t90_per_sequence"] = {k: v for k, v in d["summary"]["t90_per_sequence"].items() if k.startswith("scared")}
d["summary"]["t90_per_sequence"].update(per_seq)
path.write_text(json.dumps(d, indent=1))
print("updated gradient_statistics.json")
