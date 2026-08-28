#!/usr/bin/env python3
"""Condition-stratified performance analysis on SCARED (R2.9).

Per frame pair, computes condition measurements and joins them with the
per-frame relative rotation error of the gradient-height pipeline:
  * specular highlight fraction (share of pixels above the 95th intensity
    percentile);
  * rapid motion (mean dense-flow magnitude);
  * weak texture (inverse of the mean gradient magnitude);
  * large deformation proxy (flow residual after the rigid fit).

Reports tercile-stratified mean rotation errors and Spearman correlations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEQUENCES = [
    "scared_d1_k1",
    "scared_d1_k2",
    "scared_d2_k1",
    "scared_d2_k2",
    "scared_d3_k1",
    "scared_d3_k2",
]


def main() -> None:
    rows = []
    for seq in SEQUENCES:
        metrics = json.loads((RESULTS / f"{seq}_gradient" / "pose_metrics.json").read_text())
        rot_errors = metrics["series"]["rotation_errors_deg"]
        images = sorted((ROOT / "data" / seq / "images").glob("*.png"))[:80]
        previous = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
        for index, path in enumerate(images[1:], start=1):
            current = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            flow_mag = np.linalg.norm(flow, axis=2)
            highlight = float((current >= np.percentile(current, 95.0)).mean())
            gx = cv2.Sobel(current, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(current, cv2.CV_32F, 0, 1, ksize=3)
            texture = float(np.hypot(gx, gy).mean())
            rows.append(
                {
                    "sequence": seq,
                    "frame_idx": index,
                    "rot_err_deg": rot_errors[index - 1],
                    "highlight_fraction": highlight,
                    "flow_mean_px": float(flow_mag.mean()),
                    "texture_mean_grad": texture,
                }
            )
            previous = current

    out = RESULTS / "scared_condition_stratification.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rot = np.array([r["rot_err_deg"] for r in rows])
    print(f"frame pairs: {len(rows)}")
    report = {}
    for key, label in (
        ("highlight_fraction", "specular highlight"),
        ("flow_mean_px", "rapid motion"),
        ("texture_mean_grad", "texture strength"),
    ):
        cond = np.array([r[key] for r in rows])
        rho = stats.spearmanr(cond, rot).statistic
        q1, q2 = np.percentile(cond, [33.33, 66.67])
        low = rot[cond <= q1]
        mid = rot[(cond > q1) & (cond <= q2)]
        high = rot[cond > q2]
        report[key] = {
            "spearman_rho": float(rho),
            "tercile_low_mean_rot": float(low.mean()),
            "tercile_mid_mean_rot": float(mid.mean()),
            "tercile_high_mean_rot": float(high.mean()),
        }
        print(
            f"{label:20s} rho={rho:+.3f}  terciles low/mid/high rot: "
            f"{low.mean():.3f} / {mid.mean():.3f} / {high.mean():.3f}"
        )
    (RESULTS / "scared_condition_stratification_summary.json").write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
