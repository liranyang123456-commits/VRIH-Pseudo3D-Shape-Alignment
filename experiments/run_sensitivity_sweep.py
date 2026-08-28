#!/usr/bin/env python3
"""Hyperparameter sensitivity sweeps for the pseudo-3D alignment pipeline (R2.5).

Sweeps on two SCARED sequences (gradient height variant), one parameter at a
time around the reported default configuration (percentile 90, grid step 8,
height scale 96, RANSAC threshold 8 px):

* percentile threshold t_p: 85 / 90 / 95
* correspondence grid step (mesh stride proxy): 4 / 8 / 16
* pseudo-height scale: 48 / 96 / 192
* RANSAC inlier threshold: 4 / 8 / 16 px

Outputs sensitivity_sweep.csv + sensitivity_sweep.json with per-config
relative rotation error and Sim(3) ATE.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from run_scared_ae_e2e import kabsch_ransac  # noqa: E402

SEQUENCES = ["scared_d1_k1", "scared_d2_k1"]
DEFAULTS = dict(percentile=90.0, grid_step=8, height_scale=96.0, ransac_thresh=8.0)
SWEEPS = {
    "percentile": [85.0, 90.0, 95.0],
    "grid_step": [4, 8, 16],
    "height_scale": [48.0, 96.0, 192.0],
    "ransac_thresh": [4.0, 8.0, 16.0],
}


def height_field_param(gray: np.ndarray, percentile: float, height_scale: float) -> np.ndarray:
    gray = gray.astype(np.float32) / 255.0
    filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=0.1, sigmaSpace=3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    thr = np.percentile(magnitude, percentile)
    magnitude = np.where(magnitude > thr, magnitude, 0.0)
    maximum = float(magnitude.max())
    return (magnitude / maximum * height_scale) if maximum > 1e-8 else magnitude


def run_sequence(seq: str, percentile: float, grid_step: int, height_scale: float, ransac_thresh: float) -> dict:
    data_dir = ROOT / "data" / seq
    images = sorted((data_dir / "images").glob("*.png"))[:80]
    gt = data_dir / "scared_camera_pose_as_released_4x4.txt"
    transforms = []
    for i in range(len(images) - 1):
        prev = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
        curr = cv2.imread(str(images[i + 1]), cv2.IMREAD_GRAYSCALE)
        if prev.shape[1] > 320:
            scale = 320 / prev.shape[1]
            prev_s = cv2.resize(prev, None, fx=scale, fy=scale)
            curr_s = cv2.resize(curr, None, fx=scale, fy=scale)
        else:
            prev_s, curr_s = prev, curr
        h_prev = height_field_param(prev_s, percentile, height_scale)
        h_curr = height_field_param(curr_s, percentile, height_scale)
        flow = cv2.calcOpticalFlowFarneback(prev_s, curr_s, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        ys, xs = np.mgrid[0 : prev_s.shape[0] : grid_step, 0 : prev_s.shape[1] : grid_step]
        xs, ys = xs.ravel(), ys.ravel()
        flow_at = flow[ys, xs]
        xc, yc = xs + flow_at[:, 0], ys + flow_at[:, 1]
        valid = (xc >= 1) & (xc < curr_s.shape[1] - 2) & (yc >= 1) & (yc < curr_s.shape[0] - 2)
        xs, ys, xc, yc = xs[valid], ys[valid], xc[valid], yc[valid]
        xi = np.clip(xc.astype(int), 0, h_curr.shape[1] - 1)
        yi = np.clip(yc.astype(int), 0, h_curr.shape[0] - 1)
        src = np.column_stack([xc, yc, h_curr[yi, xi]])
        dst = np.column_stack([xs.astype(float), ys.astype(float), h_prev[ys, xs]])
        R, t, _ = kabsch_ransac(src, dst, iterations=300, thresh=ransac_thresh, seed=0)
        T = np.eye(4)
        if R is not None:
            T[:3, :3] = R
            T[:3, 3] = t
        transforms.append(T)

    out_dir = RESULTS / f"sens_{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"p{percentile:g}_g{grid_step}_h{height_scale:g}_r{ransac_thresh:g}"
    traj = out_dir / f"traj_{tag}.txt"
    with traj.open("w", encoding="utf-8") as fh:
        cumulative = np.eye(4)
        fh.write("frame_idx=0\n" + "\n".join(" ".join(f"{v:.9f}" for v in row) for row in cumulative) + "\n")
        for k, T in enumerate(transforms, start=1):
            cumulative = cumulative @ T
            fh.write(f"frame_idx={k}\n" + "\n".join(" ".join(f"{v:.9f}" for v in row) for row in cumulative) + "\n")
    out_json = out_dir / f"metrics_{tag}.json"
    subprocess.run(
        [sys.executable, str(ROOT / "evaluate_pose.py"), "--predicted", str(traj),
         "--ground-truth", str(gt), "--output", str(out_json)],
        check=True, capture_output=True,
    )
    return json.loads(out_json.read_text())


def main() -> None:
    rows = []
    for param, values in SWEEPS.items():
        for value in values:
            cfg = dict(DEFAULTS)
            cfg[param] = value
            rots, ates = [], []
            for seq in SEQUENCES:
                metrics = run_sequence(seq, **cfg)
                rots.append(metrics["relative_rotation_error_deg"]["mean"])
                ates.append(metrics["ate_after_sim3_gt_units"]["rmse"])
            row = {
                "parameter": param,
                "value": value,
                "is_default": value == DEFAULTS[param],
                "rel_rot_mean_deg": float(np.mean(rots)),
                "rel_rot_std_deg": float(np.std(rots, ddof=1)),
                "ate_rmse_mean": float(np.mean(ates)),
            }
            rows.append(row)
            print(f"{param}={value}: rot={row['rel_rot_mean_deg']:.4f}±{row['rel_rot_std_deg']:.4f}  ate={row['ate_rmse_mean']:.3f}", flush=True)

    with (RESULTS / "sensitivity_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "sensitivity_sweep.json").write_text(json.dumps(rows, indent=1))
    print("written sensitivity_sweep.csv/json")


if __name__ == "__main__":
    main()
