#!/usr/bin/env python3
"""Hyperparameter sensitivity sweep for the pseudo-3D pipeline on SCARED.

Sweeps (one factor at a time around the default configuration) on two
sequences (d1_k1, d3_k2):
  * gradient percentile threshold: 85 / 90 / 95
  * correspondence grid stride (mesh sampling analog): 4 / 8 / 16
  * pseudo-height scale: 48 / 96 / 192
  * RANSAC inlier threshold: 4 / 8 / 16 px (RANSAC variant)
  * Farneback window size: 15 / 21 / 27

Output: scared_sensitivity.csv + console aggregate.
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
SEQUENCES = ["scared_d1_k1", "scared_d3_k2"]

DEFAULTS = dict(percentile=90.0, grid_step=8, height_scale=96.0, ransac_thresh=8.0, winsize=21)
SWEEPS = {
    "percentile": [85.0, 90.0, 95.0],
    "grid_step": [4, 8, 16],
    "height_scale": [48.0, 96.0, 192.0],
    "ransac_thresh": [4.0, 8.0, 16.0],
    "winsize": [15, 21, 27],
}


def height_field(gray: np.ndarray, percentile: float, height_scale: float) -> np.ndarray:
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    threshold = np.percentile(magnitude, percentile)
    magnitude = np.where(magnitude > threshold, magnitude, 0.0)
    maximum = float(magnitude.max())
    return (magnitude / maximum * height_scale) if maximum > 1e-8 else magnitude


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = ct - rotation @ cs
    return transform


def kabsch_ransac(source: np.ndarray, target: np.ndarray, thresh: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(source)
    best = None
    for _ in range(300):
        idx = rng.choice(n, size=min(6, n), replace=False)
        transform = kabsch(source[idx], target[idx])
        residual = np.linalg.norm((transform[:3, :3] @ source.T).T + transform[:3, 3] - target, axis=1)
        inliers = residual < thresh
        if best is None or inliers.sum() > best[1].sum():
            best = (transform, inliers)
    transform, inliers = best
    return kabsch(source[inliers], target[inliers])


def run_config(seq: str, cfg: dict, tag: str) -> dict:
    data_dir = ROOT / "data" / seq
    paths = sorted((data_dir / "images").glob("*.png"))[:80]
    out_dir = RESULTS / f"sensitivity_{tag}_{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "traj_4x4.txt"

    def load(p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        scale = 320 / img.shape[1]
        return cv2.resize(img, (320, round(img.shape[0] * scale)))

    previous = load(paths[0])
    previous_height = height_field(previous, cfg["percentile"], cfg["height_scale"])
    step = cfg["grid_step"]
    grid_y, grid_x = np.mgrid[step // 2:previous.shape[0]:step, step // 2:previous.shape[1]:step]
    xy_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    poses = [np.eye(4)]
    for path in paths[1:]:
        current = load(path)
        current_height = height_field(current, cfg["percentile"], cfg["height_scale"])
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, cfg["winsize"], 3, 5, 1.2, 0)
        flow_at = flow[grid_y.ravel(), grid_x.ravel()]
        xy_cur = xy_grid + flow_at
        valid = (
            (xy_cur[:, 0] >= 1) & (xy_cur[:, 0] < current.shape[1] - 2)
            & (xy_cur[:, 1] >= 1) & (xy_cur[:, 1] < current.shape[0] - 2)
        )
        src_xy, tgt_xy = xy_cur[valid], xy_grid[valid]
        src_h = cv2.remap(current_height, src_xy[:, 0].reshape(-1, 1), src_xy[:, 1].reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1)
        tgt_h = previous_height[tgt_xy[:, 1].astype(int), tgt_xy[:, 0].astype(int)]
        source = np.column_stack([src_xy, src_h])
        target = np.column_stack([tgt_xy, tgt_h])
        if tag.startswith("ransac_thresh"):
            transform = kabsch_ransac(source, target, cfg["ransac_thresh"])
        else:
            transform = kabsch(source, target)
        poses.append(poses[-1] @ transform)
        previous, previous_height = current, current_height

    with traj_path.open("w", encoding="utf-8") as fh:
        for i, pose in enumerate(poses):
            fh.write(f"frame_idx={i}\n" + "\n".join(" ".join(f"{v:.10f}" for v in row) for row in pose) + "\n")
    metrics_path = out_dir / "pose_metrics.json"
    subprocess.run(
        [sys.executable, str(ROOT / "evaluate_pose.py"), "--predicted", str(traj_path),
         "--ground-truth", str(data_dir / "scared_camera_pose_as_released_4x4.txt"),
         "--output", str(metrics_path)],
        check=True, capture_output=True,
    )
    return json.loads(metrics_path.read_text())


def main() -> None:
    rows = []
    for factor, values in SWEEPS.items():
        for value in values:
            cfg = dict(DEFAULTS)
            cfg[factor] = value
            tag = f"{factor}_{value}"
            for seq in SEQUENCES:
                payload = run_config(seq, cfg, tag)
                rows.append({
                    "factor": factor, "value": value, "sequence": seq,
                    "ate_rmse": payload["ate_after_sim3_gt_units"]["rmse"],
                    "rot_mean_deg": payload["relative_rotation_error_deg"]["mean"],
                    "tdir_mean_deg": payload["relative_translation_direction_error_deg"]["mean"],
                })
                print(f"[ok] {tag} {seq} rot={payload['relative_rotation_error_deg']['mean']:.3f}")

    out = RESULTS / "scared_sensitivity.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nfactor          value   rot mean (2 seqs)   ATE mean")
    for factor, values in SWEEPS.items():
        for value in values:
            sel = [r for r in rows if r["factor"] == factor and r["value"] == value]
            rot = sum(r["rot_mean_deg"] for r in sel) / len(sel)
            ate = sum(r["ate_rmse"] for r in sel) / len(sel)
            marker = " (default)" if DEFAULTS[factor] == value else ""
            print(f"{factor:15s} {str(value):6s} {rot:8.3f}          {ate:6.3f}{marker}")
    print("wrote", out)


if __name__ == "__main__":
    main()
