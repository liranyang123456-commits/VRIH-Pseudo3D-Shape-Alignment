#!/usr/bin/env python3
"""Run the released dense-flow + Kabsch pipeline on the extended SCARED
keyframes (datasets 5-7) and report relative rotation error vs. released pose.

This is the Plan-2 addition for R2.9: additional released SCARED sequences with
pose ground truth (more subjects, more deformation), evaluated with the same
Sim(3) protocol as Table 4. Output: scared_extended.csv + per-sequence metrics.
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
SEQUENCES = ["scared_d5_k2", "scared_d6_k4", "scared_d7_k4"]
RESIZE_WIDTH = 320


def load(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    scale = RESIZE_WIDTH / img.shape[1]
    return cv2.resize(img, (RESIZE_WIDTH, round(img.shape[0] * scale)))


def height_field(gray: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 90.0)
    mag = np.where(mag > thr, mag, 0.0)
    mx = float(mag.max())
    return (mag / mx * 96.0) if mx > 1e-8 else mag


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = ct - r @ cs
    return t


def run_sequence(seq: str) -> dict:
    data_dir = ROOT / "data" / seq
    paths = sorted((data_dir / "images").glob("*.png"))[:80]
    out_dir = RESULTS / f"extended_{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "traj_4x4.txt"

    previous = load(paths[0])
    previous_height = height_field(previous)
    grid_y, grid_x = np.mgrid[4:previous.shape[0]:8, 4:previous.shape[1]:8]
    xy_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    poses = [np.eye(4)]
    for path in paths[1:]:
        current = load(path)
        current_height = height_field(current)
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
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
        poses.append(poses[-1] @ kabsch(source, target))
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
    for seq in SEQUENCES:
        payload = run_sequence(seq)
        meta = json.loads((ROOT / "data" / seq / "metadata.json").read_text())
        rows.append({
            "sequence": seq,
            "source": meta.get("source", ""),
            "gt_rot_mean_deg": meta.get("rot_mean_deg", ""),
            "rot_mean_deg": payload["relative_rotation_error_deg"]["mean"],
            "rot_median_deg": payload["relative_rotation_error_deg"]["median"],
            "ate_rmse": payload["ate_after_sim3_gt_units"]["rmse"],
        })
        print(f"[done] {seq} rot={payload['relative_rotation_error_deg']['mean']:.3f} "
              f"(GT motion {meta.get('rot_mean_deg', 0):.3f})")
    out = RESULTS / "scared_extended.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rot = [r["rot_mean_deg"] for r in rows]
    print(f"\nextended SCARED (d5-7) mean rot: {np.mean(rot):.3f} +- {np.std(rot, ddof=1):.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
