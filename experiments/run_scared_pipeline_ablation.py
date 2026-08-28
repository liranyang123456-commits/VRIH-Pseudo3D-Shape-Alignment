#!/usr/bin/env python3
"""Pipeline-component ablation on the six SCARED sequences (80 frames each).

Variants of the flow-guided pseudo-3D shape-alignment pipeline:
  base        : dense Farneback flow + plain Kabsch (the reported configuration)
  ransac      : + RANSAC outlier rejection before Kabsch re-fitting
  ransac_gate : + estimator-aware quality gating (freeze update when the inlier
                ratio is too low or the median inlier residual too high)

All variants use the gradient height field and identical sampling. Output:
per-sequence pose metrics + aggregate CSV for the ablation table.
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
SEQUENCES = [
    "scared_d1_k1",
    "scared_d1_k2",
    "scared_d2_k1",
    "scared_d2_k2",
    "scared_d3_k1",
    "scared_d3_k2",
]

RANSAC_ITERS = 300
RANSAC_THRESH_PX = 8.0
GATE_MIN_INLIER_RATIO = 0.15
GATE_MAX_MEDIAN_RESIDUAL = 12.0


def height_field(gray: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    threshold = np.percentile(magnitude, 90.0)
    magnitude = np.where(magnitude > threshold, magnitude, 0.0)
    maximum = float(magnitude.max())
    return (magnitude / maximum * 96.0) if maximum > 1e-8 else magnitude


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


def kabsch_ransac(source: np.ndarray, target: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(source)
    best_inliers = None
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(n, size=min(6, n), replace=False)
        transform = kabsch(source[idx], target[idx])
        residual = np.linalg.norm((transform[:3, :3] @ source.T).T + transform[:3, 3] - target, axis=1)
        inliers = residual < RANSAC_THRESH_PX
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    transform = kabsch(source[best_inliers], target[best_inliers])
    residual = np.linalg.norm((transform[:3, :3] @ source[best_inliers].T).T + transform[:3, 3] - target[best_inliers], axis=1)
    return transform, float(best_inliers.sum() / n), float(np.median(residual))


def run_sequence(seq: str, variant: str) -> None:
    data_dir = ROOT / "data" / seq
    paths = sorted((data_dir / "images").glob("*.png"))[:80]
    out_dir = RESULTS / f"pipeline_{variant}_{seq}"
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / f"{variant}_cumulative_4x4.txt"

    def load(p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        scale = 320 / img.shape[1]
        return cv2.resize(img, (320, round(img.shape[0] * scale)))

    previous = load(paths[0])
    previous_height = height_field(previous)
    grid_y, grid_x = np.mgrid[4:previous.shape[0]:8, 4:previous.shape[1]:8]
    xy_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    poses = [np.eye(4)]
    diag = []
    for index, path in enumerate(paths[1:], start=1):
        current = load(path)
        current_height = height_field(current)
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        flow_at = flow[grid_y.ravel(), grid_x.ravel()]
        xy_cur = xy_grid + flow_at
        valid = (
            (xy_cur[:, 0] >= 1) & (xy_cur[:, 0] < current.shape[1] - 2)
            & (xy_cur[:, 1] >= 1) & (xy_cur[:, 1] < current.shape[0] - 2)
        )
        src_xy = xy_cur[valid]
        tgt_xy = xy_grid[valid]
        src_h = cv2.remap(current_height, src_xy[:, 0].reshape(-1, 1), src_xy[:, 1].reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1)
        tgt_h = previous_height[tgt_xy[:, 1].astype(int), tgt_xy[:, 0].astype(int)]
        source = np.column_stack([src_xy, src_h])
        target = np.column_stack([tgt_xy, tgt_h])

        if variant == "base":
            transform = kabsch(source, target)
            inlier_ratio, med_res = 1.0, float("nan")
        else:
            transform, inlier_ratio, med_res = kabsch_ransac(source, target)
            if variant == "ransac_gate" and (
                inlier_ratio < GATE_MIN_INLIER_RATIO or med_res > GATE_MAX_MEDIAN_RESIDUAL
            ):
                transform = np.eye(4)  # reject-and-freeze
        poses.append(poses[-1] @ transform)
        diag.append({"frame_idx": index, "inlier_ratio": inlier_ratio, "median_residual": med_res})
        previous, previous_height = current, current_height

    with traj_path.open("w", encoding="utf-8") as fh:
        fh.write("# Pseudo-3D shape-alignment trajectory (non-metric)\n")
        for i, pose in enumerate(poses):
            fh.write(f"frame_idx={i}\n" + "\n".join(" ".join(f"{v:.10f}" for v in row) for row in pose) + "\n")
    (out_dir / "gating_diagnostics.json").write_text(json.dumps(diag))
    subprocess.run(
        [sys.executable, str(ROOT / "evaluate_pose.py"), "--predicted", str(traj_path),
         "--ground-truth", str(data_dir / "scared_camera_pose_as_released_4x4.txt"),
         "--output", str(out_dir / "pose_metrics.json")],
        check=True, capture_output=True,
    )
    print(f"[done] {variant}/{seq}")


def main() -> None:
    for variant in ("base", "ransac", "ransac_gate"):
        for seq in SEQUENCES:
            run_sequence(seq, variant)

    rows = []
    for variant in ("base", "ransac", "ransac_gate"):
        for seq in SEQUENCES:
            payload = json.loads((RESULTS / f"pipeline_{variant}_{seq}" / "pose_metrics.json").read_text())
            rows.append({
                "variant": variant, "sequence": seq,
                "ate_rmse": payload["ate_after_sim3_gt_units"]["rmse"],
                "rot_mean_deg": payload["relative_rotation_error_deg"]["mean"],
                "rot_median_deg": payload["relative_rotation_error_deg"]["median"],
                "tdir_mean_deg": payload["relative_translation_direction_error_deg"]["mean"],
            })
    out = RESULTS / "scared_pipeline_ablation.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import statistics

    print("\nvariant      ATE(rmse)      rot mean+-std        tdir")
    for variant in ("base", "ransac", "ransac_gate"):
        sel = [r for r in rows if r["variant"] == variant]
        ate = [r["ate_rmse"] for r in sel]
        rot = [r["rot_mean_deg"] for r in sel]
        tdir = [r["tdir_mean_deg"] for r in sel]
        print(f"{variant:12s} {statistics.mean(ate):6.3f}        {statistics.mean(rot):.3f} +- {statistics.stdev(rot):.3f}   {statistics.mean(tdir):.1f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
