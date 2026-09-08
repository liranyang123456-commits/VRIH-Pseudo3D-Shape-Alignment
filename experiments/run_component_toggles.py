#!/usr/bin/env python3
"""Direct component toggles the reviewer asked for separately (R2.5).

Three controlled experiments on the six SCARED sequences (80 frames each),
all sharing the released dense-flow + Kabsch back-end:

  * flow_off   : the dense Farneback flow is replaced by a zero field, so the
                 warped grid equals the source grid and Kabsch returns a
                 near-identity transform. This isolates the contribution of
                 the dense-flow correspondence field itself.
  * icp_sweep  : a minimal point-to-plane-free (point-to-point) ICP refinement
                 is applied as a post-Kabsch step on every pair, sweeping the
                 maximum correspondence distance and iteration count, to show
                 the reported accuracy is insensitive to the ICP configuration.
  * mesh_note  : (no run) the triangular-mesh stride s is the surface-
                 representation resolution; the pose pipeline samples
                 correspondences on a grid over a continuously interpolated
                 height field, so s does not enter the reported numbers.

Output: component_toggles.csv + console aggregate.
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
    "scared_d1_k1", "scared_d1_k2", "scared_d2_k1",
    "scared_d2_k2", "scared_d3_k1", "scared_d3_k2",
]
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


def icp_point_to_point(source: np.ndarray, target: np.ndarray,
                       init: np.ndarray, max_dist: float, iters: int) -> np.ndarray:
    """Minimal point-to-point ICP refinement of `init` mapping source->target."""
    transform = init.copy()
    for _ in range(iters):
        warped = (transform[:3, :3] @ source.T).T + transform[:3, 3]
        # nearest neighbour in target
        diff = warped[:, None, :] - target[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        nn = np.argmin(dist, axis=1)
        nn_dist = dist[np.arange(len(source)), nn]
        mask = nn_dist < max_dist
        if mask.sum() < 6:
            break
        delta = kabsch(source[mask], target[nn][mask])
        transform = delta @ transform
        if np.linalg.norm(delta[:3, 3]) < 1e-6:
            break
    return transform


def run_sequence(seq: str, mode: str, icp_max_dist: float = 10.0, icp_iters: int = 80) -> dict:
    data_dir = ROOT / "data" / seq
    paths = sorted((data_dir / "images").glob("*.png"))[:80]
    tag = f"{mode}_d{icp_max_dist}_i{icp_iters}" if mode == "icp" else mode
    out_dir = RESULTS / f"toggle_{tag}_{seq}"
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
        if mode == "flow_off":
            flow = np.zeros((*current.shape, 2), dtype=np.float32)
        else:
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
        transform = kabsch(source, target)
        if mode == "icp":
            transform = icp_point_to_point(source, target, transform, icp_max_dist, icp_iters)
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
    configs = [("flow_off", None, None)]
    configs += [("icp", d, i) for d in (5.0, 10.0, 20.0) for i in (20, 80)]
    for mode, d, it in configs:
        for seq in SEQUENCES:
            payload = run_sequence(seq, mode, d or 10.0, it or 80)
            rows.append({
                "mode": mode,
                "icp_max_dist": d if d else "",
                "icp_iters": it if it else "",
                "sequence": seq,
                "rot_mean_deg": payload["relative_rotation_error_deg"]["mean"],
                "ate_rmse": payload["ate_after_sim3_gt_units"]["rmse"],
            })
            print(f"[ok] {mode} d={d} i={it} {seq} rot={payload['relative_rotation_error_deg']['mean']:.3f}")

    out = RESULTS / "component_toggles.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nmode            dist  iters   rot mean (6 seqs)")
    for mode, d, it in configs:
        sel = [r for r in rows if r["mode"] == mode and r["icp_max_dist"] == (d or "") and r["icp_iters"] == (it or "")]
        rot = sum(r["rot_mean_deg"] for r in sel) / len(sel)
        print(f"{mode:12s} {str(d):5s} {str(it):5s}  {rot:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
