#!/usr/bin/env python3
"""Plan-3 for R2.9: evaluate the abrupt-motion window of SCARED d6_k1.

d6/keyframe_1 contains a genuine abrupt-motion event (a 6.55-degree relative
rotation between consecutive frames). We extract the 80-frame window centred on
the highest-motion region and run the released dense-flow + Kabsch pipeline,
reporting relative rotation error vs. released pose. This is the
hardest-with-GT case available on disk.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
HUB = ROOT.parent / "vrih_experiment_hub" / "links" / "scared"
OUT = ROOT / "data"
RESULTS = ROOT / "results"
KF = HUB / "dataset_6" / "keyframe_1"
WINDOW = 80
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


def kabsch(source, target):
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


def main() -> None:
    frame_data = KF / "data" / "frame_data.tar.gz"
    rgb = KF / "data" / "rgb.mp4"
    # load all poses, find hardest 80-frame window
    with tarfile.open(frame_data, "r:gz") as a:
        members = sorted([m for m in a.getmembers() if m.name.endswith(".json")], key=lambda x: x.name)
        poses = [np.asarray(json.load(a.extractfile(m))["camera-pose"], dtype=float) for m in members]
    rots = []
    for A, B in zip(poses[:-1], poses[1:]):
        R = A[:3, :3].T @ B[:3, :3]
        c = np.clip((np.trace(R) - 1) / 2, -1, 1)
        rots.append(np.degrees(np.arccos(c)))
    rots = np.array(rots)
    best = max(range(len(rots) - WINDOW + 2), key=lambda s: rots[s:s + WINDOW - 1].mean())
    print(f"d6_k1 hardest window: frames {best}..{best+WINDOW-1}, "
          f"mean GT rot {rots[best:best+WINDOW-1].mean():.3f}, max {rots[best:best+WINDOW-1].max():.2f}")

    # extract that window's frames + poses
    out_dir = OUT / "scared_d6_k1_abrupt"
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(rgb))
    all_frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        all_frames.append(fr)
    cap.release()
    window_frames = all_frames[best:best + WINDOW]
    window_poses = poses[best:best + WINDOW]
    for i, fr in enumerate(window_frames):
        cv2.imwrite(str(img_dir / f"frame_{i:06d}.png"), fr)
    pose_path = out_dir / "scared_camera_pose_as_released_4x4.txt"
    lines = []
    for i, pose in enumerate(window_poses):
        lines.append(f"frame_idx={i}")
        lines.extend(" ".join(f"{v:.10f}" for v in row) for row in pose)
        lines.append("")
    pose_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({
        "dataset": "SCARED", "source": "dataset_6/keyframe_1",
        "window_start_frame": int(best), "window_frames": WINDOW,
        "gt_rot_mean_deg": float(rots[best:best+WINDOW-1].mean()),
        "gt_rot_max_deg": float(rots[best:best+WINDOW-1].max()),
    }, indent=2), encoding="utf-8")

    # run pipeline
    paths = sorted(img_dir.glob("*.png"))
    previous = load(paths[0])
    previous_height = height_field(previous)
    grid_y, grid_x = np.mgrid[4:previous.shape[0]:8, 4:previous.shape[1]:8]
    xy_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    est = [np.eye(4)]
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
        est.append(est[-1] @ kabsch(np.column_stack([src_xy, src_h]), np.column_stack([tgt_xy, tgt_h])))
        previous, previous_height = current, current_height

    out = RESULTS / "extended_scared_d6_k1_abrupt"
    out.mkdir(parents=True, exist_ok=True)
    traj = out / "traj_4x4.txt"
    with traj.open("w", encoding="utf-8") as fh:
        for i, pose in enumerate(est):
            fh.write(f"frame_idx={i}\n" + "\n".join(" ".join(f"{v:.10f}" for v in row) for row in pose) + "\n")
    subprocess.run(
        [sys.executable, str(ROOT / "evaluate_pose.py"), "--predicted", str(traj),
         "--ground-truth", str(pose_path), "--output", str(out / "pose_metrics.json")],
        check=True, capture_output=True,
    )
    m = json.loads((out / "pose_metrics.json").read_text())
    print(f"d6_k1 abrupt window: rot mean={m['relative_rotation_error_deg']['mean']:.3f} "
          f"median={m['relative_rotation_error_deg']['median']:.3f} "
          f"(GT mean {rots[best:best+WINDOW-1].mean():.3f} max {rots[best:best+WINDOW-1].max():.2f})")
    (RESULTS / "scared_abrupt.json").write_text(json.dumps({
        "sequence": "scared_d6_k1_abrupt",
        "window_start_frame": int(best),
        "gt_rot_mean_deg": float(rots[best:best+WINDOW-1].mean()),
        "gt_rot_max_deg": float(rots[best:best+WINDOW-1].max()),
        "rot_mean_deg": m["relative_rotation_error_deg"]["mean"],
        "rot_median_deg": m["relative_rotation_error_deg"]["median"],
        "ate_rmse": m["ate_after_sim3_gt_units"]["rmse"],
    }, indent=2), encoding="utf-8")
    print("wrote", RESULTS / "scared_abrupt.json")


if __name__ == "__main__":
    main()
