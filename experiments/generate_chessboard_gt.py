#!/usr/bin/env python3
"""Standalone checkerboard GT generation (Zhang calibration + solvePnP).

Writes chessboard_camera_poses_c2w_4x4.txt and chessboard_intrinsics.json in
the same format as the main pipeline, for board sizes the default run misses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-corners", type=str, required=True, help="e.g. 11x8")
    parser.add_argument("--square-size", type=float, required=True, help="mm")
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()

    cols, rows_n = (int(v) for v in args.inner_corners.split("x"))
    pattern = (cols, rows_n)
    objp = np.zeros((cols * rows_n, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows_n].T.reshape(-1, 2) * args.square_size

    paths = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.max_frames]

    detections: dict[int, np.ndarray] = {}
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    for index, path in enumerate(paths):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        image_size = (gray.shape[1], gray.shape[0])
        ok, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        detections[index] = corners
    print(f"detected {len(detections)}/{len(paths)} frames")
    if len(detections) < 10:
        raise SystemExit("not enough detections for calibration")

    object_points = [objp] * len(detections)
    image_points = list(detections.values())
    rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    print(f"calibrateCamera rms={rms:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Chessboard solvePnP camera poses (c2w), standalone GT generation",
        f"# board={args.inner_corners} square={args.square_size}mm calib_rms={rms:.4f}",
    ]
    for index in sorted(detections):
        ok, rvec, tvec = cv2.solvePnP(objp, detections[index], K, dist)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = tvec.ravel()
        c2w = np.linalg.inv(w2c)
        lines.append(f"frame_idx={index}")
        lines.extend(" ".join(f"{v:.8f}" for v in row) for row in c2w)
        lines.append("")
    gt_path = args.output_dir / "chessboard_camera_poses_c2w_4x4.txt"
    gt_path.write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "chessboard_intrinsics.json").write_text(
        json.dumps(
            {
                "K": K.tolist(),
                "dist": dist.ravel().tolist(),
                "image_size_wh": list(image_size),
                "board": args.inner_corners,
                "square_size_mm": args.square_size,
                "calib_rms": rms,
                "n_detected": len(detections),
                "source": "generate_chessboard_gt.py standalone",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {gt_path}")


if __name__ == "__main__":
    main()
