#!/usr/bin/env python3
"""Run reproducible pseudo-3D relative-motion ablations on image sequences.

Variants differ only in the auxiliary height field:
  gradient: bilateral-filtered Sobel magnitude, truncated at percentile 90;
  intensity: normalized grayscale intensity;
  constant: zero height (the 2D-only control).

Dense Farneback flow supplies correspondences and a Kabsch solve estimates the
rigid transform from current to previous (u, v, h) points.  The output is an
explicit non-metric shape-alignment trajectory, not a metric camera pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def height_field(gray: np.ndarray, variant: str, height_scale: float) -> np.ndarray:
    gray = gray.astype(np.float32) / 255.0
    if variant == "constant":
        return np.zeros_like(gray, dtype=np.float32)
    if variant == "intensity":
        return gray * height_scale
    filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=0.1, sigmaSpace=3.0)
    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    threshold = np.percentile(magnitude, 90.0)
    magnitude = np.where(magnitude > threshold, magnitude, 0.0)
    maximum = float(magnitude.max())
    return (magnitude / maximum * height_scale) if maximum > 1e-8 else magnitude


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return T mapping source Nx3 points to target Nx3 points."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def write_poses(path: Path, poses: list[np.ndarray]) -> None:
    lines = [
        "# Pseudo-3D shape-alignment trajectory in (u, v, h)",
        "# Non-metric; do not interpret as direct physical camera poses.",
    ]
    for index, pose in enumerate(poses):
        lines.append(f"frame_idx={index}")
        lines.extend(" ".join(f"{value:.10f}" for value in row) for row in pose)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("gradient", "intensity", "constant"), required=True)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--resize-width", type=int, default=320)
    parser.add_argument("--grid-step", type=int, default=8)
    parser.add_argument("--height-scale", type=float, default=96.0)
    args = parser.parse_args()

    paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.max_frames]
    if len(paths) < 2:
        raise ValueError("Need at least two images")

    def load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot load {path}")
        scale = args.resize_width / image.shape[1]
        return cv2.resize(image, (args.resize_width, round(image.shape[0] * scale)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous = load(paths[0])
    previous_height = height_field(previous, args.variant, args.height_scale)
    poses = [np.eye(4, dtype=np.float64)]
    diagnostics: list[dict[str, object]] = []
    grid_y, grid_x = np.mgrid[
        args.grid_step // 2 : previous.shape[0] : args.grid_step,
        args.grid_step // 2 : previous.shape[1] : args.grid_step,
    ]
    xy_previous = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)

    for index, path in enumerate(paths[1:], start=1):
        current = load(path)
        if current.shape != previous.shape:
            raise RuntimeError("All input frames must share a common resolution")
        current_height = height_field(current, args.variant, args.height_scale)
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        flow_at_grid = flow[grid_y.ravel(), grid_x.ravel()]
        xy_current = xy_previous + flow_at_grid
        valid = (
            (xy_current[:, 0] >= 1)
            & (xy_current[:, 0] < current.shape[1] - 2)
            & (xy_current[:, 1] >= 1)
            & (xy_current[:, 1] < current.shape[0] - 2)
        )
        source_xy = xy_current[valid]
        target_xy = xy_previous[valid]
        source_h = cv2.remap(
            current_height,
            source_xy[:, 0].reshape(-1, 1),
            source_xy[:, 1].reshape(-1, 1),
            cv2.INTER_LINEAR,
        ).reshape(-1)
        target_h = previous_height[target_xy[:, 1].astype(int), target_xy[:, 0].astype(int)]
        source = np.column_stack([source_xy, source_h])
        target = np.column_stack([target_xy, target_h])
        transform = kabsch(source, target)
        poses.append(poses[-1] @ transform)
        aligned = (transform[:3, :3] @ source.T).T + transform[:3, 3]
        residual = np.linalg.norm(aligned - target, axis=1)
        diagnostics.append(
            {
                "frame_idx": index,
                "image": path.name,
                "n_correspondences": int(valid.sum()),
                "alignment_rmse_uvh": float(np.sqrt(np.mean(residual**2))),
                "mean_flow_px": float(np.linalg.norm(flow_at_grid[valid], axis=1).mean()),
            }
        )
        previous, previous_height = current, current_height

    write_poses(args.output_dir / f"{args.variant}_cumulative_4x4.txt", poses)
    (args.output_dir / f"{args.variant}_manifest.json").write_text(
        json.dumps(
            {
                "method": "pseudo3d_shape_alignment",
                "variant": args.variant,
                "coordinate_system": "uvh_heightfield",
                "non_metric_warning": (
                    "The trajectory is a pseudo-3D shape-alignment proxy, not a "
                    "direct physical camera pose."
                ),
                "n_frames": len(paths),
                "input_images": [path.name for path in paths],
                "config": {
                    "resize_width": args.resize_width,
                    "grid_step": args.grid_step,
                    "height_scale": args.height_scale,
                    "gradient_percentile": 90 if args.variant == "gradient" else None,
                },
                "per_pair": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
