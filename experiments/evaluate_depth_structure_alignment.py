"""Evaluate whether image-gradient saliency agrees with depth discontinuities.

This experiment does not treat image gradients as metric depth.  It evaluates the
weaker, intended claim: after bilateral smoothing, high image-gradient responses
are spatially associated with depth-structure changes on a dataset with released
depth maps.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def process(image_path: Path, depth_path: Path) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if image is None or depth is None:
        raise RuntimeError(f"Cannot read {image_path} or {depth_path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.shape != image.shape:
        depth = cv2.resize(depth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth) & (depth > 0)

    gray = image.astype(np.float32) / 255.0
    smooth = cv2.bilateralFilter(gray, 7, 0.1, 3.0)
    image_gradient = np.hypot(
        cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3),
    )
    depth_smooth = cv2.bilateralFilter(depth, 7, 20.0, 3.0)
    depth_gradient = np.hypot(
        cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3),
    )
    image_values = image_gradient[valid]
    depth_values = depth_gradient[valid]
    top_image = image_values >= np.percentile(image_values, 90.0)
    top_depth = depth_values >= np.percentile(depth_values, 90.0)
    intersection = int(np.logical_and(top_image, top_depth).sum())
    union = int(np.logical_or(top_image, top_depth).sum())
    return {
        "image": str(image_path),
        "depth": str(depth_path),
        "n_valid": int(valid.sum()),
        "spearman_rho": pearson(rank_values(image_values), rank_values(depth_values)),
        "top10_overlap_precision": intersection / max(1, int(top_image.sum())),
        "top10_overlap_recall": intersection / max(1, int(top_depth.sum())),
        "top10_iou": intersection / max(1, union),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset_dir in sorted(args.scared_root.glob("dataset_*")):
        for keyframe_dir in sorted(dataset_dir.glob("keyframe_*")):
            image = keyframe_dir / "Left_Image.png"
            depth = keyframe_dir / "left_depth_map.tiff"
            if image.exists() and depth.exists():
                rows.append(process(image, depth))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        metric: float(np.nanmean([float(row[metric]) for row in rows]))
        for metric in ("spearman_rho", "top10_overlap_precision", "top10_overlap_recall", "top10_iou")
    }
    summary["n_keyframes"] = len(rows)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
