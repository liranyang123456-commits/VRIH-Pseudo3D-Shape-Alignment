#!/usr/bin/env python3
"""Reproducibly evaluate contour density and highlight contamination.

The highlight mask is defined per image as pixels at or above the 95th percentile
of grayscale intensity.  NPC is the number of contour pixels, HPC is the number
of contour pixels falling in the highlight mask, and HCR = HPC / NPC.

This script deliberately reports only classical methods implemented here plus the
paper's gradient-percentile contour, avoiding unverified learned-model claims.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def normalized_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def percentile_binary(values: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    threshold = float(np.percentile(values, percentile))
    return (values > threshold).astype(np.uint8)


def contours(gray: np.ndarray) -> dict[str, np.ndarray]:
    filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=0.1, sigmaSpace=3.0)
    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(grad_x, grad_y)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.hypot(sobel_x, sobel_y)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    dog = np.abs(
        cv2.GaussianBlur(gray, (0, 0), 1.0) - cv2.GaussianBlur(gray, (0, 0), 2.0)
    )

    ours = percentile_binary(gradient, 90.0)
    ours = cv2.morphologyEx(
        ours, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=2
    )
    return {
        "Ours-gradient-p90": ours,
        "Canny": (cv2.Canny((gray * 255).astype(np.uint8), 50, 150) > 0).astype(np.uint8),
        "Sobel-p90": percentile_binary(sobel, 90.0),
        "Laplacian-p90": percentile_binary(laplacian, 90.0),
        "DoG-p90": percentile_binary(dog, 90.0),
    }


def image_paths(directory: Path, limit: int) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {directory}")
    return paths[:limit] if limit > 0 else paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        action="append",
        nargs=3,
        metavar=("NAME", "DIRECTORY", "MAX_IMAGES"),
        required=True,
        help="Repeatable sequence specification.",
    )
    parser.add_argument("--per-frame-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for name, directory_text, maximum in args.sequence:
        directory = Path(directory_text)
        for path in image_paths(directory, int(maximum)):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            gray = normalized_gray(image)
            highlight = gray >= np.percentile(gray, 95.0)
            for method, contour in contours(gray).items():
                npc = int(contour.sum())
                hpc = int(np.logical_and(contour.astype(bool), highlight).sum())
                rows.append(
                    {
                        "sequence": name,
                        "image": path.name,
                        "method": method,
                        "npc": npc,
                        "hpc": hpc,
                        "hcr": hpc / npc if npc else float("nan"),
                    }
                )

    args.per_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_frame_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: list[dict[str, object]] = []
    for sequence in sorted({str(row["sequence"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows}):
            subset = [
                row for row in rows if row["sequence"] == sequence and row["method"] == method
            ]
            for metric in ("npc", "hpc", "hcr"):
                values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                summary.append(
                    {
                        "sequence": sequence,
                        "method": method,
                        "metric": metric.upper(),
                        "n_images": len(values),
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values, ddof=1)),
                        "median": float(np.nanmedian(values)),
                    }
                )
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
