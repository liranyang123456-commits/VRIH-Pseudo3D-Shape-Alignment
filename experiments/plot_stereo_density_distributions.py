"""Visualize per-keyframe density--quality distributions on SCARED."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    "SIFT",
    "SIFT-dense",
    "SIFT+gradient-mask",
    "Pseudo3D-dense",
    "Pseudo3D-balanced",
    "Ours-selective",
)

LABELS = (
    "SIFT",
    "SIFT dense",
    "Gradient\nmask",
    "Pseudo3D\ndense",
    "Pseudo3D\nbalanced",
    "Ours\nselective",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    metrics = (
        ("matches", "Matches per stereo pair"),
        ("gt_match_precision", "GT geometric match precision"),
        ("depth_absrel", "Triangulated depth AbsRel"),
        ("depth_rmse", "Triangulated depth RMSE"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ORDER)))
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        values = [
            np.asarray([float(row[metric]) for row in rows if row["method"] == method])
            for method in ORDER
        ]
        box = axis.boxplot(values, patch_artist=True, showmeans=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.set_title(label)
        axis.set_xticks(range(1, len(ORDER) + 1), LABELS, rotation=0)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Per-keyframe density--quality distribution on 15 calibrated SCARED stereo keyframes",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
