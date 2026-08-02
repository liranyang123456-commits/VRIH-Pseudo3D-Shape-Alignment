#!/usr/bin/env python3
"""Visualize the scale-anchor feasibility study."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "scared_scale_anchor.csv"
OUTPUT = ROOT / "results" / "fig_scale_anchor.png"


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    labels = [r["keyframe"].replace("dataset_", "d").replace("keyframe_", "k") for r in rows]
    x = np.arange(len(rows))

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    axes[0].bar(x - 0.2, [float(r["spearman_h_vs_invdepth"]) for r in rows], 0.4,
                label="h vs $1/\\mathrm{depth}$")
    axes[0].bar(x + 0.2, [float(r["spearman_h_vs_depthgrad"]) for r in rows], 0.4,
                label="h vs $|\\nabla\\log\\mathrm{depth}|$")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].set_title("Correlation with metric geometry")
    axes[0].axhline(0.35, color="gray", linestyle="--", linewidth=0.8)
    axes[0].legend(fontsize=8)

    axes[1].bar(x, [float(r["top10_hit_rate"]) for r in rows], color="steelblue")
    axes[1].axhline(0.10, color="red", linestyle="--", linewidth=1.0,
                    label="chance level (10%)")
    axes[1].set_ylabel("Hit rate of top-10% $h$ in depth-discontinuity set")
    axes[1].set_title("Top-10% overlap with depth boundaries")
    axes[1].set_ylim(0, 0.2)
    axes[1].legend(fontsize=8)

    axes[2].bar(x, [float(r["affine_r2"]) for r in rows], color="seagreen")
    axes[2].set_ylabel("$R^2$ of affine map $\\mathrm{depth}=a\\,h+b$")
    axes[2].set_title("Direct calibration feasibility")
    axes[2].set_ylim(0, 0.1)

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=60, fontsize=7)
        axis.grid(True, axis="y", alpha=0.3)

    figure.suptitle(
        "Scale-anchor feasibility on 15 calibrated SCARED keyframes: the gradient pseudo-height "
        "cannot be directly calibrated to metric depth",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
