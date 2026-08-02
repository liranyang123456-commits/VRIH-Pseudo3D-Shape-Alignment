#!/usr/bin/env python3
"""Bar chart for the autoencoder fine-ranking ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "autoencoder_ablation.csv"
OUTPUT = ROOT / "results" / "fig_autoencoder_ablation.png"

METHODS = ["ae", "iou", "chamfer"]
LABELS = ["Autoencoder similarity", "Mask IoU (geometry)", "Chamfer (geometry)"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c"]


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    seqs = [r["sequence"].replace("scared_", "") for r in rows]
    x = np.arange(len(seqs))
    width = 0.26

    fig, (ax_top1, ax_rank) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (m, lab, c) in enumerate(zip(METHODS, LABELS, COLORS)):
        top1 = [float(r[f"{m}_top1"]) for r in rows]
        rank = [float(r[f"{m}_mean_rank"]) for r in rows]
        ax_top1.bar(x + (i - 1) * width, top1, width, label=lab, color=c)
        ax_rank.bar(x + (i - 1) * width, rank, width, label=lab, color=c)

    ax_top1.set_ylabel("Top-1 accuracy")
    ax_top1.set_title("Top-1 ranking accuracy")
    ax_top1.set_ylim(0, 0.25)
    ax_rank.axhline(24.5, color="gray", linestyle="--", linewidth=0.8, label="chance (49 cands)")
    ax_rank.set_ylabel("Mean rank of true ROI (lower is better)")
    ax_rank.set_title("Mean rank of the true ROI")
    for ax in (ax_top1, ax_rank):
        ax.set_xticks(x)
        ax.set_xticklabels(seqs)
        ax.grid(True, axis="y", alpha=0.3)
    ax_top1.legend(fontsize=8)
    ax_rank.legend(fontsize=8)
    fig.suptitle(
        "Autoencoder fine-ranking ablation on three SCARED sequences (40 pairs each, 49 candidates/pair)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
