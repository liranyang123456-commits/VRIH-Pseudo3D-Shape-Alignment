#!/usr/bin/env python3
"""Grouped bar charts for the five-sequence chessboard benchmark (seg100 protocol)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "chessboard_multi_sequence_summary.csv"
OUTPUT = ROOT / "results" / "fig_chessboard_multi_sequence.png"

METHODS = ["Ours", "DetectorFreeSfM", "Reloc3r", "SIFT", "AKAZE", "ORB"]
LABELS = {
    "Ours": "Ours",
    "DetectorFreeSfM": "DetectorFreeSfM (coarse-only)",
    "Reloc3r": "Reloc3r",
    "SIFT": "SIFT",
    "AKAZE": "AKAZE",
    "ORB": "ORB",
}
SEQ_ORDER = ["seq1", "seq2", "seq3", "seq4", "line2"]
SEQ_LABELS = {"seq1": "Seq-1", "seq2": "Seq-2", "seq3": "Seq-3", "seq4": "Seq-4", "line2": "Line2"}


def main() -> None:
    ate: dict[str, dict[str, float]] = defaultdict(dict)
    rot: dict[str, dict[str, float]] = defaultdict(dict)
    with SUMMARY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["protocol"] != "seg100":
                continue
            ate[row["method"]][row["sequence"]] = float(row["ate_sim3_rmse"])
            rot[row["method"]][row["sequence"]] = float(row["rel_rot_err_mean_deg"])

    x = np.arange(len(SEQ_ORDER))
    width = 0.13
    figure, (ax_rot, ax_ate) = plt.subplots(1, 2, figsize=(13, 4.6))
    for offset, method in enumerate(METHODS):
        positions = x + (offset - (len(METHODS) - 1) / 2) * width
        rot_vals = [rot[method].get(seq, np.nan) for seq in SEQ_ORDER]
        ate_vals = [ate[method].get(seq, np.nan) for seq in SEQ_ORDER]
        ax_rot.bar(positions, rot_vals, width, label=LABELS[method])
        ax_ate.bar(positions, ate_vals, width, label=LABELS[method])

    ax_rot.set_yscale("log")
    ax_rot.set_ylabel("Relative rotation error, mean (deg, log scale)")
    ax_rot.set_title("Relative rotation error per sequence (fair, scale-free metric)")
    ax_ate.set_yscale("log")
    ax_ate.set_ylabel("Sim(3)-aligned ATE RMSE (non-metric, GT coord units)")
    ax_ate.set_title("Sim(3)-aligned ATE per sequence (NOT mm; shape only)")
    for axis in (ax_rot, ax_ate):
        axis.set_xticks(x)
        axis.set_xticklabels([SEQ_LABELS[s] for s in SEQ_ORDER])
        axis.set_xlabel("Checkerboard sequence (first 100 GT-covered frames)")
        axis.grid(True, axis="y", alpha=0.3)
    handles, labels = ax_rot.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    figure.suptitle(
        "Five-sequence checkerboard evaluation. ATE is Sim(3)-aligned and non-metric (not mm) "
        "because the pseudo-3D transform and monocular baselines are scale-ambiguous; "
        "rotation error is the fair comparison.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    figure.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
