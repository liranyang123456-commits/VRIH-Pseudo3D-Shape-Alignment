#!/usr/bin/env python3
"""Grouped bar charts summarizing the six-sequence SCARED evaluation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "scared_multi_sequence_summary.csv"
OUTPUT = ROOT / "results" / "fig_scared_multi_sequence.png"

METHOD_ORDER = ["GradientHeight", "IntensityHeight", "ConstantHeight", "Reloc3r", "SIFT", "AKAZE"]
METHOD_LABELS = {
    "GradientHeight": "Ours (gradient h)",
    "IntensityHeight": "Intensity h",
    "ConstantHeight": "Constant h (2D)",
    "Reloc3r": "Reloc3r",
    "SIFT": "SIFT",
    "AKAZE": "AKAZE",
}
SEQ_LABELS = {
    "scared_d1_k1": "d1/k1",
    "scared_d1_k2": "d1/k2",
    "scared_d2_k1": "d2/k1",
    "scared_d2_k2": "d2/k2",
    "scared_d3_k1": "d3/k1",
    "scared_d3_k2": "d3/k2",
}


def main() -> None:
    ate: dict[str, dict[str, float]] = defaultdict(dict)
    rot: dict[str, dict[str, float]] = defaultdict(dict)
    with SUMMARY.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ate[row["method"]][row["sequence"]] = float(row["ate_sim3_rmse_gt_units"])
            rot[row["method"]][row["sequence"]] = float(row["relative_rotation_error_mean_deg"])

    sequences = list(SEQ_LABELS)
    x = np.arange(len(sequences))
    width = 0.13
    figure, (ax_rot, ax_ate) = plt.subplots(1, 2, figsize=(13, 4.6))

    for offset, method in enumerate(METHOD_ORDER):
        positions = x + (offset - (len(METHOD_ORDER) - 1) / 2) * width
        ax_rot.bar(
            positions,
            [rot[method][seq] for seq in sequences],
            width,
            label=METHOD_LABELS[method],
        )
        ax_ate.bar(
            positions,
            [ate[method][seq] for seq in sequences],
            width,
            label=METHOD_LABELS[method],
        )

    ax_rot.set_yscale("log")
    ax_rot.set_ylabel("Relative rotation error, mean (deg, log scale)")
    ax_rot.set_title("Relative rotation error per sequence")
    ax_ate.set_ylabel("Sim(3)-aligned ATE RMSE (GT units)")
    ax_ate.set_title("Sim(3)-aligned ATE per sequence")
    for axis in (ax_rot, ax_ate):
        axis.set_xticks(x)
        axis.set_xticklabels([SEQ_LABELS[seq] for seq in sequences])
        axis.set_xlabel("SCARED sequence (dataset/keyframe, 80 frames each)")
        axis.grid(True, axis="y", alpha=0.3)
    handles, labels = ax_rot.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    figure.suptitle(
        "Six-sequence SCARED evaluation (Sim(3)-aligned; non-metric shape alignment)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.94))
    figure.savefig(OUTPUT, dpi=300)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
