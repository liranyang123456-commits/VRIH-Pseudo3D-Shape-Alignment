#!/usr/bin/env python3
"""Plot Sim(3)-aligned trajectory shapes with explicit non-metric caveats."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_pose import read_poses, umeyama_similarity  # noqa: E402


def parse_method(value: str) -> tuple[str, Path]:
    label, separator, file_name = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected LABEL=POSE_FILE")
    return label, Path(file_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--method", type=parse_method, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    gt = read_poses(args.ground_truth)
    figure = plt.figure(figsize=(10, 5))
    ax_xy = figure.add_subplot(1, 2, 1)
    ax_xz = figure.add_subplot(1, 2, 2)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    common = sorted(gt)[: args.max_frames]
    gt_positions = np.stack([gt[index][:3, 3] for index in common])
    ax_xy.plot(gt_positions[:, 0], gt_positions[:, 1], "k-", linewidth=2, label="Chessboard GT")
    ax_xz.plot(gt_positions[:, 0], gt_positions[:, 2], "k-", linewidth=2, label="Chessboard GT")

    for index, (label, pose_path) in enumerate(args.method):
        poses = read_poses(pose_path)
        frames = [frame for frame in common if frame in poses]
        if len(frames) < 3:
            raise ValueError(f"{label}: fewer than three shared frames")
        predicted = np.stack([poses[frame][:3, 3] for frame in frames])
        target = np.stack([gt[frame][:3, 3] for frame in frames])
        scale, rotation, translation = umeyama_similarity(predicted, target)
        aligned = scale * (rotation @ predicted.T).T + translation
        color = colors[index % len(colors)]
        ax_xy.plot(aligned[:, 0], aligned[:, 1], color=color, label=label)
        ax_xz.plot(aligned[:, 0], aligned[:, 2], color=color, label=label)

    ax_xy.set_title("XY trajectory after Sim(3) alignment (non-metric)")
    ax_xz.set_title("XZ trajectory after Sim(3) alignment (non-metric)")
    for axis in (ax_xy, ax_xz):
        axis.set_xlabel("GT coordinate units (not mm)")
        axis.set_ylabel("GT coordinate units (not mm)")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
    handles, labels = ax_xy.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=max(2, len(labels)))
    figure.suptitle(
        "Chessboard trajectory-shape comparison. Positions are Sim(3)-aligned to GT for "
        "visualization only; the alignment absorbs scale, so this is NOT metric pose recovery.",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
