#!/usr/bin/env python3
"""Convert EndoNeRF LLFF-style poses_bounds.npy to the project pose text format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses-bounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=80)
    args = parser.parse_args()

    values = np.load(args.poses_bounds)
    if values.ndim != 2 or values.shape[1] < 15:
        raise ValueError(f"Expected Nx17 LLFF poses_bounds array, got {values.shape}")
    poses_hwf = values[:, :15].reshape(-1, 3, 5)
    poses = poses_hwf[: args.max_frames, :, :4]
    lines = [
        "# EndoNeRF LLFF camera-to-world poses converted from poses_bounds.npy",
        "# Coordinate convention retained from the released EndoNeRF dataset.",
    ]
    for index, pose_3x4 in enumerate(poses):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :4] = pose_3x4
        lines.append(f"frame_idx={index}")
        lines.extend(" ".join(f"{entry:.10f}" for entry in row) for row in pose)
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "source": str(args.poses_bounds),
        "format": "EndoNeRF LLFF poses_bounds.npy",
        "n_poses": len(poses),
        "height_width_focal_first_frame": poses_hwf[0, :, 4].tolist(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
