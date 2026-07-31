"""Export a COLMAP reconstruction to the project's frame-indexed c2w format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pycolmap


def frame_index(name: str) -> int:
    matches = re.findall(r"\d+", Path(name).stem)
    if not matches:
        raise ValueError(f"Cannot infer frame index from {name!r}")
    index = int(matches[-1])
    # The self-acquired video extractor uses frame_000000.jpg, whereas the
    # chessboard sequence uses photo_001.jpg.
    return index if Path(name).stem.startswith("frame_") else index - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reconstruction = pycolmap.Reconstruction(str(args.model))
    poses: dict[int, np.ndarray] = {}
    image_names: dict[int, str] = {}
    for image in reconstruction.images.values():
        w2c_3x4 = image.cam_from_world().matrix()
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = w2c_3x4
        c2w = np.linalg.inv(w2c)
        index = frame_index(image.name)
        poses[index] = c2w
        image_names[index] = image.name

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DetectorFreeSfM coarse reconstruction exported from COLMAP",
        "# c2w poses; only registered images are present.",
    ]
    for index in sorted(poses):
        lines.append(f"frame_idx={index}")
        lines.extend(" ".join(f"{value:.10f}" for value in row) for row in poses[index])
        lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "registered_frames": sorted(poses),
                "n_registered": len(poses),
                "image_names": image_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
