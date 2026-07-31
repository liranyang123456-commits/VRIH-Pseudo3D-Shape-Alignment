#!/usr/bin/env python3
"""Run Reloc3r on consecutive frames and export a traceable pseudo-trajectory.

Reloc3r predicts scale-ambiguous pairwise relative poses.  The output trajectory
is therefore evaluated only with Sim(3)-aligned ATE and relative rotation/
translation-direction metrics, never as direct metric pose recovery.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def write_poses(path: Path, poses: list[np.ndarray]) -> None:
    lines = [
        "# Reloc3r scale-ambiguous cumulative trajectory",
        "# T_{0<-t}; relative translations are normalized by the official model demo.",
    ]
    for index, pose in enumerate(poses):
        lines.append(f"frame_idx={index}")
        lines.extend(" ".join(f"{value:.10f}" for value in row) for row in pose)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reloc3r-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--image-resolution", choices=("224", "512"), default="224")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.reloc3r_root))
    from reloc3r.reloc3r_relpose import inference_relpose, setup_reloc3r_relpose_model
    from reloc3r.utils.device import to_numpy
    from reloc3r.utils.image import check_images_shape_format, load_images

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.max_frames]
    if len(image_paths) < 2:
        raise ValueError("Need at least two input images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_reloc3r_relpose_model(args.image_resolution, device)
    poses = [np.eye(4, dtype=np.float64)]
    pairs: list[dict[str, object]] = []

    for index, (first, second) in enumerate(zip(image_paths[:-1], image_paths[1:]), start=1):
        started = time.perf_counter()
        views = load_images([str(first), str(second)], size=int(args.image_resolution))
        views = check_images_shape_format(views, device)
        pair_pose = to_numpy(inference_relpose(views, model, device, args.amp)[0])
        translation_norm = float(np.linalg.norm(pair_pose[:3, 3]))
        if not np.isfinite(pair_pose).all() or translation_norm <= 1e-10:
            raise RuntimeError(f"Invalid Reloc3r pose for pair {first.name} -> {second.name}")
        pair_pose[:3, 3] /= translation_norm
        poses.append(poses[-1] @ pair_pose)
        pairs.append(
            {
                "frame_idx": index,
                "first_image": first.name,
                "second_image": second.name,
                "translation_norm_before_normalization": translation_norm,
                "runtime_s": time.perf_counter() - started,
            }
        )

    pose_path = args.output_dir / "reloc3r_cumulative_4x4.txt"
    write_poses(pose_path, poses)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "method": "Reloc3r",
                "device": str(device),
                "image_resolution": int(args.image_resolution),
                "translation_scale": "unit-normalized per pair (official demo convention)",
                "n_frames": len(image_paths),
                "input_images": [path.name for path in image_paths],
                "pairs": pairs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pose_path)


if __name__ == "__main__":
    main()
