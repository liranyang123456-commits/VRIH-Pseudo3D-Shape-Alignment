#!/usr/bin/env python3
"""Extract a SCARED RGB sequence and its released camera-pose metadata."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-video", type=Path, required=True)
    parser.add_argument("--frame-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=80)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    poses: list[np.ndarray] = []
    intrinsics: dict | None = None

    with tarfile.open(args.frame_data, "r:gz") as archive:
        members = sorted(
            (member for member in archive.getmembers() if member.name.endswith(".json")),
            key=lambda member: member.name,
        )[: args.max_frames]
        for member in members:
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"Cannot extract {member.name}")
            payload = json.load(handle)
            pose = np.asarray(payload["camera-pose"], dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError(f"Unexpected camera pose shape {pose.shape}")
            poses.append(pose)
            intrinsics = payload.get("camera-calibration", intrinsics)

    capture = cv2.VideoCapture(str(args.rgb_video))
    frames_written = 0
    while frames_written < len(poses):
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(str(image_dir / f"frame_{frames_written:06d}.png"), frame)
        frames_written += 1
    capture.release()
    if frames_written != len(poses):
        raise RuntimeError(
            f"RGB video yielded {frames_written} frames but {len(poses)} pose records were selected"
        )

    pose_path = args.output_dir / "scared_camera_pose_as_released_4x4.txt"
    lines = [
        "# SCARED released `camera-pose` matrices, retained as provided.",
        "# Coordinate convention is documented in run metadata; validate before claiming metric extrinsics.",
    ]
    for index, pose in enumerate(poses):
        lines.append(f"frame_idx={index}")
        lines.extend(" ".join(f"{entry:.10f}" for entry in row) for row in pose)
        lines.append("")
    pose_path.write_text("\n".join(lines), encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "SCARED",
                "n_frames": frames_written,
                "rgb_video": str(args.rgb_video),
                "frame_data_archive": str(args.frame_data),
                "pose_field": "camera-pose (as released)",
                "camera_calibration_first_frame": intrinsics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
