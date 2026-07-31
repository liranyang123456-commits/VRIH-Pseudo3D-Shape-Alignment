#!/usr/bin/env python3
"""Evaluate classical feature-based relative-motion baselines on a sequence.

Each pair is estimated with calibrated essential-matrix RANSAC.  Translation is
unit-norm by construction, so the exported trajectory is scale ambiguous and is
only suitable for Sim(3)-aligned and relative-motion evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def detector(name: str):
    if name == "SIFT":
        return cv2.SIFT_create(nfeatures=4000), cv2.NORM_L2
    if name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if name == "ORB":
        return cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING
    raise ValueError(name)


def write_poses(path: Path, poses: list[np.ndarray]) -> None:
    lines = ["# Scale-ambiguous feature-based cumulative trajectory", "# T_{0<-t}"]
    for index, pose in enumerate(poses):
        lines.append(f"frame_idx={index}")
        lines.extend(" ".join(f"{value:.10f}" for value in row) for row in pose)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("SIFT", "AKAZE", "ORB"), required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    args = parser.parse_args()

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )[: args.max_frames]
    if len(image_paths) < 2:
        raise ValueError("Need at least two images")
    calibration = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    k = np.asarray(calibration["K"], dtype=np.float64)
    extractor, norm = detector(args.method)
    matcher = cv2.BFMatcher(norm)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    poses = [np.eye(4, dtype=np.float64)]
    diagnostics: list[dict[str, object]] = []
    prior = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
    if prior is None:
        raise RuntimeError(f"Cannot load {image_paths[0]}")

    for index, path in enumerate(image_paths[1:], start=1):
        current = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if current is None:
            raise RuntimeError(f"Cannot load {path}")
        keypoints_1, descriptors_1 = extractor.detectAndCompute(prior, None)
        keypoints_2, descriptors_2 = extractor.detectAndCompute(current, None)
        inliers = 0
        status = "ok"
        transform = np.eye(4, dtype=np.float64)
        try:
            raw_matches = matcher.knnMatch(descriptors_1, descriptors_2, k=2)
            matches = [first for first, second in raw_matches if first.distance < 0.75 * second.distance]
            if len(matches) < 8:
                raise RuntimeError("fewer than 8 ratio-test matches")
            pts_1 = np.float64([keypoints_1[match.queryIdx].pt for match in matches])
            pts_2 = np.float64([keypoints_2[match.trainIdx].pt for match in matches])
            essential, mask = cv2.findEssentialMat(
                pts_1, pts_2, k, method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
            if essential is None:
                raise RuntimeError("findEssentialMat failed")
            _, rotation, translation, pose_mask = cv2.recoverPose(essential, pts_1, pts_2, k)
            inliers = int(pose_mask.astype(bool).sum())
            current_from_prior = np.eye(4, dtype=np.float64)
            current_from_prior[:3, :3] = rotation
            current_from_prior[:3, 3] = translation.reshape(3)
            transform = np.linalg.inv(current_from_prior)
        except Exception as error:
            matches = []
            status = f"failure:{error}"
        poses.append(poses[-1] @ transform)
        diagnostics.append(
            {
                "frame_idx": index,
                "first_image": image_paths[index - 1].name,
                "second_image": path.name,
                "keypoints_first": len(keypoints_1),
                "keypoints_second": len(keypoints_2),
                "ratio_matches": len(matches),
                "essential_inliers": inliers,
                "status": status,
            }
        )
        prior = current

    write_poses(args.output_dir / f"{args.method.lower()}_cumulative_4x4.txt", poses)
    with (args.output_dir / "per_pair_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "n_frames": len(image_paths),
                "intrinsics_source": str(args.intrinsics),
                "translation_scale": "unit-norm from recoverPose",
                "input_images": [path.name for path in image_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
