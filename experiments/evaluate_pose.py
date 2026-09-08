#!/usr/bin/env python3
"""Evaluate pseudo-3D trajectories against chessboard pose ground truth.

The script intentionally distinguishes pseudo-3D shape-alignment trajectories
from metric camera trajectories.  It reports:

* ATE after a similarity (Sim(3)) trajectory alignment, in GT units;
* relative rotation error, which is invariant to a global coordinate transform;
* relative translation-direction error, also scale invariant.

It must not be used to claim a direct metric pose recovery from (u, v, h).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


FRAME_PATTERN = re.compile(r"^frame_idx=(\d+)\s*$")


def read_poses(path: Path) -> dict[int, np.ndarray]:
    """Read the project's `frame_idx` + 4x4 pose text format."""
    poses: dict[int, np.ndarray] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0
    while cursor < len(lines):
        match = FRAME_PATTERN.match(lines[cursor].strip())
        if match is None:
            cursor += 1
            continue
        frame_idx = int(match.group(1))
        rows: list[list[float]] = []
        cursor += 1
        while cursor < len(lines) and len(rows) < 4:
            line = lines[cursor].strip()
            cursor += 1
            if not line:
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 4:
                raise ValueError(f"{path}: expected 4 matrix values, got {line!r}")
            rows.append(values)
        if len(rows) != 4:
            raise ValueError(f"{path}: incomplete matrix for frame {frame_idx}")
        pose = np.asarray(rows, dtype=np.float64)
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"{path}: invalid homogeneous row at frame {frame_idx}")
        poses[frame_idx] = pose
    if not poses:
        raise ValueError(f"{path}: no poses found")
    return poses


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def umeyama_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scale, rotation, translation mapping source positions to target."""
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("Sim(3) alignment requires matching trajectories with >=3 poses")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        signs[-1] = -1
    rotation = u @ np.diag(signs) @ vt
    source_variance = np.sum(source_centered**2) / source.shape[0]
    target_variance = np.sum(target_centered**2) / target.shape[0]
    if source_variance <= np.finfo(float).eps:
        raise ValueError("Degenerate source trajectory: zero positional variance")
    if target_variance <= np.finfo(float).eps:
        raise ValueError("Degenerate ground-truth trajectory: zero positional variance")
    scale = float(np.sum(singular_values * signs) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def relative_transform(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    return np.linalg.inv(pose_a) @ pose_b


def evaluate(
    predicted: dict[int, np.ndarray],
    ground_truth: dict[int, np.ndarray],
    max_frames: int | None = None,
) -> dict:
    frames = sorted(set(predicted) & set(ground_truth))
    if max_frames is not None:
        frames = frames[:max_frames]
    if len(frames) < 3:
        raise ValueError("Need at least three overlapping predicted/GT poses")
    predicted_positions = np.stack([predicted[idx][:3, 3] for idx in frames])
    gt_positions = np.stack([ground_truth[idx][:3, 3] for idx in frames])
    scale, alignment_rotation, alignment_translation = umeyama_similarity(
        predicted_positions, gt_positions
    )
    aligned_positions = (
        scale * (alignment_rotation @ predicted_positions.T).T + alignment_translation
    )
    ate_errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)

    rotation_errors: list[float] = []
    translation_direction_errors: list[float] = []
    for prior, current in zip(frames[:-1], frames[1:]):
        predicted_relative = relative_transform(predicted[prior], predicted[current])
        gt_relative = relative_transform(ground_truth[prior], ground_truth[current])
        rotation_errors.append(
            rotation_angle_deg(predicted_relative[:3, :3].T @ gt_relative[:3, :3])
        )
        predicted_translation = predicted_relative[:3, 3]
        gt_translation = gt_relative[:3, 3]
        predicted_norm = np.linalg.norm(predicted_translation)
        gt_norm = np.linalg.norm(gt_translation)
        if predicted_norm > 1e-10 and gt_norm > 1e-10:
            cosine = np.clip(
                float(np.dot(predicted_translation, gt_translation))
                / (predicted_norm * gt_norm),
                -1.0,
                1.0,
            )
            translation_direction_errors.append(math.degrees(math.acos(cosine)))

    def summary(values: list[float] | np.ndarray) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "rmse": float(np.sqrt(np.mean(array**2))),
        }

    return {
        "n_poses": len(frames),
        "frame_indices": frames,
        "trajectory_alignment": {
            "type": "Sim3",
            "scale_pred_to_gt": scale,
            "rotation_pred_to_gt": alignment_rotation.tolist(),
            "translation_pred_to_gt": alignment_translation.tolist(),
            "warning": (
                "Sim(3)-aligned ATE evaluates trajectory-shape agreement only; "
                "it does not turn a uvh pseudo-height trajectory into metric pose."
            ),
        },
        "ate_after_sim3_gt_units": summary(ate_errors),
        "relative_rotation_error_deg": summary(rotation_errors),
        "relative_translation_direction_error_deg": (
            summary(translation_direction_errors)
            if translation_direction_errors
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    result = evaluate(
        read_poses(args.predicted),
        read_poses(args.ground_truth),
        args.max_frames,
    )
    result["predicted_path"] = str(args.predicted)
    result["ground_truth_path"] = str(args.ground_truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
