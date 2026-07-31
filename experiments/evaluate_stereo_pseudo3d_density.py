"""Density--quality analysis for pseudo-height-assisted stereo matching.

The experiment tests whether pseudo-height can preserve or improve matching
density while improving geometric quality.  Metric geometry is always supplied
by released stereo calibration and depth ground truth; pseudo-height is only a
matching descriptor.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Profile:
    name: str
    ratio: float
    gradient_mask: bool
    height_max_difference: float | None
    orientation_max_degrees: float | None
    mutual: bool


PROFILES = (
    Profile("SIFT", 0.75, False, None, None, False),
    Profile("SIFT-dense", 0.90, False, None, None, False),
    Profile("SIFT+gradient-mask", 0.75, True, None, None, False),
    Profile("Pseudo3D-dense", 0.90, False, 0.80, 150.0, False),
    Profile("Pseudo3D-balanced", 0.85, False, 0.50, 120.0, False),
    Profile("Ours-selective", 0.75, False, 0.35, 75.0, True),
)


def rotation_error_deg(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    cosine = np.clip((np.trace(predicted.T @ ground_truth) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-12:
        return float("nan")
    return math.degrees(
        math.acos(float(np.clip(np.dot(first.ravel(), second.ravel()) / denominator, -1.0, 1.0)))
    )


def gradient_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = gray.astype(np.float32) / 255.0
    smooth = cv2.bilateralFilter(normalized, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    maximum = float(magnitude.max())
    height = magnitude / maximum if maximum > 1e-12 else magnitude
    return height, np.arctan2(gy, gx), height >= np.percentile(height, 90.0)


def structures(keypoints: list[cv2.KeyPoint], height: np.ndarray, angle: np.ndarray) -> np.ndarray:
    h, w = height.shape
    data = []
    for keypoint in keypoints:
        x = int(np.clip(round(keypoint.pt[0]), 0, w - 1))
        y = int(np.clip(round(keypoint.pt[1]), 0, h - 1))
        data.append((height[y, x], angle[y, x]))
    return np.asarray(data, dtype=np.float32)


def select_mask(
    keypoints: list[cv2.KeyPoint], descriptors: np.ndarray, mask: np.ndarray, enabled: bool
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    if not enabled:
        return keypoints, descriptors
    h, w = mask.shape
    selected = [
        index
        for index, keypoint in enumerate(keypoints)
        if mask[
            int(np.clip(round(keypoint.pt[1]), 0, h - 1)),
            int(np.clip(round(keypoint.pt[0]), 0, w - 1)),
        ]
    ]
    return (
        [keypoints[index] for index in selected],
        descriptors[np.asarray(selected, dtype=int)],
    ) if len(selected) >= 8 else (keypoints, descriptors)


def filtered_matches(
    profile: Profile,
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
    left_struct: np.ndarray,
    right_struct: np.ndarray,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    matches = [first for first, second in forward if first.distance < profile.ratio * second.distance]
    if profile.height_max_difference is not None:
        filtered = []
        angle_limit = math.radians(profile.orientation_max_degrees or 180.0)
        for item in matches:
            height_difference = abs(float(left_struct[item.queryIdx, 0] - right_struct[item.trainIdx, 0]))
            angle_difference = abs(
                math.atan2(
                    math.sin(float(left_struct[item.queryIdx, 1] - right_struct[item.trainIdx, 1])),
                    math.cos(float(left_struct[item.queryIdx, 1] - right_struct[item.trainIdx, 1])),
                )
            )
            if height_difference <= profile.height_max_difference and angle_difference <= angle_limit:
                filtered.append(item)
        matches = filtered
    if profile.mutual:
        reverse = matcher.match(right_descriptors, left_descriptors)
        reverse_pairs = {(item.trainIdx, item.queryIdx) for item in reverse}
        matches = [item for item in matches if (item.queryIdx, item.trainIdx) in reverse_pairs]
    return matches


def calibration(path: Path):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    r = storage.getNode("R").mat().astype(np.float64)
    t = storage.getNode("T").mat().astype(np.float64).reshape(3, 1)
    k1 = storage.getNode("M1").mat().astype(np.float64)
    d1 = storage.getNode("D1").mat().astype(np.float64)
    k2 = storage.getNode("M2").mat().astype(np.float64)
    d2 = storage.getNode("D2").mat().astype(np.float64)
    storage.release()
    return r, t, k1, d1, k2, d2


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.ravel()
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def evaluate(keyframe: Path, profile: Profile) -> dict[str, object]:
    r_gt, t_gt, k1, d1, k2, d2 = calibration(keyframe / "endoscope_calibration.yaml")
    left = cv2.imread(str(keyframe / "Left_Image.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(keyframe / "Right_Image.png"), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(keyframe / "left_depth_map.tiff"), cv2.IMREAD_UNCHANGED)[..., 0]
    sift = cv2.SIFT_create(nfeatures=4000)
    left_kp, left_desc = sift.detectAndCompute(left, None)
    right_kp, right_desc = sift.detectAndCompute(right, None)
    left_h, left_angle, left_mask = gradient_features(left)
    right_h, right_angle, right_mask = gradient_features(right)
    left_kp, left_desc = select_mask(left_kp, left_desc, left_mask, profile.gradient_mask)
    right_kp, right_desc = select_mask(right_kp, right_desc, right_mask, profile.gradient_mask)
    left_struct = structures(left_kp, left_h, left_angle)
    right_struct = structures(right_kp, right_h, right_angle)
    matches = filtered_matches(profile, left_desc, right_desc, left_struct, right_struct)
    row: dict[str, object] = {
        "keyframe": f"{keyframe.parent.name}/{keyframe.name}",
        "method": profile.name,
        "ratio_threshold": profile.ratio,
        "height_threshold": profile.height_max_difference,
        "orientation_threshold_deg": profile.orientation_max_degrees,
        "mutual": profile.mutual,
        "matches": len(matches),
    }
    if len(matches) < 8:
        row.update({key: np.nan for key in ("gt_match_precision", "inliers", "rotation_error_deg", "translation_direction_error_deg", "depth_absrel", "depth_rmse", "triangulated_points")})
        return row
    points_left = np.asarray([left_kp[item.queryIdx].pt for item in matches], dtype=np.float64)
    points_right = np.asarray([right_kp[item.trainIdx].pt for item in matches], dtype=np.float64)
    norm_left = cv2.undistortPoints(points_left.reshape(-1, 1, 2), k1, d1).reshape(-1, 2)
    norm_right = cv2.undistortPoints(points_right.reshape(-1, 1, 2), k2, d2).reshape(-1, 2)
    e_gt = skew(t_gt) @ r_gt
    x1 = np.column_stack([norm_left, np.ones(len(norm_left))])
    x2 = np.column_stack([norm_right, np.ones(len(norm_right))])
    ex1 = (e_gt @ x1.T).T
    etx2 = (e_gt.T @ x2.T).T
    numerator = np.sum(x2 * ex1, axis=1) ** 2
    sampson = numerator / (ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2 + 1e-12)
    focal = float((k1[0, 0] + k1[1, 1] + k2[0, 0] + k2[1, 1]) / 4.0)
    gt_precision = float(np.mean(sampson <= (1.5 / focal) ** 2))
    essential, _ = cv2.findEssentialMat(norm_left, norm_right, 1.0, (0.0, 0.0), cv2.RANSAC, 0.999, 0.001)
    if essential is None:
        row.update({"gt_match_precision": gt_precision, "inliers": 0, "rotation_error_deg": np.nan, "translation_direction_error_deg": np.nan, "depth_absrel": np.nan, "depth_rmse": np.nan, "triangulated_points": 0})
        return row
    inliers, r_est, t_est, mask = cv2.recoverPose(essential, norm_left, norm_right)
    valid = mask.reshape(-1) > 0
    points_h = cv2.triangulatePoints(
        np.hstack([np.eye(3), np.zeros((3, 1))]),
        np.hstack([r_gt, t_gt]),
        norm_left[valid].T,
        norm_right[valid].T,
    )
    points_3d = (points_h[:3] / points_h[3]).T
    prediction = points_3d[:, 2]
    left_valid = points_left[valid]
    xs = np.clip(np.round(left_valid[:, 0]).astype(int), 0, depth.shape[1] - 1)
    ys = np.clip(np.round(left_valid[:, 1]).astype(int), 0, depth.shape[0] - 1)
    target = depth[ys, xs].astype(np.float64)
    valid_depth = np.isfinite(prediction) & np.isfinite(target) & (prediction > 0) & (target > 0)
    prediction, target = prediction[valid_depth], target[valid_depth]
    row.update(
        {
            "gt_match_precision": gt_precision,
            "inliers": int(inliers),
            "rotation_error_deg": rotation_error_deg(r_est, r_gt),
            "translation_direction_error_deg": vector_angle_deg(t_est, t_gt),
            "depth_absrel": float(np.mean(np.abs(prediction - target) / target)),
            "depth_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "triangulated_points": int(len(prediction)),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for dataset in sorted(args.scared_root.glob("dataset_*")):
        for keyframe in sorted(dataset.glob("keyframe_*")):
            if (keyframe / "Left_Image.png").exists():
                for profile in PROFILES:
                    rows.append(evaluate(keyframe, profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    metrics = ("matches", "gt_match_precision", "inliers", "rotation_error_deg", "translation_direction_error_deg", "depth_absrel", "depth_rmse", "triangulated_points")
    for profile in PROFILES:
        group = [row for row in rows if row["method"] == profile.name]
        result = {"method": profile.name, "n_keyframes": len(group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            result[f"mean_{metric}"] = float(np.nanmean(values))
            result[f"std_{metric}"] = float(np.nanstd(values, ddof=1))
        summary.append(result)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for row in summary:
        axes[0].scatter(row["mean_matches"], row["mean_gt_match_precision"], label=row["method"])
        axes[1].scatter(row["mean_matches"], row["mean_depth_absrel"], label=row["method"])
    axes[0].set(xlabel="Mean matches per stereo pair", ylabel="GT geometric match precision")
    axes[1].set(xlabel="Mean matches per stereo pair", ylabel="Triangulated depth AbsRel")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.legend(loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(args.output.with_suffix(".png"), dpi=300)


if __name__ == "__main__":
    main()
