"""Calibrated stereo validation of pseudo-height-assisted matching.

The pseudo-height is only an auxiliary local structural descriptor. Metric depth
is reconstructed by calibrated stereo triangulation and compared with the
released SCARED depth map. Four progressively constrained SIFT pipelines are
evaluated:

* SIFT;
* SIFT + gradient mask;
* SIFT + pseudo-height descriptor;
* Ours-structural: pseudo-height descriptor + mutual matching + robust gating.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


METHODS = ("SIFT", "SIFT+gradient-mask", "SIFT+pseudo-height", "Ours-structural")


def rotation_error_deg(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    cosine = np.clip((np.trace(predicted.T @ ground_truth) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def vector_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first.reshape(-1)
    second = second.reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-12:
        return float("nan")
    cosine = np.clip(float(np.dot(first, second) / denominator), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def gradient_features(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = gray.astype(np.float32) / 255.0
    smooth = cv2.bilateralFilter(normalized, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    maximum = float(magnitude.max())
    height = magnitude / maximum if maximum > 1e-12 else magnitude
    orientation = np.arctan2(gy, gx)
    mask = height >= np.percentile(height, 90.0)
    return height, orientation, mask


def sample_descriptor(
    keypoints: list[cv2.KeyPoint], height: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    sampled: list[tuple[float, float]] = []
    h, w = height.shape
    for keypoint in keypoints:
        x = int(np.clip(round(keypoint.pt[0]), 0, w - 1))
        y = int(np.clip(round(keypoint.pt[1]), 0, h - 1))
        sampled.append((float(height[y, x]), float(orientation[y, x])))
    return np.asarray(sampled, dtype=np.float32)


def select_keypoints(
    keypoints: list[cv2.KeyPoint], descriptors: np.ndarray, mask: np.ndarray, enabled: bool
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    if not enabled:
        return keypoints, descriptors
    selected = []
    selected_descriptors = []
    h, w = mask.shape
    for keypoint, descriptor in zip(keypoints, descriptors):
        x = int(np.clip(round(keypoint.pt[0]), 0, w - 1))
        y = int(np.clip(round(keypoint.pt[1]), 0, h - 1))
        if mask[y, x]:
            selected.append(keypoint)
            selected_descriptors.append(descriptor)
    if len(selected) < 8:
        return keypoints, descriptors
    return selected, np.asarray(selected_descriptors)


def match(
    method: str,
    left_keypoints: list[cv2.KeyPoint],
    left_descriptors: np.ndarray,
    right_keypoints: list[cv2.KeyPoint],
    right_descriptors: np.ndarray,
    left_structure: np.ndarray,
    right_structure: np.ndarray,
) -> list[cv2.DMatch]:
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    selected = [first for first, second in forward if first.distance < 0.75 * second.distance]
    if method in ("SIFT+pseudo-height", "Ours-structural"):
        filtered = []
        for item in selected:
            h_left, theta_left = left_structure[item.queryIdx]
            h_right, theta_right = right_structure[item.trainIdx]
            height_difference = abs(float(h_left - h_right))
            orientation_difference = abs(math.atan2(math.sin(theta_left - theta_right), math.cos(theta_left - theta_right)))
            if height_difference <= 0.35 and orientation_difference <= math.radians(75):
                filtered.append(item)
        selected = filtered
    if method == "Ours-structural":
        reverse = matcher.match(right_descriptors, left_descriptors)
        reverse_pairs = {(item.trainIdx, item.queryIdx) for item in reverse}
        selected = [item for item in selected if (item.queryIdx, item.trainIdx) in reverse_pairs]
    return selected


def read_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise RuntimeError(f"Cannot open calibration file {path}")
    r = storage.getNode("R").mat().astype(np.float64)
    t = storage.getNode("T").mat().astype(np.float64).reshape(3, 1)
    k_left = storage.getNode("M1").mat().astype(np.float64)
    d_left = storage.getNode("D1").mat().astype(np.float64)
    k_right = storage.getNode("M2").mat().astype(np.float64)
    storage.release()
    return r, t, k_left, d_left, k_right


def evaluate_keyframe(keyframe: Path, method: str) -> dict[str, object]:
    calibration = keyframe / "endoscope_calibration.yaml"
    left_path = keyframe / "Left_Image.png"
    right_path = keyframe / "Right_Image.png"
    depth_path = keyframe / "left_depth_map.tiff"
    r_gt, t_gt, k_left, d_left, k_right = read_calibration(calibration)
    left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
    depth_map = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if left is None or right is None or depth_map is None:
        raise RuntimeError(f"Missing SCARED files in {keyframe}")
    depth_gt = depth_map[..., 0] if depth_map.ndim == 3 else depth_map
    sift = cv2.SIFT_create(nfeatures=4000)
    left_keypoints, left_descriptors = sift.detectAndCompute(left, None)
    right_keypoints, right_descriptors = sift.detectAndCompute(right, None)
    left_height, left_orientation, left_mask = gradient_features(left)
    right_height, right_orientation, right_mask = gradient_features(right)
    use_mask = method == "SIFT+gradient-mask"
    left_keypoints, left_descriptors = select_keypoints(
        left_keypoints, left_descriptors, left_mask, use_mask
    )
    right_keypoints, right_descriptors = select_keypoints(
        right_keypoints, right_descriptors, right_mask, use_mask
    )
    left_structure = sample_descriptor(left_keypoints, left_height, left_orientation)
    right_structure = sample_descriptor(right_keypoints, right_height, right_orientation)
    matches = match(
        method,
        left_keypoints,
        left_descriptors,
        right_keypoints,
        right_descriptors,
        left_structure,
        right_structure,
    )
    output: dict[str, object] = {
        "keyframe": f"{keyframe.parent.name}/{keyframe.name}",
        "method": method,
        "keypoints_left": len(left_keypoints),
        "keypoints_right": len(right_keypoints),
        "matches": len(matches),
    }
    if len(matches) < 8:
        output.update({"inliers": 0, "rotation_error_deg": np.nan, "translation_direction_error_deg": np.nan, "depth_absrel": np.nan, "depth_rmse": np.nan, "triangulated_points": 0})
        return output
    points_left = np.asarray([left_keypoints[item.queryIdx].pt for item in matches], dtype=np.float64)
    points_right = np.asarray([right_keypoints[item.trainIdx].pt for item in matches], dtype=np.float64)
    normalized_left = cv2.undistortPoints(points_left.reshape(-1, 1, 2), k_left, d_left).reshape(-1, 2)
    # SCARED right distortion is small; load directly from the calibration file.
    storage = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_READ)
    d_right = storage.getNode("D2").mat().astype(np.float64)
    storage.release()
    normalized_right = cv2.undistortPoints(points_right.reshape(-1, 1, 2), k_right, d_right).reshape(-1, 2)
    essential, mask = cv2.findEssentialMat(normalized_left, normalized_right, 1.0, (0.0, 0.0), cv2.RANSAC, 0.999, 0.001)
    if essential is None or mask is None:
        output.update({"inliers": 0, "rotation_error_deg": np.nan, "translation_direction_error_deg": np.nan, "depth_absrel": np.nan, "depth_rmse": np.nan, "triangulated_points": 0})
        return output
    inliers, r_est, t_est, pose_mask = cv2.recoverPose(
        essential, normalized_left, normalized_right
    )
    valid = pose_mask.reshape(-1) > 0
    p1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = np.hstack([r_gt, t_gt])
    homogeneous = cv2.triangulatePoints(
        p1,
        p2,
        normalized_left[valid].T,
        normalized_right[valid].T,
    )
    points_3d = (homogeneous[:3] / homogeneous[3]).T
    depth_pred = points_3d[:, 2]
    sampled_left = points_left[valid]
    x = np.clip(np.round(sampled_left[:, 0]).astype(int), 0, depth_gt.shape[1] - 1)
    y = np.clip(np.round(sampled_left[:, 1]).astype(int), 0, depth_gt.shape[0] - 1)
    depth_reference = depth_gt[y, x].astype(np.float64)
    valid_depth = (
        np.isfinite(depth_pred)
        & np.isfinite(depth_reference)
        & (depth_pred > 0)
        & (depth_reference > 0)
    )
    depth_pred = depth_pred[valid_depth]
    depth_reference = depth_reference[valid_depth]
    output.update(
        {
            "inliers": int(inliers),
            "rotation_error_deg": rotation_error_deg(r_est, r_gt),
            "translation_direction_error_deg": vector_angle_deg(t_est, t_gt),
            "depth_absrel": float(np.mean(np.abs(depth_pred - depth_reference) / depth_reference)) if len(depth_pred) else np.nan,
            "depth_rmse": float(np.sqrt(np.mean((depth_pred - depth_reference) ** 2))) if len(depth_pred) else np.nan,
            "triangulated_points": int(len(depth_pred)),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for dataset in sorted(args.scared_root.glob("dataset_*")):
        for keyframe in sorted(dataset.glob("keyframe_*")):
            if (keyframe / "Left_Image.png").exists():
                for method in METHODS:
                    rows.append(evaluate_keyframe(keyframe, method))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: list[dict[str, object]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "n_keyframes": len(method_rows),
                "mean_matches": float(np.nanmean([row["matches"] for row in method_rows])),
                "mean_inliers": float(np.nanmean([row["inliers"] for row in method_rows])),
                "mean_rotation_error_deg": float(np.nanmean([row["rotation_error_deg"] for row in method_rows])),
                "mean_translation_direction_error_deg": float(np.nanmean([row["translation_direction_error_deg"] for row in method_rows])),
                "mean_depth_absrel": float(np.nanmean([row["depth_absrel"] for row in method_rows])),
                "mean_depth_rmse": float(np.nanmean([row["depth_rmse"] for row in method_rows])),
                "mean_triangulated_points": float(np.nanmean([row["triangulated_points"] for row in method_rows])),
            }
        )
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
