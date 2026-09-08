#!/usr/bin/env python3
"""Internal alignment diagnostics on the six existing self-acquired sequences.

These clips are the Table-1 training corpus (3,002 frames). They have no
camera-pose ground truth and are not newly collected hard sequences.

For each sequence:
  * score the same six condition-severity proxies as Table 8;
  * assign a dominant proxy by z-score of the sequence mean vs. the pooled pairs;
  * take the contiguous 80-frame window with the highest mean dominant severity;
  * run the released flow+Kabsch / RANSAC / quality-gate pipeline (no GT);
  * report Kabsch residual, RANSAC inlier ratio, gate freezes, and HCR.

Output: self_acquired_case_pairs.csv, self_acquired_case_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HUB = ROOT.parent / "vrih_experiment_hub" / "links" / "self_acquired_endoscopy"
RESIZE_WIDTH = 320
WINDOW = 80
RANSAC_ITERS = 300
RANSAC_THRESH_PX = 8.0
GATE_MIN_INLIER_RATIO = 0.15
GATE_MAX_MEDIAN_RESIDUAL = 12.0

SEQUENCES = [
    ("self_seq1", HUB / "extracted_frames1" / "extracted_frames"),
    ("self_seq2", HUB / "extracted_frames2" / "extracted_frames"),
    ("self_seq3", HUB / "extracted_frames3" / "extracted_frames"),
    ("self_seq4", HUB / "extracted_frames4" / "extracted_frames"),
    ("self_seq5", HUB / "extracted_frames5" / "extracted_frames"),
    ("self_seq6", HUB / "extracted_frames6" / "extracted_frames"),
]

SPECS = [
    ("weak_texture", "texture_mean_grad", True, "Weak texture (low mean |∇I|)"),
    ("specular", "highlight_fraction", False, "Specular reflection (highlight fraction)"),
    ("deformation", "mask_iou_change", False, "Large deformation (mask-IoU change)"),
    ("rapid_motion", "flow_mean_px", False, "Rapid camera motion (mean flow)"),
    ("motion_blur", "laplacian_var", True, "Motion blur (low Laplacian variance)"),
    ("occlusion", "support_change", False, "Partial occlusion (support change)"),
]


def list_images(directory: Path) -> list[Path]:
    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"No images in {directory}")
    return paths


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    scale = RESIZE_WIDTH / image.shape[1]
    return cv2.resize(image, (RESIZE_WIDTH, round(image.shape[0] * scale)))


def structural_mask(gray: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    return magnitude > np.percentile(magnitude, 90.0), float(magnitude.mean()), magnitude


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def height_field_from_magnitude(magnitude: np.ndarray) -> np.ndarray:
    threshold = np.percentile(magnitude, 90.0)
    clipped = np.where(magnitude > threshold, magnitude, 0.0)
    maximum = float(clipped.max())
    return (clipped / maximum * 96.0) if maximum > 1e-8 else clipped


def kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = ct - rotation @ cs
    return transform


def kabsch_ransac(source: np.ndarray, target: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(source)
    best_inliers = None
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(n, size=min(6, n), replace=False)
        transform = kabsch(source[idx], target[idx])
        residual = np.linalg.norm(
            (transform[:3, :3] @ source.T).T + transform[:3, 3] - target, axis=1
        )
        inliers = residual < RANSAC_THRESH_PX
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    transform = kabsch(source[best_inliers], target[best_inliers])
    residual = np.linalg.norm(
        (transform[:3, :3] @ source[best_inliers].T).T
        + transform[:3, 3]
        - target[best_inliers],
        axis=1,
    )
    return transform, float(best_inliers.sum() / n), float(np.median(residual))


def highlight_contamination(gray: np.ndarray) -> float:
    # Same HCR definition as evaluate_contour_hcr.py (top-5% intensity).
    highlight = gray >= np.percentile(gray, 95.0)
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    contour = magnitude > np.percentile(magnitude, 90.0)
    contour = cv2.morphologyEx(
        contour.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=2,
    ).astype(bool)
    npc = int(contour.sum())
    if npc == 0:
        return float("nan")
    hpc = int(np.logical_and(contour, highlight).sum())
    return hpc / npc


def pair_proxies(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    prev_mask, _, _ = structural_mask(previous)
    cur_mask, texture, _ = structural_mask(current)
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    _, prev_otsu = cv2.threshold(previous, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {
        "texture_mean_grad": texture,
        "highlight_fraction": float((current >= 243).mean()),
        "mask_iou_change": 1.0 - iou(prev_mask, cur_mask),
        "flow_mean_px": float(np.linalg.norm(flow, axis=2).mean()),
        "laplacian_var": float(cv2.Laplacian(current, cv2.CV_32F).var()),
        "support_change": abs(float((otsu > 0).mean()) - float((prev_otsu > 0).mean())),
        "flow": flow,
    }


def alignment_diagnostics(
    previous: np.ndarray, current: np.ndarray, flow: np.ndarray, seed: int
) -> dict[str, float]:
    _, _, prev_mag = structural_mask(previous)
    _, _, cur_mag = structural_mask(current)
    previous_height = height_field_from_magnitude(prev_mag)
    current_height = height_field_from_magnitude(cur_mag)
    grid_y, grid_x = np.mgrid[4 : previous.shape[0] : 8, 4 : previous.shape[1] : 8]
    xy_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
    flow_at = flow[grid_y.ravel(), grid_x.ravel()]
    xy_cur = xy_grid + flow_at
    valid = (
        (xy_cur[:, 0] >= 1)
        & (xy_cur[:, 0] < current.shape[1] - 2)
        & (xy_cur[:, 1] >= 1)
        & (xy_cur[:, 1] < current.shape[0] - 2)
    )
    src_xy = xy_cur[valid]
    tgt_xy = xy_grid[valid]
    if len(src_xy) < 6:
        return {
            "kabsch_median_residual_px": float("nan"),
            "ransac_inlier_ratio": float("nan"),
            "ransac_median_residual_px": float("nan"),
            "gate_freeze": 1.0,
            "n_points": 0,
        }
    src_h = cv2.remap(
        current_height,
        src_xy[:, 0].reshape(-1, 1),
        src_xy[:, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).reshape(-1)
    tgt_h = previous_height[tgt_xy[:, 1].astype(int), tgt_xy[:, 0].astype(int)]
    source = np.column_stack([src_xy, src_h])
    target = np.column_stack([tgt_xy, tgt_h])
    transform = kabsch(source, target)
    residual = np.linalg.norm(
        (transform[:3, :3] @ source.T).T + transform[:3, 3] - target, axis=1
    )
    _, inlier_ratio, ransac_med = kabsch_ransac(source, target, seed=seed)
    gate_freeze = float(
        inlier_ratio < GATE_MIN_INLIER_RATIO or ransac_med > GATE_MAX_MEDIAN_RESIDUAL
    )
    return {
        "kabsch_median_residual_px": float(np.median(residual)),
        "ransac_inlier_ratio": inlier_ratio,
        "ransac_median_residual_px": ransac_med,
        "gate_freeze": gate_freeze,
        "n_points": int(len(source)),
    }


def severity_vector(rows: list[dict], key: str, invert: bool) -> np.ndarray:
    raw = np.array([r[key] for r in rows], dtype=np.float64)
    return -raw if invert else raw


def main() -> None:
    all_pairs: list[dict] = []
    per_sequence: dict[str, list[dict]] = {}
    for name, directory in SEQUENCES:
        paths = list_images(directory)
        print(f"[proxies] {name} n={len(paths)}")
        previous = load_gray(paths[0])
        rows = []
        for index, path in enumerate(paths[1:], start=1):
            current = load_gray(path)
            proxies = pair_proxies(previous, current)
            proxies.pop("flow")
            rows.append({"sequence": name, "frame_idx": index, **proxies})
            previous = current
        per_sequence[name] = rows
        all_pairs.extend(rows)

    pair_path = RESULTS / "self_acquired_case_pairs.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with pair_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_pairs[0]))
        writer.writeheader()
        writer.writerows(all_pairs)

    pool = {key: severity_vector(all_pairs, field, invert) for key, field, invert, _ in SPECS}
    pool_mean = {k: float(v.mean()) for k, v in pool.items()}
    pool_std = {k: float(v.std(ddof=1)) for k, v in pool.items()}

    cases = []
    for name, _directory in SEQUENCES:
        rows = per_sequence[name]
        zscores = {}
        for key, field, invert, label in SPECS:
            sev = severity_vector(rows, field, invert)
            zscores[key] = {
                "label": label,
                "mean": float(sev.mean()),
                "z": float((sev.mean() - pool_mean[key]) / (pool_std[key] + 1e-12)),
            }
        dominant_key = max(zscores, key=lambda k: zscores[k]["z"])
        dominant_field = next(s[1] for s in SPECS if s[0] == dominant_key)
        dominant_invert = next(s[2] for s in SPECS if s[0] == dominant_key)
        dominant_label = zscores[dominant_key]["label"]
        severity = severity_vector(rows, dominant_field, dominant_invert)
        if len(rows) + 1 < WINDOW:
            raise RuntimeError(f"{name}: only {len(rows)+1} frames, need {WINDOW}")
        best_start = 0
        best_score = -1e18
        # rows[i] is the pair ending at frame i (1-based). Window of 80 frames
        # uses pairs [start, start+78] where start is 0-based pair index.
        n_pairs_needed = WINDOW - 1
        for start in range(0, len(rows) - n_pairs_needed + 1):
            score = float(severity[start : start + n_pairs_needed].mean())
            if score > best_score:
                best_score = score
                best_start = start
        directory = next(d for n, d in SEQUENCES if n == name)
        paths = list_images(directory)
        window_paths = paths[best_start : best_start + WINDOW]
        residuals = []
        inliers = []
        ransac_res = []
        freezes = []
        hcrs = []
        print(f"[align] {name} window_start_frame={best_start} dominant={dominant_key}")
        previous = load_gray(window_paths[0])
        hcrs.append(highlight_contamination(previous))
        for offset, path in enumerate(window_paths[1:], start=1):
            current = load_gray(path)
            proxies = pair_proxies(previous, current)
            flow = proxies.pop("flow")
            diag = alignment_diagnostics(previous, current, flow, seed=offset)
            residuals.append(diag["kabsch_median_residual_px"])
            inliers.append(diag["ransac_inlier_ratio"])
            ransac_res.append(diag["ransac_median_residual_px"])
            freezes.append(diag["gate_freeze"])
            hcrs.append(highlight_contamination(current))
            previous = current
        case = {
            "sequence": name,
            "n_frames_available": len(rows) + 1,
            "window_frames": WINDOW,
            "window_start_frame": best_start,
            "window_end_frame": best_start + WINDOW - 1,
            "dominant_key": dominant_key,
            "dominant_label": dominant_label,
            "dominant_z": zscores[dominant_key]["z"],
            "zscores": zscores,
            "window_severity_mean": best_score,
            "kabsch_median_residual_px": float(np.nanmean(residuals)),
            "ransac_inlier_ratio": float(np.nanmean(inliers)),
            "ransac_median_residual_px": float(np.nanmean(ransac_res)),
            "gate_freezes": int(np.nansum(freezes)),
            "gate_pairs": len(residuals),
            "hcr_ours_p90": float(np.nanmean(hcrs)),
            "note": (
                "Existing self-acquired corpus; no camera-pose ground truth; "
                "not a newly collected hard-sequence set."
            ),
        }
        cases.append(case)
        print(
            f"  residual={case['kabsch_median_residual_px']:.3f}px  "
            f"inlier={case['ransac_inlier_ratio']:.3f}  "
            f"gates={case['gate_freezes']}/{case['gate_pairs']}  "
            f"HCR={case['hcr_ours_p90']:.3f}"
        )

    summary = {
        "note": (
            "Six 80-frame windows from the existing self-acquired training corpus "
            "(Table 1). Not a new camera-pose ground-truth set."
        ),
        "resize_width": RESIZE_WIDTH,
        "window_frames": WINDOW,
        "n_sequences": len(cases),
        "pool_pairs": len(all_pairs),
        "cases": cases,
    }
    summary_path = RESULTS / "self_acquired_case_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", pair_path)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
