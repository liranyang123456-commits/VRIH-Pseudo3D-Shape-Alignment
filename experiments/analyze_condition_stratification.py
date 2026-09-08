#!/usr/bin/env python3
"""Six-condition stratification on SCARED (R2.9), matching Table tab:conditions.

Per consecutive pair of the gradient-height trajectory (474 pairs):
  * weak texture          — severity = −mean |∇I|  (low gradient is harder)
  * specular reflection   — near-saturation highlight fraction (I ≥ 243)
  * large deformation     — 1 − IoU of t90 structural masks
  * rapid camera motion   — mean Farneback flow magnitude
  * motion blur           — severity = −Laplacian variance
  * partial occlusion     — |Δ| of Otsu foreground support

Median split on each severity; Spearman ρ of severity vs. relative rotation
error; plus hard-quartile subsets of the same 474 pairs (not new sequences).
Output: scared_condition_pairs.csv + scared_condition_summary.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEQUENCES = [
    "scared_d1_k1",
    "scared_d1_k2",
    "scared_d2_k1",
    "scared_d2_k2",
    "scared_d3_k1",
    "scared_d3_k2",
]
RESIZE_WIDTH = 320


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    scale = RESIZE_WIDTH / image.shape[1]
    return cv2.resize(image, (RESIZE_WIDTH, round(image.shape[0] * scale)))


def structural_mask(gray: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    return magnitude > np.percentile(magnitude, 90.0), float(magnitude.mean())


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def main() -> None:
    rows = []
    for seq in SEQUENCES:
        metrics = json.loads((RESULTS / f"{seq}_gradient" / "pose_metrics.json").read_text(encoding="utf-8"))
        rot_errors = metrics["series"]["rotation_errors_deg"]
        images = sorted((ROOT / "data" / seq / "images").glob("*.png"))[:80]
        previous = load_gray(images[0])
        prev_mask, _ = structural_mask(previous)
        _, prev_otsu = cv2.threshold(previous, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        prev_support = float((prev_otsu > 0).mean())
        for index, path in enumerate(images[1:], start=1):
            current = load_gray(path)
            cur_mask, texture = structural_mask(current)
            flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            _, otsu = cv2.threshold(current, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            support = float((otsu > 0).mean())
            laplacian_var = float(cv2.Laplacian(current, cv2.CV_32F).var())
            rows.append(
                {
                    "sequence": seq,
                    "frame_idx": index,
                    "rot_err_deg": float(rot_errors[index - 1]),
                    "texture_mean_grad": texture,
                    "highlight_fraction": float((current >= 243).mean()),
                    "mask_iou_change": 1.0 - iou(prev_mask, cur_mask),
                    "flow_mean_px": float(np.linalg.norm(flow, axis=2).mean()),
                    "laplacian_var": laplacian_var,
                    "support_change": abs(support - prev_support),
                }
            )
            previous, prev_mask, prev_support = current, cur_mask, support

    pair_path = RESULTS / "scared_condition_pairs.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rot = np.array([r["rot_err_deg"] for r in rows])
    specs = [
        ("weak_texture", "texture_mean_grad", True, "Weak texture (low mean |∇I|)"),
        ("specular", "highlight_fraction", False, "Specular reflection (highlight fraction)"),
        ("deformation", "mask_iou_change", False, "Large deformation (mask-IoU change)"),
        ("rapid_motion", "flow_mean_px", False, "Rapid camera motion (mean flow)"),
        ("motion_blur", "laplacian_var", True, "Motion blur (low Laplacian variance)"),
        ("occlusion", "support_change", False, "Partial occlusion (support change)"),
    ]
    report = {"n_pairs": len(rows), "resize_width": RESIZE_WIDTH}
    print(f"frame pairs: {len(rows)}")
    print(f"{'condition':42s} more   less    rho     p")
    for key, field, invert, label in specs:
        raw = np.array([r[field] for r in rows], dtype=np.float64)
        severity = -raw if invert else raw
        rho, p_value = stats.spearmanr(severity, rot)
        median = float(np.median(severity))
        more = rot[severity >= median]
        less = rot[severity < median]
        if invert:
            # more-affected = lower raw value; equivalent to severity >= median
            pass
        entry = {
            "label": label,
            "more_affected_mean_rot": float(more.mean()),
            "less_affected_mean_rot": float(less.mean()),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "n_more": int(more.size),
            "n_less": int(less.size),
        }
        report[key] = entry
        stars = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
        print(
            f"{key:16s} {more.mean():.3f}  {less.mean():.3f}  "
            f"{rho:+.2f}{stars:2s}  {p_value:.4g}"
        )

    # Hard-quartile subsets of the same 474 pairs (not new sequences).
    deformation = np.array([r["mask_iou_change"] for r in rows], dtype=np.float64)
    flow = np.array([r["flow_mean_px"] for r in rows], dtype=np.float64)
    q_def = float(np.quantile(deformation, 0.75))
    q_flow = float(np.quantile(flow, 0.75))
    mask_def = deformation >= q_def
    mask_flow = flow >= q_flow
    mask_union = mask_def | mask_flow
    mask_inter = mask_def & mask_flow
    hard = {
        "note": "Upper quartile of the same 474 SCARED pairs; not newly collected sequences.",
        "all_pairs": {"n": int(rot.size), "mean_rot": float(rot.mean())},
        "deformation_q4": {
            "n": int(mask_def.sum()),
            "threshold": q_def,
            "mean_rot": float(rot[mask_def].mean()),
        },
        "rapid_motion_q4": {
            "n": int(mask_flow.sum()),
            "threshold": q_flow,
            "mean_rot": float(rot[mask_flow].mean()),
        },
        "union_q4": {
            "n": int(mask_union.sum()),
            "mean_rot": float(rot[mask_union].mean()),
        },
        "intersection_q4": {
            "n": int(mask_inter.sum()),
            "mean_rot": float(rot[mask_inter].mean()),
        },
        "complement_of_union": {
            "n": int((~mask_union).sum()),
            "mean_rot": float(rot[~mask_union].mean()),
        },
    }
    report["hard_quartile"] = hard
    print("hard quartile (same 474 pairs; not new sequences)")
    for key in (
        "all_pairs",
        "deformation_q4",
        "rapid_motion_q4",
        "union_q4",
        "intersection_q4",
        "complement_of_union",
    ):
        entry = hard[key]
        print(f"  {key:22s} n={entry['n']:3d}  mean={entry['mean_rot']:.3f}")

    summary_path = RESULTS / "scared_condition_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", pair_path)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
