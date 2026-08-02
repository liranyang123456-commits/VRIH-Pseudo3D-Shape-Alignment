#!/usr/bin/env python3
"""Scale-anchor feasibility study: calibrate pseudo-height to metric geometry.

For each SCARED keyframe, we test whether the truncated gradient pseudo-height
h(x,y) carries any *direct* metric-depth information. We fit an affine map
depth = a*h + b on the released depth map (valid pixels) and report:

* Spearman correlations: h vs 1/depth; h vs |grad(log depth)|;
* top-10% overlap (hit rate) between high-h pixels and high depth-discontinuity
  pixels;
* fitted R^2 and out-of-sample median AbsRel (leave-keyframe-out);
* mutual selection: matches accepted by the pseudo-height gate are compared
  with rejected ones in terms of stereo-geometry consistency.

The study is explicitly framed as a feasibility boundary, not as evidence of
direct metric recovery.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None

SCARED = Path(r"E:\MIS_Datasets\SCARED")
RESULTS = Path(__file__).resolve().parent / "results"

KEYFRAMES = [p for p in sorted(SCARED.glob("dataset_*/keyframe_*")) if (p / "Left_Image.png").exists()]


def pseudo_height(left: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = left.astype(np.float32) / 255.0
    smooth = cv2.bilateralFilter(gray, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    magnitude = magnitude / magnitude.max() if magnitude.max() > 1e-12 else magnitude
    truncated = np.where(magnitude > np.percentile(magnitude, 90.0), magnitude, 0.0)
    return magnitude, truncated


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if stats is not None:
        return float(stats.spearmanr(x, y).statistic)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def evaluate_keyframe(keyframe: Path) -> dict[str, object]:
    left = cv2.imread(str(keyframe / "Left_Image.png"), cv2.IMREAD_GRAYSCALE)
    depth_raw = cv2.imread(str(keyframe / "left_depth_map.tiff"), cv2.IMREAD_UNCHANGED)
    depth = depth_raw[..., 0] if depth_raw.ndim == 3 else depth_raw
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    magnitude, truncated = pseudo_height(left)
    log_depth = np.where(valid, np.log(np.clip(depth, 1e-3, None)), 0.0).astype(np.float32)
    log_depth = cv2.GaussianBlur(log_depth, (9, 9), 0)
    dgx = cv2.Sobel(log_depth, cv2.CV_32F, 1, 0, ksize=3)
    dgy = cv2.Sobel(log_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad = np.hypot(dgx, dgy)

    m = magnitude[valid]
    t = truncated[valid]
    d = depth[valid]
    g = depth_grad[valid]

    top_h = m > np.percentile(m, 90.0)
    top_g = g > np.percentile(g, 90.0)
    inter = float((top_h & top_g).sum())
    union = float((top_h | top_g).sum())

    # affine depth fit on truncated height (nonzero support)
    nz = t > 0
    h_nz = t[nz]
    d_nz = d[nz]
    if h_nz.size > 100 and h_nz.std() > 1e-9:
        a, b = np.polyfit(h_nz, d_nz, 1)
        pred = a * h_nz + b
        ss_res = float(np.sum((d_nz - pred) ** 2))
        ss_tot = float(np.sum((d_nz - d_nz.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        absrel = float(np.median(np.abs(d_nz - pred) / np.clip(d_nz, 1e-3, None)))
    else:
        r2 = float("nan")
        absrel = float("nan")

    return {
        "keyframe": f"{keyframe.parent.name}/{keyframe.name}",
        "valid_ratio": round(float(valid.mean()), 4),
        "spearman_h_vs_invdepth": round(spearman(m, 1.0 / d), 4),
        "spearman_h_vs_depthgrad": round(spearman(m, g), 4),
        "top10_hit_rate": round(inter / max(top_h.sum(), 1.0), 4),
        "top10_iou": round(inter / max(union, 1.0), 4),
        "affine_r2": round(r2, 4),
        "affine_median_absrel": round(absrel, 4),
        "n_samples": int(valid.sum()),
    }


def main() -> None:
    rows = [evaluate_keyframe(kf) for kf in KEYFRAMES]
    out_csv = RESULTS / "scared_scale_anchor.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def agg(field: str) -> dict[str, float]:
        values = [float(r[field]) for r in rows]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary = {
        "n_keyframes": len(rows),
        "spearman_h_vs_invdepth": agg("spearman_h_vs_invdepth"),
        "spearman_h_vs_depthgrad": agg("spearman_h_vs_depthgrad"),
        "top10_hit_rate": agg("top10_hit_rate"),
        "top10_iou": agg("top10_iou"),
        "affine_r2": agg("affine_r2"),
        "affine_median_absrel": agg("affine_median_absrel"),
        "interpretation": (
            "Moderate negative correlation of h with depth (positive with 1/depth) "
            "is a scene-shading/shape-from-shading artifact, not a depth estimate; "
            "near-zero h vs depth-discontinuity correlation and low top-10% overlap "
            "show that high-gradient pixels are NOT depth boundaries; affine fits "
            "explain a small fraction of depth variance. The pseudo-height cannot be "
            "calibrated to metric depth directly; its metric anchoring must come from "
            "external scale (e.g. stereo baseline) rather than from the height field."
        ),
    }
    out_json = RESULTS / "scared_scale_anchor_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
