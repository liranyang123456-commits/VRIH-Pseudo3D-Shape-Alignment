#!/usr/bin/env python3
"""Empirical percentile-threshold sweep for the deformation-detection stage (R2.3).

On the same 3,482-frame pool used by analyze_gradient_statistics.py
(self-acquired clips + SCARED), sweep the gradient-magnitude percentile
threshold p in {50, 60, 68, 75, 80, 85, 90, 95} and measure, per frame:

  1. energy_share(p): fraction of total gradient magnitude carried by the
     retained pixels (mag > t_p) -- the "information coverage" of the mask.
  2. tnorm(p) = t_p / max(mag): normalized threshold, for cross-frame CV.
  3. hcr(p): fraction of retained pixels falling in the specular-highlight
     mask (grayscale >= 95th intensity percentile) -- contamination proxy.
  4. dice(p): Dice overlap of the retained mask with the previous frame of
     the same sequence -- cross-frame repeatability under identical motion.
  5. mean_rank: percentile rank of the frame-mean gradient magnitude in the
     magnitude distribution -- if this sits near 68, t_main coincides with
     an "above-average information" cut.

Aggregates (mean / median / CV across frames) are written to JSON + CSV and
an energy-coverage figure is saved.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HUB = ROOT.parent / "vrih_experiment_hub" / "links"

FRAME_DIRS = [
    HUB / "self_acquired_endoscopy" / f"extracted_frames{i}" / "extracted_frames" for i in range(1, 7)
] + [
    ROOT / "data" / s / "images"
    for s in ("scared_d1_k1", "scared_d1_k2", "scared_d2_k1", "scared_d2_k2", "scared_d3_k1", "scared_d3_k2")
]

PERCENTILES = [50, 60, 68, 75, 80, 85, 90, 95]


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def main() -> None:
    per_frame = []
    example_curve = None
    for directory in FRAME_DIRS:
        if not directory.exists():
            print("[skip]", directory)
            continue
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg"})
        prev_masks: dict[int, np.ndarray] = {}
        for path in paths:  # full pool: 3,482 frames, same as gradient_statistics.json
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            mag = gradient_magnitude(gray)
            flat = mag.ravel()
            total = float(flat.sum()) + 1e-12
            mmax = float(flat.max()) + 1e-12
            highlight = gray >= np.percentile(gray, 95.0)

            row = {"frame": f"{directory.parent.name}/{path.name}"}
            row["mean_rank"] = float(np.mean(flat < flat.mean()) * 100.0)
            thresholds = np.percentile(flat, PERCENTILES)
            for p, t in zip(PERCENTILES, thresholds):
                mask = mag > t
                row[f"energy_share_p{p}"] = float(flat[flat > t].sum() / total)
                row[f"tnorm_p{p}"] = float(t / mmax)
                n_ret = int(mask.sum())
                row[f"hcr_p{p}"] = float((mask & highlight).sum() / max(n_ret, 1))
                prev = prev_masks.get(p)
                if prev is not None and prev.shape == mask.shape:
                    inter = float((mask & prev).sum())
                    row[f"dice_p{p}"] = 2.0 * inter / max(n_ret + int(prev.sum()), 1)
                prev_masks[p] = mask
            per_frame.append(row)
            if example_curve is None and "scared_d1_k1" in row["frame"]:
                dense_p = np.arange(0, 100)
                dense_t = np.percentile(flat, dense_p)
                example_curve = (dense_p, [float(flat[flat > t].sum() / total) for t in dense_t])

    n = len(per_frame)
    print(f"frames analyzed: {n}")

    def collect(key):
        return np.array([r[key] for r in per_frame if key in r])

    summary: dict[str, object] = {"n_frames": n}
    mr = collect("mean_rank")
    summary["mean_rank_median"] = float(np.median(mr))
    summary["mean_rank_iqr"] = [float(np.percentile(mr, 25)), float(np.percentile(mr, 75))]
    for p in PERCENTILES:
        es = collect(f"energy_share_p{p}")
        tn = collect(f"tnorm_p{p}")
        hc = collect(f"hcr_p{p}")
        dc = collect(f"dice_p{p}")
        summary[f"p{p}"] = {
            "energy_share_median": float(np.median(es)),
            "tnorm_cv": float(np.std(tn) / (np.mean(tn) + 1e-12)),
            "hcr_median": float(np.median(hc)),
            "dice_median": float(np.median(dc)) if dc.size else None,
        }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "threshold_sweep.json").write_text(json.dumps({"summary": summary, "per_frame": per_frame}, indent=1))
    with open(RESULTS / "threshold_sweep_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["percentile", "energy_share_median", "tnorm_cv", "hcr_median", "dice_median"])
        for p in PERCENTILES:
            s = summary[f"p{p}"]
            writer.writerow([p, s["energy_share_median"], s["tnorm_cv"], s["hcr_median"], s["dice_median"]])
    print(json.dumps(summary, indent=1))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    med_curve = [summary[f"p{p}"]["energy_share_median"] for p in PERCENTILES]
    axes[0].plot(PERCENTILES, med_curve, "o-", label="median over frames")
    if example_curve is not None:
        axes[0].plot(example_curve[0], example_curve[1], "-", alpha=0.4, label="example frame")
    axes[0].axvline(68, color="k", ls=":", label="p68 (t_main)")
    axes[0].axvline(90, color="gray", ls=":", label="p90 (t_90)")
    axes[0].set(xlabel="Percentile threshold p", ylabel="Retained gradient-energy share",
                title="Information coverage vs. threshold")
    axes[0].legend()
    axes[1].hist(mr, bins=60, density=True, alpha=0.75)
    axes[1].axvline(68, color="k", ls=":", label="p68")
    axes[1].axvline(float(np.median(mr)), color="r", ls="-", label="median rank of frame mean")
    axes[1].set(xlabel="Percentile rank of frame-mean |gradient|", ylabel="Density",
                title="Where the mean gradient sits in the distribution")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RESULTS / "fig_threshold_sweep.png", dpi=300)
    print("wrote fig_threshold_sweep.png")


if __name__ == "__main__":
    main()
