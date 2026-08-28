#!/usr/bin/env python3
"""Empirical gradient statistics on real endoscopic images (R2.3).

Tests on a large frame pool (self-acquired clips + SCARED):
  1. Are the bilateral-filtered Gaussian-derivative components Gx, Gy
     approximately symmetric zero-mean (bulk-Gaussian)? -> skewness, excess
     kurtosis, D'Agostino-Pearson normal test.
  2. Is the gradient MAGNITUDE Gaussian? (expected: no, heavy-tailed)
     -> skewness/kurtosis/normal test + Rayleigh-fit KS distance.
  3. Stability of the t90 percentile threshold across frames (CV of the
     normalized threshold) — supports a fixed percentile rule.

Outputs scared/contour aggregate CSV + a diagnostic figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HUB = ROOT.parent / "vrih_experiment_hub" / "links"

FRAME_DIRS = [
    HUB / "self_acquired_endoscopy" / f"extracted_frames{i}" / "extracted_frames" for i in range(1, 7)
] + [ROOT / "data" / s / "images" for s in ("scared_d1_k1", "scared_d2_k1", "scared_d3_k1")]


def gradient_components(gray: np.ndarray):
    filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def main() -> None:
    per_frame = []
    example = None
    for directory in FRAME_DIRS:
        if not directory.exists():
            print("[skip]", directory)
            continue
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg"})
        for path in paths[::3]:  # every 3rd frame
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            gx, gy = gradient_components(gray)
            mag = np.hypot(gx, gy).ravel()
            gxv = gx.ravel()
            t90 = float(np.percentile(mag, 90.0))
            row = {
                "frame": f"{directory.parent.name}/{path.name}",
                "gx_skew": float(stats.skew(gxv)),
                "gx_kurtosis": float(stats.kurtosis(gxv)),
                "gx_normaltest_p": float(stats.normaltest(gxv[::97]).pvalue),
                "mag_skew": float(stats.skew(mag)),
                "mag_kurtosis": float(stats.kurtosis(mag)),
                "mag_normaltest_p": float(stats.normaltest(mag[::97]).pvalue),
                "t90_over_max": t90 / float(mag.max() + 1e-12),
            }
            # Rayleigh fit on magnitude (MLE: sigma = sqrt(mean(r^2)/2))
            sigma_r = float(np.sqrt(np.mean(mag**2) / 2.0))
            row["rayleigh_ks"] = float(stats.kstest(mag[::97], "rayleigh", args=(0, sigma_r)).statistic)
            per_frame.append(row)
            if example is None and "scared_d1_k1" in row["frame"]:
                example = (gxv, mag)
    n = len(per_frame)
    print(f"frames analyzed: {n}")

    def agg(key):
        v = np.array([r[key] for r in per_frame])
        return float(np.mean(v)), float(np.median(v)), float(np.std(v))

    summary = {
        "n_frames": n,
        "gx_skew_mean": agg("gx_skew")[0],
        "gx_kurtosis_mean": agg("gx_kurtosis")[0],
        "gx_normaltest_reject_fraction_p05": float(np.mean([r["gx_normaltest_p"] < 0.05 for r in per_frame])),
        "mag_skew_mean": agg("mag_skew")[0],
        "mag_kurtosis_mean": agg("mag_kurtosis")[0],
        "mag_normaltest_reject_fraction_p05": float(np.mean([r["mag_normaltest_p"] < 0.05 for r in per_frame])),
        "rayleigh_ks_median": float(np.median([r["rayleigh_ks"] for r in per_frame])),
        "t90_over_max_mean": agg("t90_over_max")[0],
        "t90_over_max_cv": agg("t90_over_max")[2] / agg("t90_over_max")[0],
    }
    (RESULTS / "gradient_statistics_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    # figure: example-frame component vs magnitude histograms
    if example is not None:
        gxv, mag = example
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(gxv, bins=200, density=True, alpha=0.7, label="Gx (empirical)")
        xs = np.linspace(gxv.min(), gxv.max(), 400)
        axes[0].plot(xs, stats.norm.pdf(xs, gxv.mean(), gxv.std()), "r-", label="Gaussian fit")
        axes[0].set_xlim(np.percentile(gxv, 0.5), np.percentile(gxv, 99.5))
        axes[0].set(xlabel="Gx response", ylabel="Density", title="Gradient component (bulk approx. symmetric)")
        axes[0].legend()
        axes[1].hist(mag, bins=200, density=True, alpha=0.7, label="|grad| (empirical)")
        xs = np.linspace(0, np.percentile(mag, 99.5), 400)
        axes[1].plot(xs, stats.norm.pdf(xs, mag.mean(), mag.std()), "r-", label="Gaussian fit")
        sigma_r = float(np.sqrt(np.mean(mag**2) / 2.0))
        axes[1].plot(xs, stats.rayleigh.pdf(xs, scale=sigma_r), "g--", label="Rayleigh fit")
        axes[1].axvline(np.percentile(mag, 90), color="k", ls=":", label="t90 threshold")
        axes[1].set(xlabel="|gradient|", ylabel="Density", title="Gradient magnitude (heavy-tailed, non-Gaussian)")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(RESULTS / "fig_gradient_statistics.png", dpi=300)
        print("wrote fig_gradient_statistics.png")


if __name__ == "__main__":
    main()
