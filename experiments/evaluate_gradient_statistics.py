#!/usr/bin/env python3
"""Empirical gradient statistics on real endoscopic images (R2.3 evidence).

On 3,002 self-acquired endoscopic frames and 480 SCARED frames, compute:

1. Component normality: skewness and excess kurtosis of the Gaussian-smoothed
   horizontal/vertical gradient components Gx, Gy (Gaussian model check).
2. Magnitude distribution: skew/kurtosis of |grad I| plus histogram fit
   comparison of a Gaussian model vs a Rayleigh model (the theoretically
   correct model when components are approximately Gaussian).
3. Threshold stability: distribution of the t90 value across frames/sequences.
4. Threshold robustness: Dice overlap between contours extracted at
   t85/t90/t95 (sensitivity of the contour set to the percentile choice).

Outputs gradient_statistics.json and fig_gradient_statistics.png.
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
SELF_ACQUIRED = [
    ROOT.parent / "vrih_experiment_hub" / "links" / "self_acquired_endoscopy" / f"extracted_frames{i}" / "extracted_frames"
    for i in range(1, 7)
]
SCARED = [ROOT / "data" / s / "images" for s in ["scared_d1_k1", "scared_d1_k2", "scared_d2_k1", "scared_d2_k2", "scared_d3_k1", "scared_d3_k2"]]


def gradients(gray: np.ndarray):
    g = gray.astype(np.float32) / 255.0
    smooth = cv2.bilateralFilter(g, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy, np.hypot(gx, gy)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return float(2.0 * inter / total) if total > 0 else 1.0


def analyze_frame(path: Path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    gx, gy, mag = gradients(gray)
    gx_v, mag_v = gx.ravel(), mag.ravel()
    # Rayleigh MLE: sigma = sqrt(mean(m^2)/2); Gaussian MLE on magnitude
    sigma_r = float(np.sqrt(np.mean(mag_v**2) / 2.0))
    mu_g, sigma_g = float(mag_v.mean()), float(mag_v.std())
    # histogram log-likelihood comparison (per-pixel mean log-likelihood)
    eps = 1e-12
    ll_ray = float(stats.rayleigh.logpdf(mag_v, scale=sigma_r).mean())
    ll_gau = float(stats.norm.logpdf(mag_v, loc=mu_g, scale=max(sigma_g, eps)).mean())
    t85, t90, t95 = (np.percentile(mag_v, p) for p in (85, 90, 95))
    c85, c90, c95 = mag_v > t85, mag_v > t90, mag_v > t95
    return {
        "gx_skew": float(stats.skew(gx_v)),
        "gx_kurt": float(stats.kurtosis(gx_v)),
        "gy_skew": float(stats.skew(gy.ravel())),
        "gy_kurt": float(stats.kurtosis(gy.ravel())),
        "mag_skew": float(stats.skew(mag_v)),
        "mag_kurt": float(stats.kurtosis(mag_v)),
        "ll_rayleigh": ll_ray,
        "ll_gauss": ll_gau,
        "t90": float(t90),
        "dice_85_90": dice(c85, c90),
        "dice_90_95": dice(c90, c95),
    }


def main() -> None:
    per_frame = []
    per_seq_t90: dict[str, list[float]] = {}
    example = None
    for group, dirs in (("self", SELF_ACQUIRED), ("scared", SCARED)):
        for d in dirs:
            paths = sorted([p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".png")])
            if not paths:
                print("skip empty", d)
                continue
            for p in paths:
                r = analyze_frame(p)
                if r is None:
                    continue
                r["group"] = group
                r["seq"] = d.parent.name if group == "scared" else d.parent.parent.name
                per_frame.append(r)
                per_seq_t90.setdefault(r["seq"], []).append(r["t90"])
                if example is None and group == "self":
                    example = (p, r)
            print(f"{d.parent.name}: {len(paths)} frames done", flush=True)

    def agg(field: str) -> dict[str, float]:
        v = np.array([r[field] for r in per_frame])
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}

    n = len(per_frame)
    ray_wins = sum(1 for r in per_frame if r["ll_rayleigh"] > r["ll_gauss"])
    seq_stability = {s: {"mean": float(np.mean(v)), "cv": float(np.std(v) / max(np.mean(v), 1e-12)), "n": len(v)} for s, v in per_seq_t90.items()}
    summary = {
        "n_frames": n,
        "component_skew_abs_mean": float(np.mean([abs(r["gx_skew"]) for r in per_frame] + [abs(r["gy_skew"]) for r in per_frame])),
        "component_kurt": agg("gx_kurt"),
        "magnitude_skew": agg("mag_skew"),
        "magnitude_kurt": agg("mag_kurt"),
        "rayleigh_better_fraction": ray_wins / n,
        "dice_85_90": agg("dice_85_90"),
        "dice_90_95": agg("dice_90_95"),
        "t90_per_sequence": seq_stability,
    }
    (RESULTS / "gradient_statistics.json").write_text(json.dumps({"summary": summary, "per_frame": per_frame[:200]}, indent=1))
    print(json.dumps(summary, indent=1)[:1200])

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # (a) example component histogram with Gaussian fit
    p, _ = example
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    gx, gy, mag = gradients(gray)
    ax = axes[0]
    ax.hist(gx.ravel(), bins=200, density=True, alpha=0.6, label="Gx (real)")
    xs = np.linspace(gx.min(), gx.max(), 400)
    ax.plot(xs, stats.norm.pdf(xs, 0, gx.std()), "r-", label="Gaussian fit")
    ax.set_title("Gradient component vs Gaussian")
    ax.set_xlabel("Gx response")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    # (b) magnitude histogram with Gaussian vs Rayleigh
    ax = axes[1]
    ax.hist(mag.ravel(), bins=200, density=True, alpha=0.6, label="|grad I| (real)")
    xs = np.linspace(0, mag.max(), 400)
    sigma_r = np.sqrt(np.mean(mag.ravel() ** 2) / 2.0)
    ax.plot(xs, stats.rayleigh.pdf(xs, scale=sigma_r), "r-", label="Rayleigh fit")
    ax.plot(xs, stats.norm.pdf(xs, mag.mean(), mag.std()), "g--", label="Gaussian fit")
    ax.set_title("Gradient magnitude: Rayleigh vs Gaussian")
    ax.set_xlabel("|grad I|")
    ax.legend(fontsize=8)
    # (c) per-sequence t90 stability + dice
    ax = axes[2]
    seqs = sorted(seq_stability)
    cvs = [seq_stability[s]["cv"] for s in seqs]
    ax.bar(range(len(seqs)), cvs)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels([s.replace("extracted_frames", "S").replace("scared_", "SC-") for s in seqs], rotation=45, fontsize=7)
    ax.set_ylabel("CV of t90 across frames")
    ax.set_title("Threshold stability per sequence")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_gradient_statistics.png", dpi=300)
    print("wrote fig_gradient_statistics.png")


if __name__ == "__main__":
    main()
