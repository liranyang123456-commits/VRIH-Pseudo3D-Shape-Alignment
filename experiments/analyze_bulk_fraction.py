#!/usr/bin/env python3
"""Estimate the per-frame 'smooth bulk' fraction of gradient responses (R2.3).

Question: what fraction of pixels lies at or below the noise floor of the
gradient field? If this fraction clusters near 0.68, the 68th-percentile
detection threshold acquires an interpretable anchor (t_main ~ bulk fraction).

Noise floor estimation (robust, edge-insensitive):
  sigma_n = 1.4826 * MAD(Gx)   (components are zero-mean; edges are sparse,
                                so MAD estimates the noise/texture floor)
  Under a zero-mean bivariate noise model the magnitude of pure-noise pixels
  is Rayleigh(sigma_n); its 99th percentile is sigma_n*sqrt(2 ln 100) ~ 3.03.
  bulk_fraction = P(|grad| <= 3 sigma_n).

Run on every 3rd frame of the same pool as the other R2.3 analyses.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
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


def main() -> None:
    fractions = []
    per_seq: dict[str, list[float]] = {}
    for directory in FRAME_DIRS:
        if not directory.exists():
            print("[skip]", directory)
            continue
        seq = directory.parent.name
        paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg"})
        for path in paths[::3]:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            filtered = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
            gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            sigma_n = 1.4826 * float(np.median(np.abs(gx - np.median(gx))))
            if sigma_n <= 0:
                continue
            frac = float(np.mean(mag <= 3.0 * sigma_n))
            fractions.append(frac)
            per_seq.setdefault(seq, []).append(frac)

    arr = np.array(fractions)
    summary = {
        "n_frames": int(arr.size),
        "bulk_fraction_median": float(np.median(arr)),
        "bulk_fraction_iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "bulk_fraction_p05_p95": [float(np.percentile(arr, 5)), float(np.percentile(arr, 95))],
        "per_sequence_median": {k: float(np.median(v)) for k, v in sorted(per_seq.items())},
    }
    (RESULTS / "bulk_fraction.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
