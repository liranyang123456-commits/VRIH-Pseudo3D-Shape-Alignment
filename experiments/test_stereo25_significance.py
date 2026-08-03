"""Paired significance for the 25-keyframe stereo validation.

Ours-selective vs SIFT and vs the new SIFT+mutual-gated control on
gt_match_precision and depth_absrel (per-keyframe paired Wilcoxon).
Writes stereo25_significance.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).parent / "results"
SRC = RESULTS / "scared_stereo_density_quality_full25.csv"

PAIRS = [("SIFT", "Ours-selective"), ("SIFT+mutual-gated", "Ours-selective")]
METRICS = ["gt_match_precision", "depth_absrel", "depth_rmse"]


def main() -> None:
    rows = list(csv.DictReader(SRC.open()))
    data: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        data[(r["keyframe"], r["method"])] = {
            m: float(r[m]) for m in METRICS if r[m] not in ("", "nan")
        }
    keyframes = sorted({k for k, _ in data})
    report = {}
    for base, ours in PAIRS:
        entry = {}
        for metric in METRICS:
            xs, ys = [], []
            for kf in keyframes:
                a = data.get((kf, ours), {}).get(metric)
                b = data.get((kf, base), {}).get(metric)
                if a is None or b is None or np.isnan(a) or np.isnan(b):
                    continue
                xs.append(a)
                ys.append(b)
            w = stats.wilcoxon(xs, ys)
            entry[metric] = {
                "n_pairs": len(xs),
                "ours_mean": float(np.mean(xs)),
                "baseline_mean": float(np.mean(ys)),
                "wilcoxon_p": float(w.pvalue),
                "ours_better_fraction": float(np.mean([1.0 if (a < b) == ("depth" in metric or "absrel" in metric or "rmse" in metric) else (a > b) for a, b in zip(xs, ys)])),
            }
        report[f"{ours}_vs_{base}"] = entry
        for metric in METRICS:
            e = entry[metric]
            print(f"{ours} vs {base} | {metric}: p={e['wilcoxon_p']:.4g} (n={e['n_pairs']})")
    (RESULTS / "stereo25_significance.json").write_text(json.dumps(report, indent=1))
    print("written stereo25_significance.json")


if __name__ == "__main__":
    main()
