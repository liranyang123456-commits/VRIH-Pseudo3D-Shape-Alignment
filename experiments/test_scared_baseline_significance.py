"""Paired significance tests: gradient-height Ours vs external baselines on SCARED.

Compares per-sequence relative rotation error (n=6 matched sequences) between
the gradient-height variant and each external baseline (Reloc3r, SIFT, AKAZE).
Writes scared_baseline_significance.json next to the input CSV.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from scipy import stats

RESULTS = Path(r"E:/elsarticle-template-TMI_Revised/experiments/results")
SRC = RESULTS / "scared_multi_sequence_summary.csv"
OUT = RESULTS / "scared_baseline_significance.json"


def main() -> None:
    rows = list(csv.DictReader(SRC.open()))
    by_method: dict[str, dict[str, float]] = {}
    for r in rows:
        method = r["method"]
        seq = r["sequence"]
        try:
            by_method.setdefault(method, {})[seq] = float(r["relative_rotation_error_mean_deg"])
        except ValueError:
            continue  # e.g. DetectorFreeSfM 'n/a' failure rows

    ours = by_method["GradientHeight"]
    report: dict[str, dict] = {}
    for baseline in ("Reloc3r", "SIFT", "AKAZE"):
        other = by_method[baseline]
        seqs = sorted(set(ours) & set(other))
        x = [ours[s] for s in seqs]
        y = [other[s] for s in seqs]
        diffs = [a - b for a, b in zip(x, y)]
        t_stat, t_p = stats.ttest_rel(x, y)
        w_stat, w_p = stats.wilcoxon(x, y)
        report[f"gradient_vs_{baseline.lower()}"] = {
            "sequences": seqs,
            "x_mean": sum(x) / len(x),
            "y_mean": sum(y) / len(y),
            "mean_diff": sum(diffs) / len(diffs),
            "all_same_sign": len({math.copysign(1, d) for d in diffs}) == 1,
            "paired_t_p": float(t_p),
            "wilcoxon_p": float(w_p),
        }
        print(
            f"Ours vs {baseline}: x={sum(x)/len(x):.3f} y={sum(y)/len(y):.3f} "
            f"diff={sum(diffs)/len(diffs):.3f} t_p={t_p:.4g} w_p={w_p:.4g}"
        )
    OUT.write_text(json.dumps(report, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main()
