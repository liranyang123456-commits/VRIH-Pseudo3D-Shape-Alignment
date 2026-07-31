#!/usr/bin/env python3
"""Paired Wilcoxon tests for HCR using per-frame contour measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="Ours-gradient-p90")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    results: list[dict[str, object]] = []
    sequences = sorted({row["sequence"] for row in rows})
    methods = sorted({row["method"] for row in rows if row["method"] != args.reference})
    for sequence in sequences:
        reference = {
            row["image"]: float(row["hcr"])
            for row in rows
            if row["sequence"] == sequence and row["method"] == args.reference
        }
        for method in methods:
            comparator = {
                row["image"]: float(row["hcr"])
                for row in rows
                if row["sequence"] == sequence and row["method"] == method
            }
            images = sorted(set(reference) & set(comparator))
            ours = np.asarray([reference[image] for image in images])
            baseline = np.asarray([comparator[image] for image in images])
            difference = ours - baseline
            if np.allclose(difference, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                test = wilcoxon(ours, baseline, alternative="two-sided", method="auto")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            results.append(
                {
                    "sequence": sequence,
                    "reference": args.reference,
                    "baseline": method,
                    "n_paired_frames": len(images),
                    "mean_hcr_difference_reference_minus_baseline": float(difference.mean()),
                    "wilcoxon_statistic": statistic,
                    "p_value_two_sided": p_value,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
