#!/usr/bin/env python3
"""Block-level paired significance for HCR (cluster-robust to frame autocorrelation).

Per-frame Wilcoxon tests treat temporally correlated frames as i.i.d.; this
script instead averages contiguous frame blocks (default 16 blocks per
sequence) and runs paired Wilcoxon tests on the block means, which is the
standard cluster-robust remedy. Writes hcr_block_wilcoxon.csv/json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(r"E:/elsarticle-template-TMI_Revised/experiments/results")
SRC = RESULTS / "contour_hcr_per_frame.csv"
N_BLOCKS = 16

OURS = "Ours-gradient-p90"
BASELINES = ["Canny", "Sobel-p90", "Laplacian-p90", "DoG-p90"]


def block_means(values: np.ndarray, n_blocks: int) -> np.ndarray:
    edges = np.linspace(0, len(values), n_blocks + 1).astype(int)
    return np.array([values[edges[i]:edges[i + 1]].mean() for i in range(n_blocks)])


def main() -> None:
    rows = list(csv.DictReader(SRC.open()))
    cols = rows[0].keys()
    print("columns:", list(cols))
    # expect columns: sequence, frame (or index), method, hcr (adjust below)
    data: dict[tuple[str, str], list[float]] = {}
    seq_col = "sequence" if "sequence" in cols else list(cols)[0]
    method_col = "method" if "method" in cols else list(cols)[1]
    hcr_col = "hcr" if "hcr" in cols else [c for c in cols if "hcr" in c.lower()][0]
    for r in rows:
        data.setdefault((r[seq_col], r[method_col]), []).append(float(r[hcr_col]))

    out_rows = []
    worst_p = 0.0
    for seq in sorted({k[0] for k in data}):
        ours = np.array(data[(seq, OURS)])
        ours_b = block_means(ours, N_BLOCKS)
        for base in BASELINES:
            other = np.array(data[(seq, base)])
            other_b = block_means(other, N_BLOCKS)
            w = stats.wilcoxon(ours_b, other_b)
            out_rows.append(
                {
                    "sequence": seq,
                    "baseline": base,
                    "n_blocks": N_BLOCKS,
                    "ours_block_mean": float(ours_b.mean()),
                    "baseline_block_mean": float(other_b.mean()),
                    "wilcoxon_p": float(w.pvalue),
                }
            )
            worst_p = max(worst_p, float(w.pvalue))
            print(f"{seq} vs {base}: p={w.pvalue:.4g}")

    with (RESULTS / "hcr_block_wilcoxon.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    (RESULTS / "hcr_block_wilcoxon.json").write_text(
        json.dumps({"n_blocks_per_sequence": N_BLOCKS, "n_tests": len(out_rows), "max_p": worst_p}, indent=1)
    )
    print(f"max p across {len(out_rows)} block-level tests: {worst_p:.4g}")


if __name__ == "__main__":
    main()
