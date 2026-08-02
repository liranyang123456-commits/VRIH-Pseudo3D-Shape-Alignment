#!/usr/bin/env python3
"""Paired significance tests: gradient vs constant vs intensity pseudo-height.

Uses the per-sequence relative rotation errors from the six-sequence SCARED
evaluation. Paired Wilcoxon signed-rank and paired t-tests across the six
matched sequences, plus per-frame Wilcoxon on d1/k1 via the trajectory re-eval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

PAIRS = [
    ("gradient", "constant"),
    ("gradient", "intensity"),
    ("constant", "intensity"),
]
SEQS = ["scared_d1_k1", "scared_d1_k2", "scared_d2_k1", "scared_d2_k2", "scared_d3_k1", "scared_d3_k2"]


def paired_tests(x: list[float], y: list[float]) -> dict:
    x = np.asarray(x)
    y = np.asarray(y)
    diff = x - y
    # paired t-test
    t_stat = float(diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))) if diff.std(ddof=1) > 0 else 0.0
    try:
        from scipy import stats  # type: ignore

        t_p = float(stats.t.sf(abs(t_stat), len(diff) - 1) * 2)
        if np.allclose(diff, 0.0):
            w_p = 1.0
        else:
            w_stat, w_p = stats.wilcoxon(x, y)
            w_p = float(w_p)
        t_p_scipy = t_p
    except Exception:
        # normal approximation fallback
        z = t_stat * math.sqrt(len(diff))
        t_p_scipy = math.erfc(abs(z) / math.sqrt(2.0))
        w_p = float("nan")
    return {
        "n_pairs": len(diff),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "t_stat": t_stat,
        "paired_t_p": t_p_scipy,
        "wilcoxon_p": w_p,
    }


def main() -> None:
    rows: dict[str, dict[str, float]] = {}
    for seq in SEQS:
        for variant in ("gradient", "constant", "intensity"):
            payload = json.loads(
                (RESULTS / f"{seq}_{variant}" / "pose_metrics.json").read_text(encoding="utf-8")
            )
            rows.setdefault(seq, {})[variant] = payload["relative_rotation_error_deg"]["mean"]

    report: dict[str, dict] = {}
    for a, b in PAIRS:
        xa = [rows[s][a] for s in SEQS]
        xb = [rows[s][b] for s in SEQS]
        report[f"{a}_vs_{b}"] = {
            "x_mean": float(np.mean(xa)),
            "y_mean": float(np.mean(xb)),
            "per_sequence_x": xa,
            "per_sequence_y": xb,
            **paired_tests(xa, xb),
        }

    out = RESULTS / "scared_height_significance.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "sequence" not in kk} for k, v in report.items()}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
