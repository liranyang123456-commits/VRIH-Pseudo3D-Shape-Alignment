#!/usr/bin/env python3
"""Generate LaTeX table fragments from traceable CSV experiment summaries."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def fmt(value: str, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pose_table(rows: list[dict[str, str]], title: str, label: str, gt_units: str) -> str:
    body = []
    for row in rows:
        body.append(
            f"{row['method']} & {row['n_poses']} & {fmt(row['ate_sim3_rmse_gt_units'])} & "
            f"{fmt(row['relative_rotation_error_mean_deg'])} $\\pm$ {fmt(row['relative_rotation_error_std_deg'])} & "
            f"{fmt(row['relative_translation_direction_error_mean_deg'])} $\\pm$ "
            f"{fmt(row['relative_translation_direction_error_std_deg'])} \\\\"
        )
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{title} ATE is reported after Sim(3) alignment in {gt_units}; it measures trajectory-shape agreement and does not imply metric pose recovery.}}",
            f"\\label{{{label}}}",
            "\\small",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Method & Frames & ATE RMSE & Relative rotation error ($^\\circ$) & Translation-direction error ($^\\circ$) \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def hcr_table(rows: list[dict[str, str]]) -> str:
    wanted = [row for row in rows if row["metric"] == "HCR"]
    sequences = sorted({row["sequence"] for row in wanted})
    methods = ["Ours-gradient-p90", "Canny", "Sobel-p90", "Laplacian-p90", "DoG-p90"]
    lookup = {(row["method"], row["sequence"]): row for row in wanted}
    body = []
    for method in methods:
        label = {
            "Ours-gradient-p90": "Ours (gradient $p_{90}$)",
            "Sobel-p90": "Sobel ($p_{90}$)",
            "Laplacian-p90": "Laplacian ($p_{90}$)",
            "DoG-p90": "DoG ($p_{90}$)",
        }.get(method, method)
        values = []
        for sequence in sequences:
            row = lookup[(method, sequence)]
            values.append(f"{float(row['mean']):.4f} $\\pm$ {float(row['std']):.4f}")
        body.append(f"{label} & " + " & ".join(values) + " \\\\")
    headers = " & ".join(sequence.replace("Contour-", "") for sequence in sequences)
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Highlight-contamination ratio (HCR, lower is better) computed per frame using the top 5\\% grayscale-intensity pixels as the highlight mask. Values are mean $\\pm$ standard deviation.}",
            "\\label{tab:hcr}",
            "\\small",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            f"Method & {headers} \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chess", type=Path, required=True)
    parser.add_argument("--scared", type=Path, required=True)
    parser.add_argument("--hcr", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "table_chessboard_pose.tex").write_text(
        pose_table(
            read_csv(args.chess),
            "Chessboard comparison on 30 synchronized frames.",
            "tab:chessboard_pose",
            "the checkerboard ground-truth coordinate units",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "table_scared_pose.tex").write_text(
        pose_table(
            read_csv(args.scared),
            "Public SCARED dataset~1/keyframe~1 comparison on 80 frames.",
            "tab:scared_pose",
            "the released SCARED coordinate units",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "table_hcr.tex").write_text(
        hcr_table(read_csv(args.hcr)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
