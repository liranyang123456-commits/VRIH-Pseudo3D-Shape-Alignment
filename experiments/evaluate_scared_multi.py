#!/usr/bin/env python3
"""Evaluate all SCARED sequences x methods and build an aggregate summary."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SEQUENCES = [
    "scared_d1_k1",
    "scared_d1_k2",
    "scared_d2_k1",
    "scared_d2_k2",
    "scared_d3_k1",
    "scared_d3_k2",
]

# method label -> (result dir template, trajectory file name)
METHODS = {
    "GradientHeight": ("{seq}_gradient", "gradient_cumulative_4x4.txt"),
    "IntensityHeight": ("{seq}_intensity", "intensity_cumulative_4x4.txt"),
    "ConstantHeight": ("{seq}_constant", "constant_cumulative_4x4.txt"),
    "Reloc3r": ("reloc3r_{seq}", "reloc3r_cumulative_4x4.txt"),
    "SIFT": ("sift_{seq}", "sift_cumulative_4x4.txt"),
    "AKAZE": ("akaze_{seq}", "akaze_cumulative_4x4.txt"),
}


def main() -> None:
    rows: list[dict[str, object]] = []
    for seq in SEQUENCES:
        gt = ROOT / "data" / seq / "scared_camera_pose_as_released_4x4.txt"
        if not gt.exists():
            print(f"[warn] {seq}: missing GT, skipped")
            continue
        for method, (dir_template, traj_name) in METHODS.items():
            result_dir = RESULTS / dir_template.format(seq=seq)
            trajectory = result_dir / traj_name
            if not trajectory.exists():
                print(f"[warn] {seq}/{method}: missing trajectory {trajectory}")
                continue
            metrics_path = result_dir / "pose_metrics.json"
            if not metrics_path.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "evaluate_pose.py"),
                        "--predicted", str(trajectory),
                        "--ground-truth", str(gt),
                        "--output", str(metrics_path),
                        "--max-frames", "80",
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            rotation = payload["relative_rotation_error_deg"]
            direction = payload["relative_translation_direction_error_deg"]
            rows.append(
                {
                    "sequence": seq,
                    "method": method,
                    "n_poses": payload["n_poses"],
                    "ate_sim3_rmse_gt_units": payload["ate_after_sim3_gt_units"]["rmse"],
                    "relative_rotation_error_mean_deg": rotation["mean"],
                    "relative_translation_direction_error_mean_deg": (
                        direction["mean"] if direction else ""
                    ),
                }
            )
            print(f"[ok] {seq}/{method}")

    output = RESULTS / "scared_multi_sequence_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} with {len(rows)} rows")

    # aggregate per method across sequences
    aggregate: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        entry = aggregate.setdefault(str(row["method"]), {"ate": [], "rot": [], "dir": []})
        entry["ate"].append(float(row["ate_sim3_rmse_gt_units"]))
        entry["rot"].append(float(row["relative_rotation_error_mean_deg"]))
        if row["relative_translation_direction_error_mean_deg"] != "":
            entry["dir"].append(float(row["relative_translation_direction_error_mean_deg"]))

    agg_output = RESULTS / "scared_multi_sequence_aggregate.csv"
    with agg_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "n_sequences",
                "ate_sim3_rmse_mean",
                "ate_sim3_rmse_std",
                "rel_rot_err_mean_deg",
                "rel_rot_err_std_deg",
                "rel_tdir_err_mean_deg",
                "rel_tdir_err_std_deg",
            ]
        )
        import statistics

        for method, entry in aggregate.items():
            writer.writerow(
                [
                    method,
                    len(entry["ate"]),
                    f"{statistics.mean(entry['ate']):.4f}",
                    f"{statistics.stdev(entry['ate']):.4f}" if len(entry["ate"]) > 1 else "0",
                    f"{statistics.mean(entry['rot']):.4f}",
                    f"{statistics.stdev(entry['rot']):.4f}" if len(entry["rot"]) > 1 else "0",
                    f"{statistics.mean(entry['dir']):.4f}" if entry["dir"] else "",
                    f"{statistics.stdev(entry['dir']):.4f}" if len(entry["dir"]) > 1 else "",
                ]
            )
    print(f"wrote {agg_output}")


if __name__ == "__main__":
    main()
