#!/usr/bin/env python3
"""Aggregate per-sequence pose metric JSON files into a traceable CSV/JSON table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="chess_")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob(f"{args.prefix}*_pose_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "sequence": path.stem.replace("_ours_pose_metrics", ""),
                "n_poses": payload["n_poses"],
                "sim3_scale_pred_to_gt": payload["trajectory_alignment"][
                    "scale_pred_to_gt"
                ],
                "ate_rmse_gt_units": payload["ate_after_sim3_gt_units"]["rmse"],
                "ate_mean_gt_units": payload["ate_after_sim3_gt_units"]["mean"],
                "rot_rel_mean_deg": payload["relative_rotation_error_deg"]["mean"],
                "rot_rel_std_deg": payload["relative_rotation_error_deg"]["std"],
                "trans_dir_mean_deg": (
                    payload["relative_translation_direction_error_deg"]["mean"]
                    if payload["relative_translation_direction_error_deg"]
                    else None
                ),
                "trans_dir_std_deg": (
                    payload["relative_translation_direction_error_deg"]["std"]
                    if payload["relative_translation_direction_error_deg"]
                    else None
                ),
                "source_metrics_file": str(path),
            }
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sequence"])
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
