#!/usr/bin/env python3
"""Build a concise, traceable pose-evaluation CSV from metric JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_entry(value: str) -> tuple[str, Path]:
    label, separator, filename = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected METHOD=METRICS_JSON")
    return label, Path(filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", type=parse_entry, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for method, file_name in args.entry:
        payload = json.loads(file_name.read_text(encoding="utf-8"))
        rotation = payload["relative_rotation_error_deg"]
        direction = payload["relative_translation_direction_error_deg"]
        rows.append(
            {
                "method": method,
                "n_poses": payload["n_poses"],
                "ate_sim3_rmse_gt_units": payload["ate_after_sim3_gt_units"]["rmse"],
                "ate_sim3_mean_gt_units": payload["ate_after_sim3_gt_units"]["mean"],
                "relative_rotation_error_mean_deg": rotation["mean"],
                "relative_rotation_error_std_deg": rotation["std"],
                "relative_translation_direction_error_mean_deg": (
                    direction["mean"] if direction else ""
                ),
                "relative_translation_direction_error_std_deg": (
                    direction["std"] if direction else ""
                ),
                "source_metrics_json": str(file_name),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
