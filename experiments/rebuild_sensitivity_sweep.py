#!/usr/bin/env python3
"""Rebuild sensitivity_sweep.json/csv from scared_sensitivity.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results" / "scared_sensitivity.csv"
DEFAULTS = {"percentile": 90.0, "grid_step": 8, "height_scale": 96.0, "ransac_thresh": 8.0}


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    out = []
    for factor in ("percentile", "grid_step", "height_scale", "ransac_thresh"):
        values = sorted({float(r["value"]) for r in rows if r["factor"] == factor})
        for value in values:
            sel = [r for r in rows if r["factor"] == factor and float(r["value"]) == value]
            rot = sum(float(r["rot_mean_deg"]) for r in sel) / len(sel)
            ate = sum(float(r["ate_rmse"]) for r in sel) / len(sel)
            out.append(
                {
                    "parameter": factor,
                    "value": value,
                    "is_default": DEFAULTS[factor] == value,
                    "rel_rot_mean_deg": rot,
                    "ate_rmse_mean": ate,
                    "sequences": ["scared_d1_k1", "scared_d3_k2"],
                }
            )
    (ROOT / "results" / "sensitivity_sweep.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["parameter,value,is_default,rel_rot_mean_deg,ate_rmse_mean"]
    for row in out:
        lines.append(
            f"{row['parameter']},{row['value']},{row['is_default']},"
            f"{row['rel_rot_mean_deg']},{row['ate_rmse_mean']}"
        )
    (ROOT / "results" / "sensitivity_sweep.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} rows from {CSV.name}")


if __name__ == "__main__":
    main()
