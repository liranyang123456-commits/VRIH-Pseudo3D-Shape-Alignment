#!/usr/bin/env python3
"""Run the ORB feature baseline on all six prepared SCARED sequences and evaluate.

Fills the ORB gap in the SCARED benchmark (Table: scared_pose). Produces
orb_{seq}/orb_cumulative_4x4.txt trajectories plus pose_metrics.json under
experiments/results/, then refreshes the multi-sequence summary/aggregate.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEQUENCES = [
    "scared_d1_k1",
    "scared_d1_k2",
    "scared_d2_k1",
    "scared_d2_k2",
    "scared_d3_k1",
    "scared_d3_k2",
]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    for seq in SEQUENCES:
        data = ROOT / "data" / seq
        out = ROOT / "results" / f"orb_{seq}"
        run(
            [
                sys.executable,
                str(ROOT / "run_feature_pose_baseline.py"),
                "--images",
                str(data / "images"),
                "--intrinsics",
                str(data / "intrinsics.json"),
                "--output-dir",
                str(out),
                "--method",
                "ORB",
                "--max-frames",
                "80",
            ]
        )
        run(
            [
                sys.executable,
                str(ROOT / "evaluate_pose.py"),
                "--predicted",
                str(out / "orb_cumulative_4x4.txt"),
                "--ground-truth",
                str(data / "scared_camera_pose_as_released_4x4.txt"),
                "--output",
                str(out / "pose_metrics.json"),
            ]
        )
    run([sys.executable, str(ROOT / "evaluate_scared_multi.py")])


if __name__ == "__main__":
    main()
