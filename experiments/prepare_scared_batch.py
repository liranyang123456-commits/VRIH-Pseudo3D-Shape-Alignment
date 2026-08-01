#!/usr/bin/env python3
"""Prepare multiple SCARED sequences with the same 80-frame protocol as d1_k1."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCARED = Path(r"E:\MIS_Datasets\SCARED")

SEQUENCES = [
    ("dataset_1", "keyframe_2", "scared_d1_k2"),
    ("dataset_2", "keyframe_1", "scared_d2_k1"),
    ("dataset_2", "keyframe_2", "scared_d2_k2"),
    ("dataset_3", "keyframe_1", "scared_d3_k1"),
    ("dataset_3", "keyframe_2", "scared_d3_k2"),
]


def main() -> None:
    for dataset, keyframe, name in SEQUENCES:
        out_dir = ROOT / "data" / name
        if (out_dir / "metadata.json").exists():
            print(f"[skip] {name} already prepared")
            continue
        base = SCARED / dataset / keyframe / "data"
        rgb = base / "rgb.mp4"
        frame_data = base / "frame_data.tar.gz"
        if not rgb.exists() or not frame_data.exists():
            print(f"[warn] {name}: missing rgb.mp4 or frame_data.tar.gz, skipping")
            continue
        print(f"[prep] {name}")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "prepare_scared_sequence.py"),
                "--rgb-video", str(rgb),
                "--frame-data", str(frame_data),
                "--output-dir", str(out_dir),
                "--max-frames", "80",
            ],
            check=True,
        )
        metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
        calib = metadata["camera_calibration_first_frame"]
        kl = calib["KL"] if isinstance(calib, dict) and "KL" in calib else calib
        (out_dir / "intrinsics.json").write_text(
            json.dumps({"K": kl, "source": "SCARED released camera-calibration.KL"}, indent=2),
            encoding="utf-8",
        )
        print(f"[done] {name}: intrinsics written")


if __name__ == "__main__":
    main()
