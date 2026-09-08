#!/usr/bin/env python3
"""Extract additional SCARED keyframes (datasets 5-7) with pose GT (R2.9).

Selects keyframes from SCARED datasets 5, 6, 7 (which remain on disk with
released per-frame camera-pose), writes 80-frame image folders + pose files in
the same layout as the existing scared_d* sequences, so the released pipeline
and evaluate_pose.py run unchanged.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HUB = ROOT.parent / "vrih_experiment_hub" / "links" / "scared"
OUT = ROOT / "data"

# Pick one keyframe per dataset (the first with a usable video + poses).
PICKS = [
    ("dataset_5", "keyframe_2", "scared_d5_k2"),
    ("dataset_6", "keyframe_4", "scared_d6_k4"),
    ("dataset_7", "keyframe_4", "scared_d7_k4"),
]


def motion_stats(frame_data: Path, maxn: int = 400):
    with tarfile.open(frame_data, "r:gz") as a:
        members = sorted([m for m in a.getmembers() if m.name.endswith(".json")], key=lambda x: x.name)[:maxn]
        poses = [np.asarray(json.load(a.extractfile(m))["camera-pose"], dtype=float) for m in members]
    rots = []
    for A, B in zip(poses[:-1], poses[1:]):
        R = A[:3, :3].T @ B[:3, :3]
        c = np.clip((np.trace(R) - 1) / 2, -1, 1)
        rots.append(np.degrees(np.arccos(c)))
    return float(np.mean(rots)), float(np.max(rots)), len(poses)


def main() -> None:
    for ds, kf, name in PICKS:
        kf_dir = HUB / ds / kf
        frame_data = kf_dir / "data" / "frame_data.tar.gz"
        rgb = kf_dir / "data" / "rgb.mp4"
        out_dir = OUT / name
        if not frame_data.exists() or not rgb.exists():
            print("[skip]", name, "missing data")
            continue
        mean_r, max_r, nposes = motion_stats(frame_data)
        print(f"{name}: {ds}/{kf} poses={nposes} rot mean={mean_r:.3f} max={max_r:.3f}")
        subprocess.run(
            [sys.executable, str(ROOT / "prepare_scared_sequence.py"),
             "--rgb-video", str(rgb), "--frame-data", str(frame_data),
             "--output-dir", str(out_dir), "--max-frames", "80"],
            check=True,
        )
        # augment metadata with motion stats
        meta = json.loads((out_dir / "metadata.json").read_text())
        meta["rot_mean_deg"] = mean_r
        meta["rot_max_deg"] = max_r
        meta["source"] = f"{ds}/{kf}"
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[done]", name)


if __name__ == "__main__":
    main()
