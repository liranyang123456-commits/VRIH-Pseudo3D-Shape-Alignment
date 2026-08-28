#!/usr/bin/env python3
"""Re-evaluate all saved trajectories to add the 1-degree rotation success rate.

Reads every pose_metrics.json-producing trajectory under experiments/results
(chessboard seg100 + SCARED protocols), recomputes metrics with the updated
evaluate_pose.py (which now reports rotation_success_rate_1deg), and writes
success_rates.csv with per-sequence and aggregate values.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# (protocol, sequence, method, trajectory path, gt path)
def collect() -> list[tuple[str, str, str, Path, Path]]:
    items = []
    # SCARED
    for seq in ["scared_d1_k1", "scared_d1_k2", "scared_d2_k1", "scared_d2_k2", "scared_d3_k1", "scared_d3_k2"]:
        gt = ROOT / "data" / seq / "scared_camera_pose_as_released_4x4.txt"
        for method, dname, fname in [
            ("GradientHeight", f"{seq}_gradient", "gradient_cumulative_4x4.txt"),
            ("IntensityHeight", f"{seq}_intensity", "intensity_cumulative_4x4.txt"),
            ("ConstantHeight", f"{seq}_constant", "constant_cumulative_4x4.txt"),
            ("Reloc3r", f"reloc3r_{seq}", "reloc3r_cumulative_4x4.txt"),
            ("SIFT", f"sift_{seq}", "sift_cumulative_4x4.txt"),
            ("AKAZE", f"akaze_{seq}", "akaze_cumulative_4x4.txt"),
            ("ORB", f"orb_{seq}", "orb_cumulative_4x4.txt"),
            ("DetectorFreeSfM", f"detectorfreesfm_{seq}", "detectorfreesfm_cumulative_4x4.txt"),
        ]:
            traj = RESULTS / dname / fname
            if traj.exists():
                items.append(("scared", seq, method, traj, gt))
    # chessboard seg100
    chess_gt = {
        "chess_seq1_100f": ROOT / "data" / "chess_seq1_100f" / "chessboard_camera_poses_c2w_4x4.txt",
        "chess_seq2_100f": ROOT / "data" / "chess_seq2_100f" / "chessboard_camera_poses_c2w_4x4.txt",
        "chess_seq3_100f": ROOT / "data" / "chess_seq3_100f" / "chessboard_camera_poses_c2w_4x4.txt",
        "chess_seq4_100f": ROOT / "data" / "chess_seq4_100f" / "chessboard_camera_poses_c2w_4x4.txt",
    }
    for d in sorted(RESULTS.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        for seq, gt in chess_gt.items():
            short = seq.replace("chess_", "").replace("_100f", "")
            if name.endswith(f"_{short}_seg100") or name.endswith(f"_{short}_100f"):
                for traj in d.glob("*_cumulative_4x4.txt"):
                    method = name.rsplit(f"_{short}", 1)[0]
                    items.append(("chessboard", seq, method, traj, gt))
    return items


def main() -> None:
    rows = []
    for protocol, seq, method, traj, gt in collect():
        if not gt.exists():
            print("missing GT", gt)
            continue
        out = traj.parent / "pose_metrics_sr.json"
        r = subprocess.run(
            [sys.executable, str(ROOT / "evaluate_pose.py"), "--predicted", str(traj),
             "--ground-truth", str(gt), "--output", str(out)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("fail", traj)
            continue
        m = json.loads(out.read_text())
        rows.append({
            "protocol": protocol,
            "sequence": seq,
            "method": method,
            "n_poses": m["n_poses"],
            "success_rate_1deg": m["rotation_success_rate_1deg"],
            "rel_rot_mean_deg": m["relative_rotation_error_deg"]["mean"],
            "rel_rot_median_deg": m["relative_rotation_error_deg"]["median"],
        })
    with (RESULTS / "success_rates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # aggregate per protocol/method
    import numpy as np

    agg: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        agg.setdefault((r["protocol"], r["method"]), []).append(r["success_rate_1deg"])
    print(f"{'protocol':12s} {'method':20s} {'n':>3s} {'succ@1deg':>9s}")
    for (protocol, method), vals in sorted(agg.items()):
        print(f"{protocol:12s} {method:20s} {len(vals):3d} {np.mean(vals):9.3f}")
    print("written success_rates.csv")


if __name__ == "__main__":
    main()
