#!/usr/bin/env python3
"""Evaluate all chessboard sequences x methods; build summaries.

Two protocols:
  * seg100: the first 100 GT-covered frames of every sequence. All methods,
    including the restricted DetectorFreeSfM coarse-only reproduction, are
    evaluated on identical frames against the freshly generated solvePnP GT of
    the current Ours pipeline run (fully traceable).
  * full: the complete GT-covered sequence against the legacy full-length GT.
    Ours entries in this protocol come from the legacy full-length runs and are
    labeled accordingly.

Sequence 2's GT coverage starts at original frame 203, so its staged segment
uses original frames 203-302; baseline trajectories are re-indexed by -203
before evaluation.
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EVAL_DIR = RESULTS / "chessboard_multi_eval"
REMAP_DIR = EVAL_DIR / "remapped"

sys.path.insert(0, str(ROOT))
from evaluate_pose import read_poses  # noqa: E402

LEGACY = {
    n: Path(rf"D:\reloc3r\Data_IMU_Camera_Pose_{n}\Ours_Camera_Pose_60f_eval_vis")
    for n in (1, 2, 3, 4)
}
# seq2's segment uses original frames 351-450 (the first 100-frame window with
# full legacy-GT coverage; the board is not visible near the sequence start).
SEG_OFFSET = {"seq1": 0, "seq2": 351, "seq3": 0, "seq4": 0, "line2": 0}
SEQ_IDS = ["seq1", "seq2", "seq3", "seq4", "line2"]
METHODS = ["Ours", "DetectorFreeSfM", "Reloc3r", "SIFT", "AKAZE", "ORB"]


def fresh_run_dir(seq: str) -> Path:
    if seq == "line2":
        return RESULTS / "ours_chess_line2_100f"
    return RESULTS / f"ours_chess_{seq}_100f"


def baseline_traj(seq: str, method: str) -> Path | None:
    suffix = "chess_line2" if seq == "line2" else f"chess_{seq}"
    mapping = {
        "Reloc3r": RESULTS / f"reloc3r_{suffix}" / "reloc3r_cumulative_4x4.txt",
        "SIFT": RESULTS / f"sift_{suffix}" / "sift_cumulative_4x4.txt",
        "AKAZE": RESULTS / f"akaze_{suffix}" / "akaze_cumulative_4x4.txt",
        "ORB": RESULTS / f"orb_{suffix}" / "orb_cumulative_4x4.txt",
    }
    if method == "DetectorFreeSfM":
        if seq == "line2":
            return RESULTS / "detectorfreesfm_chess_line2_100f" / "detectorfreesfm_cumulative_4x4.txt"
        return RESULTS / f"detectorfreesfm_chess_{seq}" / "detectorfreesfm_cumulative_4x4.txt"
    return mapping.get(method)


def remapped(path: Path, offset: int, tag: str) -> Path:
    """Shift frame indices by -offset, keeping only frames >= offset."""
    if offset == 0:
        return path
    REMAP_DIR.mkdir(parents=True, exist_ok=True)
    out = REMAP_DIR / f"{tag}.txt"
    if not out.exists():
        poses = read_poses(path)
        lines = [f"# remapped from {path} (offset -{offset})"]
        for idx in sorted(poses):
            if idx < offset:
                continue
            lines.append(f"frame_idx={idx - offset}")
            lines.extend(" ".join(f"{v:.10f}" for v in row) for row in poses[idx])
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_eval(pred: Path, gt: Path, out: Path, max_frames: int | None) -> dict | None:
    if not pred.exists() or not gt.exists():
        return None
    if not out.exists():
        cmd = [
            sys.executable, str(ROOT / "evaluate_pose.py"),
            "--predicted", str(pred),
            "--ground-truth", str(gt),
            "--output", str(out),
        ]
        if max_frames:
            cmd += ["--max-frames", str(max_frames)]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[fail] {out.name}: {result.stderr.decode(errors='replace')[-200:]}")
            return None
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> None:
    EVAL_DIR.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    # --- seg100 protocol ---
    for seq in SEQ_IDS:
        run_dir = fresh_run_dir(seq)
        gt = run_dir / "chessboard_camera_poses_c2w_4x4.txt"
        if not gt.exists() and seq == "seq2":
            # The staged seq2 segment has poor board visibility for the default
            # detector, so its GT falls back to the legacy full-sequence
            # solvePnP GT, re-indexed to the segment (offset -203).
            gt = remapped(
                LEGACY[2] / "chessboard_camera_poses_c2w_4x4.txt",
                SEG_OFFSET["seq2"],
                "seq2_GT_legacy",
            )
        if not gt.exists():
            print(f"[skip] seg100/{seq}: fresh GT not ready")
            continue
        offset = SEG_OFFSET[seq]
        for method in METHODS:
            if method == "Ours":
                pred = run_dir / "pseudo_camera_poses_c2w_4x4.txt"
            else:
                raw = baseline_traj(seq, method)
                if raw is None or not raw.exists():
                    print(f"[skip] seg100/{seq}/{method}: no trajectory")
                    continue
                pred = remapped(raw, offset, f"{seq}_{method}")
            out = EVAL_DIR / f"seg100_{seq}_{method}.json"
            payload = run_eval(pred, gt, out, 100)
            if payload is None:
                print(f"[skip] seg100/{seq}/{method}: eval failed or missing")
                continue
            rot = payload["relative_rotation_error_deg"]
            tdir = payload["relative_translation_direction_error_deg"]
            rows.append({
                "protocol": "seg100", "sequence": seq, "method": method,
                "n_poses": payload["n_poses"],
                "ate_sim3_rmse": payload["ate_after_sim3_gt_units"]["rmse"],
                "rel_rot_err_mean_deg": rot["mean"],
                "rel_tdir_err_mean_deg": tdir["mean"] if tdir else "",
            })
            print(f"[ok] seg100/{seq}/{method} n={payload['n_poses']}")

    # --- full protocol (legacy GT; legacy Ours full-length runs) ---
    for seq in SEQ_IDS:
        if seq == "line2":
            gt = RESULTS / "ours_chess_line2_100f" / "chessboard_camera_poses_c2w_4x4.txt"
            ours_pred = RESULTS / "ours_chess_line2_100f" / "pseudo_camera_poses_c2w_4x4.txt"
        else:
            n = int(seq[3:])
            gt = LEGACY[n] / "chessboard_camera_poses_c2w_4x4.txt"
            ours_pred = LEGACY[n] / "pseudo_camera_poses_c2w_4x4.txt"
        for method in METHODS:
            if method == "DetectorFreeSfM" and seq != "line2":
                continue
            pred = ours_pred if method == "Ours" else baseline_traj(seq, method)
            if method == "DetectorFreeSfM":
                pred = RESULTS / "detectorfreesfm_chess_line2_100f" / "detectorfreesfm_cumulative_4x4.txt"
            if pred is None or not pred.exists():
                print(f"[skip] full/{seq}/{method}: no trajectory")
                continue
            out = EVAL_DIR / f"full_{seq}_{method}.json"
            payload = run_eval(pred, gt, out, None)
            if payload is None:
                continue
            rot = payload["relative_rotation_error_deg"]
            tdir = payload["relative_translation_direction_error_deg"]
            rows.append({
                "protocol": "full", "sequence": seq, "method": method,
                "n_poses": payload["n_poses"],
                "ate_sim3_rmse": payload["ate_after_sim3_gt_units"]["rmse"],
                "rel_rot_err_mean_deg": rot["mean"],
                "rel_tdir_err_mean_deg": tdir["mean"] if tdir else "",
            })
            print(f"[ok] full/{seq}/{method} n={payload['n_poses']}")

    summary = RESULTS / "chessboard_multi_sequence_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary} ({len(rows)} rows)")

    aggregate = RESULTS / "chessboard_multi_sequence_aggregate.csv"
    with aggregate.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "protocol", "method", "n_sequences",
            "ate_rmse_mean", "ate_rmse_std",
            "rel_rot_err_mean_deg", "rel_rot_err_std_deg",
            "rel_tdir_err_mean_deg", "rel_tdir_err_std_deg",
        ])
        for protocol in ("seg100", "full"):
            for method in METHODS:
                sel = [r for r in rows if r["protocol"] == protocol and r["method"] == method]
                if not sel:
                    continue
                ate = [float(r["ate_sim3_rmse"]) for r in sel]
                rot = [float(r["rel_rot_err_mean_deg"]) for r in sel]
                tdir = [float(r["rel_tdir_err_mean_deg"]) for r in sel if r["rel_tdir_err_mean_deg"] != ""]
                std = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
                writer.writerow([
                    protocol, method, len(sel),
                    f"{statistics.mean(ate):.3f}", f"{std(ate):.3f}",
                    f"{statistics.mean(rot):.3f}", f"{std(rot):.3f}",
                    f"{statistics.mean(tdir):.3f}" if tdir else "", f"{std(tdir):.3f}" if tdir else "",
                ])
    print(f"wrote {aggregate}")


if __name__ == "__main__":
    main()
