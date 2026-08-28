#!/usr/bin/env python3
"""Condition-stratified evaluation on the six SCARED sequences (R2.9).

Runs the gradient-height pipeline per frame pair and records, for each pair,
the relative rotation error together with condition proxies:

* weak texture: mean gradient magnitude of the frame (lower = weaker texture)
* specular reflection: fraction of pixels in the top-5% intensity band
* large deformation: 1 - IoU of consecutive t90 foreground masks
* rapid camera motion: mean Farneback flow magnitude
* motion blur: variance of Laplacian (lower = blurrier)
* partial occlusion / support change: |r_t - r_{t-1}| foreground ratio change

Reports per-condition median-split error ratios and Spearman correlations
between each condition proxy and the pair-wise rotation error.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SEQUENCES = ["scared_d1_k1", "scared_d1_k2", "scared_d2_k1", "scared_d2_k2", "scared_d3_k1", "scared_d3_k2"]


def height_field(gray: np.ndarray, height_scale: float = 96.0) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    filtered = cv2.bilateralFilter(g, 7, 0.1, 3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 90.0)
    mag = np.where(mag > thr, mag, 0.0)
    mx = float(mag.max())
    return (mag / mx * height_scale) if mx > 1e-8 else mag


def kabsch(src: np.ndarray, dst: np.ndarray):
    cs, cd = src.mean(0), dst.mean(0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, cd - R @ cs


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def read_gt(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("frame_idx="):
            idx = int(lines[i].strip().split("=")[1])
            M = []
            i += 1
            while i < len(lines) and len(M) < 4:
                if lines[i].strip():
                    M.append([float(v) for v in lines[i].split()])
                i += 1
            poses[idx] = np.array(M)
        else:
            i += 1
    return poses


def main() -> None:
    rows = []
    for seq in SEQUENCES:
        data_dir = ROOT / "data" / seq
        images = sorted((data_dir / "images").glob("*.png"))[:80]
        gt = read_gt(data_dir / "scared_camera_pose_as_released_4x4.txt")
        prev = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
        h_prev = height_field(prev)
        mask_prev = h_prev > 0
        for i in range(1, len(images)):
            curr = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
            h_curr = height_field(curr)
            mask_curr = h_curr > 0
            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            ys, xs = np.mgrid[0:prev.shape[0]:8, 0:prev.shape[1]:8]
            xs, ys = xs.ravel(), ys.ravel()
            f = flow[ys, xs]
            xc, yc = xs + f[:, 0], ys + f[:, 1]
            valid = (xc >= 1) & (xc < curr.shape[1] - 2) & (yc >= 1) & (yc < curr.shape[0] - 2)
            xs2, ys2, xc2, yc2 = xs[valid], ys[valid], xc[valid], yc[valid]
            xi = np.clip(xc2.astype(int), 0, h_curr.shape[1] - 1)
            yi = np.clip(yc2.astype(int), 0, h_curr.shape[0] - 1)
            src = np.column_stack([xc2, yc2, h_curr[yi, xi]])
            dst = np.column_stack([xs2.astype(float), ys2.astype(float), h_prev[ys2, xs2]])
            R, _ = kabsch(src, dst)
            # GT relative rotation
            if i in gt and i - 1 in gt:
                Rg = gt[i - 1][:3, :3].T @ gt[i][:3, :3]
                err = rot_deg(R, Rg)
            else:
                continue
            # condition proxies
            gx = cv2.Sobel(prev, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(prev, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            texture = float(mag.mean())
            specular = float((prev >= np.percentile(prev, 95)).mean())
            deform = float(1.0 - (np.logical_and(mask_prev, mask_curr).sum() / max(np.logical_or(mask_prev, mask_curr).sum(), 1)))
            motion = float(np.linalg.norm(flow, axis=2).mean())
            blur = float(cv2.Laplacian(prev, cv2.CV_32F).var())
            support = float(abs(mask_curr.mean() - mask_prev.mean()))
            rows.append({"seq": seq, "pair": i, "rot_err": err, "texture": texture, "specular": specular,
                         "deformation": deform, "motion": motion, "blur": blur, "support_change": support})
            prev, h_prev, mask_prev = curr, h_curr, mask_curr
        print(f"{seq} done ({len(images)} frames)", flush=True)

    with (RESULTS / "condition_stratified_pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {}
    errs = np.array([r["rot_err"] for r in rows])
    for cond in ("texture", "specular", "deformation", "motion", "blur", "support_change"):
        v = np.array([r[cond] for r in rows])
        med = np.median(v)
        lo, hi = errs[v <= med], errs[v > med]
        spear = stats.spearmanr(v, errs).statistic
        report[cond] = {
            "median_split": float(med),
            "err_low_group": float(lo.mean()),
            "err_high_group": float(hi.mean()),
            "spearman_vs_error": float(spear),
            "spearman_p": float(stats.spearmanr(v, errs).pvalue),
        }
        print(f"{cond}: low={lo.mean():.3f} high={hi.mean():.3f} spearman={spear:+.3f}")
    (RESULTS / "condition_stratified.json").write_text(json.dumps(report, indent=1))
    print("written condition_stratified.json")


if __name__ == "__main__":
    main()
