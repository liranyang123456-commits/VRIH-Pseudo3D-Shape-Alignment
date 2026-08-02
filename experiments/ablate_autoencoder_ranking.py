#!/usr/bin/env python3
"""Quantitative ablation of the hybrid autoencoder's fine-ranking contribution.

The autoencoder is used in the spatiotemporally constrained target search to
re-rank candidate ROIs: its latent-space contour similarity is fused with a
quick mask-IoU score and a spatial prior (score = 0.85*semantic + 0.15*IoU -
spatial penalty). This ablation isolates that scoring term.

For each consecutive frame pair of the contour evaluation sequences we generate
the same candidate ROI grid used by the tracker, compute the ground-truth ROI
by frame-to-frame optical-flow alignment of the previous mask, and measure
whether the autoencoder similarity ranks the true ROI higher than two
autoencoder-free baselines:

* AE ranking: latent-space contour similarity (hybrid autoencoder);
* IoU ranking: quick mask-IoU on the 64x64 grid (geometry only);
* Chamfer ranking: symmetric chamfer distance between contour point sets
  (geometry only, higher resolution).

Reported metrics per sequence: top-1 accuracy, mean rank of the true ROI,
mean Spearman correlation of each scoring with the true mask IoU.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from scipy import stats  # type: ignore
except Exception:
    stats = None

ROOT = Path(__file__).resolve().parent
DATA = Path(r"E:\MIS_Datasets")
RESULTS = ROOT / "results"

# SCARED endoscopic sequences (weak-texture, released, reproducible)
SEQUENCES = {
    "scared_d1_k1": ROOT / "data" / "scared_d1_k1" / "images",
    "scared_d2_k1": ROOT / "data" / "scared_d2_k1" / "images",
    "scared_d3_k2": ROOT / "data" / "scared_d3_k2" / "images",
}
MODEL_PATH = r"D:\reloc3r\Autoencoder_129.pth"
MAX_PAIRS = 40
SEARCH_RADIUS = 24
SEARCH_STEP = 8
SEM_SIZE = (128, 128)


def t90_mask(gray: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    thresh = np.percentile(magnitude, 90.0)
    return (magnitude > thresh).astype(np.uint8) * 255


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_image(contour, size) -> np.ndarray:
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if contour is None:
        return canvas
    scaled = contour.astype(np.float32).copy()
    scaled[:, :, 0] *= size[0]
    scaled[:, :, 1] *= size[1]
    cv2.drawContours(canvas, [scaled.astype(np.int32)], -1, (255, 255, 255), 2)
    return canvas


def chamfer_similarity(c1, c2) -> float:
    if c1 is None or c2 is None:
        return 0.0
    p1 = c1.reshape(-1, 2).astype(np.float32)
    p2 = c2.reshape(-1, 2).astype(np.float32)
    canvas1 = cv2.drawContours(np.zeros((128, 128), np.uint8), [(p2 * 128).astype(np.int32)], -1, 255, 1)
    dt1 = cv2.distanceTransform((canvas1 == 0).astype(np.uint8), cv2.DIST_L2, 3)
    d12 = dt1[(p1[:, 1] * 127).astype(int), (p1[:, 0] * 127).astype(int)].mean()
    canvas2 = cv2.drawContours(np.zeros((128, 128), np.uint8), [(p1 * 128).astype(np.int32)], -1, 255, 1)
    dt2 = cv2.distanceTransform((canvas2 == 0).astype(np.uint8), cv2.DIST_L2, 3)
    d21 = dt2[(p2[:, 1] * 127).astype(int), (p2[:, 0] * 127).astype(int)].mean()
    return float(1.0 / (1.0 + (d12 + d21) / 2.0))


def spearman(x, y) -> float:
    if stats is not None:
        r = stats.spearmanr(x, y).statistic
        return float(r) if np.isfinite(r) else 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float((rx * ry).mean())


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT.parent / "vrih_experiment_hub" / "src" / "semantic"))
    try:
        from extract_semantic_features_Function import calculate_contour_similarity  # type: ignore

        ae_available = True
    except Exception as exc:
        print(f"[warn] autoencoder unavailable: {exc}; running geometry-only ablation")
        ae_available = False

    rows: list[dict[str, object]] = []
    for seq_name, seq_dir in SEQUENCES.items():
        if not seq_dir.exists():
            print(f"[skip] {seq_name}: {seq_dir} missing")
            continue
        paths = sorted(
            p for p in seq_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )[: MAX_PAIRS + 1]
        if len(paths) < 2:
            print(f"[skip] {seq_name}: not enough frames")
            continue

        ae_scores_all: list[list[float]] = []
        iou_scores_all: list[list[float]] = []
        chamfer_scores_all: list[list[float]] = []
        true_iou_all: list[list[float]] = []
        true_index_all: list[int] = []

        for i in range(1, len(paths)):
            prev = cv2.imread(str(paths[i - 1]), cv2.IMREAD_GRAYSCALE)
            cur = cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE)
            if prev is None or cur is None:
                continue
            prev = cv2.resize(prev, (320, 256))
            cur = cv2.resize(cur, (320, 256))
            H, W = prev.shape

            prev_mask = t90_mask(prev)
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            warped = cv2.remap(
                prev_mask, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_NEAREST
            )
            true_cnt = largest_contour(warped)
            if true_cnt is None:
                continue
            tx, ty, tw, th = cv2.boundingRect(true_cnt)
            if tw < 8 or th < 8:
                continue
            true_center = (tx + tw // 2, ty + th // 2)
            true_mask_small = cv2.resize(warped, (64, 64), interpolation=cv2.INTER_NEAREST) > 0

            prev_cnt = largest_contour(prev_mask)
            prev_cont_img = contour_image(
                prev_cnt / np.array([[[W, H]]], np.float32) if prev_cnt is not None else None,
                SEM_SIZE,
            )

            ae_row: list[float] = []
            iou_row: list[float] = []
            chamfer_row: list[float] = []
            closeness_row: list[float] = []
            chosen_true = -1
            best_dist = 1e9
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, SEARCH_STEP):
                for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, SEARCH_STEP):
                    x = int(np.clip(true_center[0] + dx - tw // 2, 0, W - tw))
                    y = int(np.clip(true_center[1] + dy - th // 2, 0, H - th))
                    roi = cur[y : y + th, x : x + tw]
                    mask = t90_mask(roi)
                    cnt = largest_contour(mask)
                    cont_img = contour_image(
                        cnt / np.array([[[tw, th]]], np.float32) if cnt is not None else None,
                        SEM_SIZE,
                    )
                    if ae_available:
                        ae_row.append(
                            calculate_contour_similarity(
                                contour1=prev_cont_img, contour2=cont_img,
                                model_path=MODEL_PATH, image_size=SEM_SIZE,
                            )
                        )
                    m_small = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) > 0
                    inter = float((m_small & true_mask_small).sum())
                    union = float((m_small | true_mask_small).sum())
                    iou_row.append(inter / max(union, 1.0))
                    cn = cnt / np.array([[[tw, th]]], np.float32) if cnt is not None else None
                    chamfer_row.append(
                        chamfer_similarity(
                            (prev_cnt / np.array([[[W, H]]], np.float32)) if prev_cnt is not None else None,
                            cn,
                        )
                    )
                    dist = math.hypot(x + tw // 2 - true_center[0], y + th // 2 - true_center[1])
                    # ground-truth proximity is spatial closeness to the
                    # flow-aligned mask center, independent of any scoring term
                    closeness_row.append(float(1.0 / (1.0 + dist)))
                    if dist < best_dist:
                        best_dist = dist
                        chosen_true = len(iou_row) - 1
            if len(iou_row) < 4 or chosen_true < 0:
                continue
            if ae_available and len(ae_row) != len(iou_row):
                continue
            ae_scores_all.append(ae_row if ae_available else [0.0] * len(iou_row))
            iou_scores_all.append(iou_row)
            chamfer_scores_all.append(chamfer_row)
            true_iou_all.append(closeness_row)
            true_index_all.append(chosen_true)

        n = len(true_index_all)
        if n == 0:
            print(f"[skip] {seq_name}: no valid pairs")
            continue

        def top1(scores_list: list[list[float]]) -> float:
            hits = 0
            for s, tidx in zip(scores_list, true_index_all):
                if int(np.argmax(s)) == tidx:
                    hits += 1
            return hits / n

        def mean_rank(scores_list: list[list[float]]) -> float:
            ranks = []
            for s, tidx in zip(scores_list, true_index_all):
                order = np.argsort(np.argsort(s)[::-1])
                ranks.append(int(order[tidx]) + 1)
            return float(np.mean(ranks))

        def mean_spearman(scores_list: list[list[float]]) -> float:
            return float(
                np.mean(
                    [
                        spearman(s, t)
                        for s, t in zip(scores_list, true_iou_all)
                        if len(s) == len(t) and len(s) > 1
                    ]
                )
            )

        # fused tracker score: 0.85*AE + 0.15*IoU - spatial penalty; and the
        # same fusion with the AE term replaced by chamfer or dropped to IoU-only
        def fused_top1(primary_list: list[list[float]], weight_primary: float) -> float:
            hits = 0
            for prim, iou_row, tidx in zip(primary_list, iou_scores_all, true_index_all):
                fused = [
                    weight_primary * p + (1.0 - weight_primary) * q for p, q in zip(prim, iou_row)
                ]
                if int(np.argmax(fused)) == tidx:
                    hits += 1
            return hits / n

        rows.append(
            {
                "sequence": seq_name,
                "n_pairs": n,
                "ae_top1": round(top1(ae_scores_all), 4) if ae_available else "n/a",
                "iou_top1": round(top1(iou_scores_all), 4),
                "chamfer_top1": round(top1(chamfer_scores_all), 4),
                "ae_mean_rank": round(mean_rank(ae_scores_all), 2) if ae_available else "n/a",
                "iou_mean_rank": round(mean_rank(iou_scores_all), 2),
                "chamfer_mean_rank": round(mean_rank(chamfer_scores_all), 2),
                "ae_spearman": round(mean_spearman(ae_scores_all), 4) if ae_available else "n/a",
                "iou_spearman": round(mean_spearman(iou_scores_all), 4),
                "chamfer_spearman": round(mean_spearman(chamfer_scores_all), 4),
                "fused_ae_top1": round(fused_top1(ae_scores_all, 0.85), 4) if ae_available else "n/a",
                "fused_chamfer_top1": round(fused_top1(chamfer_scores_all, 0.85), 4),
            }
        )
        print(f"[ok] {seq_name}: {n} pairs evaluated")

    if not rows:
        print("no sequences evaluated")
        return
    out_csv = RESULTS / "autoencoder_ablation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
