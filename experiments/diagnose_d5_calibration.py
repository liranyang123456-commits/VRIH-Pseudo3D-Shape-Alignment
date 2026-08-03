"""Diagnose near-zero GT epipolar precision on SCARED dataset_5.

Hypothesis: released Left/Right images (or T sign) are swapped for dataset_5.
Tests four interpretations on one keyframe: as-is, T negated, images swapped,
and both. Reports GT Sampson precision for plain SIFT matches in each case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_stereo_pseudo3d_density import calibration, skew

KF = Path(r"E:/MIS_Datasets/SCARED/dataset_5/keyframe_1")


def precision(pts_l: np.ndarray, pts_r: np.ndarray, k1, d1, k2, d2, r, t) -> float:
    nl = cv2.undistortPoints(pts_l.reshape(-1, 1, 2), k1, d1).reshape(-1, 2)
    nr = cv2.undistortPoints(pts_r.reshape(-1, 1, 2), k2, d2).reshape(-1, 2)
    e = skew(t) @ r
    x1 = np.column_stack([nl, np.ones(len(nl))])
    x2 = np.column_stack([nr, np.ones(len(nr))])
    ex1 = (e @ x1.T).T
    etx2 = (e.T @ x2.T).T
    num = np.sum(x2 * ex1, axis=1) ** 2
    samp = num / (ex1[:, 0] ** 2 + ex1[:, 1] ** 2 + etx2[:, 0] ** 2 + etx2[:, 1] ** 2 + 1e-12)
    focal = float((k1[0, 0] + k1[1, 1] + k2[0, 0] + k2[1, 1]) / 4.0)
    return float(np.mean(samp <= (1.5 / focal) ** 2))


def main() -> None:
    r, t, k1, d1, k2, d2 = calibration(KF / "endoscope_calibration.yaml")
    left = cv2.imread(str(KF / "Left_Image.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(KF / "Right_Image.png"), cv2.IMREAD_GRAYSCALE)
    print("image shapes:", left.shape, right.shape)
    sift = cv2.SIFT_create(nfeatures=4000)
    kl, dl = sift.detectAndCompute(left, None)
    kr, dr = sift.detectAndCompute(right, None)
    bf = cv2.BFMatcher()
    matches = [m for m, n in bf.knnMatch(dl, dr, k=2) if m.distance < 0.75 * n.distance]
    pl = np.asarray([kl[m.queryIdx].pt for m in matches])
    pr = np.asarray([kr[m.trainIdx].pt for m in matches])
    print("matches:", len(matches))
    print("as-is:                %.4f" % precision(pl, pr, k1, d1, k2, d2, r, t))
    print("T negated:            %.4f" % precision(pl, pr, k1, d1, k2, d2, r, -t))
    print("images swapped:       %.4f" % precision(pr, pl, k2, d2, k1, d1, r, t))
    print("swapped + T negated:  %.4f" % precision(pr, pl, k2, d2, k1, d1, r, -t))


if __name__ == "__main__":
    main()
