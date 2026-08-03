"""Depth-based check: does dataset_5's calibration yaml actually fit its images?

Projects left-image SIFT keypoints to 3D via the released depth map and M1,
transforms them into the right camera via R/T, projects with M2/D2, and
reports the median reprojection error against the matched right keypoints.
A working calibration yields errors of a few pixels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_stereo_pseudo3d_density import calibration

KF = Path(r"E:/MIS_Datasets/SCARED/dataset_5/keyframe_1")


def main() -> None:
    r, t, k1, d1, k2, d2 = calibration(KF / "endoscope_calibration.yaml")
    left = cv2.imread(str(KF / "Left_Image.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(KF / "Right_Image.png"), cv2.IMREAD_GRAYSCALE)
    depth_raw = cv2.imread(str(KF / "left_depth_map.tiff"), cv2.IMREAD_UNCHANGED)[..., 0]
    print("depth dtype/range:", depth_raw.dtype, depth_raw.min(), depth_raw.max())
    sift = cv2.SIFT_create(nfeatures=4000)
    kl, dl = sift.detectAndCompute(left, None)
    kr, dr = sift.detectAndCompute(right, None)
    bf = cv2.BFMatcher()
    matches = [m for m, n in bf.knnMatch(dl, dr, k=2) if m.distance < 0.75 * n.distance]
    pl = np.asarray([kl[m.queryIdx].pt for m in matches])
    pr = np.asarray([kr[m.trainIdx].pt for m in matches])

    xs = np.clip(pl[:, 0].astype(int), 0, depth_raw.shape[1] - 1)
    ys = np.clip(pl[:, 1].astype(int), 0, depth_raw.shape[0] - 1)
    z = depth_raw[ys, xs].astype(np.float64)
    scale = 1.0 if z.max() < 1000 else 0.001  # mm-encoded TIFF heuristic
    z = z * scale
    valid = z > 0.1
    print(f"valid depths: {valid.sum()}/{len(z)}  z range {z[valid].min():.2f}..{z[valid].max():.2f}")
    pl_n = cv2.undistortPoints(pl[valid].reshape(-1, 1, 2), k1, d1).reshape(-1, 2)
    pts3 = np.column_stack([pl_n * z[valid, None], z[valid]])
    pts3_r = (r @ pts3.T).T + t
    proj, _ = cv2.projectPoints(pts3_r.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k2, d2)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - pr[valid], axis=1)
    print("reprojection error px: median=%.1f mean=%.1f p90=%.1f" % (np.median(err), err.mean(), np.percentile(err, 90)))
    row_diff = np.abs(proj[:, 1] - pr[valid][:, 1])
    print("vertical disparity px: median=%.1f" % np.median(row_diff))


if __name__ == "__main__":
    main()
