#!/usr/bin/env python3
"""Report registered image counts for DetectorFreeSfM COLMAP models."""

import sys

import pycolmap

base = sys.argv[1]
for sub in ("0", "1", "."):
    path = f"{base}/{sub}"
    try:
        reconstruction = pycolmap.Reconstruction(path)
        stats = reconstruction.compute_mean_reprojection_error()
        print(sub, "images:", len(reconstruction.images), "points:", len(reconstruction.points3D), "mean_reproj:", round(stats, 4))
    except Exception as exc:
        print(sub, "error:", exc)
