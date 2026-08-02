#!/usr/bin/env python3
"""Convert a SCARED keyframe sequence into LumenGSLAM input format.

LumenGSLAM expects:
    <seq>/color/0000.png ... (RGB)
    <seq>/depth/0000.tiff ... (uint16, depth_mm = raw / depth_scale, depth_scale=256)
    <seq>/pose.txt  (one 4x4 c2w row-major matrix per line, cv2 convention)

We derive per-frame metric depth from the released scene_points XYZ maps
(2048x1280x3 float32, in mm). The depth channel of the XYZ map is the z
component in the left-camera frame; invalid points are zero-filled and then
densified by a nearest-neighbor inpainting pass to approximate the surfel-mesh
densification used by the authors. Poses are converted from the released
camera-pose matrices.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def read_poses(path: Path) -> dict[int, np.ndarray]:
    import re

    frame_re = re.compile(r"^frame_idx=(\d+)\s*$")
    poses: dict[int, np.ndarray] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0
    while cursor < len(lines):
        m = frame_re.match(lines[cursor].strip())
        if m is None:
            cursor += 1
            continue
        idx = int(m.group(1))
        rows = []
        cursor += 1
        while cursor < len(lines) and len(rows) < 4:
            line = lines[cursor].strip()
            cursor += 1
            if not line:
                continue
            rows.append([float(v) for v in line.split()])
        poses[idx] = np.asarray(rows, dtype=np.float64)
    return poses


def densify(depth_mm: np.ndarray) -> np.ndarray:
    """Fill zero (invalid) depth by nearest-valid inpainting."""
    valid = (depth_mm > 0) & np.isfinite(depth_mm)
    if valid.all() or (~valid).all():
        return depth_mm
    mask = (valid.astype(np.uint8)) * 255
    # distance transform to nearest valid pixel
    inv = cv2.bitwise_not(mask)
    dist, labels = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # build lookup of valid pixel values
    ys, xs = np.nonzero(mask)
    # labels index into arrays of zeros then valid points
    filled = depth_mm.copy()
    # distanceTransformWithLabels: label 0 = zero pixel; labels map to nearest valid
    label_to_val = {0: 0.0}
    for i, (y, x) in enumerate(zip(ys, xs), start=1):
        label_to_val[i] = float(depth_mm[y, x])
    flat_labels = labels.ravel()
    filled = np.vectorize(label_to_val.get, otypes=[np.float32])(flat_labels).reshape(depth_mm.shape)
    return filled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, help="e.g. scared_d1_k1")
    parser.add_argument("--scared", type=Path, default=Path(r"E:\MIS_Datasets\SCARED"))
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path(r"D:\LumenGSLAM\dataset\SCARED"))
    args = parser.parse_args()

    # map scared_d1_k1 -> dataset_1/keyframe_1
    parts = args.seq.replace("scared_", "").split("_")
    dataset = f"dataset_{parts[0][1:]}"
    keyframe = f"keyframe_{parts[1][1:]}"
    src = args.scared / dataset / keyframe / "data"
    prep = ROOT / "data" / args.seq

    images = sorted((prep / "images").glob("*.png"))[: args.max_frames]
    poses = read_poses(prep / "scared_camera_pose_as_released_4x4.txt")
    out_dir = args.out / f"{parts[0]}_{parts[1].replace('k', 'key')}"
    (out_dir / "color").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)

    pose_lines = []
    with tarfile.open(src / "scene_points.tar.gz", "r:gz") as tar:
        members = sorted(
            (m for m in tar.getmembers() if m.name.endswith(".tiff")),
            key=lambda m: m.name,
        )[: args.max_frames]
        for i, member in enumerate(members):
            f = tar.extractfile(member)
            buf = np.frombuffer(f.read(), dtype=np.uint8)
            xyz = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if xyz is None:
                raise RuntimeError(f"cannot decode {member.name}")
            depth_mm = xyz[..., 2].astype(np.float32)
            depth_mm = densify(depth_mm)
            depth_mm = np.clip(depth_mm, 0, 65535.0 / 256.0)
            raw = np.round(depth_mm * 256.0).astype(np.uint16)
            cv2.imwrite(str(out_dir / "depth" / f"{i:04d}.tiff"), raw)
            # copy color
            img = cv2.imread(str(images[i]))
            cv2.imwrite(str(out_dir / "color" / f"{i:04d}.png"), img)
            pose_lines.append(" ".join(f"{v:.8f}" for v in poses[i].ravel()))

    (out_dir / "pose.txt").write_text("\n".join(pose_lines), encoding="utf-8")
    print(f"wrote {out_dir} with {len(pose_lines)} frames")


if __name__ == "__main__":
    main()
