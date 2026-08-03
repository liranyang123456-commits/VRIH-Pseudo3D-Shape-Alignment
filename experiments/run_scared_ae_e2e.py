#!/usr/bin/env python3
"""End-to-end ablation of the autoencoder re-ranking on final rotation error.

Two variants of the flow-guided shape-alignment pipeline on six SCARED
sequences (80 frames each):

* ``wo_ae`` — current pipeline: foreground sampling, Farneback flow,
  RANSAC + Kabsch rigid fit in (u, v, h). Identical to the reported results.
* ``w_ae``  — same, but each sampled correspondence is additionally gated by
  the hybrid autoencoder's latent contour similarity between the local
  t90-contour patches around the previous-frame and warped current-frame
  points (batched GPU encoding, keep top ``--keep`` fraction).

Also records per-stage wall-clock timings (gradient+height, flow, AE encode,
RANSAC+Kabsch) for the runtime analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("VAL_DISABLE_TORCH_COMPILE", "1")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEMANTIC = ROOT.parent / "vrih_experiment_hub" / "src" / "semantic"
sys.path.insert(0, str(SEMANTIC))

SEQUENCES = [
    "scared_d1_k1",
    "scared_d1_k2",
    "scared_d2_k1",
    "scared_d2_k2",
    "scared_d3_k1",
    "scared_d3_k2",
]

PATCH = 48  # local patch side (px) around a correspondence point
GRID_STEP = 8


def height_field(gray: np.ndarray) -> np.ndarray:
    smooth = cv2.bilateralFilter(gray.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    truncated = np.where(magnitude > np.percentile(magnitude, 90.0), magnitude, 0.0)
    return truncated / (magnitude.max() + 1e-12), truncated


def contour_patch(mask: np.ndarray, cx: float, cy: float) -> np.ndarray:
    h, w = mask.shape
    r = PATCH // 2
    x0 = int(np.clip(round(cx) - r, 0, max(0, w - PATCH)))
    y0 = int(np.clip(round(cy) - r, 0, max(0, h - PATCH)))
    patch = mask[y0:y0 + PATCH, x0:x0 + PATCH]
    patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)


def kabsch_ransac(src: np.ndarray, dst: np.ndarray, iterations: int = 300, thresh: float = 8.0, seed: int = 0):
    """RANSAC + Kabsch rigid fit src -> dst (both Nx3)."""
    rng = np.random.default_rng(seed)
    n = len(src)
    if n < 3:
        return None, None, 0
    best_inliers = None
    for _ in range(iterations):
        idx = rng.choice(n, size=min(6, n), replace=False)
        R, t = kabsch(src[idx], dst[idx])
        residual = np.linalg.norm((src @ R.T + t) - dst, axis=1)
        inliers = residual < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        return None, None, 0
    R, t = kabsch(src[best_inliers], dst[best_inliers])
    return R, t, int(best_inliers.sum())


def kabsch(src: np.ndarray, dst: np.ndarray):
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cd - R @ cs
    return R, t


def load_ae(device_str: str = "cuda"):
    import torch  # noqa: F401
    from extract_semantic_features_Function import ModelManager  # type: ignore

    manager = ModelManager()
    manager.initialize(model_path=r"D:\reloc3r\Autoencoder_129.pth", image_size=(128, 128), device=device_str)
    model = manager.get_model()
    transform = manager.get_transform()
    device = manager.get_device()
    return model, transform, device


def encode_patches(model, transform, device, patches: list[np.ndarray], batch: int = 128) -> np.ndarray:
    import torch

    feats = []
    with torch.no_grad():
        for i in range(0, len(patches), batch):
            tensors = [transform(image=p)["image"] for p in patches[i:i + batch]]
            x = torch.stack(tensors).to(device)
            z = model.encode(x)
            feats.append(z.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=float, default=0.7, help="Fraction of correspondences kept by AE gating")
    parser.add_argument("--sequences", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    sequences = SEQUENCES if args.sequences is None else [s.strip() for s in args.sequences.split(",")]

    model, transform, device = load_ae(args.device)
    timings: list[dict[str, float]] = []
    per_sequence: dict[str, dict[str, dict[str, float]]] = {}

    for seq in sequences:
        data_dir = ROOT / "data" / seq
        images = sorted((data_dir / "images").glob("*.png"))[:80]
        gt = data_dir / "scared_camera_pose_as_released_4x4.txt"
        variants = {name: [] for name in ("wo_ae", "w_ae")}
        for pair_idx in range(len(images) - 1):
            prev = cv2.imread(str(images[pair_idx]), cv2.IMREAD_GRAYSCALE)
            curr = cv2.imread(str(images[pair_idx + 1]), cv2.IMREAD_GRAYSCALE)

            t0 = time.perf_counter()
            h_prev, mask_prev = height_field(prev)
            h_curr, mask_curr = height_field(curr)
            t1 = time.perf_counter()

            flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 21, 3, 5, 1.2, 0)
            t2 = time.perf_counter()

            ys, xs = np.nonzero(mask_prev > 0)
            if len(xs) < 10:
                continue
            sel = np.arange(0, len(xs), max(1, len(xs) // 400))
            xs, ys = xs[sel], ys[sel]
            flow_at = flow[ys, xs]
            xc = xs + flow_at[:, 0]
            yc = ys + flow_at[:, 1]
            valid = (xc >= 2) & (xc < curr.shape[1] - 2) & (yc >= 2) & (yc < curr.shape[0] - 2)
            xs, ys, xc, yc = xs[valid], ys[valid], xc[valid], yc[valid]
            xi, yi = np.clip(xc.astype(int), 0, h_curr.shape[1] - 1), np.clip(yc.astype(int), 0, h_curr.shape[0] - 1)
            src = np.column_stack([xc, yc, h_curr[yi, xi]])
            dst = np.column_stack([xs.astype(float), ys.astype(float), h_prev[ys, xs]])

            t3 = time.perf_counter()
            sims = np.ones(len(src))
            patches_prev = [contour_patch(mask_prev, x, y) for x, y in zip(xs, ys)]
            patches_curr = [contour_patch(mask_curr, x, y) for x, y in zip(xc, yc)]
            z_prev = encode_patches(model, transform, device, patches_prev)
            z_curr = encode_patches(model, transform, device, patches_curr)
            dist = np.linalg.norm(z_prev - z_curr, axis=1)
            sims = 1.0 / (1.0 + dist)
            t4 = time.perf_counter()

            for name in ("wo_ae", "w_ae"):
                if name == "w_ae":
                    thresh = np.quantile(sims, 1.0 - args.keep)
                    keep = sims >= thresh
                    if keep.sum() >= 10:
                        R, t, inl = kabsch_ransac(src[keep], dst[keep])
                    else:
                        R, t, inl = kabsch_ransac(src, dst)
                else:
                    R, t, inl = kabsch_ransac(src, dst)
                if R is None:
                    T = np.eye(4)
                else:
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t
                variants[name].append(T)
            t5 = time.perf_counter()

            timings.append(
                {
                    "seq": seq,
                    "pair": pair_idx,
                    "n_points": len(src),
                    "t_gradient": t1 - t0,
                    "t_flow": t2 - t1,
                    "t_sample": t3 - t2,
                    "t_ae": t4 - t3,
                    "t_fit": t5 - t4,
                    "t_total_wo": (t5 - t0) - (t4 - t3),
                    "t_total_w": t5 - t0,
                }
            )

        for name, transforms in variants.items():
            out_dir = RESULTS / f"ae_e2e_{name}_{seq}"
            out_dir.mkdir(parents=True, exist_ok=True)
            traj = out_dir / f"{name}_cumulative_4x4.txt"
            with traj.open("w", encoding="utf-8") as fh:
                cumulative = np.eye(4)
                fh.write("frame_idx=0\n" + "\n".join(" ".join(f"{v:.9f}" for v in row) for row in cumulative) + "\n")
                for i, T in enumerate(transforms, start=1):
                    cumulative = cumulative @ T
                    fh.write(f"frame_idx={i}\n" + "\n".join(" ".join(f"{v:.9f}" for v in row) for row in cumulative) + "\n")
            import subprocess

            subprocess.run(
                [sys.executable, str(ROOT / "evaluate_pose.py"), "--predicted", str(traj),
                 "--ground-truth", str(gt), "--output", str(out_dir / "pose_metrics.json")],
                check=True,
                capture_output=True,
            )
            per_sequence.setdefault(seq, {})[name] = json.loads((out_dir / "pose_metrics.json").read_text())
        print(f"[done] {seq}")

    with (RESULTS / "ae_e2e_timings.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(timings[0].keys()))
        writer.writeheader()
        writer.writerows(timings)
    (RESULTS / "ae_e2e_per_sequence.json").write_text(json.dumps(per_sequence, indent=1))

    print("\n=== aggregate relative rotation error (deg) ===")
    for name in ("wo_ae", "w_ae"):
        vals = [per_sequence[s][name]["relative_rotation_error_deg"]["mean"] for s in per_sequence]
        print(f"{name}: mean={np.mean(vals):.4f} std={np.std(vals, ddof=1):.4f} per-seq={['%.3f' % v for v in vals]}")
    t = np.array([[r["t_gradient"], r["t_flow"], r["t_ae"], r["t_fit"], r["t_total_wo"], r["t_total_w"]] for r in timings])
    print("\n=== per-frame stage timings (ms, mean over %d pairs) ===" % len(timings))
    for label, col in zip(("gradient+height", "flow", "AE encode (batched)", "RANSAC+Kabsch", "total w/o AE", "total w/ AE"), t.T):
        print(f"{label}: {col.mean() * 1000:.1f}")


if __name__ == "__main__":
    main()
