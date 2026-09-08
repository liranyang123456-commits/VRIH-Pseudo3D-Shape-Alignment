#!/usr/bin/env python3
"""Threshold-visualization figures for the R2 response (Comment 2.3).

Pipeline identical to the manuscript's t_main operator (and to the GT-mask
operator family of the companion tracking study): bilateral filter ->
Sobel gradient magnitude -> percentile threshold -> morphological closing.

Figure 1 (fig_threshold_panels.png): three example frames (self-acquired
endoscopy, SCARED public data, standardized checkerboard) thresholded at the
median (p50), p60, p68 (t_main), the frame mean, p90 (t_90) and p95.

Figure 2 (fig_gradient_3d_cutplanes.png): the gradient magnitude of one
example frame rendered as a 3D surface with horizontal cut planes at the same
six levels, visualizing how each threshold slices the gradient landscape.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
HUB = ROOT.parent / "vrih_experiment_hub" / "links"

EXAMPLES = [
    ("Self-acquired", HUB / "self_acquired_endoscopy" / "extracted_frames1" / "extracted_frames", 120),
    ("SCARED (public)", ROOT / "data" / "scared_d2_k1" / "images", 40),
    ("Checkerboard", ROOT / "data" / "chess_seq1_100f" / "images", 50),
]

LEVELS = [("Median (p50)", "p", 50), ("p60", "p", 60), (r"$t_{\mathrm{main}}$ (p68)", "p", 68),
          ("Mean", "mean", None), (r"$t_{90}$ (p90)", "p", 90), ("p95", "p", 95)]


def gradient_magnitude(gray_u8: np.ndarray) -> np.ndarray:
    filtered = cv2.bilateralFilter(gray_u8.astype(np.float32) / 255.0, 7, 0.1, 3.0)
    gx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def mask_at(mag: np.ndarray, kind: str, value) -> tuple[np.ndarray, float]:
    flat = mag.ravel()
    t = float(np.mean(flat)) if kind == "mean" else float(np.percentile(flat, value))
    mask = (mag > t).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    retained = float(np.mean(mask))
    return mask, retained


def load_example(directory: Path, index: int) -> np.ndarray:
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg"})
    return cv2.imread(str(paths[min(index, len(paths) - 1)]))


def make_panels() -> None:
    n_rows, n_cols = len(EXAMPLES), 2 + len(LEVELS)
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.15 * n_rows))
    for r, (name, directory, index) in enumerate(EXAMPLES):
        bgr = load_example(directory, index)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mag = gradient_magnitude(gray)
        axes[r, 0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        axes[r, 0].set_ylabel(name, fontsize=9)
        if r == 0:
            axes[r, 0].set_title("Input", fontsize=9)
        axes[r, 1].imshow(mag, cmap="magma")
        if r == 0:
            axes[r, 1].set_title(r"$|\nabla I|$", fontsize=9)
        for c, (label, kind, value) in enumerate(LEVELS):
            mask, retained = mask_at(mag, kind, value)
            axes[r, 2 + c].imshow(mask, cmap="gray")
            title = f"{label}\n({retained * 100:.0f}% kept)" if r == 0 else f"({retained * 100:.0f}% kept)"
            axes[r, 2 + c].set_title(title, fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    figure.tight_layout()
    figure.savefig(RESULTS / "fig_threshold_panels.png", dpi=250, bbox_inches="tight")
    plt.close(figure)
    print("wrote fig_threshold_panels.png")


def make_3d_cutplanes() -> None:
    bgr = load_example(EXAMPLES[0][1], EXAMPLES[0][2])
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mag = gradient_magnitude(gray)
    h, w = mag.shape
    cy, cx, half = h // 2, w // 2, 128
    crop = mag[cy - half:cy + half, cx - half:cx + half]
    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    crop = crop / (crop.max() + 1e-12)

    flat = crop.ravel()
    levels = []
    for label, kind, value in LEVELS:
        t = float(np.mean(flat)) if kind == "mean" else float(np.percentile(flat, value))
        clean = label.replace(r"$t_{\mathrm{main}}$ ", "t_main ").replace(r"$t_{90}$ ", "t_90 ")
        levels.append((clean, t))

    # Clip at the 99.5th percentile and display on a square-root z scale so the
    # near-floor bulk and the closely spaced lower cut planes stay readable
    # despite the heavy upper tail.
    zmax = float(np.percentile(flat, 99.5)) * 1.15
    surface_z = np.sqrt(np.minimum(crop, zmax) / zmax)

    xs, ys = np.meshgrid(np.arange(crop.shape[1]), np.arange(crop.shape[0]))
    figure = plt.figure(figsize=(10.5, 7.2))
    ax = figure.add_subplot(111, projection="3d")
    ax.plot_surface(xs, ys, surface_z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.8, rcount=96, ccount=96)
    plane_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]
    px, py = np.meshgrid([0, crop.shape[1] - 1], [0, crop.shape[0] - 1])
    handles = []
    for (label, t), color in sorted(zip(levels, plane_colors), key=lambda item: -item[0][1]):
        z_disp = float(np.sqrt(min(t, zmax) / zmax))
        ax.plot_surface(px, py, np.full_like(px, z_disp, dtype=float), color=color, alpha=0.3)
        handles.append(matplotlib.patches.Patch(color=color, alpha=0.6, label=f"{label}: {t:.3f}"))
    tick_values = [0.0, 0.01, 0.03, 0.07, 0.15, 0.3, 0.6]
    ax.set_zticks([float(np.sqrt(min(v, zmax) / zmax)) for v in tick_values])
    ax.set_zticklabels([f"{v:g}" for v in tick_values])
    ax.set_zlim(0, 1.0)
    ax.set(xlabel="x", ylabel="y")
    ax.set_zlabel(r"normalized $|\nabla I|$ (sqrt display scale)", labelpad=8)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.92), fontsize=9, framealpha=0.9,
              title="Threshold cut planes")
    ax.set_title("Gradient magnitude surface with threshold cut planes\n(self-acquired endoscopic frame, central crop)", fontsize=11)
    ax.view_init(elev=20, azim=-55)
    figure.tight_layout()
    figure.savefig(RESULTS / "fig_gradient_3d_cutplanes.png", dpi=250, bbox_inches="tight")
    plt.close(figure)
    print("wrote fig_gradient_3d_cutplanes.png")


if __name__ == "__main__":
    make_panels()
    make_3d_cutplanes()
