"""Regenerate the gradient-statistics figure with the honest heavy-tail framing.

Panels: (a) log-density histogram of a real gradient component with a Gaussian
overlay (the tail mismatch is the evidence); (b) per-sequence t90 stability
(coefficient of variation); (c) downstream rotation error vs percentile
threshold (from the sensitivity sweep).
"""
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# (a) example frame component histogram
frame = sorted((ROOT / "data" / "scared_d1_k1" / "images").glob("*.png"))[10]
gray = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
g = gray.astype(np.float32) / 255.0
smooth = cv2.bilateralFilter(g, 7, 0.1, 3.0)
gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3).ravel()

# (b) t90 stability from the statistics JSON
stats_json = json.loads((RESULTS / "gradient_statistics.json").read_text())
seq_cv = stats_json["summary"]["t90_per_sequence"]

# (c) percentile sweep values
sweep = list(csv.DictReader((RESULTS / "sensitivity_sweep.csv").open()))
pct_rows = [r for r in sweep if r["parameter"] == "percentile"]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
ax = axes[0]
ax.hist(gx, bins=300, density=True, alpha=0.65, label="Gx (real endoscopic)")
xs = np.linspace(gx.min(), gx.max(), 500)
ax.plot(xs, stats.norm.pdf(xs, 0, gx.std()), "r--", label="Gaussian model")
ax.set_yscale("log")
ax.set_ylim(1e-1, None)
ax.set_title("Heavy-tailed gradient component (log scale)")
ax.set_xlabel("Gx response")
ax.set_ylabel("Density (log)")
ax.legend(fontsize=8)

ax = axes[1]
seqs = sorted(seq_cv)
cvs = [seq_cv[s]["cv"] for s in seqs]
labels = [s.replace("self_seq", "Self-").replace("scared_", "SC-").replace("scared_d", "D").replace("_k", "/K") for s in seqs]
ax.bar(range(len(seqs)), cvs, color="steelblue")
ax.set_xticks(range(len(seqs)))
ax.set_xticklabels(labels, rotation=45, fontsize=7)
ax.set_ylabel("Coefficient of variation of t90")
ax.set_title("t90 stability across frames per sequence")

ax = axes[2]
xs2 = [float(r["value"]) for r in pct_rows]
ys2 = [float(r["rel_rot_mean_deg"]) for r in pct_rows]
es2 = [float(r["rel_rot_std_deg"]) for r in pct_rows]
ax.errorbar(xs2, ys2, yerr=es2, marker="o", capsize=4)
ax.set_xlabel("Gradient percentile threshold")
ax.set_ylabel("Relative rotation error (deg)")
ax.set_title("Insensitivity to the percentile choice")
ax.set_xticks([85, 90, 95])
ax.set_xticklabels(["t85", "t90", "t95"])

fig.tight_layout()
fig.savefig(RESULTS / "fig_gradient_statistics.png", dpi=300)
print("wrote fig_gradient_statistics.png")
