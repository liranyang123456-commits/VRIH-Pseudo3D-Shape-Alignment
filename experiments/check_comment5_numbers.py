#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

rows = list(
    csv.DictReader(
        Path(r"E:/elsarticle-template-TMI_Revised/experiments/results/scared_pipeline_ablation.csv").open(
            encoding="utf-8"
        )
    )
)
grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"rot": [], "ate": []})
for row in rows:
    grouped[row["variant"]]["rot"].append(float(row["rot_mean_deg"]))
    grouped[row["variant"]]["ate"].append(float(row["ate_rmse"]))
for key, vals in grouped.items():
    print(
        key,
        f"rot {np.mean(vals['rot']):.3f} +/- {np.std(vals['rot'], ddof=1):.3f}",
        f"ate {np.mean(vals['ate']):.3f} +/- {np.std(vals['ate'], ddof=1):.3f}",
    )
ransac = [r for r in rows if r["variant"] == "ransac"]
gate = [r for r in rows if r["variant"] == "ransac_gate"]
print(
    "ransac==gate",
    all(
        a["rot_mean_deg"] == b["rot_mean_deg"] and a["ate_rmse"] == b["ate_rmse"]
        for a, b in zip(ransac, gate)
    ),
)

paper = Path(r"E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex").read_text(
    encoding="utf-8"
)
marked = Path(r"E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2_markedup.tex").read_text(
    encoding="utf-8"
)
clean = Path(r"E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2_clean.tex").read_text(
    encoding="utf-8"
)
print("clean==paper", clean == paper)
start = paper.find("Ablation and sensitivity analysis")
chunk = paper[start : start + 2800]
print("recovery in ablation section", "recovery" in chunk.lower())
print("ICP-not-swept sentence", "invariant to the ICP parameters by construction" in paper)
i = marked.find("Geometric back-end decomposition")
print("marked near decomp", repr(marked[max(0, i - 40) : i + 20]))
print("tab:pipeline_ablation in marked", "tab:pipeline_ablation" in marked)
print("tab:sensitivity in marked", "tab:sensitivity" in marked)
