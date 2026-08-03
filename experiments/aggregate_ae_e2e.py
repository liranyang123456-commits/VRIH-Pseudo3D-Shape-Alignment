"""Aggregate AE end-to-end ablation results from saved pose_metrics.json files."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).parent / "results"
SEQS = ["scared_d1_k1", "scared_d1_k2", "scared_d2_k1", "scared_d2_k2", "scared_d3_k1", "scared_d3_k2"]

report = {}
for name in ("wo_ae", "w_ae"):
    vals, ates = [], []
    for s in SEQS:
        p = RESULTS / f"ae_e2e_{name}_{s}" / "pose_metrics.json"
        if not p.exists():
            print("missing", p)
            continue
        d = json.loads(p.read_text())
        vals.append(d["relative_rotation_error_deg"]["mean"])
        ates.append(d["ate_after_sim3_gt_units"]["rmse"])
    report[name] = {"rot_mean": float(np.mean(vals)), "rot_std": float(np.std(vals, ddof=1)), "ate_mean": float(np.mean(ates)), "per_seq_rot": vals}
    print(f"{name}: rot={np.mean(vals):.4f}+-{np.std(vals, ddof=1):.4f}  ate={np.mean(ates):.3f}  per-seq={['%.3f' % v for v in vals]}")

if all(k in report for k in ("wo_ae", "w_ae")):
    x = report["w_ae"]["per_seq_rot"]
    y = report["wo_ae"]["per_seq_rot"]
    if len(x) == len(y) and len(x) > 1:
        t_p = stats.ttest_rel(x, y).pvalue
        w_p = stats.wilcoxon(x, y).pvalue
        report["paired"] = {"n": len(x), "t_p": float(t_p), "wilcoxon_p": float(w_p), "mean_diff": float(np.mean(np.array(x) - np.array(y)))}
        print(f"paired w/ - w/o: diff={report['paired']['mean_diff']:+.4f} deg, t_p={t_p:.4g}, wilcoxon_p={w_p:.4g}")

(RESULTS / "ae_e2e_aggregate.json").write_text(json.dumps(report, indent=1))
print("written ae_e2e_aggregate.json")
