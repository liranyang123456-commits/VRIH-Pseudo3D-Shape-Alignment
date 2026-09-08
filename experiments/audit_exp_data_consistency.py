from pathlib import Path
import json

root = Path(r'E:/elsarticle-template-TMI_Revised')
res = root / 'experiments' / 'results'

print("=== 1. Checking gradient statistics JSON ===")
gs = json.loads((res / 'gradient_statistics.json').read_text(encoding='utf-8'))
summary = gs['summary']
print("n_frames:", summary['n_frames'])
print("component_kurt median:", round(summary['component_kurt']['median'], 1))
print("magnitude_skew median:", round(summary['magnitude_skew']['median'], 1))

cvs = [v['cv'] for k, v in summary['t90_per_sequence'].items()]
print(f"t90 CV range: {min(cvs):.3f} -- {max(cvs):.3f}")

print("\n=== 2. Checking bulk fraction JSON ===")
bf = json.loads((res / 'bulk_fraction.json').read_text(encoding='utf-8'))
print("p68 retained:", bf['p68']['retained_fraction'])
print("p68 energy median:", round(bf['p68']['energy_fraction']['median'], 2))
print("p68 dice median:", round(bf['p68']['consecutive_dice']['median'], 2))
print("p68 hcr median:", round(bf['p68']['highlight_fraction']['median'], 3))
print("noise floor percentile median:", round(bf['noise_floor']['percentile_of_threshold']['median'], 1))

print("\n=== 3. Checking scale anchor JSON ===")
sa = json.loads((res / 'scale_anchor_summary.json').read_text(encoding='utf-8'))
print("R^2 affine fit:", round(sa['affine_r2'], 3))
print("Spearman rho mean:", round(sa['spearman_mean'], 2))

print("\n=== 4. Checking AE end-to-end aggregate JSON ===")
ae = json.loads((res / 'ae_e2e_aggregate.json').read_text(encoding='utf-8'))
print("wo_ae rot_mean:", round(ae['wo_ae']['rot_mean'], 3), "ate_mean:", round(ae['wo_ae']['ate_mean'], 2))
print("w_ae rot_mean:", round(ae['w_ae']['rot_mean'], 3), "ate_mean:", round(ae['w_ae']['ate_mean'], 2))
print("t-test p:", round(ae['paired']['t_p'], 2), "wilcoxon p:", round(ae['paired']['wilcoxon_p'], 2))
