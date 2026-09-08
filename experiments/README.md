# Revision experiment records

This directory contains the scripts and primary outputs used for the VRIH major
revision. It separates pseudo-3D shape alignment from metric camera-pose recovery.

## Core scripts

- `evaluate_pose.py`: evaluates a trajectory against ground truth using
  Sim(3)-aligned ATE, relative rotation error, and relative
  translation-direction error. Sim(3)-aligned ATE is a trajectory-shape measure,
  not metric pose accuracy for a pseudo-height trajectory.
- `evaluate_contour_hcr.py` and `test_hcr_significance.py`: compute NPC, HPC,
  HCR, uncertainty, and paired Wilcoxon tests (reproduces Table 2).
- `evaluate_chessboard_multi.py`: evaluates shape alignment on the five
  checkerboard sequences against calibration ground truth (reproduces Table 3).
- `evaluate_scared_multi.py`: evaluates shape alignment on the six public SCARED
  sequences against released camera poses (reproduces Table 4).
- `prepare_scared_extended.py`: extracts held-out SCARED keyframes from
  datasets 5, 6, and 7 with released camera-pose metadata (for Table 5).
- `evaluate_scared_extended.py`: evaluates the pipeline on extended SCARED
  datasets 5--7 keyframes (reproduces Table 5).
- `evaluate_scared_abrupt.py`: evaluates the pipeline on the abrupt-motion
  window of dataset 6 keyframe 1 containing a 6.55-degree jump (reproduces Table 5).
- `evaluate_stereo_pseudo3d.py`: evaluates the four SIFT/pseudo-height matching
  variants against released SCARED stereo calibration and depth ground truth (reproduces Table 6).
- `evaluate_stereo_pseudo3d_density.py`: sweeps dense, balanced, and selective
  pseudo-height matching profiles to quantify the density--quality tradeoff.
- `run_scared_pipeline_ablation.py`: evaluates nested geometric back-end configurations
  (dense flow + Kabsch, + RANSAC, + quality gating) on SCARED (reproduces Table 7).
- `run_scared_sensitivity.py`: sweeps gradient percentile, correspondence grid stride,
  pseudo-height scale, and RANSAC threshold (reproduces Table 8).
- `analyze_condition_stratification.py`: computes six-condition stratification and
  hard-quartile subsets across 474 SCARED frame pairs (reproduces Table 9).
- `analyze_self_acquired_cases.py`: evaluates internal alignment diagnostics
  (Kabsch residual, inlier ratio, gate freezes, HCR) on clinical sequences
  without pose ground truth (reproduces Table 10).
- `retrain_ae_variants.py`: retrains the full 225.6M SwinUNetLarge autoencoder
  under controlled variants (positional encoding: 2D vs (x+y) vs none;
  diffusion block: on vs off; split: frame vs sequence-level).
- `run_component_toggles.py`: evaluates direct component toggles (zero-flow
  control and ICP parameter sweep) on the six SCARED sequences.
- `analyze_gradient_statistics.py`: measures empirical gradient statistics,
  Rayleigh test, and threshold stability across 3,482 real endoscopic frames (Figure 4).
- `run_reloc3r_pairs.py`: runs official Reloc3r pairwise inference and exports a
  scale-ambiguous cumulative trajectory.
- `run_feature_pose_baseline.py`: calibrated SIFT, AKAZE, and ORB essential-matrix
  baselines.
- `run_pseudo3d_ablation.py`: controlled gradient-, intensity-, and
  constant-height pseudo-3D alignment ablation.
- `prepare_scared_sequence.py`: extracts RGB and released pose metadata from
  SCARED dataset 1/keyframe 1.

## Result provenance

- `results/chessboard_pose_summary_30f.csv`: 30-frame checkerboard comparison.
- `results/scared_d1_k1_pose_summary.csv`: 80-frame public SCARED comparison.
- `results/contour_hcr_summary.csv`: HCR mean, standard deviation, and median
  across 498 self-acquired endoscopic frames.
- `results/contour_hcr_wilcoxon.csv`: paired HCR significance tests.
- `results/fig_chessboard_trajectory_30f.png`: revised Figure 7 source.
- `results/scared_depth_structure_alignment.csv`: image-gradient versus released
  depth-discontinuity analysis on 15 SCARED keyframes. This diagnostic evaluates
  structural association only; it does not support metric-depth claims.
- `results/scared_stereo_pseudo3d_summary.csv`: calibrated stereo pose and
  triangulated-depth results for SIFT/pseudo-height matching variants.
- `results/scared_stereo_density_quality_summary.csv`: match density, GT
  epipolar-geometry precision, pose, and triangulated-depth tradeoff results.

## DetectorFreeSfM

`results/detectorfreesfm_chess_line2/REPRODUCTION_STATUS.md` records the WSL2
coarse-only reproduction. The Windows environment was incompatible with the
upstream dependency stack; the WSL run completed detector-free coarse matching
and COLMAP reconstruction after minimal HLoc/pycolmap compatibility adapters.
Fine refinement and post-optimization were disabled and are not claimed.
