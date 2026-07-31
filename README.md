# Pseudo-3D Shape Alignment for Endoscopic Image Sequences

This repository contains the implementation and evaluation scripts used for the
major revision of the manuscript *Nonrigid-Assisted Pseudo-3D Shape Alignment
for Endoscopic Image Sequences*.

## Scope

The central pipeline estimates a **relative shape-alignment transform** in the
non-metric pseudo-3D coordinate system `(u, v, h)`. The pseudo-height `h` is a
structural matching cue derived from image gradients; it is not metric depth and
the method does not directly recover a calibrated camera extrinsic.

## Repository layout

- `vrih_experiment_hub/src/ours/`: pseudo-height, mesh-alignment, checkerboard,
  and diagnostic implementation.
- `vrih_experiment_hub/src/semantic/`: hybrid autoencoder feature modules.
- `experiments/`: reproducible evaluation utilities for HCR, trajectory error,
  SCARED preparation, stereo matching, pseudo-height ablations, and plots.

## Evaluation protocol

The evaluation scripts support:

1. HCR (highlight-contamination ratio) plus paired Wilcoxon tests;
2. Sim(3)-aligned ATE, relative rotation error, and translation-direction error;
3. gradient/intensity/constant pseudo-height ablations;
4. calibrated SCARED stereo matching and triangulated-depth evaluation;
5. Reloc3r and classical feature baselines.

See [`experiments/README.md`](experiments/README.md) for the script map and
result provenance.

## Dependencies

The main pipeline was developed with Python, PyTorch, OpenCV, Open3D, NumPy,
Matplotlib, Albumentations, Pillow, SciPy, and tqdm. Install versions matching
your CUDA/PyTorch platform before running the neural modules. The basic
evaluation scripts require at least:

```text
numpy
opencv-python
matplotlib
scipy
open3d
torch
torchvision
albumentations
pillow
tqdm
```

DetectorFreeSfM helper files are compatibility patches for its upstream project;
they additionally require its released dependencies, HLoc, pycolmap, Ray,
PyTorch Lightning, OmegaConf, h5py, and loguru.

## Data, weights, and baselines

No image data, patient-related material, model checkpoint, or private absolute
path is included. Configure your own image, output, and checkpoint locations
through environment variables such as `INPUT_DIR`, `OUTPUT_DIR`, and
`MODEL_PATH`. Obtain public datasets and upstream baselines under their
respective licenses:

- [SCARED](https://endovissub2019-scared.grand-challenge.org/)
- [Reloc3r](https://github.com/ffrivera0/reloc3r)
- [DetectorFreeSfM](https://github.com/zju3dv/DetectorFreeSfM)

## Reproducibility note

The paper distinguishes pseudo-3D alignment from metric pose estimation.
Metric depth in the stereo experiment is reconstructed only through released
stereo calibration and triangulation; pseudo-height is used solely for matching.

## Citation

If you use this code, please cite the associated manuscript after publication.

