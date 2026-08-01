#!/bin/bash
# Run DetectorFreeSfM (restricted coarse-only) on staged chessboard segments 1-4.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vrih_dfsfm
cd /root/vrih_dfsfm
export COLMAP_PATH=/usr/bin/colmap
for n in 1 2 3 4; do
    echo "=== seq$n start $(date) ==="
    python eval_dataset.py +demo=dfsfm.yaml \
        dataset_base_dir=/root/vrih_dfsfm/SfM_dataset \
        dataset_name=vrih_chess_seq$n \
        "scene_list=[seq${n}_100f]" \
        output_base_dir=/root/vrih_dfsfm/outputs \
        use_prior_intrin=False \
        colmap_cfg.ImageReader_single_camera=True \
        neuralsfm.NEUSFM_enable_post_optimization=False \
        neuralsfm.NEUSFM_fine_match_type=coarse_only \
        neuralsfm.redo_all=True
    echo "=== seq$n exit=$? $(date) ==="
done
echo "ALL_DONE"
