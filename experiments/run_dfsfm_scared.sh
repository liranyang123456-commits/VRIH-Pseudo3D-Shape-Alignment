#!/bin/bash
# Run DetectorFreeSfM (restricted coarse-only) on six SCARED sequences.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vrih_dfsfm
cd /root/vrih_dfsfm
export COLMAP_PATH=/usr/bin/colmap
for name in d1_k1 d1_k2 d2_k1 d2_k2 d3_k1 d3_k2; do
    echo "=== $name start $(date) ==="
    python eval_dataset.py +demo=dfsfm.yaml \
        dataset_base_dir=/root/vrih_dfsfm/SfM_dataset \
        dataset_name=vrih_scared_full \
        "scene_list=[$name]" \
        output_base_dir=/root/vrih_dfsfm/outputs \
        use_prior_intrin=False \
        colmap_cfg.ImageReader_single_camera=True \
        neuralsfm.NEUSFM_enable_post_optimization=False \
        neuralsfm.NEUSFM_fine_match_type=coarse_only \
        neuralsfm.redo_all=True
    echo "=== $name exit=$? $(date) ==="
done
echo "ALL_DONE"
