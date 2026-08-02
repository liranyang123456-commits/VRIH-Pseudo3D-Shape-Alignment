#!/bin/bash
# Export DetectorFreeSfM coarse models for SCARED sequences (as they finish).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vrih_dfsfm
for name in d1_k1 d1_k2 d2_k1 d2_k2 d3_k1 d3_k2; do
    model=/root/vrih_dfsfm/SfM_dataset/vrih_scared_full/$name/DetectorFreeSfM_loftr_official_coarse_only__scratch_no_intrin/colmap_coarse
    out=/mnt/e/elsarticle-template-TMI_Revised/experiments/results/detectorfreesfm_scared_$name/detectorfreesfm_cumulative_4x4.txt
    if [ -d "$model" ]; then
        python /mnt/e/elsarticle-template-TMI_Revised/experiments/export_colmap_poses.py --model "$model" --output "$out"
        echo "$name exported"
    else
        echo "$name not ready"
    fi
done
