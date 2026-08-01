#!/bin/bash
# Export DetectorFreeSfM coarse models for chessboard segments 1-4.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vrih_dfsfm
for n in 1 2 3 4; do
    model=/root/vrih_dfsfm/SfM_dataset/vrih_chess_seq$n/seq${n}_100f/DetectorFreeSfM_loftr_official_coarse_only__scratch_no_intrin/colmap_coarse
    out=/mnt/e/elsarticle-template-TMI_Revised/experiments/results/detectorfreesfm_chess_seq$n/detectorfreesfm_cumulative_4x4.txt
    python /mnt/e/elsarticle-template-TMI_Revised/experiments/export_colmap_poses.py --model "$model" --output "$out"
    echo "seq$n exported: $?"
done
