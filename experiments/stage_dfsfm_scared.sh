#!/bin/bash
# Stage six SCARED sequences for DetectorFreeSfM coarse-only runs.
set -e
mkdir -p /root/vrih_dfsfm/SfM_dataset/vrih_scared_full
for name in d1_k1 d1_k2 d2_k1 d2_k2 d3_k1 d3_k2; do
    d=/root/vrih_dfsfm/SfM_dataset/vrih_scared_full/$name/images
    mkdir -p "$d"
    cp /mnt/e/elsarticle-template-TMI_Revised/experiments/data/scared_$name/images/*.png "$d/"
    echo "$name: $(ls "$d" | wc -l) frames"
done
