#!/bin/bash
# Stage 100-frame GT-covered segments of chessboard sequences 1-4 for DetectorFreeSfM.
set -e
for cfg in "1 1 100" "2 204 303" "3 1 100" "4 1 100"; do
    set -- $cfg
    n=$1; a=$2; b=$3
    d=/root/vrih_dfsfm/SfM_dataset/vrih_chess_seq$n/seq${n}_100f/images
    mkdir -p "$d"
    i=$a
    while [ "$i" -le "$b" ]; do
        f=$(printf 'photo_%03d.jpg' "$i")
        cp "/mnt/d/reloc3r/Data_IMU_Camera_Pose_$n/captured_photos/$f" "$d/" 2>/dev/null || true
        i=$((i+1))
    done
    echo "seq$n: $(ls "$d" | wc -l) files"
done
