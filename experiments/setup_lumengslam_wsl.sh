#!/bin/bash
# Setup LumenGSLAM in WSL (reuse torch, install pip deps + CUDA extensions)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vrih_dfsfm
cd /root
if [ ! -d LumenGSLAM ]; then
    git clone --depth 1 https://github.com/FrancescoLeni/LumenGSLAM.git
fi
cd LumenGSLAM
pip install imageio==2.37.0 kornia==0.8.0 lpips==0.1.4 pytorch_msssim==1.0.0 torchmetrics==1.7.1 tqdm opencv-python==4.11.0.86 2>&1 | tail -3
echo "=== installing CUDA extensions (this may take several minutes) ==="
pip install "git+https://github.com/VladimirYugay/simple-knn.git@c7e51a06a4cd84c25e769fee29ab391fe5d5ff8d" 2>&1 | tail -5
echo "SETUP_DONE"
