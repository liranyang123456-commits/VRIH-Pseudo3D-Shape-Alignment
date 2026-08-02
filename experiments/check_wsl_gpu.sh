#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vrih_dfsfm
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
