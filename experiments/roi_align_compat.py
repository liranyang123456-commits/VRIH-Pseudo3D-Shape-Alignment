"""Compatibility replacement for the legacy RoIAlign.pytorch extension.

DetectorFreeSfM imports ``roi_align.roi_align.RoIAlign``.  The original CUDA
extension is incompatible with modern PyTorch; this adapter delegates to the
maintained torchvision implementation while preserving the call signature used
by DetectorFreeSfM.
"""

import torch
from torch import nn
from torchvision.ops import roi_align as torchvision_roi_align


class RoIAlign(nn.Module):
    def __init__(self, crop_height, crop_width, transform_fpcoor=False):
        super().__init__()
        self.output_size = (int(crop_height), int(crop_width))
        self.aligned = bool(transform_fpcoor)

    def forward(self, features, boxes, box_index):
        rois = torch.cat(
            [box_index.to(dtype=features.dtype).reshape(-1, 1), boxes.to(features.dtype)],
            dim=1,
        )
        return torchvision_roi_align(
            features,
            rois,
            output_size=self.output_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=self.aligned,
        )
