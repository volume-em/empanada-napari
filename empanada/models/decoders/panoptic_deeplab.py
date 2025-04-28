"""
Copied with modification from:
https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/decoder/panoptic_deeplab.py
"""

# ------------------------------------------------------------------------------
# Panoptic-DeepLab decoder.
# Written by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F
from empanada.models.decoders.aspp import ASPP
from empanada.models.blocks import *
from typing import List


__all__ = [
    'PanopticDeepLabDecoder'
]

class PanopticDeepLabDecoder(nn.Module):
    def __init__(
        self, 
        in_channels,
        decoder_channels,
        low_level_stages,
        low_level_channels, 
        low_level_channels_project,
        atrous_rates, 
        aspp_channels=None,
        aspp_dropout=0.5
    ):
        super(PanopticDeepLabDecoder, self).__init__()
        
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels, out_channels=aspp_channels, 
                         atrous_rates=atrous_rates, dropout_p=aspp_dropout)
        
        assert len(low_level_stages) == len(low_level_channels)
        self.low_level_stages = low_level_stages
        self.low_level_channels = low_level_channels
        self.decoder_stages = len(low_level_channels)
        
        # Transform low-level feature
        # Top-down direction, i.e. starting from largest stride
        project = []
        fuse = []
        for i in range(self.decoder_stages):
            project.append(conv_bn_act(low_level_channels[i], low_level_channels_project[i], 1))
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
                
            fuse.append(separable_conv_bn_act(fuse_in_channels, decoder_channels, 5))
            
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)
        
        self.apply(init_weights)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, pyramid_features: List[torch.Tensor]):
        x = pyramid_features[-1]
        x = self.aspp(x)

        # run decoder
        for i, (proj, fuse) in enumerate(zip(self.project, self.fuse)):
            l = pyramid_features[self.low_level_stages[i]]
            l = proj(l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = fuse(x)

        return x
    
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
