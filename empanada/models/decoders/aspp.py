"""
Copied with minor modifications from:
https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/decoder/aspp.py
"""

# ------------------------------------------------------------------------------
# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'ASPP'
]


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.aspp_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.ReLU()
        )

    def set_image_pooling(self, pool_size=None):
        if pool_size is None:
            self.aspp_pooling[0] = nn.AdaptiveAvgPool2d(1)
        else:
            self.aspp_pooling[0] = nn.AvgPool2d(kernel_size=pool_size, stride=1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pooling(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        atrous_rates=(2, 4, 6),
        dropout_p=0.5
    ):
        r"""A atrous spatial pyramid pooling module.

        Args:
            in_channels (int) - Number of channels in the input image
            out_channels (int) - Number of channels produced by the convolution
            atrous_rates (list, tuple) - Convolution dilation rates for each pyramid
            level (there are typically 3 pyramid levels). Default (2, 4, 6).
            dropout_p (float) - Dropout probability for projection module
        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{in}, W_{in})`

        """
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

    def set_image_pooling(self, pool_size):
        self.convs[-1].set_image_pooling(pool_size)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
            
        res = torch.cat(res, dim=1)
        return self.project(res)