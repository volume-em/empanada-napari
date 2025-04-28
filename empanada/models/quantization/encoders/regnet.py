"""
Copied with modification from: 
https://github.com/pytorch/vision/blob/master/torchvision/models/quantization/resnet.py.
"""

import torch
import torch.nn as nn
from torch.quantization import fuse_modules
from empanada.models.encoders.regnet import (
    Stem, Bottleneck, BottleneckBlock, RegNet, RegNetConfig
)

__all__ = [
    'QuantizableRegNet',
    'regnetx_6p4gf',
    'regnety_6p4gf'
]

class QuantizableStem(Stem):
    def __init__(self, *args, **kwargs):
        super(QuantizableStem, self).__init__(*args, **kwargs)

    def fuse_model(self) -> None:
        fuse_modules(
            self.cbr, ['0', '1', '2'], inplace=True
        )
        
class QuantizableBottleneckBlock(BottleneckBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneckBlock, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        return self.skip_add_relu.add_relu(self.downsample(x), self.bottleneck(x))

    def fuse_model(self) -> None:
        fuse_modules(self.bottleneck.a, ['0', '1', '2'], inplace=True)
        fuse_modules(self.bottleneck.b, ['0', '1', '2'], inplace=True)
        if self.bottleneck.se is not None:
            fuse_modules(self.bottleneck.se.se, ['0', '1'], inplace=True)
        fuse_modules(self.bottleneck.c, ['0', '1'], inplace=True)
        if not isinstance(self.downsample.conv, nn.Identity):
            fuse_modules(self.downsample.conv, ['0', '1'], inplace=True)


class QuantizableRegNet(RegNet):
    def __init__(self, *args, **kwargs):
        super(QuantizableRegNet, self).__init__(*args, **kwargs)

    def fuse_model(self) -> None:
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(
            self.stem.cbr, ['0', '1', '2'], inplace=True
        )
        
        for m in self.modules():
            if type(m) == QuantizableBottleneckBlock:
                m.fuse_model()

def regnetx_6p4gf(**kwargs):
    params = {
        'depth': 17, 'w_0': 184, 'w_a': 60.83,
        'w_m': 2.07, 'group_w': 56
    }
    return QuantizableRegNet(RegNetConfig(**params, **kwargs), block=QuantizableBottleneckBlock)

def regnety_6p4gf(**kwargs):
    params = {
        'depth': 25, 'w_0': 112, 'w_a': 33.22,
        'w_m': 2.27, 'group_w': 72, 'use_se': True
    }
    return QuantizableRegNet(RegNetConfig(**params, **kwargs), block=QuantizableBottleneckBlock)
