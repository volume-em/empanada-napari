"""
RegNet models from https://arxiv.org/abs/2103.06877 and
https://github.com/facebookresearch/pycls/blob/main/pycls/models/anynet.py

TODO:
Add scaling rules from RegNetZ
Add correct initialization for ResNet
"""

import numpy as np
import torch
import torch.nn as nn
from empanada.models.blocks import *

__all__ = [
    'RegNet',
    'regnetx_6p4gf',
    'regnety_200mf',
    'regnety_800mf',
    'regnety_3p2gf',
    'regnety_4gf',
    'regnety_6p4gf',
    'regnety_8gf',
    'regnety_16gf'
]

def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()

class Stem(nn.Module):
    """
    Simple input stem.
    """
    def __init__(self, w_in, w_out, kernel_size=3):
        super(Stem, self).__init__()
        self.cbr = conv_bn_act(w_in, w_out, kernel_size, stride=2)

    def forward(self, x):
        x = self.cbr(x)
        return x

class Bottleneck(nn.Module):
    """
    ResNet-style bottleneck block.
    """
    def __init__(
        self,
        w_in,
        w_out,
        bottle_ratio=1,
        groups=1,
        stride=1,
        use_se=False
    ):
        super(Bottleneck, self).__init__()
        w_b = int(round(w_out * bottle_ratio))
        self.a = conv_bn_act(w_in, w_b, 1)
        self.b = conv_bn_act(w_b, w_b, 3, stride=stride, groups=groups)
        self.se = SqueezeExcite(w_b) if use_se else None
        self.c = conv_bn_act(w_b, w_out, 1, activation=None)
        self.c[1].final_bn = True # layer 1 is the BN layer

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
            
        return x

class BottleneckBlock(nn.Module):
    def __init__(
        self,
        w_in,
        w_out,
        bottle_ratio,
        groups=1,
        stride=1,
        use_se=False
    ):
        super(BottleneckBlock, self).__init__()
        self.bottleneck = Bottleneck(w_in, w_out, bottle_ratio, groups, stride, use_se)
        self.downsample = Resample2d(w_in, w_out, stride=stride)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.downsample(x) + self.bottleneck(x))

class Stage(nn.Module):
    def __init__(
        self,
        block,
        w_in,
        w_out,
        depth,
        bottle_r=1,
        groups=1,
        stride=1,
        use_se=False
    ):
        super(Stage, self).__init__()

        assert depth > 0, "Each stage has minimum depth of 1 layer."

        for i in range(depth):
            if i == 0:
                # only the first layer in a stage 
                # has expansion and downsampling
                layer = block(w_in, w_out, bottle_r, groups, stride, use_se=use_se)
            else:
                layer = block(w_out, w_out, bottle_r, groups, use_se=use_se)

            self.add_module(f'block{i + 1}', layer)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)

        return x

class RegNet(nn.Module):
    """
    Simplest RegNetX/Y-like encoder without classification head
    """
    def __init__(
        self,
        cfg,
        im_channels=1,
        output_stride=32,
        block=BottleneckBlock
    ):
        super(RegNet, self).__init__()
        
        assert output_stride in [16, 32]
        if output_stride == 16:
            cfg.strides[-1] = 1

        # make the stages with correct widths and depths
        self.cfg = cfg
        groups = cfg.groups
        depths = cfg.depths
        w_ins = [cfg.w_stem] + cfg.widths[:-1]
        w_outs = cfg.widths
        strides = cfg.strides
        use_se = cfg.use_se
        
        self.stem = Stem(im_channels, cfg.w_stem, kernel_size=3)
        
        for i in range(cfg.num_stages):
            stage = Stage(block, w_ins[i], w_outs[i], depths[i],
                          groups=groups[i], stride=strides[i], use_se=use_se)

            self.add_module(f'stage{i + 1}', stage)
            
        self.apply(init_weights)

    def forward(self, x):
        pyramid_features = []
        for layer in self.children():
            x = layer(x)
            pyramid_features.append(x)

        return pyramid_features
    
class RegNetConfig:
    w_stem = 32
    bottle_ratio = 1
    strides = [2, 2, 2, 2]

    def __init__(
        self,
        depth,
        w_0,
        w_a,
        w_m,
        group_w,
        q=8,
        use_se=False,
        **kwargs
    ):
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
        self.depth = depth
        self.w_0 = w_0
        self.w_a = w_a
        self.w_m = w_m
        self.group_w = group_w
        self.q = q
        self.use_se = use_se
        
        for k,v in kwargs.items():
            setattr(self, k, v)

        self.set_params()
        self.adjust_params()

    def adjust_params(self):
        """
        Adjusts widths and groups to guarantee compatibility.
        """
        ws = self.widths
        gws = self.group_widths
        b = self.bottle_ratio

        adj_ws = []
        adj_groups = []
        for w, gw in zip(ws, gws):
            # group width can't exceed width
            # in the bottleneck
            w_b = int(max(1, w * b))
            gw = int(min(gw, w_b))

            # fix width s.t. it is always divisible by
            # group width for any bottleneck_ratio
            m = np.lcm(gw, b) if b > 1 else gw
            w_b = max(m, int(m * round(w_b / m)))
            w = int(w_b / b)

            adj_ws.append(w)
            adj_groups.append(w_b // gw)

        assert all(w * b % g == 0 for w, g in zip(adj_ws, adj_groups))
        self.widths = adj_ws
        self.groups = adj_groups

    def set_params(self):
        """
        Generates RegNet parameters following:
        https://arxiv.org/pdf/2003.13678.pdf
        """
        # capitals for complete sets
        # widths of blocks
        U = self.w_0 + np.arange(self.depth) * self.w_a # eqn (2)

        # quantize stages by solving eqn (3) for sj
        S = np.round(
            np.log(U / self.w_0) / np.log(self.w_m)
        )

        # block widths from eqn (4)
        W = self.w_0 * np.power(self.w_m, S)

        # round the widths to nearest factor of q
        # (makes best use of tensor cores)
        W = self.q * np.round(W / self.q).astype(int)

        # group stages by the quantized widths, use
        # as many stages as there are unique widths
        W, D = np.unique(W, return_counts=True)
        assert len(W) == 4, "Bad parameters, only 4 stage networks allowed!"

        self.num_stages = len(W)
        self.group_widths = len(W) * [self.group_w]
        self.widths = W.tolist()
        self.depths = D.tolist()
        
def regnetx_6p4gf(**kwargs):
    params = {
        'depth': 17, 'w_0': 184, 'w_a': 60.83,
        'w_m': 2.07, 'group_w': 56
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock)

def regnety_200mf(**kwargs):
    params = {
        'depth': 13, 'w_0': 24, 'w_a': 36.44,
        'w_m': 2.49, 'group_w': 8
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock)

def regnety_800mf(**kwargs):
    params = {
        'depth': 14, 'w_0': 56, 'w_a': 38.84,
        'w_m': 2.4, 'group_w': 16
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock)

def regnety_3p2gf(**kwargs):
    params = {
        'depth': 21, 'w_0': 80, 'w_a': 42.63,
        'w_m': 2.66, 'group_w': 24
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock)

def regnety_4gf(**kwargs):
    params = {
        'depth': 22, 'w_0': 96, 'w_a': 31.41,
        'w_m': 2.24, 'group_w': 64
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock, **kwargs)

def regnety_6p4gf(**kwargs):
    params = {
        'depth': 25, 'w_0': 112, 'w_a': 33.22,
        'w_m': 2.27, 'group_w': 72, 'use_se': True
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock)

def regnety_8gf(**kwargs):
    params = {
        'depth': 17, 'w_0': 192, 'w_a': 76.82,
        'w_m': 2.19, 'group_w': 56, 'use_se': True
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock)

def regnety_16gf(**kwargs):
    params = {
        'depth': 18, 'w_0': 200, 'w_a': 106.23,
        'w_m': 2.48, 'group_w': 112, 'use_se': True
    }
    return RegNet(RegNetConfig(**params, **kwargs), block=BottleneckBlock)
