"""
Simplification of https://github.com/rwightman/efficientdet-pytorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from empanada.models.blocks import *
from typing import List

__all__ = [
    'BiFPN',
    'BiFPNDecoder'
]

class TopDownFPN(nn.Module):
    def __init__(
        self,
        pyramid_nins,
        fpn_dim,
        depthwise=True
    ):
        super(TopDownFPN, self).__init__()
        
        # channel resampling layers for each scale
        self.resamplings = nn.ModuleList()
        for nin in pyramid_nins:
            self.resamplings.append(Resample2d(nin, fpn_dim))
            
        # resizing layer
        self.resize_up = Resize2d(2, up_or_down='up')
        
        if depthwise:
            conv_block = separable_conv_bn_act(fpn_dim, fpn_dim, 3, activation=nn.SiLU(inplace=True))
        else:
            conv_block = conv_bn_act(fpn_dim, fpn_dim, 3, activation=nn.ReLU(inplace=True))
            
        # convolution layers after combining feature maps
        self.after_combines = nn.ModuleList()
        for _ in range(len(pyramid_nins)):
            self.after_combines.append(conv_block)
            
        # fast-fusion weights, plus 1 for non resampled feature map
        self.weights = nn.Parameter(torch.ones(len(pyramid_nins) + 1), requires_grad=True)
        self.eps = 1e-4
        
    def forward(self, pyramid_features: List[torch.Tensor]):
        td_features = []
        
        # rescale and clip weights to positive
        weights = F.relu(self.weights)
        weight_sum = weights.sum()
        weights = weights / (weight_sum + self.eps)
        
        td_features = [pyramid_features[0]]
        
        # loop through the fpn
        for i, (resampling, after_combine) in enumerate(zip(self.resamplings, self.after_combines)):
            low_res_x = td_features[-1]
            # resample the high res features
            high_res_x = resampling(pyramid_features[i + 1])
            
            # fuse the features
            w1, w2 = weights[i], weights[i + 1]
            fused_numer = w1 * self.resize_up(low_res_x) + w2 * high_res_x
            fused_denom = w1 + w2 + self.eps
            fused = after_combine(fused_numer / fused_denom)
            td_features.append(fused)
            
        return td_features
    
class BottomUpFPN(nn.Module):
    def __init__(
        self,
        pyramid_nins,
        fpn_dim,
        depthwise=True
    ):
        super(BottomUpFPN, self).__init__()
        
        # channel resampling layers for each scale
        self.resamplings = nn.ModuleList()
        for nin in pyramid_nins:
            self.resamplings.append(Resample2d(nin, fpn_dim))
            
        # resizing layer
        self.resize_down = Resize2d(2, up_or_down='down')
        
        if depthwise:
            conv_block = separable_conv_bn_act(fpn_dim, fpn_dim, 3, activation=nn.SiLU(inplace=True))
        else:
            conv_block = conv_bn_act(fpn_dim, fpn_dim, 3, activation=nn.ReLU(inplace=True))
            
        # convolution layers after combining feature maps
        self.after_combines = nn.ModuleList()
        for _ in range(len(pyramid_nins)):
            self.after_combines.append(conv_block)
            
        # fast-fusion weights
        self.weights = nn.Parameter(torch.ones(len(pyramid_nins) + 1), requires_grad=True)
        self.eps = 1e-4
        
    def forward(self, pyramid_features: List[torch.Tensor], top_down_features: List[torch.Tensor]):
        # rescale and clip weights to positive
        weights = F.relu(self.weights)
        weight_sum = weights.sum()
        weights = weights / (weight_sum + self.eps)
        
        bu_features = [top_down_features[0]]
        
        # loop through the fpn
        for i, (resampling, after_combine) in enumerate(zip(self.resamplings, self.after_combines)):
            high_res_x = bu_features[-1]
            td_low_res_x = top_down_features[i + 1]
            
            # resample the high res features
            pyr_low_res_x = resampling(pyramid_features[i])
            
            # fuse the features
            if i < len(self.resamplings) - 1:
                # 3 feature maps to merge
                w1, w2, w3 = weights[i], weights[i + 1], weights[i + 2]
                fused_numer = w1 * self.resize_down(high_res_x) + w2 * pyr_low_res_x + w3 * td_low_res_x
                fused_denom = w1 + w2 + w3 + self.eps
            else:
                # only 2 feature maps to merge at the lowest
                # resolution level
                w1, w2 = weights[i], weights[i + 1]
                fused_numer = w1 * self.resize_down(high_res_x) + w2 * pyr_low_res_x
                fused_denom = w1 + w2 + self.eps
            
            fused = after_combine(fused_numer / fused_denom)
            bu_features.append(fused)
            
        return bu_features
        
class BiFPNLayer(nn.Module):
    def __init__(
        self,
        pyramid_nins, 
        fpn_dim,
        depthwise=True
    ):
        super(BiFPNLayer, self).__init__()
        self.top_down_fpn = TopDownFPN(pyramid_nins[::-1][1:], fpn_dim, depthwise=depthwise)
        self.bottom_up_fpn = BottomUpFPN(pyramid_nins[1:], fpn_dim, depthwise=depthwise)
        
    def forward(self, pyramid_features: List[torch.Tensor]):
        # order features from smallest to largest
        top_down_features: List[torch.Tensor] = self.top_down_fpn(pyramid_features[::-1])
        
        # first pyramid level features were already fused
        # in top-down FPN so we skip them
        # order top_down_features from largest to smallest
        bottom_up_features: List[torch.Tensor] = self.bottom_up_fpn(pyramid_features[1:], top_down_features[::-1])
        
        return bottom_up_features
    
class BiFPN(nn.Module):
    def __init__(
        self,
        pyramid_nins,
        fpn_dim,
        num_layers=3,
        depthwise=True
    ):
        super(BiFPN, self).__init__()
        
        # Adds additional coarse-grained feature maps at 1/64 and 1/128
        # the original image resolutions (P6 and P7, see paper)
        self.p6_resample = Resample2d(pyramid_nins[-1], fpn_dim)
        self.downsize = Resize2d(2, up_or_down='down')
        
        pyramid_nins += [fpn_dim, fpn_dim]
        self.bifpns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                bifpn = BiFPNLayer(pyramid_nins, fpn_dim, depthwise=depthwise)
            else:
                bifpn = BiFPNLayer(len(pyramid_nins) * [fpn_dim], fpn_dim, depthwise=depthwise)
            
            self.bifpns.append(bifpn)
                
    def forward(self, pyramid_features: List[torch.Tensor]):
        # calculate P6 and P7 features
        p6 = self.downsize(self.p6_resample(pyramid_features[-1]))
        p7 = self.downsize(p6)
        
        #pyramid_features.extend([p6, p7])
        pyramid_features = pyramid_features + [p6, p7]

        for bifpn in self.bifpns:
            pyramid_features = bifpn(pyramid_features)
            
        return pyramid_features
    
class BiFPNDecoder(nn.Module):
    def __init__(
        self, 
        fpn_dim,
        n_fpn_scales=5,
        conv_transpose=False
    ):
        """
        See https://github.com/google/automl/blob/be5340b61e84ae998765ad3340b633fcf57da87a/efficientdet/tf2/efficientdet_keras.py#L651
        """
        super(BiFPNDecoder, self).__init__()
        
        self.n_fpn_scales = n_fpn_scales
        self.upsamplings = nn.ModuleList()
        for i in range(n_fpn_scales):
            if i == 0:
                # first layer has nothing concatenated
                upsample = conv_transpose_bn_act(fpn_dim, fpn_dim, 2)
            else:
                upsample = conv_transpose_bn_act(2 * fpn_dim, fpn_dim, 2)
                
            self.upsamplings.append(upsample)
            
        # 5x5 like Panoptic DeepLab
        self.fusion = separable_conv_bn_act(2 * fpn_dim, fpn_dim, 5)
        
        self.apply(init_weights)
        
    def forward(self, fpn_features: List[torch.Tensor]):
        assert len(fpn_features) == self.n_fpn_scales + 1
        
        x = fpn_features[0]
        skips = fpn_features[1:]
        
        for i, upsample in enumerate(self.upsamplings):
            x = upsample(x)
            x = torch.cat([x, skips[i]], dim=1)
            
        return self.fusion(x)
    
def init_weights(m):
    """
    Weight initialization as per https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/efficientdet.py.
    """
    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., (fan_in + fan_out) / 2.)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., fan_in)  # fan in
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, SeparableConv2d):
        _glorot_uniform(m.sepconv[0].weight, groups=m.sepconv[0].groups) # grouped conv
        _glorot_uniform(m.sepconv[1].weight) # depth conv
    elif isinstance(m, nn.Conv2d):
        _glorot_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
