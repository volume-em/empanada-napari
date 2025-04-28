import torch
import torch.nn as nn
from empanada.models import encoders
from empanada.models.decoders import BiFPN, BiFPNDecoder
from empanada.models.heads import PanopticDeepLabHead
from empanada.models.point_rend import PointRendSemSegHead
from empanada.models.blocks import *
from empanada.models import encoders
from typing import List, Tuple
from copy import deepcopy

backbones = sorted(name for name in encoders.__dict__
    if not name.startswith("__")
    and callable(encoders.__dict__[name])
)

__all__ = [
    'PanopticBiFPN',
    'PanopticBiFPNPR'
]

class _BaseModel(nn.Module):
    def __init__(
        self,
        encoder='regnety_6p4gf',
        num_classes=1,
        fpn_dim=160,
        fpn_layers=3,
        ins_decoder=False,
        depthwise=True,
        **kwargs
    ):
        super(_BaseModel, self).__init__()
        
        assert (encoder in backbones), \
        f'Invalid encoder name {encoder}, choices are {backbones}'
        
        self.encoder = encoders.__dict__[encoder]()
        self.p2_resample = Resample2d(int(self.encoder.cfg.widths[0]), fpn_dim)
        self.num_classes = num_classes
        self.fpn_dim = fpn_dim

        # pass input channels from stages 2-4 only (1/8->1/32 resolutions)
        # N.B. EfficientDet BiFPN uses compound scaling rules that we ignore
        if 'resnet' in encoder:
            widths = self.encoder.cfg.widths[1:].tolist()
        else:
            widths = self.encoder.cfg.widths[1:]
            
        self.semantic_fpn = BiFPN(deepcopy(widths), fpn_dim, fpn_layers, depthwise=depthwise)
        self.semantic_decoder = BiFPNDecoder(fpn_dim)
        
        # separate BiFPN for instance-level predictions
        # following PanopticDeepLab
        if ins_decoder:
            self.instance_fpn = BiFPN(deepcopy(widths), fpn_dim, fpn_layers, depthwise=depthwise)
            self.instance_decoder = BiFPNDecoder(fpn_dim)
        else:
            self.instance_fpn = None
            
        self.semantic_head = PanopticDeepLabHead(fpn_dim, num_classes)
        self.ins_center = PanopticDeepLabHead(fpn_dim, 1)
        self.ins_xy = PanopticDeepLabHead(fpn_dim, 2)
        
        self.interpolate = Interpolate2d(4, mode='bilinear', align_corners=True)
            
    def _forward_encoder(self, x):
        return self.encoder(x)
            
    def _forward_decoders(self, x: List[torch.Tensor], p2_features):
        semantic_pyr = self.semantic_fpn(x)
        semantic_pyr = [p2_features] + semantic_pyr
        semantic_x = self.semantic_decoder(semantic_pyr[::-1])

        if self.instance_fpn is not None:
            instance_pyr = self.instance_fpn(x)
            instance_pyr = [p2_features] + instance_pyr
            instance_x = self.instance_decoder(instance_pyr[::-1])
        else:
            instance_x = semantic_x

        return semantic_x, instance_x
            
    def _apply_heads(self, semantic_x, instance_x):
        # apply the semantic head
        sem = self.semantic_head(semantic_x)
        ctr_hmp = self.ins_center(instance_x)
        offsets = self.ins_xy(instance_x)
       
        # return at original image resolution (4x)
        output = {}
        output['sem_logits'] = self.interpolate(sem)
        output['ctr_hmp'] = self.interpolate(ctr_hmp)
        output['offsets'] = self.interpolate(offsets)

        return output

    def forward(self, x):
        pyramid_features: List[torch.Tensor] = self._forward_encoder(x)
        p2_features = self.p2_resample(pyramid_features[1])
        
        # only passes features from
        # 1/8 -> 1/32 resolutions (i.e. P3-P5)
        semantic_x,  instance_x  = self._forward_decoders(pyramid_features[2:], p2_features)

        output = self._apply_heads(semantic_x, instance_x)
        
        return output
    
class PanopticBiFPN(_BaseModel):
    def __init__(
        self,
        encoder='regnety_6p4gf',
        num_classes=1,
        fpn_dim=160,
        fpn_layers=3,
        ins_decoder=False,
        **kwargs
    ):
        super(PanopticBiFPN, self).__init__(
            encoder,
            num_classes,
            fpn_dim,
            fpn_layers,
            ins_decoder,
            **kwargs
        )

class PanopticBiFPNPR(PanopticBiFPN):
    def __init__(
        self,
        num_fc=3,
        train_num_points=1024,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        subdivision_steps=2,
        subdivision_num_points=8192,
        **kwargs
    ):
        super(PanopticBiFPNPR, self).__init__(**kwargs)
        
        # change semantic head from regular PDL head to 
        # PDL head + PointRend
        self.semantic_pr = PointRendSemSegHead(
            self.fpn_dim, self.num_classes, num_fc,
            train_num_points, oversample_ratio, 
            importance_sample_ratio, subdivision_steps,
            subdivision_num_points
        )
        
    def _apply_heads(self, semantic_x, instance_x):
        heads_out = {}
        
        sem = self.semantic_head(semantic_x)
        ctr_hmp = self.ins_center(instance_x)
        offsets = self.ins_xy(instance_x)
        pr_out = self.semantic_pr(sem, semantic_x)
        
        if self.training:
            # interpolate to original resolution (4x)
            heads_out['sem_logits'] = self.interpolate(pr_out['sem_seg_logits'])
            heads_out['sem_points'] = pr_out['point_logits']
            heads_out['point_coords'] = pr_out['point_coords']
        else:
            # in eval mode interpolation is handled by point rend
            heads_out['sem_logits'] = pr_out['sem_seg_logits']
            
        # resize to original image resolution (4x)
        heads_out['ctr_hmp'] = self.interpolate(ctr_hmp)
        heads_out['offsets'] = self.interpolate(offsets)
        
        return heads_out
