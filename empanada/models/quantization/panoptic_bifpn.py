import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from empanada.models.quantization import encoders
from empanada.models.quantization.point_rend import QuantizablePointRendSemSegHead
from empanada.models import PanopticBiFPN
from empanada.models.blocks import *
from typing import List, Dict

__all__ = [
    'QuantizablePanopticBiFPN'
]

def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value

class QuantizablePanopticBiFPN(PanopticBiFPN):
    def __init__(
        self,
        quantize=False,
        **kwargs,
    ):
        super(QuantizablePanopticBiFPN, self).__init__(**kwargs)
        
        encoder = kwargs['encoder']
        # only the encoder is quantizable
        self.encoder = encoders.__dict__[encoder]()
        
        _replace_relu(self)
        
        self.interpolate = Interpolate2d(4, mode='bilinear', align_corners=True)

        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
    
    def fix_qconfig(self, observer='fbgemm'):
        # only the encoder gets quantized
        self.encoder.qconfig = torch.quantization.get_default_qconfig(observer)
        self.quant.qconfig = torch.quantization.get_default_qconfig(observer)
        self.dequant.qconfig = torch.quantization.get_default_qconfig(observer)
        
    def prepare_quantization(self):
        torch.quantization.prepare(self.encoder, inplace=True)
        torch.quantization.prepare(self.quant, inplace=True)
        torch.quantization.prepare(self.dequant, inplace=True)
    
    def _forward_encoder(self, x: torch.Tensor):
        x = self.quant(x)
        features: List[torch.Tensor] = self.encoder(x)
        return [self.dequant(t) for t in features]
    
    def _apply_heads(self, semantic_x, instance_x):
        heads_out = {}
        sem = self.semantic_head(semantic_x)
        ctr_hmp = self.ins_center(instance_x)
        offsets = self.ins_xy(instance_x)
        
        # return at quarter resolution
        heads_out['sem_logits'] = sem
        heads_out['ctr_hmp'] = ctr_hmp
        heads_out['offsets'] = offsets
        heads_out['semantic_x'] = semantic_x
        
        return heads_out
    
    def fuse_model(self):
        self.encoder.fuse_model()
            
class QuantizablePanopticBiFPNPR(QuantizablePanopticBiFPN):
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
        super(QuantizablePanopticBiFPNPR, self).__init__(**kwargs)
        
        # change semantic head from regular PDL head to 
        # PDL head + PointRend
        self.semantic_pr = QuantizablePointRendSemSegHead(
            self.fpn_dim, self.num_classes, num_fc,
            train_num_points, oversample_ratio, 
            importance_sample_ratio, subdivision_steps,
            subdivision_num_points, quantize=kwargs['quantize']
        )
        
    def _apply_heads(
        self, 
        semantic_x, 
        instance_x, 
        render_steps: int,
        interpolate_ins: bool
    ):
        heads_out = {}

        sem = self.semantic_head(semantic_x)
        ctr_hmp = self.ins_center(instance_x)
        offsets = self.ins_xy(instance_x)
        
        if self.training:
            pr_out: Dict[str, torch.Tensor] = self.semantic_pr(sem, semantic_x)
            heads_out['sem_points'] = pr_out['point_logits']
            heads_out['point_coords'] = pr_out['point_coords']
            
            # interpolate to original resolution (4x)
            heads_out['sem_logits'] = self.interpolate(pr_out['sem_seg_logits'])
            heads_out['ctr_hmp'] = self.interpolate(ctr_hmp)
            heads_out['offsets'] = self.interpolate(offsets)
            
            # dequant all outputs
            heads_out = {k: self.dequant(v) for k,v in heads_out.items()}
        else:
            # update the number of subdivisions
            self.semantic_pr.subdivision_steps = render_steps
            
            sem = self.dequant(sem)
            semantic_x = self.dequant(semantic_x)
            pr_out: Dict[str, torch.Tensor] = self.semantic_pr(
                sem, semantic_x
            )
            
            # in eval mode interpolation is handled by point rend
            heads_out['sem_logits'] = pr_out['sem_seg_logits']
            heads_out['ctr_hmp'] = self.interpolate(ctr_hmp) if interpolate_ins else ctr_hmp
            heads_out['offsets'] = self.interpolate(offsets) if interpolate_ins else offsets
        
        return heads_out
    
    def forward(self, x, render_steps: int=2, interpolate_ins: bool=True):
        if self.training:
            assert isinstance(self.quant, nn.Identity), \
            "Quantized trainining not supported!"
        
        pyramid_features: List[torch.Tensor] = self._forward_encoder(x)
        p2_features = self.p2_resample(pyramid_features[1])
        
        # only passes features from
        # 1/8 -> 1/32 resolutions (i.e. P3-P5)
        semantic_x,  instance_x  = self._forward_decoders(pyramid_features[2:], p2_features)

        output = self._apply_heads(semantic_x, instance_x, render_steps, interpolate_ins)
        
        return output
    
    def fuse_model(self):
        self.encoder.fuse_model()