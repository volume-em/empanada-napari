import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import fuse_modules, QuantStub, DeQuantStub
from empanada.models.quantization import encoders
from empanada.models.quantization.point_rend import QuantizablePointRendSemSegHead
from empanada.models.quantization.decoders import QuantizablePanopticDeepLabDecoder
from empanada.models.heads import PanopticDeepLabHead
from empanada.models.blocks import *
from typing import List, Dict

backbones = sorted(name for name in encoders.__dict__
    if not name.startswith("__")
    and callable(encoders.__dict__[name])
)

__all__ = [
    'QuantizablePanopticDeepLab',
    'QuantizablePanopticDeepLabPR'
]

def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6 or type(mod) == nn.SiLU:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value
    
class QuantizablePanopticDeepLab(nn.Module):
    def __init__(
        self,
        encoder='resnet50',
        num_classes=1,
        stage4_stride=16,
        decoder_channels=256,
        low_level_stages=[3, 2, 1],
        low_level_channels_project=[128, 64, 32],
        atrous_rates=[2, 4, 6],
        aspp_channels=None,
        ins_decoder=False,
        ins_ratio=0.5,
        quantize=False,
        **kwargs
    ):
        super(QuantizablePanopticDeepLab, self).__init__()
        
        assert (encoder in backbones), \
        f'Invalid encoder name {encoder}, choices are {backbones}'
        assert stage4_stride in [16, 32]
        assert min(low_level_stages) > 0
        
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes
        self.encoder = encoders.__dict__[encoder](output_stride=stage4_stride)
        
        self.semantic_decoder = QuantizablePanopticDeepLabDecoder(
            int(self.encoder.cfg.widths[-1]),
            decoder_channels,
            low_level_stages,
            [int(self.encoder.cfg.widths[i - 1]) for i in low_level_stages], 
            low_level_channels_project,
            atrous_rates, 
            aspp_channels
        )
        
        if ins_decoder:
            self.instance_decoder = QuantizablePanopticDeepLabDecoder(
                int(self.encoder.cfg.widths[-1]),
                decoder_channels,
                low_level_stages,
                [int(self.encoder.cfg.widths[i - 1]) for i in low_level_stages], 
                [int(s * ins_ratio) for s in low_level_channels_project],
                atrous_rates, 
                aspp_channels
            )
        else:
            self.instance_decoder = None
        
        self.semantic_head = PanopticDeepLabHead(decoder_channels, num_classes)
        self.ins_center = PanopticDeepLabHead(decoder_channels, 1)
        self.ins_xy = PanopticDeepLabHead(decoder_channels, 2)
        
        self.interpolate = Interpolate2d(4, mode='bilinear', align_corners=True)
            
        _replace_relu(self)
        
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()
    
    def fix_qconfig(self, observer='fbgemm'):
        self.qconfig = torch.quantization.get_default_qconfig(observer)

    def _encode_decode(self, x):
        pyramid_features: List[torch.Tensor] = self.encoder(x)
        
        semantic_x = self.semantic_decoder(pyramid_features)
        sem = self.semantic_head(semantic_x)
        
        if self.instance_decoder is not None:
            instance_x = self.instance_decoder(pyramid_features)
        else:
            # this shouldn't make a copy!
            instance_x = semantic_x
            
        return pyramid_features, semantic_x, instance_x
    
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
            
    def forward(self, x):
        x = self.quant(x)
        
        pyramid_features, semantic_x, instance_x = self._encode_decode(x)
        output: Dict[str, torch.Tensor] = self._apply_heads(semantic_x, instance_x)
            
        # dequant all outputs
        output = {k: self.dequant(v) for k,v in output.items()}
        
        return output
    
    def fuse_model(self):
        self.encoder.fuse_model()
        self.semantic_decoder.fuse_model()
        if self.instance_decoder is not None:
            self.instance_decoder.fuse_model()

class QuantizablePanopticDeepLabPR(QuantizablePanopticDeepLab):
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
        super(QuantizablePanopticDeepLabPR, self).__init__(**kwargs)
        
        # change semantic head from regular PDL head to 
        # PDL head + PointRend
        self.semantic_pr = QuantizablePointRendSemSegHead(
            self.decoder_channels, self.num_classes, num_fc,
            train_num_points, oversample_ratio, 
            importance_sample_ratio, subdivision_steps,
            subdivision_num_points, quantize=kwargs['quantize']
        )
        
    def fix_qconfig(self, observer='fbgemm'):
        self.encoder.qconfig = torch.quantization.get_default_qconfig(observer)
        self.semantic_decoder.qconfig = torch.quantization.get_default_qconfig(observer)
        if self.instance_decoder is not None:
            self.instance_decoder.qconfig = torch.quantization.get_default_qconfig(observer)
            
        self.semantic_head.qconfig = torch.quantization.get_default_qconfig(observer)
        self.ins_center.qconfig = torch.quantization.get_default_qconfig(observer)
        
        self.quant.qconfig = torch.quantization.get_default_qconfig(observer)
        self.dequant.qconfig = torch.quantization.get_default_qconfig(observer)
        
    def prepare_quantization(self):
        torch.quantization.prepare(self.encoder, inplace=True)
        torch.quantization.prepare(self.semantic_decoder, inplace=True)
        if self.instance_decoder is not None:
            torch.quantization.prepare(self.instance_decoder, inplace=True)
            
        torch.quantization.prepare(self.semantic_head, inplace=True)
        torch.quantization.prepare(self.ins_center, inplace=True)
        
        torch.quantization.prepare(self.quant, inplace=True)
        torch.quantization.prepare(self.dequant, inplace=True)
        
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
        offsets = self.ins_xy(self.dequant(instance_x))
        
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
            
            ctr_hmp = self.dequant(ctr_hmp)
            heads_out['ctr_hmp'] = self.interpolate(ctr_hmp) if interpolate_ins else ctr_hmp
            heads_out['offsets'] = self.interpolate(offsets) if interpolate_ins else offsets
        
        return heads_out
    
    def forward(self, x, render_steps: int=2, interpolate_ins: bool=True):
        if self.training:
            assert isinstance(self.quant, nn.Identity), \
            "Quantized trainining not supported!"
        
        x = self.quant(x)
            
        pyramid_features, semantic_x, instance_x = self._encode_decode(x)
        output: Dict[str, torch.Tensor] = self._apply_heads(
            semantic_x, instance_x, render_steps, interpolate_ins
        )
        
        return output
    
    def fuse_model(self):
        self.encoder.fuse_model()
        self.semantic_decoder.fuse_model()
        self.semantic_pr.fuse_model()
        if self.instance_decoder is not None:
            self.instance_decoder.fuse_model()
