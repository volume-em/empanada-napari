import torch
import torch.nn as nn
from torch.quantization import fuse_modules
from empanada.models.decoders.aspp import ASPPConv, ASPPPooling
from empanada.models.decoders.panoptic_deeplab import PanopticDeepLabDecoder

__all__ = [
    'QuantizablePanopticDeepLabDecoder'
]

class QuantizablePanopticDeepLabDecoder(PanopticDeepLabDecoder):
    def __init__(self, *args, **kwargs):
        super(QuantizablePanopticDeepLabDecoder, self).__init__(*args, **kwargs)

    def fuse_model(self):
        # fuse the ASPP layers
        fuse_modules(self.aspp.convs[0], ['0', '1', '2'], inplace=True)
        fuse_modules(self.aspp.project, ['0', '1', '2'], inplace=True)
        for m in self.modules():
            if type(m) == ASPPPooling:
                fuse_modules(m.aspp_pooling, ['1', '2'], inplace=True)
            elif type(m) == ASPPConv:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
                
        # fuse projection layer
        fuse_modules(self.project[0], ['0', '1', '2'], inplace=True)
