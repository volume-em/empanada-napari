import torch
import torch.nn as nn
from empanada.models.blocks import *

__all__ = [
    'PanopticDeepLabHead'
]

class PanopticDeepLabHead(nn.Module):
    def __init__(self, nin, n_classes):
        super(PanopticDeepLabHead, self).__init__()
        self.head = nn.Sequential(
            separable_conv_bn_act(nin, nin, 5),
            nn.Conv2d(nin, n_classes, 1, bias=True)
        )
        self.apply(init_weights)
        
    def forward(self, x):
        return self.head(x)
    
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.001)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
