import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'SeparableConv2d',
    'SqueezeExcite',
    'Resample2d',
    'Interpolate2d',
    'Resize2d',
    'separable_conv_bn_act',
    'conv_bn_act',
    'conv_transpose_bn_act'
]

class SeparableConv2d(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        kernel_size=3,
        stride=1,
        bias=True
    ):
        super(SeparableConv2d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.sepconv = nn.Sequential(
            nn.Conv2d(nin, nin, kernel_size, stride=stride, padding=padding, groups=nin, bias=bias),
            nn.Conv2d(nin, nout, 1, stride=1, bias=bias),
        )
        
    def forward(self, x):
        x = self.sepconv(x)
        return x
    
class SqueezeExcite(nn.Module):
    def __init__(self, nin):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AvgPool2d((1, 1))
        
        # hard code the squeeze factor at 4
        ns = nin // 4
        self.se = nn.Sequential(
            nn.Conv2d(nin, ns, 1, bias=True), # squeeze
            nn.ReLU(inplace=True),
            nn.Conv2d(ns, nin, 1, bias=True), # excite
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.se(self.avg_pool(x))
    
class Resample2d(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        stride=1,
        activation=None
    ):
        super(Resample2d, self).__init__()
        
        # convolution to downsample channels, if needed
        if nin != nout or stride > 1:
            self.conv = conv_bn_act(nin, nout, kernel_size=1, stride=stride, activation=activation)
        else:
            self.conv = nn.Identity()
            
    def forward(self, x):
        x = self.conv(x)
        return x
    
class Interpolate2d(nn.Module):
    def __init__(
        self,
        scale_factor,
        mode='nearest',
        align_corners=False
    ):
        super(Interpolate2d, self).__init__()
        self.scale_factor = float(scale_factor)
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, 
                          mode=self.mode, align_corners=self.align_corners)
        
        return x
    
class Resize2d(nn.Module):
    def __init__(self, scale_factor, up_or_down):
        super(Resize2d, self).__init__()
        
        assert up_or_down in ['up', 'down']
        self.up_or_down = up_or_down
        
        if up_or_down == 'up':
            resample = Interpolate2d(scale_factor, mode='nearest')
        else:
            kernel_size = 3
            resample = nn.MaxPool2d(kernel_size, stride=scale_factor, padding=kernel_size//2)
            
        self.resample = resample
            
    def forward(self, x):
        return self.resample(x)
    
def separable_conv_bn_act(
    nin,
    nout,
    kernel_size,
    stride=1,
    activation=nn.ReLU(inplace=True)
):
    """
    Following https://arxiv.org/pdf/2003.13678.pdf Swish/SiLU activation
    is preferred for depthwise convolutions (size_group = 1).
    
    Following regnet standards, also ignoring the different momentum
    and epsilon parameters for BatchNorm in EfficientDet
    """
    # separable convolution and batchnorm
    layers = [
        SeparableConv2d(nin, nout, kernel_size, stride, bias=False),
        nn.BatchNorm2d(nout),
    ]
    
    # add activation if needed
    if activation is not None:
        layers.append(activation)
        
    return nn.Sequential(*layers)

def conv_bn_act(
    nin,
    nout,
    kernel_size,
    stride=1,
    groups=1,
    activation=nn.ReLU(inplace=True)
):
    padding = (kernel_size - 1) // 2
    # regular convolution and batchnorm
    layers = [
        nn.Conv2d(nin, nout, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(nout)
    ]
    
    # add activation if necessary
    if activation:
        layers.append(activation)
        
    return nn.Sequential(*layers)

def conv_transpose_bn_act(
    nin,
    nout,
    kernel_size,
    activation=nn.ReLU(inplace=True)
):
    # regular transposed convolution and batchnorm
    layers = [
        nn.ConvTranspose2d(nin, nout, kernel_size, stride=kernel_size, bias=False),
        nn.BatchNorm2d(nout)
    ]
    
    # add activation if necessary
    if activation:
        layers.append(activation)
        
    return nn.Sequential(*layers)
