"""
Copied with modification from: 
https://github.com/pytorch/vision/blob/master/torchvision/models/quantization/resnet.py.
"""

import torch
import torch.nn as nn
from torch.quantization import fuse_modules
from empanada.models.encoders.resnet import Bottleneck, BasicBlock, ResNet

__all__ = [
    'QuantizableResNet', 
    'resnet50'
]

class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self) -> None:
        fuse_modules(self, [['conv1', 'bn1', 'relu'],
                            ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = self.skip_add_relu.add_relu(out, identity)

        return out

    def fuse_model(self) -> None:
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableResNet(ResNet):
    def __init__(self, *args, **kwargs):
        super(QuantizableResNet, self).__init__(*args, **kwargs)

    def fuse_model(self) -> None:
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                m.fuse_model()

def _resnet(
    arch,
    block,
    layers,
    pretrained=False,
    progress=True,
    **kwargs
):

    model = QuantizableResNet(block, layers, **kwargs)
    assert pretrained == False, "Loading quantized parameters not supported!"
    return model

def resnet34(
    pretrained=False,
    progress=True,
    **kwargs
):

    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _resnet('resnet34', QuantizableBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)



def resnet50(
    pretrained=False,
    progress=True,
    **kwargs
):

    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _resnet('resnet50', QuantizableBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
