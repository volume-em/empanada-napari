import torch
import torch.nn as nn
from torch.quantization import fuse_modules, QuantStub, DeQuantStub
from empanada.models.point_rend import (
    PointRendSemSegHead, calculate_uncertainty,
    get_uncertain_point_coords_on_grid, point_sample,
    get_uncertain_point_coords_with_randomness
)

__all__ = [
    'QuantizablePointRendSemSegHead'
]

class QuantizablePointRendSemSegHead(PointRendSemSegHead):
    def __init__(self, *args, **kwargs):
        super(QuantizablePointRendSemSegHead, self).__init__(*args, **kwargs)

    def fuse_model(self):
        # fuse the PointHead fc layers
        layers = range(len(self.point_head.fc_layers))
        for layer in layers:
            fuse_modules(self.point_head.fc_layers[layer], ['0', '1'], inplace=True)
