import os
import cv2
import torch
import numpy as np
from skimage import io
from empanada.data._base import _BaseDataset
from empanada.data.utils import heatmap_and_offsets

__all__ = [
    'SingleClassInstanceDataset'
]

class SingleClassInstanceDataset(_BaseDataset):
    r"""Dataset for panoptic deeplab that supports a single instance
    class only.

    Args:
        data_dir: Str. Directory containing image/mask pairs. Structure should
        be data_dir -> source_datasets -> images/masks.

        transforms: Albumentations transforms to apply to images and masks.

        heatmap_sigma: Float. The standard deviation used for the gaussian
        blurring filter when converting object centers to a heatmap.

        weight_gamma: Float (0-1). Parameter than controls sampling of images
        within different source_datasets based on the number of images
        that that directory contains. Default is 0.3.

    """
    def __init__(
        self,
        data_dir,
        transforms=None,
        heatmap_sigma=6,
        weight_gamma=0.3,
    ):
        super(SingleClassInstanceDataset, self).__init__(
            data_dir, transforms, weight_gamma
        )
        self.heatmap_sigma = heatmap_sigma

    def __getitem__(self, idx):
        # transformed and paste example
        f = self.impaths[idx]
        image = cv2.imread(f, 0)
        mask = io.imread(self.mskpaths[idx])

        # add channel dimension if needed
        if image.ndim == 2:
            image = image[..., None]

        if self.transforms is not None:
            output = self.transforms(image=image, mask=mask)
        else:
            output = {'image': image, 'mask': mask}

        mask = output['mask']
        heatmap, offsets = heatmap_and_offsets(mask, self.heatmap_sigma)
        output['ctr_hmp'] = heatmap
        output['offsets'] = offsets
        output['fname'] = f

        # the last step is to binarize the mask for semantic segmentation
        if isinstance(mask, torch.Tensor):
            output['sem'] = (mask > 0).float()
        elif isinstance(mask, np.ndarray):
            output['sem'] = (mask > 0).astype(np.float32)
        else:
            raise Exception(f'Invalid mask type {type(mask)}. Expect tensor or ndarray.')

        return output
