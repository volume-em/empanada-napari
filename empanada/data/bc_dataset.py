import os
import cv2
import torch
import numpy as np
from skimage import io
from skimage import measure
from empanada.data._base import _BaseDataset
from empanada.data.utils import seg_to_instance_bd

__all__ = [
    'BCDataset'
]

class BCDataset(_BaseDataset):
    def __init__(
        self,
        data_dir,
        transforms=None,
        weight_gamma=0.3
    ):
        r"""Dataset for boundary contour generation that supports a single instance
        class only.

        Args:
            data_dir: Str. Directory containing image/mask pairs. Structure should
            be data_dir -> source_datasets -> images/masks.

            transforms: Albumentations transforms to apply to images and masks.

            weight_gamma: Float (0-1). Parameter than controls sampling of images
            within different source_datasets based on the number of images
            that that directory contains. Default is 0.3.

        """
        super(BCDataset, self).__init__(
            data_dir, transforms, weight_gamma
        )

    def __getitem__(self, idx):
        # transformed and paste example
        f = self.impaths[idx]
        image = cv2.imread(f, 0)
        mask = cv2.imread(self.mskpaths[idx], -1)

        # add channel dimension if needed
        if image.ndim == 2:
            image = image[..., None]

        if self.transforms is not None:
            output = self.transforms(image=image, mask=mask)
        else:
            output = {'image': image, 'mask': mask}

        mask = output['mask']

        if isinstance(mask, torch.Tensor):
            contours = seg_to_instance_bd(mask.numpy()[None])[0]
            contours = torch.from_numpy(contours)
            output['sem'] = (mask > 0).float()
            output['cnt'] = (contours > 0).float()
        elif isinstance(mask, np.ndarray):
            contours = seg_to_instance_bd(mask[None])[0]
            output['sem'] = (mask > 0).astype(np.float32)
            output['cnt'] = (contours > 0).astype(np.float32)
        else:
            raise Exception(f'Invalid mask type {type(mask)}. Expect tensor or ndarray.')

        output['fname'] = f

        del output['mask']

        return output
