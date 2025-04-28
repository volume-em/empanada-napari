import os
import cv2
import torch
import numpy as np
from skimage import io
from skimage import measure
from empanada.data._base import _BaseDataset
from empanada.data.utils import heatmap_and_offsets

__all__ = [
    'PanopticDataset'
]

class PanopticDataset(_BaseDataset):
    r"""Dataset for panoptic deeplab that supports panoptic segmentation.

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
        labels,
        thing_list,
        label_divisor,
        transforms=None,
        heatmap_sigma=6,
        weight_gamma=0.3,
    ):
        super(PanopticDataset, self).__init__(
            data_dir, transforms, weight_gamma
        )

        assert len(labels) > 1, \
        "Must be more than 1 label class! Use SingleClassInstanceDataset instead."

        assert all([l > 0 for l in labels]), \
        "Labels must be positive non-zero integers!"

        self.labels = labels
        self.thing_list = thing_list
        self.label_divisor = label_divisor
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

        # thing and semantic seg
        if isinstance(mask, torch.Tensor):
            thing_seg = torch.zeros_like(mask)
            sem_seg = torch.zeros_like(mask)
        elif isinstance(mask, np.ndarray):
            thing_seg = np.zeros_like(mask)
            sem_seg = np.zeros_like(mask)

        # fill in segmentation classes
        for class_id in self.labels:
            min_id = class_id * self.label_divisor
            max_id = min_id + self.label_divisor

            inside = (mask >= min_id) * (mask < max_id)

            sem_seg[inside] = class_id
            if class_id in self.thing_list:
                thing_seg[inside] = mask[inside]

        heatmap, offsets = heatmap_and_offsets(thing_seg, self.heatmap_sigma)
        if isinstance(mask, torch.Tensor):
            output['sem'] = sem_seg.long()
        elif isinstance(mask, np.ndarray):
            output['sem'] = sem_seg.astype(np.int32)

        output['ctr_hmp'] = heatmap
        output['offsets'] = offsets

        # useful for debugging
        output['fname'] = f

        return output
