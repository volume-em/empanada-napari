import math
import cv2
import dask.array as da
from torch.utils.data import Dataset
from empanada.array_utils import take
from empanada.data.utils import resize_by_factor

class VolumeDataset(Dataset):
    r"""Dataset for loading and transforming images cut from
    a particular index and axis of a volume.

    Args:
        array: Array of (d, h, w) defining an image volume. Can be
        np.ndarray, zarr.Array or dask.core.Array type.

        axis: Integer [0, 1, 2]. Axis to take slices from.

        tfs: Albumentations-like transforms to apply to the image slices.

        scale: Integer, exponent of 2 (e.g., 2, 4, 8, 16...). Downsampling
        to apply to the image before transformation.

    """
    def __init__(self, array, axis=0, tfs=None, scale=1):
        super(VolumeDataset, self).__init__()
        if not math.log(scale, 2).is_integer():
            raise Exception(f'Image rescaling must be log base 2, got {scale}')

        self.array = array
        self.axis = axis
        self.tfs = tfs
        self.scale = scale

    def __len__(self):
        return self.array.shape[self.axis]

    def __getitem__(self, idx):
        # load the image
        image = take(self.array, idx, self.axis)

        # if dask, then call compute
        if type(image) == da.core.Array:
            image = image.compute()

        # downsample image by scale
        h, w = image.shape
        image = resize_by_factor(image, self.scale)
        assert (image.shape[0] * self.scale) >= h
        assert (image.shape[1] * self.scale) >= w

        image = self.tfs(image=image)['image']

        return {'index': idx, 'image': image, 'size': (h, w)}
