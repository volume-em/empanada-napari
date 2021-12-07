import os
import yaml
import cv2
import dask.array as da
from torch.utils.data import Dataset

__all__ = [
    'get_configs',
    'load_config',
    'adjust_shape_by_scale',
    'resize'
    'ArrayData'
]

def get_configs():
    # get dict of all model configs
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    model_configs = {}
    for fn in os.listdir(config_path):
        if fn.endswith('.yaml'):
            model_configs[fn[:-len('.yaml')]] = os.path.join(config_path, fn)

    return model_configs

def load_config(url):
    with open(url, mode='r') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    return config

def take3d(array, indices, axis=0):
    """Similar to np.take but for 3d zarr arrays."""
    assert len(array.shape) == 3
    assert axis in [0, 1, 2]
    
    if axis == 0:
        x = array[indices]
    elif axis == 1:
        x = array[:, indices]
    else:
        x = array[..., indices]

    return x

def adjust_shape_by_scale(shape, scale):
    h, w = shape
    if h % 2 == 0:
        h = h * scale
    else:
        h = (h - 1) * scale

    if w % 2 == 0:
        w = w * scale
    else:
        w = 1 + (w - 1) * scale

    return (h, w)

def resize(image, scale_factor=1):
    # do nothing
    if scale_factor == 1:
        return image

    # cv2 expects (w, h) for image size
    h, w = image.shape
    dh = h // scale_factor
    dw = w // scale_factor

    image = cv2.resize(image, (dw, dh), cv2.INTER_LINEAR)

    return image

class ArrayData(Dataset):
    def __init__(self, array, scale=1, axis=0, tfs=None):
        super(ArrayData, self).__init__()
        self.array = array
        self.scale = scale
        self.axis = axis
        self.tfs = tfs
        
    def __len__(self):
        return self.array.shape[self.axis]
    
    def __getitem__(self, idx):
        # load the image
        image = take3d(self.array, idx, self.axis)

        # if dask, then call compute
        if type(image) == da.core.Array:
            image = image.compute()

        image = resize(image, self.scale)

        image = self.tfs(image=image)['image']
        
        return {'index': idx, 'image': image}