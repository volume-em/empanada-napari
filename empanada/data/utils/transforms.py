import cv2
import math
import numpy as np

__all__ = [
    'resize_by_factor'
]

def resize_by_factor(image, scale_factor=1):
    # do nothing
    if scale_factor == 1:
        return image

    # cv2 expects (w, h) for image size
    h, w = image.shape
    dh = math.ceil(h / scale_factor)
    dw = math.ceil(w / scale_factor)

    image = cv2.resize(image, (dw, dh), cv2.INTER_LINEAR)

    return image

def factor_pad(image, factor=128):
    h, w = image.shape[:2]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if image.ndim == 3:
        padding = ((0, pad_bottom), (0, pad_right), (0, 0))
    elif image.ndim == 2:
        padding = ((0, pad_bottom), (0, pad_right))
    else:
        raise Exception

    padded_image = np.pad(image, padding)
    return padded_image

try:
    # only necessary for model training,
    # inference-only empanada doesn't need it
    import albumentations as A

    class FactorPad(A.Lambda):
        def __init__(self, factor=128):
            super().__init__(image=self.pad_func, mask=self.pad_func)
            self.factor = factor

        def pad_func(self, x, **kwargs):
            return factor_pad(x, factor=self.factor)
                
        __all__.append('FactorPad')
                            
except ImportError:
    pass
