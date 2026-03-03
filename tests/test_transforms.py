import pytest
import numpy as np
from empanada.data.utils.transforms import resize_by_factor


@pytest.mark.parametrize(['image', 'scale_factor', 'expected'],
    [([[10., 20.], [30., 40.]], 1, [[10., 20.], [30., 40.]]),
     ([[10., 20.], [30., 40.]], 0.5, [[10., 12.5, 17.5, 20.], 
                                      [15., 17.5, 22.5, 25.],
                                      [25., 27.5, 32.5, 35.], 
                                      [30., 32.5, 37.5, 40.]]),
     ([[10., 20.], [30., 40.]], 2, [[25.]])],
     ids=["scale_factor_1", "scale_factor_0.5", "scale_factor_2"])
def test_resize_by_factor(image, scale_factor, expected):
    """
    Test `resize_by_factor` function.

    The function should:
    - read in an image array containing float64 values
    - return an image array resized based on the scale_factor

    Expected:
    - A rescaled image array with shape dimensions equal to
      the original image's dimensions divided by the scale_factor
    """
    resized_img = resize_by_factor(np.array(image), scale_factor)
    assert np.array_equal(resized_img, expected)
    assert resized_img.shape == np.array(expected).shape
