import numpy as np
import pytest

from empanada_napari.utils import Preprocessor


@pytest.mark.parametrize(
    'image',
    [
        np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        np.array([[0.0, 127.5, 255.0]], dtype=np.float32),
        np.array([[0, 127, 255]], dtype=np.uint8),
    ],
    ids=['float01', 'float255', 'uint8'],
)
def test_preprocessor_accepts_image_dtypes(image):
    preprocessor = Preprocessor(mean=0.5, std=0.25)
    result = preprocessor(image=image)

    assert 'image' in result
    assert result['image'].dtype == np.float32
    assert result['image'].shape[0] == 1
