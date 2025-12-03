import pytest
import numpy as np
from pytest import param
from empanada.zarr_utils import chunk_ranges, fill_func

@pytest.mark.parametrize(['ranges', 'modulo', 'divisor', 'expected'],
    [([[0, 20], [15, 35]], 7, 6, [[0, 6], [6, 7], [7, 13], [13, 14], 
                                  [14, 20], [15, 20], [20, 21], [21, 27], 
                                  [27, 28], [28, 34], [34, 35]])])
def test_chunk_ranges(ranges, modulo, divisor, expected):
    new_ranges = chunk_ranges(np.array(ranges), modulo, divisor)
    assert new_ranges == expected
    

@pytest.mark.parametrize(['seg1d', 'coords', 'instance_id', 'expected'],
    [([0, 0, 0, 0, 0], [(2, 5), (10, 13), (15, 18)], 7, [0, 0, 7, 7, 7])])
def test_fill_func(seg1d, coords, instance_id, expected):
    new_seg1d = fill_func(np.array(seg1d), np.array(coords), instance_id)
    assert np.array_equal(new_seg1d, expected)