import numpy as np
import pytest

dask = pytest.importorskip('dask')
da = pytest.importorskip('dask.array')
from dask import delayed

from empanada_napari._export_batch_segs import (
    _get_impaths_from_dask,
    _path_from_dask_task,
)


def test_path_from_legacy_tuple_task():
    assert _path_from_dask_task((lambda p: p, '/data/a.tif')) == '/data/a.tif'


def test_path_from_modern_task_object():
    @delayed
    def imread(path):
        return np.zeros((2, 2), dtype=np.uint8)

    arr = da.from_delayed(imread('/tmp/sample.tif'), shape=(2, 2), dtype=np.uint8)
    keys = [name for name in arr.dask.layers if 'imread' in str(name)]
    assert keys
    assert _path_from_dask_task(arr.dask[keys[0]]) == '/tmp/sample.tif'


def test_get_impaths_from_dask_stack_preserves_order():
    @delayed
    def imread(path):
        return np.zeros((4, 4), dtype=np.uint8)

    paths = ['/tmp/a.tif', '/tmp/b.tif', '/tmp/c.tif']
    stacked = da.stack([
        da.from_delayed(imread(p), shape=(4, 4), dtype=np.uint8) for p in paths
    ])
    assert _get_impaths_from_dask(stacked) == paths
