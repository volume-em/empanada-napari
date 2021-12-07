import os
import pathlib
from napari_plugin_engine import napari_hook_implementation
from ._test_model import test_widget
from ._stack_inference import stack_inference_widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def test():
    return test_widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def stack_inference():
    return stack_inference_widget

"""
@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def dims_sorter():
    return DimsSorter

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def axis_labels():
    return set_axis_labels

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def watershed():
    return watershed_split

@napari_hook_implementation(specname='napari_get_reader')
def zarr_tensorstore(path: str | pathlib.Path):
    if (str(path).endswith('.zarr') and os.path.isdir(path)
                and '.zarray' in os.listdir(path)):
        return lambda path: [
                (open_tensorstore(path), open_ts_meta(path), 'labels')
                ]

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def _add_points_callback():
    return add_points_3d_with_alt_click
"""