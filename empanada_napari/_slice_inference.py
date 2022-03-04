import sys
import yaml
import os
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

import napari
from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui

from magicgui.tqdm import tqdm

import zarr
import dask.array as da

def test_widget():
    import cv2
    from time import time
    from torch.cuda import device_count
    from napari.qt.threading import thread_worker

    # Import when users activate plugin
    from empanada_napari.inference import Engine2d
    from empanada_napari.utils import get_configs, abspath
    from empanada.config_loaders import read_yaml

    # get list of all available model configs
    model_configs = get_configs()

    @thread_worker
    def test_model(
        engine,
        image,
        axis,
        plane
    ):
        # create the inference engine
        start = time()
        seg = engine.infer(image)
        print(f'Inference time:', time() - start)
        return seg, axis, plane

    logo = abspath(__file__, 'resources/empanada_logo.png')

    gui_params = dict(
        model_config=dict(widget_type='ComboBox', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], label='Model', tooltip='Model to use for inference'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Image Downsampling', tooltip='Downsampling factor to apply before inference'),
        confidence_thr=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1, label='Segmentation Confidence Thr'),
        center_confidence_thr=dict(widget_type='FloatSpinBox', value=0.1, min=0.05, max=0.9, step=0.05, label='Center Confidence Thr'),
        min_distance_object_centers=dict(widget_type='SpinBox', value=3, min=1, max=21, step=1, label='Centers Min Distance'),
        fine_boundaries=dict(widget_type='CheckBox', text='Fine boundaries', value=False, tooltip='Finer boundaries between objects'),
        semantic_only=dict(widget_type='CheckBox', text='Semantic only', value=False, tooltip='Only run semantic segmentation for all classes.'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value='100000', label='Max objects per class'),
    )

    gui_params['use_gpu'] = dict(widget_type='CheckBox', text='Use GPU', value=device_count() >= 1, tooltip='If checked, run on GPU 0')

    @magicgui(
        label_head= dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Run 2D Inference',
        layout='vertical',
        **gui_params
    )
    def widget(
        viewer: napari.viewer.Viewer,
        label_head,
        image_layer: Image,
        model_config,
        downsampling,
        confidence_thr,
        center_confidence_thr,
        min_distance_object_centers,
        fine_boundaries,
        semantic_only,
        maximum_objects_per_class,
        use_gpu
    ):
        # load the model config
        model_config = read_yaml(model_configs[model_config])
        maximum_objects_per_class = int(maximum_objects_per_class)

        if not hasattr(widget, 'last_config'):
            widget.last_config = model_config

        if not hasattr(widget, 'using_gpu'):
            widget.using_gpu = use_gpu

        if not hasattr(widget, 'engine') or widget.last_config != model_config or use_gpu != widget.using_gpu:
            widget.engine = Engine2d(
                model_config,
                inference_scale=downsampling,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                label_divisor=maximum_objects_per_class,
                semantic_only=semantic_only,
                fine_boundaries=fine_boundaries,
                use_gpu=use_gpu
            )
            widget.last_config = model_config
            widget.using_gpu = use_gpu
        else:
            # update the parameters of the engine
            # without reloading the model
            widget.engine.update_params(
                inference_scale=downsampling,
                label_divisor=maximum_objects_per_class,
                nms_threshold=center_confidence_thr,
                nms_kernel=min_distance_object_centers,
                confidence_thr=confidence_thr,
                semantic_only=semantic_only,
                fine_boundaries=fine_boundaries
            )

        def _get_current_slice(image_layer):
            axis = viewer.dims.order[0]
            cursor_pos = viewer.cursor.position

            # handle multiscale by taking highest resolution level
            image = image_layer.data
            if image_layer.multiscale:
                print(f'Multiscale image selected, using highest resolution level!')
                image = image[0]

            if image.ndim == 3:
                plane = int(image_layer.world_to_data(cursor_pos)[axis])

                slices = [slice(None), slice(None), slice(None)]
                slices[axis] = plane
            else:
                slices = [slice(None), slice(None)]
                axis = None
                plane = None

            return image[tuple(slices)], axis, plane

        def _show_test_result(*args):
            seg, axis, plane = args[0]

            if axis is not None and plane is not None:
                seg = np.expand_dims(seg, axis=axis)
                translate = [0, 0, 0]
                translate[axis] = plane
            else:
                translate = [0, 0]

            viewer.add_labels(seg, name=f'Test Seg', visible=True, translate=tuple(translate))

        # load data for currently viewer slice of chosen image layer
        image2d, axis, plane = _get_current_slice(image_layer)
        if type(image2d) == da.core.Array:
            image2d = image2d.compute()

        test_worker = test_model(widget.engine, image2d, axis, plane)
        test_worker.returned.connect(_show_test_result)
        test_worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def slice_dock_widget():
    return test_widget, {'name': '2D Inference (Parameter Testing)'}
