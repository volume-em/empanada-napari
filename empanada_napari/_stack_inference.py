"""
cellpose dock widget module
"""
import sys
import os
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import time
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

import napari
from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui

from magicgui.tqdm import tqdm

import zarr
import dask.array as da

def stack_inference_widget():
    from napari.qt.threading import thread_worker

    # Import when users activate plugin
    from empanada_napari.orthoplane import OrthoPlaneEngine, tracker_consensus, stack_postprocessing
    from empanada_napari.utils import get_configs, load_config
    from mitonet.inference import filters
    
    # get list of all available model configs
    model_configs = get_configs()

    @thread_worker
    def stack_inference(engine, volume):
        stack, trackers = engine.infer_on_axis(volume, 'xy')
        trackers_dict = {'xy': trackers}
        return stack, trackers_dict

    @magicgui(
        call_button='Run XY Stack Inference',
        layout='vertical',
        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='Model to use for inference'),
        median_slices=dict(widget_type='ComboBox', choices=[1, 3, 5, 7, 9], value=3, label='Median filter size', tooltip='Median filter size'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Downsampling before inference', tooltip='Downsampling factor to apply before inference'),
        min_distance_object_centers=dict(widget_type='Slider', value=3, min=3, max=21, label='Minimum distance between object centers.'),
        confidence_thr=dict(widget_type='FloatSlider', value=0.5, min=0.1, max=0.9, step=0.1, label='Confidence Threshold'),
        center_confidence_thr=dict(widget_type='FloatSlider', value=0.1, min=0.1, max=0.9, label='Center Confidence Threshold'),
        merge_iou_thr=dict(widget_type='FloatSlider', value=0.25, max=0.9, label= 'IoU Threshold'),
        merge_ioa_thr=dict(widget_type='FloatSlider', value=0.25, max=0.9, label= 'IoA Threshold'),
        min_size=dict(widget_type='LineEdit', value=500, label='Minimum object size in voxels'),
        min_extent=dict(widget_type='LineEdit', value=4, label='Minimum extent of object bounding box'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value=20000, label='Max objects per class'),
        store_dir=dict(widget_type='FileEdit', value='./', label='Store Directory', mode='d', tooltip='location to store segmentations on disk'),
        overwrite=dict(widget_type='CheckBox', text='Overwrite stored files?', value=False, tooltip='whether to overwrite zarr stores in store dir'),
        use_gpu=dict(widget_type='CheckBox', text='Use GPU?', value=True, tooltip='Run inference on GPU, if available.')
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        model_config,
        median_slices,
        downsampling,
        min_distance_object_centers,
        confidence_thr,
        center_confidence_thr,
        merge_iou_thr,
        merge_ioa_thr,
        min_size,
        min_extent,
        maximum_objects_per_class,
        store_dir,
        overwrite,
        use_gpu
    ):
        # load the model config
        model_config_name = model_config
        model_config = load_config(model_configs[model_config])
        min_size = int(min_size)
        min_extent = int(min_extent)
        maximum_objects_per_class = int(maximum_objects_per_class)

        # create the storage url from layer name and model config
        store_dir = str(store_dir)
        #if store_dir:
        store_url = os.path.join(store_dir, f'{image_layer.name}_{model_config_name}_napari.zarr')
        if os.path.isdir(store_url) and not overwrite:
            raise Exception(f"{store_url} exists, if you want to overwrite it check 'Overwrite stored files?' ")
        elif os.path.isdir(store_url):
            print(f'Overwriting segmentations stored in {store_url}')
        #else:
        #    store_url = None
        #    print(f'Running without store_dir, this may use a lot of RAM!')

        if not hasattr(widget, 'last_config'):
            widget.last_config = model_config

        if not hasattr(widget, 'using_gpu'):
            widget.using_gpu = use_gpu

        # conditions where model needs to be (re)loaded
        if not hasattr(widget, 'engine') or widget.last_config != model_config or use_gpu != widget.using_gpu:
            widget.engine = OrthoPlaneEngine(
                store_url, model_config,
                inference_scale=downsampling,
                median_kernel_size=median_slices,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                merge_iou_thr=merge_iou_thr,
                merge_ioa_thr=merge_ioa_thr,
                label_divisor=maximum_objects_per_class,
                use_gpu=use_gpu
            )
            widget.last_config = model_config
            widget.using_gpu = use_gpu
        else:
            # update the parameters
            widget.engine.update_params(
                store_url=store_url,
                inference_scale=downsampling,
                median_kernel_size=median_slices,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                merge_iou_thr=merge_iou_thr,
                merge_ioa_thr=merge_ioa_thr,
                label_divisor=maximum_objects_per_class
            )

        def _new_layers(mask, description):
            layers = []

            if type(mask) == zarr.core.Array:
                mask = da.from_zarr(mask)

            viewer.add_labels(mask, name=f'{image_layer.name}-{description}', visible=True)

        def _new_segmentation(*args):
            mask = args[0][0]
            try:
                _new_layers(mask, f'panoptic-xy-stack')

                for layer in viewer.layers:
                    layer.visible = False

                viewer.layers[-1].visible = True
                image_layer.visible = True

            except Exception as e:
                print(e)

        def _new_class_stack(*args):
            masks, class_name = args[0]
            try:
                _new_layers(masks, f'{class_name}-pred-stack')

                for layer in viewer.layers:
                    layer.visible = False

                viewer.layers[-1].visible = True
                image_layer.visible = True
            except Exception as e:
                print(e)

        def start_postprocess_worker(*args):
            trackers_dict = args[0][1]
            postprocess_worker = stack_postprocessing(
                trackers_dict, store_url, model_config, label_divisor=maximum_objects_per_class,
                min_size=min_size, min_extent=min_extent
            )
            postprocess_worker.yielded.connect(_new_class_stack)
            postprocess_worker.start()

        image = image_layer.data
        if image_layer.multiscale:
            print(f'Multiscale image selected, using highest resolution level!')
            image = image[0]

        worker = stack_inference(widget.engine, image)
        worker.returned.connect(_new_segmentation)
        worker.returned.connect(start_postprocess_worker)
        worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def stack_inference_dock_widget():
    return stack_inference_widget, {'name': 'Stack Inference'}
