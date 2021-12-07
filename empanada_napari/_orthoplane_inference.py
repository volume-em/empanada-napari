"""
cellpose dock widget module
"""
import sys
import yaml
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

def orthoplane_inference_widget():
    from napari.qt.threading import thread_worker
    from napari.qt import progress

    # Import when users activate plugin
    from empanada_napari.orthoplane import OrthoPlaneEngine, tracker_consensus
    from empanada_napari.utils import get_configs, load_config

    # get list of all available model configs
    model_configs = get_configs()
    
    @thread_worker
    def orthoplane_inference(engine, volume):
        trackers_dict = {}
        for axis_name in ['xy', 'xz', 'yz']:
            stack, trackers = engine.infer_on_axis(volume, axis_name)
            trackers_dict[axis_name] = trackers
            yield stack, axis_name

        return trackers_dict
        
    @magicgui(
        call_button='Run Orthoplane Inference',
        layout='vertical',
        store_url=dict(widget_type='FileEdit', value='/Users/conradrw/Desktop/napari_gastro.zarr', label='save path', mode='d', tooltip='location to save file'),
        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='Model to use for inference'),
        median_slices=dict(widget_type='ComboBox', choices=[1, 3, 5, 7, 9], value=3, label='Median filter size', tooltip='Median filter size'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Downsampling before inference', tooltip='Downsampling factor to apply before inference'),
        min_distance_object_centers=dict(widget_type='Slider', value=3, min=3, max=21, label='Minimum distance between object centers.'),
        confidence_thr=dict(widget_type='FloatSlider', value=0.5, min=0.1, max=0.9, step=0.1, label='Confidence Threshold'),
        center_confidence_thr=dict(widget_type='FloatSlider', value=0.1, min=0.1, max=0.9, label='Center Confidence Threshold'),
        merge_iou_thr=dict(widget_type='FloatSlider', value=0.25, max=0.9, label='IoU Threshold'),
        merge_ioa_thr=dict(widget_type='FloatSlider', value=0.25, max=0.9, label='IoA Threshold'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value=1000, label='Max objects per class'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        store_url,
        model_config,
        median_slices,
        downsampling,
        min_distance_object_centers,
        confidence_thr,
        center_confidence_thr,
        merge_iou_thr,
        merge_ioa_thr,
        maximum_objects_per_class
    ):
        # load the model config
        model_config = load_config(model_configs[model_config])
        maximum_objects_per_class = int(maximum_objects_per_class)

        if not hasattr(widget, 'last_config'):
            widget.last_config = model_config

        if not hasattr(widget, 'engine') or widget.last_config != model_config:
            widget.engine = OrthoPlaneEngine(
                store_url, model_config,
                inference_scale=downsampling,
                median_kernel_size=median_slices,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                merge_iou_thr=merge_iou_thr,
                merge_ioa_thr=merge_ioa_thr,
                label_divisor=maximum_objects_per_class
            )
            widget.last_config = model_config
        else:
            # update the parameters
            widget.engine.inference_scale = downsampling
            widget.engine.median_kernel_size = median_slices
            widget.engine.nms_kernel = min_distance_object_centers
            widget.engine.nms_threshold = center_confidence_thr
            widget.engine.confidence_thr = confidence_thr
            widget.engine.merge_iou_thr = merge_iou_thr
            widget.engine.merge_ioa_thr = merge_ioa_thr
            widget.label_divisor = maximum_objects_per_class

        if not str(store_url).endswith('.zarr'):
            store_url = store_url / f'{image_layer.name}_{model_config}-napari.zarr'

        def _new_layers(masks, description):
            widget.masks_orig = masks
            layers = []

            if type(masks) == zarr.core.Array:
                masks = da.from_zarr(masks)

            viewer.add_labels(masks, name=f'{image_layer.name}-{description}', visible=True)

        def _new_segmentation(*args):
            masks, axis_name = args[0]
            try:
                _new_layers(masks, f'panoptic-{axis_name}')
                
                for layer in viewer.layers:
                    layer.visible = False
                    
                viewer.layers[-1].visible = True
                image_layer.visible = True
                    
            except Exception as e:
                print(e)
                
        def _new_consensus(*args):
            masks, class_name = args[0]
            try:
                _new_layers(masks, f'{class_name}-consensus')

                for layer in viewer.layers:
                    layer.visible = False

                viewer.layers[-1].visible = True
                image_layer.visible = True
            except Exception as e:
                print(e)
                
        def start_consensus_worker(trackers_dict):
            consensus_worker = tracker_consensus(
                trackers_dict, store_url, model_config, label_divisor=maximum_objects_per_class
            )
            consensus_worker.yielded.connect(_new_consensus)
            consensus_worker.start()
                            
        image = image_layer.data
        if image_layer.multiscale:
            print(f'Multiscale image selected, using highest resolution level!')                    
            image = image[0]

        worker = orthoplane_inference(widget.engine, image)
        worker.yielded.connect(_new_segmentation)
        worker.returned.connect(start_consensus_worker)
        worker.start()
        
    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def orthoplane_inference_dock_widget():
    return orthoplane_inference_widget, {'name': 'Orthoplane Inference'}