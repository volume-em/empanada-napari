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

def widget_wrapper():
    from napari.qt.threading import thread_worker

    # Import when users activate plugin
    from empanada_napari.orthoplane import TestEngine, OrthoPlaneEngine, tracker_consensus

    # get dict of all model configs
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
    model_configs = {}
    for fn in os.listdir(config_path):
        if fn.endswith('.yaml'):
            model_configs[fn[:-len('.yaml')]] = os.path.join(config_path, fn)

    def load_config(url):
        with open(url, mode='r') as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)

        return config
    
    @thread_worker
    def run_mitonet(volume, store_url, model_config, axes, confidence_thr, merge_iou_thr, merge_ioa_thr):
        # create the inference engine
        engine = OrthoPlaneEngine(store_url, model_config, confidence_thr, merge_iou_thr, merge_ioa_thr)
        for axis_name in axes:
            stack, trackers = engine.infer_on_axis(volume, axis_name)
            yield stack, trackers, axis_name

    @thread_worker
    def test_mitonet(image, model_config, axis, plane):
        # create the inference engine
        engine = TestEngine(model_config)
        return image, engine.infer(image), axis, plane

    @thread_worker
    def yield_dummy(volume):
        for _ in range(100):
            x = np.ones(volume.shape[1:]).astype(np.uint8)
            x[50:100, 50:100] = 0
            yield x
            time.sleep(2)
            print('Sleeping...')

    #@thread_worker
    #def create_consensus(trackers, store_url, model_config):
    #    yield tracker_consensus(trackers, store_url, model_config, label_divisor=1000)
        
    @magicgui(
        #call_button='Run Segmentation',
        auto_call=True,
        layout='vertical',
        store_url=dict(widget_type='FileEdit', value='/Users/conradrw/Desktop/napari_gastro.zarr', label='save path', mode='d', tooltip='location to save file'),
        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='Model to use for inference'),
        run_xy=dict(widget_type='CheckBox', text='Infer XY', value=True, tooltip='Run inference on xy images'),
        run_xz=dict(widget_type='CheckBox', text='Infer XZ', value=False, tooltip='Run inference on xz images'),
        run_yz=dict(widget_type='CheckBox', text='Infer YZ', value=False, tooltip='Run inference on yz images'),
        compute_consensus=dict(widget_type='PushButton', text='Compute Consensus', tooltip='Create consensus annotation from axis predictions'),
        confidence_threshold=dict(widget_type='FloatSlider', value=0.3, max=0.9, label= 'Confidence Threshold'),
        merge_iou_threshold=dict(widget_type='FloatSlider', value=0.25, max=0.9, label= 'IOU Threshold'),
        merge_ioa_threshold=dict(widget_type='FloatSlider', value=0.25, max=0.9, label= 'IOA Threshold'),
        test_image=dict(widget_type='PushButton', text='Test Image', tooltip='Test model on the current image'),
        run_segmentation=dict(widget_type='PushButton', text='Run Segmentation', tooltip='Run segmentation on the volume'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        store_url,
        model_config,
        run_xy,
        run_xz,
        run_yz, 
        confidence_threshold,
        merge_iou_threshold,
        merge_ioa_threshold,
        compute_consensus,
        test_image,
        run_segmentation
    ):
        if not hasattr(widget, 'mitonet_layers'):
            widget.mitonet_layers = []

        if not hasattr(widget, 'trackers'):
            widget.trackers = {}

        if not hasattr(widget, 'test_count'):
            widget.test_count = 0

        if not str(store_url).endswith('.zarr'):
            store_url = store_url / f'{image_layer.name}_{model_config}-napari.zarr'
            
        # load the model config
        model_config = load_config(model_configs[model_config])

        def _new_layers(masks, description):
            widget.masks_orig = masks
            layers = []

            if type(masks) == zarr.core.Array:
                masks = da.from_zarr(masks)

            layers.append(viewer.add_labels(masks, name=f'{image_layer.name}-{description}', visible=False))
            widget.mitonet_layers.append(layers)

        def _new_segmentation(*args):
            masks, tracker, axis_name = args[0]
            try:
                _new_layers(masks, f'Panoptic-{axis_name}')
                
                for layer in viewer.layers:
                    layer.visible = False
                    
                viewer.layers[-1].visible = True
                image_layer.visible = True

                widget.trackers[axis_name] = tracker
                    
            except Exception as e:
                print(e)
                
        def _append(image):
            if image is None:
                return

            if len(viewer.layers) == 2:
                # layer is present, append to its data
                layer = viewer.layers[-1]
                image_dtype = layer.data.dtype
                image = da.from_array(image).reshape((1,) + image.shape)
                layer.data = da.concatenate((layer.data, image), axis=0)
            else:
                # first run, no layer added yet
                image = da.from_array(image).reshape((1,) + image.shape)
                layer = viewer.add_image(image, rendering='attenuated_mip')

        def _get_current_slice(image_layer):
            axis = viewer.dims.order[0]
            cursor_pos = viewer.cursor.position
            plane = int(image_layer.world_to_data(cursor_pos)[axis])

            slices = [slice(None), slice(None), slice(None)]
            slices[axis] = plane

            return image_layer.data[tuple(slices)], axis, plane

        def _new_test(*args):
            image, seg, axis, plane = args[0]

            seg = np.expand_dims(seg, axis=axis)
            translate = [0, 0, 0]
            translate[axis] = plane

            mask_layer = viewer.add_labels(seg, name=f'Test Seg {widget.test_count}', visible=True, translate=tuple(translate))

            widget.mitonet_layers.append(mask_layer)
            widget.test_count += 1

        def _new_consensus(*args):
            masks, class_name = args[0]
            if(len(widget.trackers.keys()) >= 2):
                try:
                    _new_layers(masks, f'{class_name}-consensus')

                    for layer in viewer.layers:
                        layer.visible = False

                    viewer.layers[-1].visible = True
                    image_layer.visible = True    
                except Exception as e:
                    print(e)
            else:
                raise ValueError("Need to run segmentation on at least 2 axes for consensus!")
                
            #widget.call_button.enabled = True
                            
        image = image_layer.data

        axis_names = []
        if run_xy:
            axis_names.append('xy')
        if run_xz:
            axis_names.append('xz')
        if run_yz:
            axis_names.append('yz')
        
        confidence_thr = confidence_threshold["value"]
        merge_iou_thr = merge_iou_threshold["value"]
        merge_ioa_thr = merge_ioa_threshold["value"]
        
        #worker = yield_dummy(image)
        #worker.yielded.connect(_append)
        #worker.start()
        #print('calling')
        #@widget.run_segmentation.changed.connect 
        #def _run_segmentation(e: Any):
        #if run_segmentation:
        worker = run_mitonet(image, store_url, model_config, axes=axis_names)
        worker.yielded.connect(_new_segmentation)
        widget.run_segmentation.changed.connect(worker.start)

        if compute_consensus:
        #@widget.compute_consensus.changed.connect 
        #def _compute_consensus(e: Any):
            consensus_worker = tracker_consensus(widget.trackers, store_url, model_config, label_divisor=1000)
            consensus_worker.yielded.connect(_new_consensus)
            consensus_worker.start()

        if test_image:
        #@widget.test_image.changed.connect 
        #def _test_on_image(e: Any):
            # get the image
            print('testing', image_layer, image_layer.name)
            image2d, axis, plane = _get_current_slice(image_layer)
            if type(image2d) == da.core.Array:
                image2d = image2d.compute()

            #axis = 
            test_worker = test_mitonet(image2d, model_config, axis, plane)
            test_worker.returned.connect(_new_test)
            test_worker.start()

    return widget

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'empanada'}


