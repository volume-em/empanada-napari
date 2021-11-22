"""
cellpose dock widget module
"""
import sys
import yaml
from urllib.request import urlopen
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

    model_configs = {
        'MitoNet': 'https://www.dropbox.com/s/7420koff8j0te7d/mitonet_211118.yaml?dl=1',
        'MitoNet_V2': 'https://www.dropbox.com/s/t40hjkfwtc70zle/mitonet_211119.yaml?dl=1'
    }

    def download_config(url):
        with urlopen(url) as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)

        return config
    
    @thread_worker
    def run_mitonet(volume, store_url, model_config, axes):
        # create the inference engine
        engine = OrthoPlaneEngine(store_url, model_config)
        for axis_name in axes:
            stack, trackers = engine.infer_on_axis(volume, axis_name)
            yield stack, trackers, axis_name

    @thread_worker
    def test_mitonet(image, model_config):
        # create the inference engine
        engine = TestEngine(model_config)
        return image, engine.infer(image)

    @thread_worker
    def yield_dummy(volume):
        for _ in range(100):
            x = np.ones(volume.shape[1:]).astype(np.uint8)
            x[50:100, 50:100] = 0
            yield x
            time.sleep(2)
            print('Sleeping...')

    @thread_worker
    def create_consensus(trackers, store_url, model_config):
        consensus_vol = tracker_consensus(trackers, store_url, model_config, label_divisor=1000)
        return consensus_vol
        
    @magicgui(
        call_button='Run Segmentation',
        layout='vertical',
        store_url=dict(widget_type='FileEdit', label='save path', mode='d', tooltip='location to save file'),
        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()), value=list(model_configs.keys())[-1], tooltip='Model to use for inference'),
        run_xy=dict(widget_type='CheckBox', text='Infer XY', value=True, tooltip='Run inference on xy images'),
        run_xz=dict(widget_type='CheckBox', text='Infer XZ', value=True, tooltip='Run inference on xz images'),
        run_yz=dict(widget_type='CheckBox', text='Infer YZ', value=True, tooltip='Run inference on yz images'),
        compute_consensus=dict(widget_type='PushButton', text='Compute Consensus', tooltip='Create consensus annotation from axis predictions'),
        test_image=dict(widget_type='PushButton', text='Test Image', tooltip='Test model on the current image'),
        #run_segmentation=dict(widget_type='PushButton', text='Run Segmentation', tooltip='Run segmentation on the volume'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        store_url,
        model_config,
        run_xy,
        run_xz,
        run_yz,
        compute_consensus,
        test_image,
        #run_segmentation
    ):
        if not hasattr(widget, 'mitonet_layers'):
            widget.mitonet_layers = []

        if not hasattr(widget, 'trackers'):
            widget.trackers = {}

        if not hasattr(widget, 'test_count'):
            widget.test_count = 0

        """
        if not str(store_url).endswith('.zarr'):
            store_url += f'{image_layer.name}_{model_config}-napari.zarr'
        """

        # load the model config
        model_config = download_config(model_configs[model_config])

        def _new_layers(masks, axis_name):
            widget.masks_orig = masks
            layers = []

            if type(masks) == zarr.core.Array:
                masks = da.from_zarr(masks)

            layers.append(viewer.add_labels(masks, name=f'{image_layer.name}-Panoptic-{axis_name}', visible=False))
            widget.mitonet_layers.append(layers)

        def _new_segmentation(*args):
            masks, tracker, axis_name = args[0]
            try:
                _new_layers(masks, axis_name)
                
                for layer in viewer.layers:
                    layer.visible = False
                    
                viewer.layers[-1].visible = True
                image_layer.visible = True

                widget.trackers[axis_name] = tracker
                    
            except Exception as e:
                print(e)
                
            widget.call_button.enabled = True

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

        def _get_current_image():
            data = viewer.layers.selection.active.data
            axis = viewer.dims.order[0]
            plane = viewer.dims.current_step[axis]

            slices = [slice(None), slice(None), slice(None)]
            slices[axis] = plane
            return data[tuple(slices)]

        def _new_test(*args):
            print(args)
            image, seg = args[0]
            image_layer = viewer.add_image(image, name=f'Test Image {widget.test_count}', visible=True)
            mask_layer = viewer.add_labels(seg, name=f'Test Seg {widget.test_count}', visible=True)

            widget.mitonet_layers.extend([image_layer, mask_layer])

        def _new_consensus(masks):
            try:
                _new_layers(masks, 'consensus')
                
                for layer in viewer.layers:
                    layer.visible = False
                    
                viewer.layers[-1].visible = True
                image_layer.visible = True
                    
            except Exception as e:
                print(e)
                
            widget.call_button.enabled = True
            
        image = image_layer.data

        axis_names = []
        if run_xy:
            axis_names.append('xy')
        if run_xz:
            axis_names.append('xz')
        if run_yz:
            axis_names.append('yz')

        #worker = yield_dummy(image)
        #worker.yielded.connect(_append)
        #worker.start()
        #print('calling')
        #@widget.run_segmentation.changed.connect 
        #def _run_segmentation(e: Any):
        worker = run_mitonet(image, store_url, model_config, axes=axis_names)
        worker.yielded.connect(_new_segmentation)
        worker.start()

        @widget.compute_consensus.changed.connect 
        def _compute_consensus(e: Any):
            consensus_worker = create_consensus(widget.trackers, store_url, model_config)
            consensus_worker.returned.connect(_new_consensus)
            consensus_worker.start()

        @widget.test_image.changed.connect 
        def _test_on_image(e: Any):
            # get the image
            image = _get_current_image()
            if type(image) == da.core.Array:
                image = image.compute()

            test_worker = test_mitonet(image, model_config)
            test_worker.returned.connect(_new_test)
            test_worker.start()

    return widget

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'empanada'}


