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
import torch
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPlainTextEdit

import napari 
from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui

from magicgui.tqdm import tqdm

def widget_wrapper():
    from napari.qt.threading import thread_worker

    # Import when users activate plugin
    from cellpose_napari.orthoplane import OrthoPlaneEngine, tracker_consensus

    model_configs = {
        'MitoNet': 'https://www.dropbox.com/s/7420koff8j0te7d/mitonet_211118.yaml?dl=1'
    }
    
    trackers = {}

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
            yield stack[...], trackers, axis_name

        #return stack[...], trackers, axis_name

    @thread_worker
    def create_consensus(trackers, store_url, model_config):
        consensus_vol = tracker_consensus(trackers, store_url, model_config, label_divisor=1000)
        return consensus_vol[...]
        
    @magicgui(
        call_button='Run Segmentation',  
        layout='vertical',
        store_url=dict(widget_type='FileEdit', label='save path', mode='d', tooltip='location to save file'),
        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()), value='MitoNet', tooltip='Model to use for inference'),
        run_xy=dict(widget_type='CheckBox', text='Infer XY', value=True, tooltip='Run inference on xy images'),
        run_xz=dict(widget_type='CheckBox', text='Infer XZ', value=True, tooltip='Run inference on xz images'),
        run_yz=dict(widget_type='CheckBox', text='Infer YZ', value=True, tooltip='Run inference on yz images'),
        compute_consensus=dict(widget_type='PushButton', text='Compute Consensus', tooltip='Create consensus annotation from axis predictions'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        store_url,
        model_config,
        run_xy,
        run_xz,
        run_yz,
        compute_consensus
    ):
        if not hasattr(widget, 'mitonet_layers'):
            widget.mitonet_layers = []

        print(store_url)
        if not str(store_url).endswith('.zarr'):
            store_url += f'{image_layer.name}_{model_config}-napari.zarr'

        # load the model config
        model_config = download_config(model_configs[model_config])

        def _new_layers(masks, axis_name):
            widget.masks_orig = masks
            layers = []
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

                trackers[axis_name] = tracker
                    
            except Exception as e:
                print(e)
                
            widget.call_button.enabled = True

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

        cp_worker = run_mitonet(image, store_url, model_config, axes=axis_names)
        cp_worker.yielded.connect(_new_segmentation)
        cp_worker.start()

        @widget.compute_consensus.changed.connect 
        def _compute_consensus(e: Any):
            consensus_worker = create_consensus(trackers, store_url, model_config)
            consensus_worker.returned.connect(_new_consensus)
            consensus_worker.start()

    return widget

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'mitonet'}


