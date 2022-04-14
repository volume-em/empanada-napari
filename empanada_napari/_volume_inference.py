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

def volume_inference_widget():
    from napari.qt.threading import thread_worker

    # Import when users activate plugin
    import zarr
    import dask.array as da
    from torch.cuda import device_count
    from empanada_napari.inference import Engine3d, tracker_consensus, stack_postprocessing
    from empanada_napari.multigpu import MultiGPUEngine3d
    from empanada_napari.utils import get_configs, abspath
    from empanada.config_loaders import read_yaml
    from empanada.inference import filters

    logo = abspath(__file__, 'resources/empanada_logo.png')
    # get list of all available model configs
    model_configs = get_configs()

    @thread_worker
    def stack_inference(engine, volume, axis_name):
        stack, trackers = engine.infer_on_axis(volume, axis_name)
        trackers_dict = {axis_name: trackers}
        return stack, axis_name, trackers_dict

    @thread_worker
    def orthoplane_inference(engine, volume):
        trackers_dict = {}
        for axis_name in ['xy', 'xz', 'yz']:
            stack, trackers = engine.infer_on_axis(volume, axis_name)
            trackers_dict[axis_name] = trackers

            # report instances per class
            for tracker in trackers:
                class_id = tracker.class_id
                print(f'Class {class_id}, axis {axis_name}, has {len(tracker.instances.keys())} instances')

            yield stack, axis_name

        return trackers_dict

    @magicgui(
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Run 3D Inference',
        layout='vertical',
        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], tooltip='Model to use for inference'),
        store_dir=dict(widget_type='FileEdit', value='no zarr storage', label='Zarr Directory (optional)', mode='d', tooltip='location to store segmentations on disk'),
        use_gpu=dict(widget_type='CheckBox', text='Use GPU', value=device_count() >= 1, tooltip='If checked, run on GPU 0'),
        multigpu=dict(widget_type='CheckBox', text='Multi GPU', value=False, tooltip='If checked, run on all available GPUs'),

        parameters2d_head=dict(widget_type='Label', label=f'<h3 text-align="center">2D Parameters</h3>'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Image Downsampling', tooltip='Downsampling factor to apply before inference'),
        confidence_thr=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1, label='Segmentation Confidence Thr'),
        center_confidence_thr=dict(widget_type='FloatSpinBox', value=0.1, min=0.05, max=0.9, step=0.05, label='Center Confidence Thr'),
        min_distance_object_centers=dict(widget_type='SpinBox', value=3, min=1, max=21, step=1, label='Centers Min Distance'),
        fine_boundaries=dict(widget_type='CheckBox', text='Fine Boundaries', value=False, tooltip='Finer boundaries between objects'),
        semantic_only=dict(widget_type='CheckBox', text='Semantic Only', value=False, tooltip='Only run semantic segmentation for all classes.'),

        parameters_stack_head=dict(widget_type='Label', label=f'<h3 text-align="center">Stack Parameters</h3>'),
        median_slices=dict(widget_type='ComboBox', choices=[1, 3, 5, 7, 9, 11], value=3, label='Median Filter Size', tooltip='Median filter size'),
        min_size=dict(widget_type='SpinBox', value=500, min=0, max=1e6, step=100, label='Min Size (Voxels)'),
        min_extent=dict(widget_type='SpinBox', value=5, min=0, max=1000, step=1, label='Min Box Extent'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value='1000000', label='Max objects per class in 3D'),
        inference_plane=dict(widget_type='ComboBox', choices=['xy', 'xz', 'yz'], value='xy', label='Inference plane', tooltip='Image plane along which to run inference. Overwritten, if using ortho-plane.'),

        parameters_ortho_head=dict(widget_type='Label', label=f'<h3 text-align="center">Ortho-plane Parameters (Optional)</h3>'),
        orthoplane=dict(widget_type='CheckBox', text='Run ortho-plane', value=False, tooltip='Whether to run orthoplane inference'),
        return_panoptic=dict(widget_type='CheckBox', text='Return xy, xz, yz stacks', value=False, tooltip='Whether to return the inference stacks.'),
        pixel_vote_thr=dict(widget_type='SpinBox', value=2, min=1, max=3, step=1, label='Voxel Vote Thr Out of 3', tooltip='Number of votes out of 3 for a voxel to be labeled in the consensus'),
        allow_one_view=dict(widget_type='CheckBox', text='Permit detections found in 1 stack into consensus', value=False, tooltip='Whether to allow detections into consensus that were picked up by inference in just 1 stack')
    )
    def widget(
        viewer: napari.viewer.Viewer,
        label_head,
        image_layer: Image,
        model_config,
        store_dir,
        use_gpu,
        multigpu,

        parameters2d_head,
        downsampling,
        confidence_thr,
        center_confidence_thr,
        min_distance_object_centers,
        fine_boundaries,
        semantic_only,

        parameters_stack_head,
        median_slices,
        min_size,
        min_extent,
        maximum_objects_per_class,
        inference_plane,

        parameters_ortho_head,
        orthoplane,
        return_panoptic,
        pixel_vote_thr,
        allow_one_view
    ):
        # load the model config
        model_config_name = model_config
        model_config = read_yaml(model_configs[model_config])
        min_size = int(min_size)
        min_extent = int(min_extent)
        maximum_objects_per_class = int(maximum_objects_per_class)

        # create the storage url from layer name and model config
        store_dir = str(store_dir)
        if store_dir == 'no zarr storage':
            store_url = None
            print(f'Running without zarr storage directory, this may use a lot of memory!')
        else:
            store_url = os.path.join(store_dir, f'{image_layer.name}_{model_config_name}_napari.zarr')

        if not hasattr(widget, 'last_config'):
            widget.last_config = model_config

        if not hasattr(widget, 'using_gpu'):
            widget.using_gpu = use_gpu

        if multigpu:
            widget.engine = MultiGPUEngine3d(
                model_config,
                inference_scale=downsampling,
                median_kernel_size=median_slices,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                min_size=min_size,
                min_extent=min_extent,
                fine_boundaries=fine_boundaries,
                label_divisor=maximum_objects_per_class,
                semantic_only=semantic_only,
                save_panoptic=return_panoptic,
                store_url=store_url
            )
            widget.last_config = model_config
        # conditions where model needs to be (re)loaded
        elif not hasattr(widget, 'engine') or widget.last_config != model_config or use_gpu != widget.using_gpu:
            widget.engine = Engine3d(
                model_config,
                inference_scale=downsampling,
                median_kernel_size=median_slices,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                min_size=min_size,
                min_extent=min_extent,
                fine_boundaries=fine_boundaries,
                label_divisor=maximum_objects_per_class,
                use_gpu=use_gpu,
                semantic_only=semantic_only,
                save_panoptic=return_panoptic,
                store_url=store_url
            )
            widget.last_config = model_config
            widget.using_gpu = use_gpu
        else:
            # update the parameters
            widget.engine.update_params(
                inference_scale=downsampling,
                median_kernel_size=median_slices,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                min_size=min_size,
                min_extent=min_extent,
                fine_boundaries=fine_boundaries,
                label_divisor=maximum_objects_per_class,
                semantic_only=semantic_only,
                save_panoptic=return_panoptic,
                store_url=store_url
            )

        def _new_layers(mask, description):
            layers = []

            if type(mask) == zarr.core.Array:
                mask = da.from_zarr(mask)

            viewer.add_labels(mask, name=f'{image_layer.name}-{description}', visible=True)

        def _new_segmentation(*args):
            mask = args[0][0]
            axis_name = args[0][1]

            if mask is not None:
                try:
                    _new_layers(mask, f'panoptic-stack-{axis_name}')

                    for layer in viewer.layers:
                        layer.visible = False

                    viewer.layers[-1].visible = True
                    image_layer.visible = True

                except Exception as e:
                    print(e)

        def _new_class_stack(*args):
            masks, class_name = args[0]
            try:
                _new_layers(masks, f'{class_name}-prediction')

                for layer in viewer.layers:
                    layer.visible = False

                viewer.layers[-1].visible = True
                image_layer.visible = True
            except Exception as e:
                print(e)

        def start_postprocess_worker(*args):
            trackers_dict = args[0][2]
            postprocess_worker = stack_postprocessing(
                trackers_dict, store_url, model_config, label_divisor=maximum_objects_per_class,
                min_size=min_size, min_extent=min_extent, dtype=widget.engine.dtype
            )
            postprocess_worker.yielded.connect(_new_class_stack)
            postprocess_worker.start()

        def start_consensus_worker(trackers_dict):
            consensus_worker = tracker_consensus(
                trackers_dict, store_url, model_config, label_divisor=maximum_objects_per_class,
                pixel_vote_thr=pixel_vote_thr, allow_one_view=allow_one_view,
                min_size=min_size, min_extent=min_extent, dtype=widget.engine.dtype
            )
            consensus_worker.yielded.connect(_new_class_stack)
            consensus_worker.start()

        image = image_layer.data
        if image_layer.multiscale:
            print(f'Multiscale image selected, using highest resolution level!')
            image = image[0]

        if orthoplane:
            worker = orthoplane_inference(widget.engine, image)
            worker.yielded.connect(_new_segmentation)
            worker.returned.connect(start_consensus_worker)
        else:
            worker = stack_inference(widget.engine, image, inference_plane)
            worker.returned.connect(_new_segmentation)
            worker.returned.connect(start_postprocess_worker)

        worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def volume_dock_widget():
    return volume_inference_widget, {'name': '3D Inference'}
