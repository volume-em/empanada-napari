import os
import napari
from napari import Viewer
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QScrollArea

import zarr
import dask.array as da
from torch.cuda import device_count
from empanada_napari.inference import Engine3d, tracker_consensus, stack_postprocessing
from empanada_napari.multigpu import MultiGPUEngine3d
from empanada_napari.utils import get_configs, abspath
from empanada.config_loaders import read_yaml

class VolumeInferenceWidget:
    def __init__(self, 
            model_config: str,
            viewer: Viewer = None,
            label_head: dict = None,
            image_layer: Image = None,
            use_gpu: bool = False,
            use_quantized: bool = False,
            multigpu: bool = False,

            downsampling: int = 1,
            confidence_thr: float = 0.5,
            center_confidence_thr: float = 0.1,
            min_distance_object_centers: int = 3,
            fine_boundaries: bool = False,
            semantic_only: bool = False,

            median_slices: int = 3,
            min_size: int = 500,
            min_extent: int = 5,
            maximum_objects_per_class: int = 10000,
            inference_plane: str = 'xy',

            label_erosion: int = 0,
            label_dilation: int = 0,
            fill_holes_in_segmentation: bool = False,
            orthoplane: bool = False,
            return_panoptic: bool = False,
            pixel_vote_thr: int = 2,
            allow_one_view: bool = False,

            store_dir: str = 'no zarr storage',
            chunk_size: int | str = 256,

            pbar: widgets.ProgressBar = None
    ):
        self.viewer = viewer
        self.label_head = label_head
        self.image_layer = image_layer
        self.model_config = model_config
        self.use_gpu = use_gpu
        self.use_quantized = use_quantized
        self.multigpu = multigpu

        self.downsampling = downsampling
        self.confidence_thr = confidence_thr
        self.center_confidence_thr = center_confidence_thr
        self.min_distance_object_centers = min_distance_object_centers
        self.fine_boundaries = fine_boundaries
        self.semantic_only = semantic_only

        self.median_slices = median_slices
        self.min_size = min_size
        self.min_extent = min_extent
        self.maximum_objects_per_class = maximum_objects_per_class
        self.inference_plane = inference_plane

        self.label_erosion = label_erosion
        self.label_dilation = label_dilation
        self.fill_holes = fill_holes_in_segmentation
        self.orthoplane = orthoplane
        self.return_panoptic = return_panoptic
        self.pixel_vote_thr = pixel_vote_thr
        self.allow_one_view = allow_one_view

        self.store_dir = store_dir

        chunk_size = chunk_size.split(',')
        if len(chunk_size) == 1:
            self.chunk_size = tuple(int(chunk_size[0]) for _ in range(3))
        else:
            assert len(chunk_size) == 3, f"Chunk size must be 1 or 3 integers, got {chunk_size}"
            self.chunk_size = tuple(int(s) for s in chunk_size)
        
        self.pbar = pbar

# ---------------- Option handling & inference running entrypoint ----------------
    def config_and_run_inference(self, use_thread=False):  
        # Load the model config
        model_configs = get_configs()
        self.model_config = read_yaml(model_configs[self.model_config_name])

        if self.last_config is None:
            self.last_config = self.model_config_name
            
        # Create storage url from layer name and model config
        if self.store_dir == 'no zarr storage': # This is a default - 
            self.store_url = None
            print(f'Running without zarr storage directory, this may use a lot of memory!')
        else:
            self.store_url = os.path.join(self.store_dir, f'{self.image_layer.name}_{self.model_config_name}.zarr')

        self.get_engine()

        # Get the 3d slice from the image (Can mock a layer/viewer object in the tests)
        image = self.image_layer.data
        if self.image_layer.multiscale:
            print(f'Multiscale image selected, using highest resolution level!')
            image = image[0]

        # Verify that the image doesn't have extraneous channel dimensions
        assert image.ndim in [3, 4], "Only 3D and 4D input images can be handled!"
        if image.ndim == 4:
            # Channel dimensions are commonly 1, 3 and 4
            # Check for dimensions on zeroth and last axes
            shape = image.shape
            if shape[0] in [1, 3, 4]:
                image = image[0]
            elif shape[-1] in [1, 3, 4]:
                image = image[..., 0]
            else:
                raise Exception(f'Image volume must be 3D, got image of shape {shape}')

            print(f'Got 4D image of shape {shape}, extracted single channel of size {image.shape}')

        match (self.orthoplane, use_thread):
        # if use_thread is False, use the non-threaded versions of orthoplane + stack
            case True, True:
                worker = self.orthoplane_inference(self.engine, image)
                worker.yielded.connect(self._new_segmentation)
                worker.returned.connect(self.start_consensus_worker)
                worker.start()

            case True, False:
                trackers_dict = self._orthoplane_inference(self.engine, image)
                return trackers_dict

            case False, True:
                worker = self.stack_inference(self.widget.engine, image, self.inference_plane)
                worker.returned.connect(self._new_segmentation)
                worker.returned.connect(self.start_postprocess_worker)
                worker.start()

            case False, False:
                stack, axis_name, trackers_dict = self._stack_inference(self.engine, image, self.inference_plane)
                return stack, axis_name, trackers_dict       
        return

# ---------------- Engine management ----------------
    def get_engine(self):
        update_engine = (
            self.engine is None
            or self.last_config != self.model_config_name
        )

        if update_engine:
            self.widget.engine = Engine3d(
                self.model_config,
                inference_scale=self.downsampling,
                median_kernel_size=self.median_slices,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                min_size=self.min_size,
                min_extent=self.min_extent,
                fine_boundaries=self.fine_boundaries,
                label_divisor=self.maximum_objects_per_class,
                use_gpu=self.use_gpu,
                use_quantized=self.use_quantized,
                semantic_only=self.semantic_only,
                save_panoptic=self.return_panoptic,
                store_url=self.store_url,
                chunk_size=self.chunk_size,
                label_erosion=self.label_erosion,
                label_dilation=self.label_dilation,
                fill_holes_in_segmentation=self.fill_holes_in_segmentation
            )
            self.widget.last_config = self.model_config_name
            self.widget.using_gpu = self.use_gpu

        elif self.multigpu:
            self.widget.engine = MultiGPUEngine3d(
                self.model_config,
                inference_scale=self.downsampling,
                median_kernel_size=self.median_slices,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                min_size=self.min_size,
                min_extent=self.min_extent,
                fine_boundaries=self.fine_boundaries,
                label_divisor=self.maximum_objects_per_class,
                semantic_only=self.semantic_only,
                save_panoptic=self.return_panoptic,
                store_url=self.store_url,
                chunk_size=self.chunk_size
            )
            self.widget.last_config = self.model_config_name

        else:
            # update the parameters
            self.widget.engine.update_params(
                inference_scale=self.downsampling,
                median_kernel_size=self.median_slices,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                min_size=self.min_size,
                min_extent=self.min_extent,
                fine_boundaries=self.fine_boundaries,
                label_divisor=self.maximum_objects_per_class,
                semantic_only=self.semantic_only,
                save_panoptic=self.return_panoptic,
                store_url=self.store_url,
                chunk_size=self.chunk_size,
                label_erosion=self.label_erosion,
                label_dilation=self.label_dilation,
                fill_holes_in_segmentation=self.fill_holes_in_segmentation
            )
        return

# ---------------- Helper methods ----------------
    def _new_layers(self, mask, description, instances=None):
        metadata = {}
        if instances is not None:
            for label, label_attrs in instances.items():
                metadata[label] = {
                    'box': label_attrs['box'],
                    'area': label_attrs['runs'].sum(),
                }
        translate = self.image_layer.translate
        scale = self.image_layer.scale
        ndim = self.image_layer.data[0].ndim if self.image_layer.multiscale else self.image_layer.data.ndim
        if ndim:
            shape = self.image_layer.data.shape
            if shape[0] in [1, 3, 4]:
                translate = translate[1:]
                scale = scale[1:]
            elif shape[-1] in [1, 3, 4]:
                translate = translate[:-1]
                scale = scale[:-1]
        self.viewer.add_labels(
            mask, name=f'{self.image_layer.name}-{description}',
            visible=True, metadata=metadata, translate=translate,
            scale=scale
        )
        self.pbar.hide()

    def _new_segmentation(self, *args):
        mask = args[0][0]
        axis_name = args[0][1]
        if mask is not None:
            try:
                self._new_layers(mask, f'panoptic-stack-{axis_name}')
                for layer in self.viewer.layers:
                    layer.visible = False
                self.viewer.layers[-1].visible = True
                self.image_layer.visible = True
            except Exception as e:
                print(e)

    def _new_class_stack(self, *args):
        masks, class_name, instances = args[0]
        try:
            self._new_layers(masks, f'{class_name}-prediction', instances)
            for layer in self.viewer.layers:
                layer.visible = False
            self.viewer.layers[-1].visible = True
            self.image_layer.visible = True
        except Exception as e:
            print(e)

    def start_postprocess_worker(self, *args):
        trackers_dict = args[0][2]
        postprocess_worker = self.stack_postprocessing(
            trackers_dict, self.store_url, self.model_config, label_divisor=self.maximum_objects_per_class,
            min_size=self.min_size, min_extent=self.min_extent, dtype=self.widget.engine.dtype, chunk_size=self.chunk_size
        )
        postprocess_worker.yielded.connect(self._new_class_stack)
        postprocess_worker.start()

    def start_consensus_worker(self, trackers_dict):
        consensus_worker = self.tracker_consensus(
            trackers_dict, self.store_url, self.model_config, label_divisor=self.maximum_objects_per_class,
            pixel_vote_thr=self.pixel_vote_thr, allow_one_view=self.allow_one_view,
            min_size=self.min_size, min_extent=self.min_extent, dtype=self.widget.engine.dtype,
            chunk_size=self.chunk_size
        )
        consensus_worker.yielded.connect(self._new_class_stack)
        consensus_worker.start()

    # ---------------- Inference runners ----------------
    @thread_worker
    def stack_inference(self, engine, volume, axis_name):
        return self._stack_inference(engine, volume, axis_name)

    @thread_worker
    def orthoplane_inference(self, engine, volume):
        return self._orthoplane_inference(engine, volume)

    def _stack_inference(self, engine, volume, axis_name):
        stack, trackers = engine.infer_on_axis(volume, axis_name)
        trackers_dict = {axis_name: trackers}
        return stack, axis_name, trackers_dict

    def _orthoplane_inference(self, engine, volume):
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



def volume_inference_widget():
    # Import when users activate plugin
    from torch.cuda import device_count
    from empanada_napari.utils import get_configs, abspath

    logo = abspath(__file__, 'resources/empanada_logo.png')
    model_configs = get_configs()

    @magicgui(
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Run 3D Inference',
        layout='vertical',
        scrollable=True,

        model_config=dict(widget_type='ComboBox', label='model', choices=list(model_configs.keys()),
                          value=list(model_configs.keys())[0], tooltip='Model to use for inference'),
        use_gpu=dict(widget_type='CheckBox', text='Use GPU', value=device_count() >= 1,
                     tooltip='If checked, run on GPU 0'),
        use_quantized=dict(widget_type='CheckBox', text='Use quantized model', value=device_count() == 0,
                           tooltip='If checked, use the quantized model for faster CPU inference.'),
        multigpu=dict(widget_type='CheckBox', text='Multi GPU', value=False,
                      tooltip='If checked, run on all available GPUs'),

        parameters2d_head=dict(widget_type='Label', label=f'<h3 text-align="center">2D Parameters</h3>'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Image Downsampling',
                          tooltip='Downsampling factor to apply before inference'),
        confidence_thr=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1,
                            label='Segmentation Confidence Thr'),
        center_confidence_thr=dict(widget_type='FloatSpinBox', value=0.1, min=0.05, max=0.9, step=0.05,
                                   label='Center Confidence Thr'),
        min_distance_object_centers=dict(widget_type='SpinBox', value=3, min=1, max=21, step=1,
                                         label='Centers Min Distance'),
        fine_boundaries=dict(widget_type='CheckBox', text='Fine Boundaries', value=False,
                             tooltip='Finer boundaries between objects'),
        semantic_only=dict(widget_type='CheckBox', text='Semantic Only', value=False,
                           tooltip='Only run semantic segmentation for all classes.'),

        parameters_stack_head=dict(widget_type='Label', label=f'<h3 text-align="center">Stack Parameters</h3>'),
        median_slices=dict(widget_type='ComboBox', choices=[1, 3, 5, 7, 9, 11], value=3, label='Median Filter Size',
                           tooltip='Median filter size'),
        min_size=dict(widget_type='SpinBox', value=500, min=0, max=1e6, step=100, label='Min Size (Voxels)'),
        min_extent=dict(widget_type='SpinBox', value=5, min=0, max=1000, step=1, label='Min Box Extent'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value=10000, label='Max objects per class in 3D',
                                       tooltip='Maximum number of objects per class in 3D inference'),
                                       # value here was originally '10000' string, may break
        inference_plane=dict(widget_type='ComboBox', choices=['xy', 'xz', 'yz'], value='xy', label='Inference plane',
                             tooltip='Image plane along which to run inference. Overwritten, if using ortho-plane.'),

        parameters_ortho_head=dict(widget_type='Label',
                                   label=f'<h3 text-align="center">Ortho-plane Parameters (Optional)</h3>'),
        label_erosion=dict(widget_type='SpinBox', value=0, min=0, max=50, step=1, label='Erode Labels',
                           tooltip='How much to erode labels produced after inference'),
        label_dilation=dict(widget_type='SpinBox', value=0, min=0, max=50, step=1, label='Dilate Labels',
                            tooltip='How much to dilate labels produced after inference'),
        fill_holes_in_segmentation=dict(widget_type='CheckBox', text='Fill holes in segmentation', value=False,
                                            tooltip='Whether to fill holes in the segmentation after inference'),
        orthoplane=dict(widget_type='CheckBox', text='Run ortho-plane', value=False,
                        tooltip='Whether to run orthoplane inference'),
        return_panoptic=dict(widget_type='CheckBox', text='Return xy, xz, yz stacks', value=False,
                             tooltip='Whether to return the inference stacks.'),
        pixel_vote_thr=dict(widget_type='SpinBox', value=2, min=1, max=3, step=1, label='Voxel Vote Thr Out of 3',
                            tooltip='Number of votes out of 3 for a voxel to be labeled in the consensus'),
        allow_one_view=dict(widget_type='CheckBox', text='Permit detections found in 1 stack into consensus',
                            value=False,
                            tooltip='Whether to allow detections into consensus that were picked up by inference in just 1 stack'),

        storage_head=dict(widget_type='Label', label=f'<h3 text-align="center">Zarr Storage (optional)</h3>'),
        store_dir=dict(widget_type='FileEdit', value='no zarr storage', label='Directory', mode='d',
                       tooltip='location to store segmentations on disk'),
        chunk_size=dict(widget_type='LineEdit', value='256', label='Chunk size',
                        tooltip='Chunk size of the zarr array. Integer or comma separated list of 3 integers.'),
        pbar={'visible': False, 'max': 0, 'label': 'Running...'},
    )
    def widget(
            viewer: napari.viewer.Viewer,
            label_head,
            image_layer: Image,
            model_config,
            use_gpu,
            use_quantized,
            multigpu,

            # parameters2d_head,
            downsampling,
            confidence_thr,
            center_confidence_thr,
            min_distance_object_centers,
            fine_boundaries,
            semantic_only,

            # parameters_stack_head,
            median_slices,
            min_size,
            min_extent,
            maximum_objects_per_class,
            inference_plane,

            # parameters_ortho_head,
            label_erosion,
            label_dilation,
            fill_holes_in_segmentation,
            orthoplane,
            return_panoptic,
            pixel_vote_thr,
            allow_one_view,

            # storage_head,
            store_dir,
            chunk_size,

            pbar: widgets.ProgressBar
    ):
        # instantiate the class
        inference_config = VolumeInferenceWidget(viewer = viewer,
            label_head = label_head,
            image_layer = image_layer,
            model_config = model_config,
            use_gpu = use_gpu,
            use_quantized = use_quantized,
            multigpu = multigpu,
            downsampling = downsampling,
            confidence_thr = confidence_thr,
            center_confidence_thr = center_confidence_thr,
            min_distance_object_centers = min_distance_object_centers,
            fine_boundaries = fine_boundaries,
            semantic_only = semantic_only,
            median_slices = median_slices,
            min_size = min_size,
            min_extent = min_extent,
            maximum_objects_per_class = maximum_objects_per_class,
            inference_plane = inference_plane,
            label_erosion = label_erosion,
            label_dilation = label_dilation,
            fill_holes_in_segmentation = fill_holes_in_segmentation,
            orthoplane = orthoplane,
            return_panoptic = return_panoptic,
            pixel_vote_thr = pixel_vote_thr,
            allow_one_view = allow_one_view,
            store_dir = store_dir,
            chunk_size = chunk_size,
            pbar = pbar
    )
        
        # method that configures & runs inference
        # use_thread=True will output result to napari layer/viewer
        inference_config.config_and_run_inference(use_thread=True)
        pbar.show()

    # make the scroll available
    scroll = QScrollArea()
    scroll.setWidget(widget._widget._qwidget)
    widget._widget._qwidget = scroll

    return widget


@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def volume_dock_widget():
    return volume_inference_widget, {'name': '3D Inference'}