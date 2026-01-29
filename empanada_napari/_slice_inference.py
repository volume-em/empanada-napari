import numpy as np
import dask.array as da
from time import time
from tqdm import tqdm
from skimage.draw import polygon

from empanada.config_loaders import read_yaml
from empanada_napari.inference import Engine2d
from empanada_napari.utils import get_configs, abspath

from napari import Viewer
from napari.layers import Image, Labels, Shapes
from napari_plugin_engine import napari_hook_implementation

from magicgui import magicgui, widgets
from skimage import measure
from scipy.ndimage import binary_fill_holes
from qtpy.QtWidgets import QScrollArea
from torch.cuda import device_count
from napari.qt.threading import thread_worker


class SliceInferenceWidget:
    def __init__(self, 
            model_config: str,
            viewer: Viewer = None,
            label_head: dict = None,
            image_layer: Image = None,
            downsampling: int = 1,
            confidence_thr: float = 0.5,
            center_confidence_thr: float = 0.1,
            min_distance_object_centers: int = 3,
            fine_boundaries: bool = False,
            semantic_only: bool = False,
            fill_holes_in_segmentation: bool = False,
            maximum_objects_per_class: int = 10000,
            tile_size: int = 0,
            batch_mode: bool = False,
            use_gpu: bool = False,
            use_quantized: bool = False,
            viewport: bool = False,
            confine_to_roi: bool = False,
            output_to_layer: bool = False,
            output_layer: Labels = None,
            pbar: widgets.ProgressBar = None
    ):
        self.viewer = viewer
        self.label_head = label_head
        self.image_layer = image_layer
        self.model_config_name = model_config
        self.downsampling = downsampling
        self.confidence_thr = confidence_thr
        self.center_confidence_thr = center_confidence_thr
        self.min_distance_object_centers = min_distance_object_centers
        self.fine_boundaries = fine_boundaries
        self.fill_holes = fill_holes_in_segmentation
        self.tile_size = tile_size
        self.batch_mode = batch_mode
        self.semantic_only = semantic_only
        self.using_gpu = use_gpu
        self.using_quantized = use_quantized
        self.viewport = viewport
        self.confine_to_roi = confine_to_roi
        self.output_to_layer, self.output_layer = output_to_layer, output_layer
        self.pbar = pbar
        self.maximum_objects_per_class = int(maximum_objects_per_class)
        self.last_config = None
        self.engine = None

        self._check_option_compatibility()

    # ---------------- Option handling & inference running entrypoint ----------------
    def config_and_run_inference(self, use_thread=False):  
        # Load the model config
        model_configs = get_configs()
        self.model_config = read_yaml(model_configs[self.model_config_name])

        if self.last_config is None:
            self.last_config = self.model_config_name

        self.get_engine()
                
        # Get the 2d slice from the image (Can mock a layer/viewer object in the tests)
        if not self.batch_mode:
            if self.confine_to_roi:
                shapes_layer = [layer for layer in self.viewer.layers if isinstance(layer, Shapes)][0]
                image2d, y, x, y_max, x_max, binary_mask = self._get_roi_slice(self.image_layer, shapes_layer)
                image2d[binary_mask == False] = 0
                axis, plane = "overloaded", self.image_layer.data.shape
            else:
                image2d, axis, plane, y, x = self._get_current_slice(self.image_layer)
            print(f'Image of size {image2d.shape} sliced at plane {plane} from axis {axis}')
            if type(image2d) == da.core.Array:
                image2d = image2d.compute()

        
        #### The part above should return a 2D slice from the Zarr input fine

            #### Now, we need to __split the 2D slice into tiles__ because the slice is still very big!
            #### This part should only run if we have an image that is a zarr or dask arr
                # Need a condition that the array should meet to run inference over tiles... maybe arr size?
            
                #### Setup jobs:
                # if batch_mode false

                #create image2dout array
                # image2dout = np.zeros_like(image2d)

                # compute segmentation on each of the tiles
                # run in parallel
                # Pass the result to _show_test_result() or _store_test_result()

                # if batch_mode true
                # compute segmentation using the run_model_batch
                # run in parallel
                # Pass result to _show/store_test_result()


        # Run the inference methods (either threaded or synchronously)
        match (self.batch_mode, use_thread):
            case True, True:
                assert not self.output_to_layer, "Batch mode is not compatible with output to layer!"
                assert not self.image_layer.multiscale, "Batch mode is not compatible with multiscale images!"
                assert not self.viewport, "Batch mode is not compatible with viewport inference!"
                assert not self.confine_to_roi, "Batch mode is not compatible with ROI inference!"

                test_worker = self.run_model_batch(self.engine, self.image_layer.data, self.fill_holes)
                if self.image_layer.data.ndim == 2:
                    test_worker.returned.connect(self._show_test_result)
                else:
                    test_worker.returned.connect(self._show_batch_stack)
                test_worker.start()

            case True, False:# For testing batch slice inference
                seg, axis, plane, y, x = self._run_model_batch(self.engine, self.image_layer.data, self.fill_holes)
                return seg, axis, plane, y, x
            
            case False, True:
                inference_worker = self.run_model(self.engine, image2d, axis, plane, y, x, self.fill_holes)
                if self.output_to_layer:
                    inference_worker.returned.connect(self._store_test_result)
                else:
                    inference_worker.returned.connect(self._show_test_result)
                inference_worker.start()

            case False, False: # For testing non-batch slice inference
                seg, axis, plane, y, x = self._run_model(self.engine, image2d, axis, plane, y, x, self.fill_holes)
                return seg, axis, plane, y, x
        return

    # ---------------- Engine management ----------------
    def get_engine(self):
        update_engine = (
            self.engine is None
            or self.last_config != self.model_config_name
        )

        if update_engine:
            self.engine = Engine2d(
                self.model_config,
                inference_scale=self.downsampling,
                nms_kernel=self.min_distance_object_centers,
                nms_threshold=self.center_confidence_thr,
                confidence_thr=self.confidence_thr,
                label_divisor=self.maximum_objects_per_class,
                semantic_only=self.semantic_only,
                fine_boundaries=self.fine_boundaries,
                tile_size=self.tile_size,
                use_gpu=self.using_gpu,
                use_quantized=self.using_quantized,
            )
        else:
            # update the parameters of the engine
            # without reloading the model
            self.engine.update_params(
                inference_scale=self.downsampling,
                label_divisor=self.maximum_objects_per_class,
                nms_threshold=self.center_confidence_thr,
                nms_kernel=self.min_distance_object_centers,
                confidence_thr=self.confidence_thr,
                semantic_only=self.semantic_only,
                fine_boundaries=self.fine_boundaries,
                tile_size=self.tile_size,
            )
        self.last_config = self.model_config_name
        return

    # ---------------- Helper methods ----------------    
    def _fill_holes_in_segmentation(self, mask):
        unique_indices = np.unique(mask)
        rprops = measure.regionprops(mask)

        # crop labels and then apply fill holes
        for rp in tqdm(rprops, desc='filling holes in labels:'):
            if rp.label in unique_indices and rp.label > 0:
                minr, minc, maxr, maxc = rp.bbox

                tmp = mask[minr:maxr, minc:maxc]
                tmp = binary_fill_holes(tmp.astype(bool))
                mask[minr:maxr, minc:maxc] = tmp.astype(mask.dtype) * rp.label
        return mask
    
    def _viewer_slices(self, image_layer, plane=None, axis=None):
        corners = image_layer.corner_pixels.T.tolist()
        if isinstance(axis, tuple) and isinstance(plane, tuple):
            yslice = slice(*corners[2])
            xslice = slice(*corners[3])
        elif axis is not None:
            # handle all of the weird special cases
            cases12 = [(0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
            cases01 = [(1, 2, 0)]
            cases20 = [(2, 0, 1)]
            if self.viewer.dims.order in cases12:
                yslice = slice(*corners[1])
                xslice = slice(*corners[2])
            elif self.viewer.dims.order in cases01:
                yslice = slice(*corners[0])
                xslice = slice(*corners[1])
            else:  # cases20
                yslice = slice(*corners[2])
                xslice = slice(*corners[0])
        else:
            yslice = slice(*corners[0])
            xslice = slice(*corners[1])

        print(f'Corners {corners}, slices {yslice, xslice}')

        return yslice, xslice

    def _get_current_slice(self, image_layer):
        cursor_pos = self.viewer.cursor.position

        # handle multiscale by taking highest resolution level
        image = image_layer.data
        if image_layer.multiscale:
            print('Using highest resolution level from multiscale!')
            image = image[0]

        y, x = 0, 0
        if image.ndim == 4:
            axis = tuple(self.viewer.dims.order[:2])
            plane = (
                int(image_layer.world_to_data(cursor_pos)[axis[0]]),
                int(image_layer.world_to_data(cursor_pos)[axis[1]])
            )

            slices = [slice(None), slice(None), slice(None), slice(None)]

            slices[axis[0]] = plane[0]
            slices[axis[1]] = plane[1]
            if self.viewport:
                yslice, xslice = self._viewer_slices(image_layer, plane)
                slices[2] = yslice
                slices[3] = xslice
                y = yslice.start
                x = xslice.start

        elif image.ndim == 3:
            axis = self.viewer.dims.order[0]
            plane = int(image_layer.world_to_data(cursor_pos)[axis])

            slices = [slice(None), slice(None), slice(None)]
            slices[axis] = plane
            if self.viewport:
                yaxis, xaxis = [i for i in range(3) if i != axis]
                yslice, xslice = self._viewer_slices(image_layer, plane, axis)
                slices[yaxis] = yslice
                slices[xaxis] = xslice
                y = yslice.start
                x = xslice.start
                print(f'Slices {slices}')

        else:
            slices = [slice(None), slice(None)]
            axis = None
            plane = None
            if self.viewport:
                yslice, xslice = self._viewer_slices(image_layer, plane, axis)
                slices[0] = yslice
                slices[1] = xslice
                y = yslice.start
                x = xslice.start

        return image[tuple(slices)], axis, plane, y, x

    def _get_mask_from_roi(self, image_layer, shapes_layer):
        h, w = image_layer.data.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        for shape in shapes_layer.data:
            rr, cc = polygon(shape[:, 0], shape[:, 1], (h, w))
            mask[rr, cc] = True
        return mask

    def _get_roi_slice(self, image_layer, shapes_layer):
        shapes = np.array(shapes_layer.data)
        min_y, min_x = np.inf, np.inf
        max_y, max_x = -np.inf, -np.inf
        for shape in shapes:
            min_y, min_x = min(min_y, shape[:, 0].min()), min(min_x, shape[:, 1].min())
            max_y, max_x = max(max_y, shape[:, 0].max()), max(max_x, shape[:, 1].max())
        min_y, min_x, max_y, max_x = map(int, (min_y, min_x, max_y, max_x))
        roi = image_layer.data[min_y:max_y, min_x:max_x].copy()
        mask = self._get_mask_from_roi(image_layer, shapes_layer)
        return roi, min_y, min_x, max_y, max_x, mask[min_y:max_y, min_x:max_x]
    
    def _check_option_compatibility(self):
        if self.output_to_layer:
            assert self.output_layer is not None, "Must select an output layer or uncheck Output to layer!"
            assert self.output_layer.data.shape == self.image_layer.data.shape, \
                "Output layer must have the same shape as the input image."
            assert self.viewport is False, "Cannot output to layer and restrict to viewport at the same time."

        if self.batch_mode:
            assert self.viewport is False, "Cannot use batch mode and restrict to viewport at the same time."

        if self.viewport:
            assert all(s == 1 for s in
                       self.image_layer.scale), "Viewport inference only supports images with scale 1 in all dimensions!"
            assert self.viewer.dims.order[0] != 1, "Viewport inference not supported for xz planes!"

        if not all(s == 1 for s in self.image_layer.scale):
            print(f'Image has non-unit scale. 2D segmentations will disappear after rotation or axis rolling!')
        return

    # ---------------- Inference runners ----------------
    @thread_worker
    def run_model(self, engine, image, axis, plane, y, x, fill_holes):
        return self._run_model(engine, image, axis, plane, y, x, fill_holes)

    @thread_worker
    def run_model_batch(self, engine, image, fill_holes):
        return self._run_model_batch(engine, image, fill_holes)
    
    def _run_model(self, engine, image, axis, plane, y, x, fill_holes):
        # create the inference engine
        start = time()
        seg = engine.infer(image)
        if fill_holes:
            seg = self._fill_holes_in_segmentation(seg)
        print(f'Inference time:', time() - start)
        return seg, axis, plane, y, x

    def _run_model_batch(self, engine, image, fill_holes):
        # axis is always xy
        axis = 0

        # create the inference engine
        if image.ndim == 3:
            print(f'Running batch mode inference on {len(image)} images.')
            segmentations = []
            for plane, img_slice in tqdm(enumerate(image), total=len(image)):
                if type(img_slice) == da.core.Array:
                    img_slice = img_slice.compute()

                seg = engine.infer(img_slice)
                if fill_holes:
                    seg = self._fill_holes_in_segmentation(seg)
                segmentations.append(seg)

            # stack segmentations with padding
            max_h = max(seg.shape[0] for seg in segmentations)
            max_w = max(seg.shape[1] for seg in segmentations)
            padded = []
            for seg in segmentations:
                h, w = seg.shape
                padh, padw = max_h - h, max_w - w
                padded.append(np.pad(seg, ((0, padh), (0, padw))))

            padded = np.stack(padded, axis=0)
            return padded

        elif image.ndim == 2:
            start = time()
            if type(image) == da.core.Array:
                image = image.compute()

            plane = 0
            seg = engine.infer(image)
            if fill_holes:
                seg = self._fill_holes_in_segmentation(seg)
            print(f'Inference time:', time() - start)
            return seg, None, None, None, None
        
        else:
            raise Exception(f'Batch mode supports 2d and 3d, got {image.ndim}d.')
    
    # ---------------- GUI result functions ----------------
    def _show_batch_stack(self, *args):
        stack = args[0]
        self.viewer.add_labels(stack, name=self.image_layer.name + '_batch_segs')
        self.pbar.hide()

    def _show_test_result(self, *args):
        seg, axis, plane, y, x = args[0]

        if axis == "overloaded":
            out_2d = np.zeros(plane, dtype=seg.dtype)
            seg_shape = seg.shape
            out_2d[y:y + seg_shape[0], x:x + seg_shape[1]] = seg
            seg = out_2d
            translate = [0, 0]
        elif axis is not None and plane is not None:
            if isinstance(axis, tuple) and isinstance(plane, tuple):
                seg = np.expand_dims(seg, axis=axis)
                translate = [0, 0, y, x]
                translate[axis[0]] = plane[0]
                translate[axis[1]] = plane[1]
            else:
                seg = np.expand_dims(seg, axis=axis)

                # oddly translate has to be a list and
                # not an array or things break. WHY????
                translate = self.image_layer.translate.tolist()
                translate[axis] += plane
                yaxis, xaxis = [i for i in range(3) if i != axis]
                if y is not None:
                    translate[yaxis] += y
                if x is not None:
                    translate[xaxis] += x
        else:
            translate = [y, x]

        self.viewer.add_labels(seg, name=f'empanada_seg_2d', visible=True, translate=tuple(translate))
        self.viewer.layers[-1].scale = self.image_layer.scale

        self.pbar.hide()


    def _store_test_result(self, *args):
        seg, axis, plane, y, x = args[0]

        if axis == "overloaded":
            out_2d = np.zeros(plane, dtype=seg.dtype)
            seg_shape = seg.shape
            out_2d[y:y + seg_shape[0], x:x + seg_shape[1]] = seg
            seg = out_2d
            self.output_layer.data = seg
        elif axis is not None and plane is not None:
            if isinstance(axis, tuple) and isinstance(plane, tuple):
                # 4D flipbook case
                slices = [slice(None), slice(None), slice(None), slice(None)]
                slices[axis[0]] = plane[0]
                slices[axis[1]] = plane[1]
                self.output_layer.data[tuple(slices)] = seg
            else:
                # 3D case
                slices = [slice(None), slice(None), slice(None)]
                slices[axis] = plane
                self.output_layer.data[tuple(slices)] = seg
        else:
            # 2D case
            self.output_layer.data = seg

        self.output_layer.visible = False
        self.output_layer.visible = True

        self.pbar.hide()

# ---------------- Napari GUI wrapper ----------------
def slice_inference_widget():
    """
    Factory function to create the widget for Napari.
    This is what Napari will call.
    """
    from napari.layers import Image, Labels
    from magicgui import widgets

    logo = abspath(__file__, 'resources/empanada_logo.png')
    model_configs = get_configs()

    # define magicgui params
    gui_params = dict(
        model_config=dict(widget_type='ComboBox', choices=list(model_configs.keys()),
                          value=list(model_configs.keys())[0], label='Model', tooltip='Model to use for inference'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Image Downsampling',
                          tooltip='Downsampling factor to apply before inference'),
        confidence_thr=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1,
                            label='Segmentation Confidence Thr'),
        center_confidence_thr=dict(widget_type='FloatSpinBox', value=0.1, min=0.05, max=0.9, step=0.05,
                                   label='Center Confidence Thr'),
        min_distance_object_centers=dict(widget_type='SpinBox', value=3, min=1, max=35, step=1,
                                         label='Centers Min Distance'),
        fine_boundaries=dict(widget_type='CheckBox', text='Fine boundaries', value=False,
                             tooltip='Finer boundaries between objects'),
        semantic_only=dict(widget_type='CheckBox', text='Semantic only', value=False,
                           tooltip='Only run semantic segmentation for all classes.'),
        fill_holes_in_segmentation=dict(widget_type='CheckBox', text='Fill holes in segmentation', value=False,
                                        tooltip='If checked, fill holes in the segmentation mask.'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value='10000', label='Max objects per class',
                                       tooltip='Maximum number of objects per class/ label divisor for mutliclass segmentation.'),
        tile_size=dict(widget_type='SpinBox', value=0, min=0, max=128000, step=1280, label='Tile size',
                       tooltip='Tile size for inference, whole image will be segmented if 0'),
        batch_mode=dict(widget_type='CheckBox', text='Batch mode', value=False,
                        tooltip='If checked, each image in a stack is segmented independently.'),
        viewport=dict(widget_type='CheckBox', text='Confine to viewport', value=False,
                      tooltip='If checked, inference will be restricted to the current viewport.'),
        output_to_layer=dict(widget_type='CheckBox', text='Output to layer', value=False,
                             tooltip='If checked, the segmentation is output to the selected output layer.'),
    )

    gui_params['use_gpu'] = dict(widget_type='CheckBox', text='Use GPU', value=device_count() >= 1,
                                 tooltip='If checked, run on GPU 0')
    gui_params['use_quantized'] = dict(widget_type='CheckBox', text='Use quantized model', value=device_count() == 0,
                                       tooltip='If checked, run on GPU 0')
    # Add the new option to the gui_params dictionary
    gui_params['confine_to_roi'] = dict(widget_type='CheckBox', text='Confine to ROI', value=False,
                                        tooltip='If checked, inference will be restricted to the ROI defined by a shapes layer.')
    
    @magicgui(
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Run 2D Inference',
        layout='vertical',
        scrollable=True,
        pbar={'visible': False, 'max': 0, 'label': 'Running...'},
        **gui_params
    )
    def widget(
            viewer: Viewer,
            label_head,
            image_layer: Image,
            model_config,
            downsampling,
            confidence_thr,
            center_confidence_thr,
            min_distance_object_centers,
            fine_boundaries,
            semantic_only,
            fill_holes_in_segmentation,
            maximum_objects_per_class,
            tile_size,
            batch_mode,
            use_gpu,
            use_quantized,
            viewport,
            confine_to_roi,
            output_to_layer,
            output_layer: Labels,
            pbar: widgets.ProgressBar
    ):

        # instantiate the class
        inference_config = SliceInferenceWidget(viewer=viewer,
            label_head=label_head,
            image_layer=image_layer,
            model_config=model_config,
            downsampling=downsampling,
            confidence_thr=confidence_thr,
            center_confidence_thr=center_confidence_thr,
            min_distance_object_centers=min_distance_object_centers,
            fine_boundaries=fine_boundaries,
            semantic_only=semantic_only,
            fill_holes_in_segmentation=fill_holes_in_segmentation,
            maximum_objects_per_class=maximum_objects_per_class,
            tile_size=tile_size,
            batch_mode=batch_mode,
            use_gpu=use_gpu,
            use_quantized=use_quantized,
            viewport=viewport,
            confine_to_roi=confine_to_roi,
            output_to_layer=output_to_layer,
            output_layer=output_layer,
            pbar=pbar
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
def slice_dock_widget():
    return slice_inference_widget, {'name': '2D Inference (Parameter Testing)'}
