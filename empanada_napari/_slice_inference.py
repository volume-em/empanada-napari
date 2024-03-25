import math
import numpy as np
from napari_plugin_engine import napari_hook_implementation

import napari
from napari import Viewer
from napari.layers import Image, Labels
from magicgui import magicgui, widgets

#from magicgui.tqdm import tqdm
from tqdm import tqdm
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
    def run_model(
        engine,
        image,
        axis,
        plane,
        y,
        x
    ):
        # create the inference engine
        start = time()
        seg = engine.infer(image) 
        print(f'Inference time:', time() - start)
        return seg, axis, plane, y, x

    @thread_worker
    def run_model_batch(
        engine,
        image
    ):
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
            print(f'Inference time:', time() - start)
            return seg, None, None, None, None
        
        else:
            raise Exception(f'Batch mode supports 2d and 3d, got {image.ndim}d.')

    logo = abspath(__file__, 'resources/empanada_logo.png')

    gui_params = dict(
        model_config=dict(widget_type='ComboBox', choices=list(model_configs.keys()), value=list(model_configs.keys())[0], label='Model', tooltip='Model to use for inference'),
        downsampling=dict(widget_type='ComboBox', choices=[1, 2, 4, 8, 16, 32, 64], value=1, label='Image Downsampling', tooltip='Downsampling factor to apply before inference'),
        confidence_thr=dict(widget_type='FloatSpinBox', value=0.5, min=0.1, max=0.9, step=0.1, label='Segmentation Confidence Thr'),
        center_confidence_thr=dict(widget_type='FloatSpinBox', value=0.1, min=0.05, max=0.9, step=0.05, label='Center Confidence Thr'),
        min_distance_object_centers=dict(widget_type='SpinBox', value=3, min=1, max=21, step=1, label='Centers Min Distance'),
        fine_boundaries=dict(widget_type='CheckBox', text='Fine boundaries', value=False, tooltip='Finer boundaries between objects'),
        semantic_only=dict(widget_type='CheckBox', text='Semantic only', value=False, tooltip='Only run semantic segmentation for all classes.'),
        maximum_objects_per_class=dict(widget_type='LineEdit', value='10000', label='Max objects per class', tooltip='Maximum number of objects per class/ label divisor for mutliclass segmentation.'),
        tile_size=dict(widget_type='SpinBox', value=0, min=0, max=4096, step=256, label='Tile size', tooltip='Tile size for inference, whole image will be segmented if 0'),
        batch_mode=dict(widget_type='CheckBox', text='Batch mode', value=False, tooltip='If checked, each image in a stack is segmented independently.'),
        viewport=dict(widget_type='CheckBox', text='Confine to viewport', value=False, tooltip='If checked, inference will be restricted to the current viewport.'),
        output_to_layer=dict(widget_type='CheckBox', text='Output to layer', value=False, tooltip='If checked, the segmentation is output to the selected output layer.'),
    )

    gui_params['use_gpu'] = dict(widget_type='CheckBox', text='Use GPU', value=device_count() >= 1, tooltip='If checked, run on GPU 0')
    gui_params['use_quantized'] = dict(widget_type='CheckBox', text='Use quantized model', value=device_count() == 0, tooltip='If checked, run on GPU 0')

    @magicgui(
        label_head=dict(widget_type='Label', label=f'<h1 style="text-align:center"><img src="{logo}"></h1>'),
        call_button='Run 2D Inference',
        layout='vertical',
        pbar={'visible': False, 'max': 0, 'label': 'Running...'},
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
        tile_size,
        batch_mode,
        use_gpu,
        use_quantized,
        viewport,
        output_to_layer,
        output_layer: Labels,
        pbar: widgets.ProgressBar
    ):
        if output_to_layer:
            assert output_layer is not None, "Must select an output layer or uncheck Output to layer!"
            assert output_layer.data.shape == image_layer.data.shape, \
            "Output layer must have the same shape as the input image."
            assert viewport is False, "Cannot output to layer and restrict to viewport at the same time."

        if batch_mode:
            assert viewport is False, "Cannot use batch mode and restrict to viewport at the same time."

        if viewport:
            assert all(s == 1 for s in image_layer.scale), "Viewport inference only supports images with scale 1 in all dimensions!"
            assert viewer.dims.order[0] != 1, "Viewport inference not supported for xz planes!"

        if not all(s == 1 for s in image_layer.scale):
            print(f'Image has non-unit scale. 2D segmentations will disappear after rotation or axis rolling!')

        # load the model config
        model_config_name = model_config
        model_config = read_yaml(model_configs[model_config])
        maximum_objects_per_class = int(maximum_objects_per_class)

        if not hasattr(widget, 'last_config'):
            widget.last_config = model_config_name

        if not hasattr(widget, 'using_gpu'):
            widget.using_gpu = use_gpu

        if not hasattr(widget, 'using_quantized'):
            widget.using_quantized = use_quantized

        if not hasattr(widget, 'engine') or widget.last_config != model_config_name or use_gpu != widget.using_gpu or use_quantized != widget.using_quantized:
            widget.engine = Engine2d(
                model_config,
                inference_scale=downsampling,
                nms_kernel=min_distance_object_centers,
                nms_threshold=center_confidence_thr,
                confidence_thr=confidence_thr,
                label_divisor=maximum_objects_per_class,
                semantic_only=semantic_only,
                fine_boundaries=fine_boundaries,
                tile_size=tile_size,
                use_gpu=use_gpu,
                use_quantized=use_quantized
            )
            widget.last_config = model_config_name
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
                fine_boundaries=fine_boundaries,
                tile_size=tile_size,
            )

        def _viewer_slices(image_layer, axis=None):
            corners = image_layer.corner_pixels.T.tolist()
            if isinstance(axis, tuple) and isinstance(plane, tuple):
                yslice = slice(*corners[2])
                xslice = slice(*corners[3])
            elif axis is not None:
                # handle all of the weird special cases
                cases12 = [(0, 1, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
                cases01 = [(1, 2, 0)]
                cases20 = [(2, 0, 1)]
                if viewer.dims.order in cases12: 
                    yslice = slice(*corners[1])
                    xslice = slice(*corners[2])
                elif viewer.dims.order in cases01: 
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

        def _get_current_slice(image_layer):
            cursor_pos = viewer.cursor.position

            # handle multiscale by taking highest resolution level
            image = image_layer.data
            if image_layer.multiscale:
                print('Using highest resolution level from multiscale!')
                image = image[0]

            y, x = 0, 0
            if image.ndim == 4:
                axis = tuple(viewer.dims.order[:2])
                plane = (
                    int(image_layer.world_to_data(cursor_pos)[axis[0]]),
                    int(image_layer.world_to_data(cursor_pos)[axis[1]])
                )

                slices = [slice(None), slice(None), slice(None), slice(None)]
                
                slices[axis[0]] = plane[0]
                slices[axis[1]] = plane[1]
                if viewport:
                    yslice, xslice = _viewer_slices(image_layer)
                    slices[2] = yslice
                    slices[3] = xslice
                    y = yslice.start
                    x = xslice.start

            elif image.ndim == 3:
                axis = viewer.dims.order[0]
                plane = int(image_layer.world_to_data(cursor_pos)[axis])

                slices = [slice(None), slice(None), slice(None)]
                slices[axis] = plane
                if viewport:
                    yaxis, xaxis = [i for i in range(3) if i != axis]
                    yslice, xslice = _viewer_slices(image_layer, axis)
                    slices[yaxis] = yslice
                    slices[xaxis] = xslice
                    y = yslice.start
                    x = xslice.start
                    print(f'Slices {slices}')

            else:
                slices = [slice(None), slice(None)]
                axis = None
                plane = None
                if viewport:
                    yslice, xslice = _viewer_slices(image_layer, axis)
                    slices[0] = yslice
                    slices[1] = xslice
                    y = yslice.start
                    x = xslice.start

            return image[tuple(slices)], axis, plane, y, x

        def _show_test_result(*args):
            seg, axis, plane, y, x = args[0]

            if axis is not None and plane is not None:
                if isinstance(axis, tuple) and isinstance(plane, tuple):
                    seg = np.expand_dims(seg, axis=axis)
                    translate = [0, 0, y, x]
                    translate[axis[0]] = plane[0]
                    translate[axis[1]] = plane[1]
                else:
                    seg = np.expand_dims(seg, axis=axis)

                    # oddly translate has to be a list and
                    # not an array or things break. WHY????
                    translate = image_layer.translate.tolist()
                    translate[axis] += plane
                    yaxis, xaxis = [i for i in range(3) if i != axis]
                    if y is not None:
                        translate[yaxis] += y
                    if x is not None:
                        translate[xaxis] += x
            else:
                translate = [y, x]

            viewer.add_labels(seg, name=f'empanada_seg_2d', visible=True, translate=tuple(translate))
            viewer.layers[-1].scale = image_layer.scale

            pbar.hide()

        def _store_test_result(*args):
            seg, axis, plane, _, _ = args[0]

            if axis is not None and plane is not None:
                if isinstance(axis, tuple) and isinstance(plane, tuple):
                    # 4D flipbook case
                    slices = [slice(None), slice(None), slice(None), slice(None)]
                    slices[axis[0]] = plane[0]
                    slices[axis[1]] = plane[1]
                    output_layer.data[tuple(slices)] = seg
                else:
                    # 3D case
                    slices = [slice(None), slice(None), slice(None)]
                    slices[axis] = plane
                    output_layer.data[tuple(slices)] = seg
            else:
                # 2D case
                output_layer.data = seg

            output_layer.visible = False 
            output_layer.visible = True

            pbar.hide()

        def _show_batch_stack(*args):
            stack = args[0]
            viewer.add_labels(stack, name=image_layer.name + '_batch_segs')
            pbar.hide()

        # load data for currently viewer slice of chosen image layer
        if not batch_mode:
            image2d, axis, plane, y, x = _get_current_slice(image_layer)
            print(f'Image of size {image2d.shape} sliced at plane {plane} from axis {axis}')
            if type(image2d) == da.core.Array:
                image2d = image2d.compute()

            test_worker = run_model(widget.engine, image2d, axis, plane, y, x)
            if output_to_layer:
                test_worker.returned.connect(_store_test_result)
            else:
                test_worker.returned.connect(_show_test_result)
            test_worker.start()
        else:
            assert not output_to_layer, "Batch mode is not compatible with output to layer!"
            assert not image_layer.multiscale, "Batch mode is not compatible with multiscale images!"
            assert not viewport, "Batch mode is not compatible with viewport inference!"

            test_worker = run_model_batch(widget.engine, image_layer.data)
            if image_layer.data.ndim == 2:
                test_worker.returned.connect(_show_test_result)
            else:
                test_worker.returned.connect(_show_batch_stack)

            test_worker.start()

        pbar.show()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def slice_dock_widget():
    return test_widget, {'name': '2D Inference (Parameter Testing)'}
