import os
import re
import subprocess
import warnings
import pytest
import numpy as np
from tifffile import imread
import tifffile
import napari.viewer
from napari.components import ViewerModel
from empanada.array_utils import take
from empanada_napari._slice_inference import SliceInferenceWidget
from empanada_napari._volume_inference import VolumeInferenceWidget
from empanada_napari._merge_split_widget import merge_labels
from empanada_napari.utils import get_configs, enable_layer_rename_refresh

from .conftest import MODEL_NAMES, gen_slice_sanity_params, gen_slice_dset_params, \
                gen_vol_sanity_params, gen_vol_dset_params, gen_ortho_dset_params

# ---------------- Global Variables ----------------
DATA_DIR = "datasets_for_tests"
FILE_2D = "nanotomy_islet_rat375_crop1.tif"
FILE_3D = "hela_cell_em.tif"

# ---------------- Tests ----------------
class TestSliceInference:
    # ---------------- Dataset Fixtures ---------------- 
    @pytest.fixture
    def image_2d(self):
        rng = np.random.default_rng(0)
        h, w = 100, 100
        y, x = np.mgrid[0:h, 0:w]
        image = np.zeros((h, w), dtype=np.float32)

        # Parameters for blobs
        n_blobs = 8
        for _ in range(n_blobs):
            cx = rng.uniform(0, w)
            cy = rng.uniform(0, h)
            sigma = rng.uniform(4, 10)
            amplitude = rng.uniform(120, 255)

            blob = amplitude * np.exp(
                -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
            )
            image += blob

        # Add mild noise
        image += rng.normal(0, 10, size=image.shape)

        # Normalize to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    @pytest.fixture
    def tutorial_2d_image(self):
        dataset_2d = "https://zenodo.org/records/15319873"
        datapath =  os.path.join(DATA_DIR, FILE_2D)
        if os.path.isfile(datapath) is False:
            subprocess.run(["zenodo_get", dataset_2d, "-o", DATA_DIR])
        image = imread(datapath)
        if image.shape != (3000, 12600): # Rerun if incomplete download
            subprocess.run(["zenodo_get", dataset_2d, "-o", DATA_DIR])
        print(type(image), image.shape)
        return image

    # ---------------- Tests ----------------
    @pytest.mark.parametrize(("test_args", "expected_shape"), gen_slice_sanity_params(), #list(zip(slice_test_args, expect_shape)),
            ids=["tutorial_params", "DropNet", "NucleoNet", "fine_boundaries", "semantic_only", 
              "fill_holes_in_segmentation", "use_quantized", "batch_mode", "use_gpu", "viewport", 
              "confine_to_roi", "output_to_layer"])
    def test_slice_inference_sanity(self, image_2d, test_args, expected_shape):
        viewer = ViewerModel()
        image_layer = viewer.add_image(image_2d)
        if "model_config" not in test_args.keys():
            test_args["model_config"] = MODEL_NAMES['MitoNet_mini']

        if "output_to_layer" in test_args.keys():
            output_layer = viewer.add_image(np.zeros_like(image_2d))
            test_args["output_layer"] = output_layer

        if "confine_to_roi" in test_args.keys():
            triangle = np.array([[11, 13], [30, 6], [30, 20]])
            viewer.add_shapes(triangle, shape_type="polygon", edge_width=5)

        inference_config = SliceInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        **test_args)
        seg, _, _, _, _ = inference_config.config_and_run_inference(use_thread=False)

        assert isinstance(seg, np.ndarray)
        assert np.asarray(seg).shape == expected_shape

    def test_roi_slice_from_labels_layer(self, image_2d):
        """Confine-to-ROI helpers should crop/mask from a Labels layer."""
        viewer = ViewerModel()
        image_layer = viewer.add_image(image_2d)
        labels = np.zeros(image_2d.shape, dtype=np.int32)
        labels[10:25, 15:40] = 1
        labels[50:60, 50:55] = 2
        labels_layer = viewer.add_labels(labels)

        widget = SliceInferenceWidget(
            viewer=viewer,
            image_layer=image_layer,
            model_config=MODEL_NAMES['MitoNet_mini'],
            confine_to_roi=True,
            roi_layer=labels_layer,
        )

        resolved = widget._resolve_roi_layer()
        assert resolved is labels_layer

        roi, min_y, min_x, max_y, max_x, mask = widget._get_roi_slice(image_layer, labels_layer)
        assert (min_y, min_x, max_y, max_x) == (10, 15, 60, 55)
        assert roi.shape == (50, 40)
        assert mask.shape == roi.shape
        # crop origin (10, 15) is inside label 1
        assert mask[0, 0]
        # image coordinate (30, 50) is between the two cells
        assert not mask[20, 35]

    def test_roi_prefers_shapes_over_labels(self, image_2d):
        viewer = ViewerModel()
        image_layer = viewer.add_image(image_2d)
        labels = np.zeros(image_2d.shape, dtype=np.int32)
        labels[10:25, 15:40] = 1
        labels_layer = viewer.add_labels(labels)
        triangle = np.array([[11, 13], [30, 6], [30, 20]])
        shapes_layer = viewer.add_shapes(triangle, shape_type="polygon", edge_width=5)

        widget = SliceInferenceWidget(
            viewer=viewer,
            image_layer=image_layer,
            model_config=MODEL_NAMES['MitoNet_mini'],
            confine_to_roi=True,
            roi_layer=labels_layer,
        )
        assert widget._resolve_roi_layer() is shapes_layer


    @pytest.mark.slow
    @pytest.mark.parametrize(("test_args", "expected_labels"), gen_slice_dset_params(), #list(zip(slice_test_args, expect_results)),
            ids=["tutorial_params", "DropNet", "NucleoNet", "MitoNetMini"])
    def test_slice_inference_dataset(self, tutorial_2d_image, test_args, expected_labels):
        viewer = ViewerModel()
        image_layer = viewer.add_image(tutorial_2d_image)

        inference_config = SliceInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        use_gpu=True,
                                        **test_args)
        seg, _, _, _, _ = inference_config.config_and_run_inference(use_thread=False)
        seg_nonzero = seg[seg != 0]
        counts, _ = np.histogram(seg_nonzero, bins=10)

        print(seg.min(), seg.max())

        tolerance = 0.1  # 10% tolerance
        for count, expected in zip(counts, expected_labels):
            lb = expected * (1-tolerance)
            ub = expected * (1+tolerance)
            assert lb <= count <= ub


class TestVolumeInference:
    # ---------------- Dataset Fixtures ----------------
    @pytest.fixture
    def image_3d(self):
        rng = np.random.default_rng(0)
        h, w, d = 100, 100, 100
        z, y, x = np.mgrid[0:d, 0:h, 0:w]
        image = np.zeros((h, w, d), dtype=np.float32)
        # Parameters for blobs
        n_blobs = 8
        for _ in range(n_blobs):
            cx = rng.uniform(0, w)
            cy = rng.uniform(0, h)
            cz = rng.uniform(0, d)
            sigma = rng.uniform(4, 10)
            amplitude = rng.uniform(120, 255)
            blob = amplitude * np.exp(
                -((z - cz)**2 + (x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
            )
            image += blob
        # Add mild noise
        image += rng.normal(0, 10, size=image.shape)
        # Normalize to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @pytest.fixture
    def tutorial_3d_image(self):
        dataset_3d = "https://zenodo.org/records/15311513"
        datapath =  os.path.join(DATA_DIR, FILE_3D)
        if os.path.isfile(datapath) is False:
            subprocess.run(["zenodo_get", dataset_3d, "-o", DATA_DIR])
        image = imread(datapath)
        if image.shape != (256, 256, 256): # Rerun if incomplete download
            subprocess.run(["zenodo_get", dataset_3d, "-o", DATA_DIR])
        print(type(image), image.shape)
        return image


    # ---------------- Tests  ---------------- 
    @pytest.mark.parametrize(("test_args", "expected_shape"), gen_vol_sanity_params(), #list(zip(vol_test_args, expect_shape)),
        ids=["MitoNet", "DropNet", "NucleoNet", "fine_boundaries", "semantic_only", 
             "fill_holes_in_segmentation", "use_quantized", "use_gpu", "multigpu", "allow_one_view"])
    def test_volume_stack_inference_sanity(self, image_3d, test_args, expected_shape):
        viewer = ViewerModel()
        image_layer = viewer.add_image(image_3d)
        inference_plane = "xy"
        if "model_config" not in test_args.keys():
            test_args["model_config"] = MODEL_NAMES['MitoNet_mini']

        inference_config = VolumeInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        return_panoptic=True,
                                        inference_plane=inference_plane,
                                        orthoplane=False,
                                        **test_args)
        
        stack, axis_name, trackers_dict = inference_config.config_and_run_inference(use_thread=False)
        assert isinstance(stack, np.ndarray)
        assert stack.shape == expected_shape


    @pytest.mark.slow
    @pytest.mark.parametrize(("test_args", "expected_labels"), gen_vol_dset_params(), #list(zip(vol_dset_args, expect_results)),
            ids=["MitoNet", "DropNet", "NucleoNet", "MitoNetMini"])
    def test_volume_stack_inference_dataset(self, tutorial_3d_image, test_args, expected_labels):
        viewer = ViewerModel()
        image_layer = viewer.add_image(tutorial_3d_image)
        # inference_plane = "xy"

        inference_config = VolumeInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        return_panoptic=True,
                                        use_gpu=True,
                                        orthoplane=False,
                                        # inference_plane=inference_plane,
                                        **test_args)
        
        stack, axis_name, trackers_dict = inference_config.config_and_run_inference(use_thread=False)
        seg_nonzero = stack[stack != 0]
        counts, _ = np.histogram(seg_nonzero, bins=10)

        tolerance = 0.1  # 10% tolerance
        for count, expected in zip(counts, expected_labels):
            lb = expected * (1-tolerance)
            ub = expected * (1+tolerance)
            assert lb <= count <= ub


    @pytest.mark.parametrize(("test_args", "expected_shape"), gen_vol_sanity_params(), #list(zip(vol_test_args, expect_shape)),
        ids=["MitoNet", "DropNet", "NucleoNet", "fine_boundaries", "semantic_only", 
             "fill_holes_in_segmentation", "use_quantized", "use_gpu", "multigpu", "allow_one_view"])
    def test_volume_orthoplane_inference_sanity(self, image_3d, test_args, expected_shape):   
        viewer = ViewerModel()
        image_layer = viewer.add_image(image_3d)
        if "model_config" not in test_args.keys():
            test_args["model_config"] = MODEL_NAMES['MitoNet_mini']

        inference_config = VolumeInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        return_panoptic=True,
                                        orthoplane=True,
                                        **test_args)
        
        result = inference_config.config_and_run_inference(use_thread=False)
        for _, stack in result.items():
            assert isinstance(stack, np.ndarray)
            assert stack.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize(("test_args", "expected_labels"), gen_ortho_dset_params(), #list(zip(vol_dset_args, expect_results)),
            ids=["MitoNet", "DropNet", "NucleoNet", "MitoNetMini"])
    def test_volume_orthoplane_inference_dataset(self, tutorial_3d_image, test_args, expected_labels):   
        viewer = ViewerModel()
        image_layer = viewer.add_image(tutorial_3d_image)

        inference_config = VolumeInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        use_gpu=True,
                                        return_panoptic=True,
                                        orthoplane=True,
                                        **test_args)

        result = inference_config.config_and_run_inference(use_thread=False)
        tolerance = 0.1  # 10% tolerance

        for (_, stack), expected_label in zip(result.items(), expected_labels):
            seg_nonzero = stack[stack != 0]
            counts, _ = np.histogram(seg_nonzero, bins=10)

            for count, expected in zip(counts, expected_label):
                lb = expected * (1-tolerance)
                ub = expected * (1+tolerance)
                assert lb <= count <= ub


class _SpyEngine:
    """Fake inference engine that records every 2D array it is asked to segment."""
    def __init__(self):
        self.calls = []

    def infer(self, image):
        image = np.asarray(image)
        self.calls.append(image.copy())
        return (image > 0).astype(np.int32)


class TestBatchInferenceOrientation:
    """Regression tests for 2D batch inference on non-default (xz/yz) viewer orientations."""

    @pytest.fixture
    def volume(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(6, 4, 8), dtype=np.uint8)

    @pytest.mark.parametrize(("order", "axis"), [
        ((0, 1, 2), 0),  # xy: iterate over the first (z) axis
        ((1, 0, 2), 1),  # xz: iterate over the second (y) axis
        ((2, 1, 0), 2),  # yz: iterate over the third (x) axis
    ], ids=["xy", "xz", "yz"])
    def test_run_model_batch_slices_along_viewer_axis(self, volume, order, axis):
        viewer = ViewerModel()
        image_layer = viewer.add_image(volume)
        viewer.dims.order = order

        widget = SliceInferenceWidget(
            viewer=viewer,
            image_layer=image_layer,
            model_config=MODEL_NAMES['MitoNet_mini'],
            batch_mode=True,
        )

        spy = _SpyEngine()
        stacked = widget._run_model_batch(spy, volume, fill_holes=False)

        # output must match the input volume's shape/orientation
        assert stacked.shape == volume.shape

        # inference must have been run once per slice along the *viewed* axis,
        # not always along raw array axis 0
        assert len(spy.calls) == volume.shape[axis]
        for i, recorded_slice in enumerate(spy.calls):
            expected_slice = take(volume, i, axis)
            assert np.array_equal(recorded_slice, expected_slice), \
                f"Batch inference used the wrong slice at index {i} for viewer order {order}"

    def test_batch_mode_nonthreaded_3d_end_to_end(self, volume):
        """config_and_run_inference should not crash and should respect orientation
        for a 3D batch-mode run (regression for the ndim==3 tuple-unpacking bug)."""
        viewer = ViewerModel()
        image_layer = viewer.add_image(volume)
        viewer.dims.order = (1, 0, 2)  # simulate viewing the xz plane

        widget = SliceInferenceWidget(
            viewer=viewer,
            image_layer=image_layer,
            model_config=MODEL_NAMES['MitoNet_mini'],
            batch_mode=True,
        )

        spy = _SpyEngine()
        widget.engine = spy
        widget.get_engine = lambda: None  # skip loading a real model

        seg, axis, plane, y, x = widget.config_and_run_inference(use_thread=False)

        assert seg.shape == volume.shape
        assert len(spy.calls) == volume.shape[1]


class TestLayerRenameRefresh:
    """Regression tests for layer-rename not being reflected in plugin dropdowns."""

    def test_enable_layer_rename_refresh_updates_combobox_choice(self):
        from magicgui import magicgui
        from napari.layers import Labels

        viewer = ViewerModel()
        labels_layer = viewer.add_labels(np.zeros((5, 5), dtype=int), name='orig_name')

        def get_labels_layers(gui):
            return [l for l in viewer.layers if isinstance(l, Labels)]

        @magicgui(labels_layer=dict(widget_type='ComboBox', choices=get_labels_layers))
        def widget(labels_layer):
            pass

        enable_layer_rename_refresh(widget, viewer=viewer)

        assert widget.labels_layer.current_choice == 'orig_name'
        labels_layer.name = 'renamed_layer'
        assert widget.labels_layer.current_choice == 'renamed_layer'

    def test_merge_labels_dropdown_refreshes_on_rename(self, monkeypatch):
        """Merge Labels (and other widgets using enable_layer_rename_refresh) should
        reflect a layer rename immediately, without closing/reopening the widget."""
        viewer = ViewerModel()
        labels_layer = viewer.add_labels(np.zeros((5, 5), dtype=int), name='orig_name')

        # widgets resolve their viewer via napari's current_viewer() fallback
        # when not docked in a real Qt window; simulate that here.
        monkeypatch.setattr(napari.viewer, 'current_viewer', lambda: viewer)

        widget = merge_labels()
        assert widget.labels_layer.current_choice == 'orig_name'

        labels_layer.name = 'renamed_layer'
        assert widget.labels_layer.current_choice == 'renamed_layer'

    def test_enable_layer_rename_refresh_tracks_new_layers(self):
        from magicgui import magicgui
        from napari.layers import Labels

        viewer = ViewerModel()

        def get_labels_layers(gui):
            return [l for l in viewer.layers if isinstance(l, Labels)]

        @magicgui(labels_layer=dict(widget_type='ComboBox', choices=get_labels_layers))
        def widget(labels_layer):
            pass

        enable_layer_rename_refresh(widget, viewer=viewer)

        new_layer = viewer.add_labels(np.zeros((5, 5), dtype=int), name='new_layer')
        assert widget.labels_layer.current_choice == 'new_layer'

        new_layer.name = 'renamed_new_layer'
        assert widget.labels_layer.current_choice == 'renamed_new_layer'