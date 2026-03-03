import os
import re
import subprocess
import warnings
import pytest
import numpy as np
from tifffile import imread
import tifffile
from napari.components import ViewerModel
from empanada_napari._slice_inference import SliceInferenceWidget
from empanada_napari._volume_inference import VolumeInferenceWidget
from empanada_napari.utils import get_configs

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