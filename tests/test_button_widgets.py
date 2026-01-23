import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from empanada_napari._volume_inference import VolumeInferenceWidget
from empanada_napari.inference import Engine2d, Engine3d, stack_postprocessing, tracker_consensus
from empanada_napari.utils import get_configs
from empanada.config_loaders import read_yaml
from empanada_napari._slice_inference import SliceInferenceWidget


# ---------------- Dataset Fixtures ----------------
# Replace with an actual 2d img & maybe assert result == some values
@pytest.fixture
def image_2d():
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
def image_3d():
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


# ---------------- Tests ----------------
@pytest.mark.parametrize(("test_args", "expected_shape"),
    [
        ({"model_config":"MitoNet_v1", "batch_mode":True, "use_gpu":True}, (100, 100, 100))
        ({"model_config":"DropNet_base_v1"}, (100, 100)),
        ({"model_config":"MitoNet_v1"}, (100, 100)),
        ({"model_config":"NucleoNet_base_v1"}, (100, 100)),
         ({"fine_boundaries":True}, (100, 100)),
         ({"semantic_only":True}, (100, 100)),
         ({"fill_holes_in_segmentation":True}, (100, 100)),
         ({"batch_mode":True}, (100,)),
         ({"use_gpu":True}, (100, 100)),
         ({"use_quantized":True}, (100, 100)),
         ({"viewport":True}, (99, 99)),
         ({"confine_to_roi":True}, (19, 14)),
         ({"output_to_layer":True}, (100, 100))],
         ids=["tutorial_params","DropNet", "MitoNet", "NucleoNet", "fine_boundaries", "semantic_only", 
              "fill_holes_in_segmentation", "batch_mode", "use_gpu", "use_quantized", 
              "viewport", "confine_to_roi", "output_to_layer"])
def test_slice_inference(make_napari_viewer_proxy, image_2d, test_args, expected_shape):
    viewer = make_napari_viewer_proxy()
    image_layer = viewer.add_image(image_2d)
    if "model_config" not in test_args.keys():
        test_args["model_config"] = "MitoNet_v1_mini"

    if "output_to_layer" in test_args.keys():
        output_layer = viewer.add_image(np.zeros_like(image_2d))
        test_args["output_layer"] = output_layer

    if "confine_to_roi" in test_args.keys():
        triangle = np.array([[11, 13], [30, 6], [30, 20]])
        viewer.add_shapes(triangle,
                          shape_type='polygon',
                          edge_width=5,
                          edge_color='coral',
                          face_color='royalblue')

    inference_config = SliceInferenceWidget(viewer=viewer,
                                    image_layer=image_layer,
                                    **test_args)
    results = inference_config.config_and_run_inference(use_thread=False)
    seg = results[0]

    assert isinstance(seg, np.ndarray)
    assert np.asarray(seg).shape == expected_shape


@pytest.mark.parametrize(("test_args", "expected_shape"),
    [   
        ({"model_config":"MitoNet_v1", "batch_mode":True, "use_gpu":True}, (100, 100, 100))
        ({"model_config":"DropNet_base_v1"}, (100, 100, 100)),
        ({"model_config":"MitoNet_v1"}, (100, 100, 100)),
        ({"model_config":"NucleoNet_base_v1"}, (100, 100, 100)),
         ({"use_gpu":True}, (100, 100, 100)), 
         ({"use_quantized":True}, (100, 100, 100)),
         ({"multigpu":True}, (100, 100, 100)),
         ({"fine_boundaries":True}, (100, 100, 100)),
         ({"semantic_only":True}, (100, 100, 100)),
         ({"fill_holes_in_segmentation":True}, (100, 100, 100)),
         ({"allow_one_view":True}, (100, 100, 100))],
         ids=["tutorial_params", "DropNet", "MitoNet", "NucleoNet", "use_gpu", "use_quantized",
              "multigpu", "fine_boundaries", "semantic_only", 
              "fill_holes_in_segmentation", "allow_one_view"])
def test_volume_stack_inference(make_napari_viewer_proxy, image_3d, test_args, expected_shape):
    viewer = make_napari_viewer_proxy()
    image_layer = viewer.add_image(image_3d)
    inference_plane = 'xy'
    if "model_config" not in test_args.keys():
        test_args["model_config"] = "MitoNet_v1_mini"

    inference_config = VolumeInferenceWidget(viewer=viewer,
                                    image_layer=image_layer,
                                    return_panoptic=True,
                                    inference_plane=inference_plane,
                                    **test_args)
    
    stack, axis_name, trackers_dict = inference_config.config_and_run_inference(use_thread=False)
    assert isinstance(stack, np.ndarray)
    assert stack.shape == expected_shape


@pytest.mark.parametrize(("test_args", "expected_shape"),
    [
        ({"model_config":"DropNet_base_v1"}, (100, 100, 100)),
        ({"model_config":"MitoNet_v1"}, (100, 100, 100)),
        ({"model_config":"NucleoNet_base_v1"}, (100, 100, 100)),
         ({"use_gpu":True}, (100, 100, 100)), 
         ({"use_quantized":True}, (100, 100, 100)),
         ({"multigpu":True}, (100, 100, 100)),
         ({"fine_boundaries":True}, (100, 100, 100)),
         ({"semantic_only":True}, (100, 100, 100)),
         ({"fill_holes_in_segmentation":True}, (100, 100, 100)),
         ({"allow_one_view":True}, (100, 100, 100))],
         ids=["DropNet", "MitoNet", "NucleoNet", "use_gpu", "use_quantized",
              "multigpu", "fine_boundaries", "semantic_only", 
              "fill_holes_in_segmentation", "allow_one_view"])
def test_volume_inference_orthoplane_inference(make_napari_viewer_proxy, image_3d, test_args, expected_shape):   
    viewer = make_napari_viewer_proxy()
    image_layer = viewer.add_image(image_3d)
    if "model_config" not in test_args.keys():
        test_args["model_config"] = "MitoNet_v1_mini"

    inference_config = VolumeInferenceWidget(viewer=viewer,
                                    image_layer=image_layer,
                                    use_gpu=True,
                                    return_panoptic=True,
                                    orthoplane=True,
                                    **test_args)
    
    result = inference_config.config_and_run_inference(use_thread=False)
    for stack, axis_name in result:
        assert isinstance(stack, np.ndarray)
        assert stack.shape == expected_shape