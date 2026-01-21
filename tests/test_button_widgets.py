import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from empanada_napari._volume_inference import _stack_inference, _orthoplane_inference
from empanada_napari.inference import Engine2d, Engine3d, stack_postprocessing, tracker_consensus
from empanada_napari.utils import get_configs
from empanada.config_loaders import read_yaml
from empanada_napari._slice_inference import SliceInferenceWidget


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

def test_slice_inf(make_napari_viewer_proxy, image_2d):
    viewer = make_napari_viewer_proxy()
    image_layer = viewer.add_image(image_2d)
    model_config = 'MitoNet_v1_mini'

    inference_config = SliceInferenceWidget(viewer=viewer,
                                    image_layer=image_layer,
                                    model_config=model_config)
    
    seg, axis, plane, y, x = inference_config.config_and_run_inference(use_thread=False)
    assert isinstance(seg, np.ndarray)
    assert seg.shape == image_2d.shape

    

def test_volume_inference_stack_inference(image_3d):   
    model_configs = get_configs()
    # model_name = list(model_configs.keys())[2]
    model_name = 'MitoNet_v1_mini'
    model_config = read_yaml(model_configs[model_name])

    engine = Engine3d(model_config) 
    inference_plane = 'xy'

    _, axis, trackers_dict = _stack_inference(engine, image_3d, inference_plane)
    worker = stack_postprocessing(trackers=trackers_dict, store_url=None, model_config=model_config)

    results = []
    worker.yielded.connect(lambda r: results.append(r))
    worker.run()
    stack_out, class_name, instances = results[-1]

    assert stack_out is not None
    assert stack_out.shape == image_3d.shape


def test_volume_inference_orthoplane_inference(image_3d):   
    model_configs = get_configs()
    model_name = 'MitoNet_v1_mini'
    model_config = read_yaml(model_configs[model_name])

    engine = Engine3d(model_config) 
    gen = _orthoplane_inference(engine, image_3d)

    while True:
        try:
            next(gen)
        except StopIteration as e:
            trackers_dict = e.value
            break

    worker = tracker_consensus(trackers=trackers_dict, store_url=None, model_config=model_config)
    
    results = []
    worker.yielded.connect(lambda r: results.append(r))
    worker.run()
    consensus_vol_out, class_name, consensus_instances = results[-1]

    assert consensus_vol_out is not None
    assert consensus_vol_out.shape == image_3d.shape
