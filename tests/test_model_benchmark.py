import os
import subprocess
import pytest
from tifffile import imread
from empanada_napari.utils import get_configs
from empanada_napari._slice_inference import SliceInferenceWidget
from empanada_napari._volume_inference import VolumeInferenceWidget

DATA_DIR = "datasets_for_tests"
FILE_2D = "nanotomy_islet_rat375_crop1.tif"
FILE_3D = "hela_cell_em.tif"
MODEL_NAMES = list(get_configs().keys()) # DropNet, MitoNet, MitoNet_mini, NucleoNet


# ---------------- Dataset Fixture ----------------
@pytest.fixture
def tutorial_3d_image():
    dataset_3d = "https://zenodo.org/records/15311513"
    datapath =  os.path.join(DATA_DIR, FILE_3D)
    if os.path.isfile(datapath) is False:
        subprocess.run(['zenodo_get', dataset_3d, "-o", DATA_DIR])
    image = imread(datapath)
    print(type(image), image.shape)
    return image

# ---------------- Tests ----------------
class TestSliceInference:   
    @pytest.fixture
    def tutorial_2d_image(self):
        dataset_2d = "https://zenodo.org/records/15319873"
        datapath =  os.path.join(DATA_DIR, FILE_2D)
        if os.path.isfile(datapath) is False:
            subprocess.run(['zenodo_get', dataset_2d, "-o", DATA_DIR])
        image = imread(datapath)
        print(type(image), image.shape)
        return image


    @pytest.mark.parametrize("test_args",
    [
        ({"model_config":MODEL_NAMES[0]}),
        ({"model_config":MODEL_NAMES[1]}),
        ({"model_config":MODEL_NAMES[2]}),
        ({"model_config":MODEL_NAMES[3]})],
         ids=["tutorial_params", "DropNet", "MitoNetMini", "NucleoNet"])
    def test_slice_inference_benchmark(self, make_napari_viewer_proxy, tutorial_2d_image, test_args, benchmark):
        viewer = make_napari_viewer_proxy()
        image_layer = viewer.add_image(tutorial_2d_image)
        inference_config = SliceInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        use_gpu=True,
                                        **test_args)
        benchmark(inference_config.config_and_run_inference, use_thread=False)


class TestVolumeInferenceStack:
    
    @pytest.mark.parametrize("test_args",
    [
        ({"model_config":MODEL_NAMES[0]}),
        ({"model_config":MODEL_NAMES[1]}),
        ({"model_config":MODEL_NAMES[2]}),
        ({"model_config":MODEL_NAMES[3]})],
         ids=["DropNet", "MitoNet", "MitoNet_v1_mini", "NucleoNet"])
    def test_volume_stack_inference_benchmark(self, make_napari_viewer_proxy, tutorial_3d_image, test_args, benchmark):
        viewer = make_napari_viewer_proxy()
        image_layer = viewer.add_image(tutorial_3d_image)
        inference_plane = 'xy'
        inference_config = VolumeInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        return_panoptic=True,
                                        use_gpu=True,
                                        inference_plane=inference_plane,
                                        **test_args)
        benchmark(inference_config.config_and_run_inference, use_thread=False)


class TestVolumeInferenceOrthoplane:        

    @pytest.mark.parametrize("test_args",
    [
        ({"model_config":MODEL_NAMES[0]}),
        ({"model_config":MODEL_NAMES[1]}),
        ({"model_config":MODEL_NAMES[2]}),
        ({"model_config":MODEL_NAMES[3]})],
         ids=["DropNet", "MitoNet", "MitoNet_v1_mini", "NucleoNet"])
    def test_volume_stack_inference_benchmark(self, make_napari_viewer_proxy, tutorial_3d_image, test_args, benchmark):
        viewer = make_napari_viewer_proxy()
        image_layer = viewer.add_image(tutorial_3d_image)
        inference_config = VolumeInferenceWidget(viewer=viewer,
                                        image_layer=image_layer,
                                        use_gpu=True,
                                        return_panoptic=True,
                                        orthoplane=True,
                                        **test_args)
        benchmark(inference_config.config_and_run_inference, use_thread=False)