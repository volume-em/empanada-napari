import os
import re
import pytest
from empanada_napari.utils import get_configs


'''Skip Benchmarking Tests unless specifically called'''
def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    # If -m wasn't given OR it doesn't mention "benchmark"
    if not config.getoption("-m") or "benchmark" not in config.getoption("-m"):
        skip_bench = pytest.mark.skip(reason="use `-m benchmark` to run benchmark tests")

        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_bench)

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
                
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False

    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


'''Generate Parameters For test_button_widgets inference tests & benchmarking'''
MODEL_NAMES = {} 
for name in list(get_configs().keys()):
    basenm = re.sub(r'_.*[_]?[\d]' , '', name, count=1) # MitoNet, DropNet, NucleoNet, MitoNet_mini
    MODEL_NAMES[basenm] = name

common_params = [
        {"model_config":MODEL_NAMES['MitoNet']},
        {"model_config":MODEL_NAMES['DropNet']},
        {"model_config":MODEL_NAMES['NucleoNet']},
        {"fine_boundaries":True},
        {"semantic_only":True},
        {"fill_holes_in_segmentation":True},
        {"use_quantized":True}]

def gen_slice_sanity_params():
    slice_test_args = [p.copy() for p in common_params] + [{"batch_mode":True}, {"use_gpu":True},
                    {"viewport":True}, {"confine_to_roi":True}, {"output_to_layer":True}]
    slice_test_args[0].update({"use_gpu":True})
    expect_shape = [(100, 100), (100, 100), (100, 100), (100, 100),
                     (100, 100), (100, 100), (100, 100), (100, 100),
                     (100, 100), (99, 99), (19, 14), (100, 100)]
    
    return list(zip(slice_test_args, expect_shape))

def gen_slice_dset_params():
    slice_dset_args = [p.copy() for p in common_params[:3]] + [{"model_config":MODEL_NAMES['MitoNet_mini']}]
    # expect_results = [3400, 840, 830, 2400]
    expect_results = [
        [130900, 148000, 153200, 132100, 159000, 160600, 144000, 145400, 154900, 139300], #MN
        [109700, 84500, 77000, 111700, 119700, 112600, 100900, 101800, 118600, 107900], #DN
        [391500, 581000, 865600, 213800, 195100, 776800, 592000, 533600, 271100, 139600], #NN
        [111500, 108000, 130900, 110500, 137400, 125600, 113000, 115100, 111300, 116600] #MN_m
        ]
    return list(zip(slice_dset_args, expect_results))

def gen_vol_sanity_params():
    vol_test_args = common_params + [{"use_gpu":True}, {"multigpu":True}, {"allow_one_view":True}]
    expect_shape = [(100, 100, 100), (100, 100, 100), (100, 100, 100), (100, 100, 100),
                      (100, 100, 100), (100, 100, 100), (100, 100, 100), (100, 100, 100),
                      (100, 100, 100), (100, 100, 100)]
    
    return list(zip(vol_test_args, expect_shape))

def gen_vol_dset_params():
    vol_dset_args = common_params[:3] + [{"model_config":MODEL_NAMES['MitoNet_mini']}]
    # expect_results = [120, 0, 1, 100]
    expect_results = [
        [180100, 156900, 88600, 43600, 58300, 63500, 78700, 28900, 60800, 32100], #MN
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #DN
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #NN
        [106300, 104300, 81900, 27800, 50600, 67600, 96100, 65400, 38100, 37600] #MN_m
        ]

    return list(zip(vol_dset_args, expect_results))


def gen_ortho_dset_params():
    vol_dset_args = common_params[:3] + [{"model_config":MODEL_NAMES['MitoNet_mini']}]
    # expect_results = [120, 0, 1, 100]
    mito_counts = [[180100, 156900, 88600, 43600, 58300, 63500, 78700, 28900, 60800, 32100], #xy
                   [124700, 23300, 117300, 114300, 62400, 85800, 68400, 130600, 57400, 34800], #xz
                   [95200, 143300, 40300, 40000, 30000, 154500, 32500, 122500, 78100, 37000]] #yz

    drop_counts = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    nucleo_counts = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    
    mito_mini_counts = [[106300, 104300, 81900, 27800, 50600, 67600, 96100, 65400, 38100, 37600],
                        [97000, 40900, 81500, 63500, 65500, 63200, 65500, 116200, 55100, 54700],
                        [54800, 101500, 17900, 24200, 59600, 69300, 78800, 80800, 87300, 45200]]

    expect_results = [mito_counts, drop_counts, nucleo_counts, mito_mini_counts]

    return list(zip(vol_dset_args, expect_results))
