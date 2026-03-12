import pytest
import torch

@pytest.mark.gpu
@pytest.mark.dependency(depends=["test_install.py::test_torch_cuda_available"])
def test_cuda_usable():
    """Verify CUDA runtime actually works"""
    try:
        x = torch.tensor([1.0], device="cuda")
        y = x * 2
        assert y.item() == 2.0
    except Exception as e:
        raise AssertionError(f"CUDA runtime failed: {e}")

@pytest.mark.parametrize("use_quantized", [(True), (False)])
@pytest.mark.parametrize("use_gpu", [(True), (False)])
def test_model_uses_correct_device(use_quantized, use_gpu):
    from empanada.config_loaders import read_yaml
    from empanada_napari.utils import get_configs
    from empanada_napari.utils import load_model_to_device

    MODEL_NAMES = list(get_configs().keys()) # DropNet, MitoNet, MitoNet_mini, NucleoNet
    model_config_name = MODEL_NAMES[1] 

    # Load the model config
    model_configs = get_configs()
    model_config = read_yaml(model_configs[model_config_name])

    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    if use_quantized and str(device) == "cpu" and model_config.get("model_quantized") is not None:
        model_url = model_config["model_quantized"]
    else:
        model_url = model_config["model"]

    model = load_model_to_device(model_url, device)
    model = model.to(device)
    param_device = next(model.parameters()).device # Test that the model loads to the correct device
  
    if device.type == "cpu":
        assert param_device.type == "cpu"
    else:
        assert param_device.type == "cuda"

