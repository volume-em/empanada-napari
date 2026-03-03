import os
import pytest
import subprocess
from importlib.metadata import distributions
    
def pytest_report_header(config):
    import datetime
    import platform
    import sys
    return [
        f"Test run: {datetime.datetime.now().isoformat()}",
        f"Hostname: {platform.node()}",
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {sys.version.split()[0]}"
    ]

def test_module_is_installed():
    packages = [dist.metadata.get("Name") for dist in distributions()]
    required = ["empanada-napari", "torch", "napari"]
    missing = [pkg for pkg in required if pkg not in packages]
    assert not missing, f"Missing packages: {', '.join(missing)}"

def test_module_imports():
    try:
        import napari
        import torch
        import empanada_napari
    except ImportError as e:
        pytest.fail(f"Failed to import required module: {e}")

@pytest.mark.gpu
@pytest.mark.dependency(name="nvidia_driver")
def test_nvidia_driver_available():
    try:
        result = subprocess.check_output(
            ["nvidia-smi"], 
            stderr=subprocess.STDOUT,
            timeout=2
        )
        assert "CUDA" in str(result)
    except Exception as e:
        pytest.fail(f"NVIDIA driver not found ({e}) - GPU acceleration unavailable")

@pytest.mark.gpu
@pytest.mark.dependency()
def test_torch_cuda_available():
    import torch
    if torch.version.cuda is None:
        pytest.skip("PyTorch not built with CUDA - GPU acceleration unavailable")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available to PyTorch - GPU acceleration unavailable")
    
    print(f"\nPyTorch CUDA version: {torch.version.cuda}")
    print(f"CUDA devices available: {torch.cuda.device_count()}")


def test_display_set():
    import os
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping in GitHub Actions")
    if not os.environ.get("DISPLAY"):
        pytest.fail("DISPLAY unset - napari GUI unavailable")
