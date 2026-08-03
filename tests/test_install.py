import os
import sys
import platform
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


@pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="This regression only applies to macOS (Darwin) fork/CoreFoundation safety.",
)
def test_macos_forces_spawn_start_method_even_if_fork_locked_in_first():
    r"""Regression test for: after running the fine-tuning/patch-creation
    workflow, 3D inference's matcher subprocess would hang (progress bar
    stuck forever, no new layer added) with a
    'process has forked ... you MUST exec()' CoreFoundation warning.

    Root cause: on macOS, forking a process that already has Cocoa/
    CoreFoundation loaded (as any napari/Qt GUI does) is unsafe and can hang
    the forked child. `empanada_napari` guards against this by forcing the
    'spawn' multiprocessing start method (which uses fork+exec, not a bare
    fork). But Python's start method can only be set once per process
    *unless* `force=True` is used - if anything else (e.g. a dependency
    used while creating/saving training patches) implicitly locks in a
    different context first, an unguarded `set_start_method('spawn')` call
    silently no-ops (swallowed by `except RuntimeError: pass`), leaving the
    unsafe method in place.

    This runs in an isolated subprocess (multiprocessing's start method is
    process-global and can't be reset), first locking the context to
    'fork', then importing empanada_napari and asserting it still won.
    """
    script = (
        "import multiprocessing as std_mp\n"
        "std_mp.set_start_method('fork')\n"
        "import torch.multiprocessing as mp\n"
        "import empanada_napari\n"
        "assert mp.get_start_method() == 'spawn', "
        "f'expected spawn, got {mp.get_start_method()!r}'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
