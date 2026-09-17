import os
import sys
import platform
import pytest
import subprocess
from pathlib import Path
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


def test_sets_kmp_duplicate_lib_ok_before_torch_import():
    r"""Regression test for: opening any empanada-napari widget (e.g. 2D
    Inference) aborted napari with OMP Error #15 (duplicate OpenMP
    runtime). Mixed conda-forge + pip wheels (especially PyTorch) load
    two copies of OpenMP on Mac, Linux, and Windows. The plugin must set
    KMP_DUPLICATE_LIB_OK before importing torch so the second runtime is
    allowed to load.
    """
    script = (
        "import builtins, os\n"
        "os.environ.pop('KMP_DUPLICATE_LIB_OK', None)\n"
        "real_import = builtins.__import__\n"
        "def checking_import(name, *args, **kwargs):\n"
        "    if name == 'torch' or name.startswith('torch.'):\n"
        "        assert os.environ.get('KMP_DUPLICATE_LIB_OK') == 'TRUE', "
        "os.environ.get('KMP_DUPLICATE_LIB_OK')\n"
        "        builtins.__import__ = real_import\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = checking_import\n"
        "import empanada_napari\n"
        "assert os.environ.get('KMP_DUPLICATE_LIB_OK') == 'TRUE'\n"
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


def test_openmp_workaround_does_not_override_user_env():
    """Users must be able to unset/override KMP_DUPLICATE_LIB_OK themselves."""
    script = (
        "import os\n"
        "os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'\n"
        "import empanada_napari\n"
        "assert os.environ.get('KMP_DUPLICATE_LIB_OK') == 'FALSE'\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "FALSE"},
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


@pytest.mark.skipif(
    platform.system() not in ("Darwin", "Linux"),
    reason="libomp bundling clash is macOS/Linux specific.",
)
def test_points_torch_libomp_at_env_libomp_when_both_exist():
    r"""Regression for OMP Error #179 after KMP_DUPLICATE_LIB_OK alone.

    Allowing two OpenMP runtimes past Error #15 still fails during
    threaded inference (pthread_mutex_init). The plugin must make
    torch's bundled libomp resolve to the env's single libomp when
    present (typical conda-forge + pip torch install).
    """
    lib_name = "libomp.dylib" if platform.system() == "Darwin" else "libomp.so"
    env_libomp = Path(sys.prefix) / "lib" / lib_name
    if not env_libomp.is_file():
        pytest.skip(f"No env {lib_name} at {env_libomp}")

    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "import torch\n"
        f"lib_name = {lib_name!r}\n"
        "torch_omp = Path(torch.__file__).resolve().parent / 'lib' / lib_name\n"
        "env_omp = Path(sys.prefix) / 'lib' / lib_name\n"
        "assert torch_omp.resolve() == env_omp.resolve(), "
        "(str(torch_omp.resolve()), str(env_omp.resolve()))\n"
        "print('OK')\n"
    )
    # Import empanada_napari first in the subprocess so unification runs.
    script = (
        "import empanada_napari\n" + script
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={k: v for k, v in os.environ.items() if k != "CONDA_PREFIX"},
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout
