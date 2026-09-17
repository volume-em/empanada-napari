try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

import importlib.util
import os
import platform
from pathlib import Path


def _point_bundled_libomp_at_conda():
    r"""Make pip wheels reuse conda-forge's single OpenMP runtime.

    Mixed conda + pip installs leave multiple libomp copies on disk
    (conda llvm-openmp, torch/lib/libomp, sklearn/.dylibs/libomp, ...).
    KMP_DUPLICATE_LIB_OK only skips OMP Error #15; during real inference
    the two runtimes still collide (OMP Error #179 pthread_mutex_init).
    When this Python env provides libomp, point known pip-bundled copies at
    that one file so the process loads a single runtime.
    """
    system = platform.system()
    if system == "Darwin":
        lib_name = "libomp.dylib"
    elif system == "Linux":
        lib_name = "libomp.so"
    else:
        return

    # Prefer sys.prefix (the env that is actually running). CONDA_PREFIX can
    # still point at base when someone launches env/bin/python without activate.
    import sys

    libomp_candidates = [
        Path(sys.prefix) / "lib" / lib_name,
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        libomp_candidates.append(Path(conda_prefix) / "lib" / lib_name)

    shared_libomp = next((p for p in libomp_candidates if p.is_file()), None)
    if shared_libomp is None:
        return

    bundled_paths = []
    for module_name, relative in (
        ("torch", Path("lib") / lib_name),
        ("sklearn", Path(".dylibs") / lib_name),
    ):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
        if spec is None or not spec.origin:
            continue
        bundled_paths.append(Path(spec.origin).resolve().parent / relative)

    shared_resolved = shared_libomp.resolve()
    for bundled in bundled_paths:
        try:
            if not bundled.exists() and not bundled.is_symlink():
                continue
            if bundled.resolve() == shared_resolved:
                continue
            bundled.unlink(missing_ok=True)
            bundled.symlink_to(shared_libomp)
        except OSError:
            # Read-only env or missing parent dir: fall back to KMP flag below.
            continue


_point_bundled_libomp_at_conda()

# Fallback if we could not unify libomp copies (no conda, permissions, etc.).
# Prefer the symlink path above: this alone can still hit OMP Error #179
# under heavy threaded inference.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.multiprocessing as mp

# Fix macOS (Darwin) child processes (e.g. the 3D inference matcher process,
# or DataLoader/training workers) forking a process that already has Cocoa /
# CoreFoundation loaded by napari's Qt GUI. A bare fork() in that state is
# unsafe and can silently hang the child (leaving the plugin's progress bar
# stuck forever). The 'spawn' start method avoids this by using fork+exec.
#
# This must run as early as possible (before any widget code has a chance to
# implicitly create a multiprocessing object, e.g. via a joblib/dask backend),
# since Python's multiprocessing context can only be set once per process.
# `force=True` guarantees 'spawn' wins even if something already set a
# default context before this module was imported.
if platform.system() == "Darwin":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

if torch.backends.quantized.engine in (None or 'none'):
    if 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'

from ._slice_inference import slice_dock_widget
from ._volume_inference import volume_dock_widget
from ._finetune import finetuning_dock_widget, get_info_dock_widget
from ._train import training_dock_widget
from ._register_model import register_model_dock_widget
from ._pick_patches import pick_patches_widget, store_dataset_widget
from ._merge_split_widget import (
    morph_labels_widget, delete_labels_widget, split_labels_widget,
    merge_labels_widget, jump_to_label_widget,
    find_next_available_label_widget
)
from ._export_batch_segs import export_batch_segs_widget
from ._label_counter_widget import label_counter_widget

__all__ = [
    'slice_dock_widget',
    'volume_dock_widget',
    'finetune_dock_widget',
    'training_dock_widget',
    'register_model_dock_widget',
    'get_info_dock_widget',
    'pick_patches_widget',
    'store_dataset_widget',
    'merge_labels_widget',
    'split_labels_widget',
    'delete_labels_widget',
    'morph_labels_widget',
    'jump_to_label_widget',
    'find_next_available_label_widget',
    'export_batch_segs_widget',
    'label_counter_widget'
]

